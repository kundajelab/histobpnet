import os
import argparse
from lightning.pytorch.strategies import DDPStrategy
import lightning as L
import torch
import json
from toolbox.utils import get_instance_id, set_random_seed
from toolbox.logger import SimpleLogger
from histobpnet.data_loader.data_config import DataConfig
from histobpnet.model.model_config import ChromBPNetConfig

# TODO do we need to set random seed before these imports?
from histobpnet.model.model_wrappers import create_model_wrapper, load_pretrained_model, adjust_bias_model_logcounts
from histobpnet.data_loader.dataset import DataModule
from histobpnet.eval.metrics import compare_with_observed, save_predictions, load_output_to_regions

# Example usage:
# python main.py train \
# --output_dir /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_chrombpnet_tuto/training \
# --peaks /large_storage/goodarzilab/valehvpa/projects/scCisTrans/for_chrombpnet_tuto/peaks_no_blacklist.bed \
# --negatives /large_storage/goodarzilab/valehvpa/projects/scCisTrans/for_chrombpnet_tuto/negatives/lei_negatives.bed \
# --fold_path /large_storage/goodarzilab/valehvpa/projects/scCisTrans/for_chrombpnet_tuto/splits/instance-20250727_104312/fold_0.json \
# --max_epochs 2 \
# --bias_scaled /large_storage/goodarzilab/valehvpa/projects/scCisTrans/for_chrombpnet_tuto/bias_model/ENCSR868FGK_bias_fold_0.h5 \
# --gpu 0 1 2 3

def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments shared across train, predict, and interpret commands.
    
    Args:
        parser: ArgumentParser instance to add arguments to
    """
    parser.add_argument('--output_dir', '-o', type=str, required=True,
                        help='Output directory')
    parser.add_argument('--fast_dev_run', action='store_true',
                       help='Run a quick development test')
    parser.add_argument('--name', type=str, default='',
                       help='Name of the run')
    parser.add_argument('--checkpoint', '-c', type=str, default=None,
                       help='Path to model checkpoint')
    parser.add_argument('--gpu', type=int, nargs='+', default=[0],
                       help='GPU device IDs to use')
    parser.add_argument('--chrom', type=str, default='test',
                       help='Chromosome type to analyze (e.g., train, val, test, all)')
    parser.add_argument('--model_type', type=str, default='chrombpnet',
                       help='Type of model to use')
    # what s bias_scaled as opposed to regular bias model? -> see https://github.com/kundajelab/chrombpnet/wiki/Output-format
    parser.add_argument('--bias_scaled', type=str, required=True,
                       help='Path to bias scaled model')
    parser.add_argument('--chrombpnet_wo_bias', type=str, default=None,
                       help='ChromBPNet model without bias')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')
    
    # Add model-specific arguments
    ChromBPNetConfig.add_argparse_args(parser)

    # Add data configuration arguments
    DataConfig.add_argparse_args(parser)

def get_parsers():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command', required=True)

    # train sub-command
    train_parser = subparsers.add_parser('train', help='Train the histobpnet model.')
    train_parser.add_argument('--max_epochs', type=int, default=100,
                            help='Maximum number of training epochs')
    train_parser.add_argument('--precision', type=int, default=32,
                            help='Training precision (16, 32, or 64)')
    # train_parser.add_argument('--gradient_clip', type=float, default=None,
    #                         help='Gradient clipping value')
    # train_parser.add_argument('--force', action='store_true', default=False,
    #                         help='Force training even if model already exists')
    train_parser.add_argument('--skip_wandb', action='store_true',
                            help='Do not use Weights & Biases logger')
    # TODO_later: is this actually used anywhere?
    train_parser.add_argument('--alpha', type=float, default=1,
                              help='Weight for count loss (profile loss will be weighted as 1-alpha).')
    train_parser.add_argument('--adjust_bias', action='store_true',
                              help='Adjust bias model')
    add_common_args(train_parser)

    # predict sub-command
    predict_parser = subparsers.add_parser('predict', help='Test or predict with the pre-trained histobpnet model.')
    predict_parser.set_defaults(plot=True) # always plot
    add_common_args(predict_parser)

    # predict_bias sub-command
    predict_bias_parser = subparsers.add_parser('predict_bias', help='Predict bias with the ChromBPNet model.')
    add_common_args(predict_bias_parser)

    # interpret sub-command
    interpret_parser = subparsers.add_parser('interpret', help='Interpret the pre-trained histobpnet model.')
    interpret_parser.add_argument('--shap', type=str, default='counts',
                                   help='Type of SHAP analysis')
    add_common_args(interpret_parser)

    # finetune sub-command
    finetune_parser = subparsers.add_parser('finetune', help='Finetune the ChromBPNet model.')
    add_common_args(finetune_parser)

    return parser

def validate_args(args_d: dict):
    pass

def setup(instance_id: str):
    parser = get_parsers()
    args = parser.parse_args() 

    validate_args(vars(args))

    # set up instance_id and output directory
    output_dir = os.path.join(args.output_dir, instance_id)
    os.makedirs(output_dir, exist_ok=False)

    # set up logger
    script_name = os.path.basename(__file__).replace(".py", "")
    logger = SimpleLogger(os.path.join(output_dir, f"{script_name}.log"))

    # save configs to disk in output dir
    with open(os.path.join(output_dir, 'args.json'), 'w') as f:
        # convert Namespace to dict
        args_d = vars(args)
        json.dump(args_d, f, indent=4)

    return args, output_dir, logger

def train(args, output_dir: str, logger):
    logger.add_to_log(f"Training with model type: {args.model_type}")
    logger.add_to_log(f'bias: {args.bias_scaled}')
    logger.add_to_log(f'adjust_bias: {args.adjust_bias}')
    logger.add_to_log(f'precision: {args.precision}')
    logger.add_to_log(f'alpha: {args.alpha}')
    logger.add_to_log(f'n_filters: {args.n_filters}')

    data_config = DataConfig.from_argparse_args(args)
    logger.add_to_log(f'data_type: {data_config.data_type}')
    logger.add_to_log(f'in_window: {data_config.in_window}') 
    logger.add_to_log(f'data_dir: {data_config.data_dir}')
    logger.add_to_log(f'negative_sampling_ratio: {data_config.negative_sampling_ratio}')
    logger.add_to_log(f'fold: {data_config.fold}')
    logger.add_to_log(f'batch_size: {data_config.batch_size}')

    # STOPPED HERE
    datamodule = DataModule(data_config, args)
    # what is this for? -> loss weighting, see model_wrappers.py
    args.alpha = datamodule.median_count / 10
    model = create_model_wrapper(args)
    if args.adjust_bias:
        adjust_bias_model_logcounts(model.model.bias, datamodule.negative_dataloader())

    loggers = [L.pytorch.loggers.CSVLogger(output_dir, name=args.name, version=f'fold_{args.fold}')]

    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        # valeh: why not 0?
        reload_dataloaders_every_n_epochs=1,
        check_val_every_n_epoch=1, # 5
        accelerator='gpu',
        devices=args.gpu,
        val_check_interval=None,
        strategy=DDPStrategy(find_unused_parameters=True),
        # So if early stopping kicks in after (say) epoch 17, then:
        # best_model.ckpt will be from the epoch with lowest val_loss up to epoch 17
        # last.ckpt will be the model as it was at epoch 17, the early-stopped point
        callbacks=[
            L.pytorch.callbacks.EarlyStopping(monitor='val_loss', patience=5),
            L.pytorch.callbacks.ModelCheckpoint(monitor='val_loss', save_top_k=1, mode='min', filename='best_model', save_last=True),
        ],
        logger=loggers,
        fast_dev_run=args.fast_dev_run,
        precision=args.precision,
        gradient_clip_val=args.gradient_clip,
    )
    trainer.fit(model, datamodule)
    if args.model_type == 'chrombpnet' and not args.fast_dev_run:
        # save the raw weights (state_dict) of the underlying model.
        # Even though Lightning’s ModelCheckpoint already saves .ckpt files:
        # .ckpt includes Lightning-specific training state (optimizer, scheduler, etc.)
        # saving just .state_dict() gives a clean PyTorch model that you can load via model.load_state_dict(...)
        # in a plain nn.Module
        torch.save(model.model.model.state_dict(), os.path.join(output_dir, f'checkpoints/{args.model_type}.pt'))

def load_model(args, output_dir: str):
    if args.checkpoint is None:
        checkpoint = os.path.join(output_dir, 'checkpoints/best_model.ckpt')
        if not os.path.exists(checkpoint):
            print(f'No checkpoint found in {output_dir}/checkpoints/best_model.ckpt')
        else:
            args.checkpoint = checkpoint
            print(f'Loading checkpoint from {checkpoint}')
    model = load_pretrained_model(args)
    return model

# TODO review + metrics.py
def predict(args, model, logger, datamodule=None, mode='predict'):
    out_dir = os.path.join(args.out_dir, args.name, f'fold_{args.fold}')
    os.makedirs(os.path.join(out_dir, mode), exist_ok=True)

    logger.add_to_log(f'out_dir: {out_dir}')
    logger.add_to_log(f'model_type: {args.model_type}')
    logger.add_to_log(f'checkpoint: {args.checkpoint}')
    logger.add_to_log(f'chrom: {args.chrom}')

    trainer = L.Trainer(logger=False, fast_dev_run=args.fast_dev_run, devices=args.gpu, val_check_interval=None)
    logger.add_to_log(args.out_dir)
    logger.add_to_log(f'{args.model_type}')

    if datamodule is None:
        data_config = DataConfig.from_argparse_args(args)
        dm = DataModule(data_config)
        logger.add_to_log(f'peaks: {data_config.peaks}')
        logger.add_to_log(f'negatives: {data_config.negatives}')
        logger.add_to_log(f'bigwig: {data_config.bigwig}')
        logger.add_to_log(f'fasta: {data_config.fasta}')
        logger.add_to_log(f'chrom_sizes: {data_config.chrom_sizes}')
    else:
        dm = datamodule

    chrom = args.chrom
    dataloader, dataset = dm.chrom_dataloader(args.chrom)
    # log.info(f"{chrom}: {regions['is_peak'].value_counts()}")
    output = trainer.predict(model, dataloader)
    regions, parsed_output = load_output_to_regions(output, dataset.regions, os.path.join(out_dir, mode, chrom))
    if mode == 'predict_bias':
        return
    model_metrics = compare_with_observed(regions, parsed_output, os.path.join(out_dir, mode, chrom))     
    save_predictions(output, regions, data_config.chrom_sizes, os.path.join(out_dir, mode, chrom))  

# TODO review + interpret.py
def interpret(args, args_d, model, datamodule=None):
    from histobpnet.interpert.interpret import run_modisco_and_shap 
    
    if datamodule is None:
        data_config = DataConfig.from_argparse_args(args)
        datamodule = DataModule(data_config, args)
    model.to(f'cuda:{args_d["gpu"][0]}')

    out_dir = os.path.join(args_d["output_dir"], args.name, f'fold_{args.fold}')

    tasks = ['profile', 'counts'] if args.shap == 'both' else [args.shap]
    for task in tasks:
        run_modisco_and_shap(
            model.model.model,
            data_config.peaks,
            out_dir=os.path.join(out_dir, 'interpret'),
            batch_size=args_d["batch_size"],
            in_window=data_config.in_window,
            out_window=data_config.out_window,
            task=task,
            debug=True,
        )
    # out = model._mutagenesis(dataloader, debug=args.debug)
    # os.makedirs(os.path.join(out_dir, 'interpret'), exist_ok=True)
    # np.save(os.path.join(out_dir, 'interpret', 'mutagenesis.npy'), out)

# TODO review
def finetune(args, logger):
    data_config = DataConfig.from_argparse_args(args)
    loggers=[L.pytorch.loggers.CSVLogger(args.out_dir, name=args.name, version=f'fold_{args.fold}')]
    out_dir = os.path.join(args.out_dir, args.name, 'finetune', f'fold_{args.fold}')
    os.makedirs(out_dir, exist_ok=True)

    if os.path.exists(os.path.join(out_dir, 'checkpoints/best_model.ckpt')) and not args.force:
        raise ValueError(f"Model folder {out_dir}/checkpoints/best_model.ckpt already exists. Please delete the existing model or specify a new version.")
    if args.bias_scaled is None:
        args.bias_scaled = os.path.join(args.data_dir, 'bias_scaled.h5')
    logger.add_to_log(f'out_dir: {out_dir}')
    logger.add_to_log(f'bias: {args.bias_scaled}')      
    logger.add_to_log(f'adjust_bias: {args.adjust_bias}')
    logger.add_to_log(f'data_type: {data_config.data_type}')
    logger.add_to_log(f'in_window: {data_config.in_window}') 
    logger.add_to_log(f'data_dir: {data_config.data_dir}')
    logger.add_to_log(f'negative_sampling_ratio: {data_config.negative_sampling_ratio}')
    logger.add_to_log(f'fold: {args.fold}')
    logger.add_to_log(f'n_filters: {args.n_filters}')
    logger.add_to_log(f'batch_size: {data_config.batch_size}')
    logger.add_to_log(f'precision: {args.precision}')

    datamodule = DataModule(data_config)

    args.alpha = datamodule.median_count / 10
    logger.add_to_log(f'alpha: {args.alpha}')

    model = load_model(args)
    if args.adjust_bias:
        adjust_bias_model_logcounts(model.model.bias, datamodule.negative_dataloader())

    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        reload_dataloaders_every_n_epochs=1,
        check_val_every_n_epoch=1, # 5
        accelerator='gpu',
        devices=args.gpu,
        val_check_interval=None,
        strategy=DDPStrategy(find_unused_parameters=True),
        callbacks=[
            L.pytorch.callbacks.EarlyStopping(monitor='val_loss', patience=5),
            L.pytorch.callbacks.ModelCheckpoint(monitor='val_loss', save_top_k=1, mode='min', filename='best_model', save_last=True),
        ],
        logger=loggers, # L.pytorch.loggers.TensorBoardLogger
        fast_dev_run=args.fast_dev_run,
        precision=args.precision,
        gradient_clip_val=args.gradient_clip,
        # precision="bf16"
    )
    trainer.fit(model, datamodule)
    if args.model_type == 'chrombpnet' and not args.fast_dev_run:
        torch.save(model.model.model.state_dict(), os.path.join(out_dir, 'checkpoints/chrombpnet_wo_bias.pt'))

def main(instance_id: str):
    args, output_dir, logger = setup(instance_id)

    # TODO_later
    # config["run_pid"] = os.getpid()
    # wandb.init(project="expression-models", name=config["run_name"]+"|"+instance_id, config=config)

    if args.command == 'train':
        train(args, output_dir, logger)
        model = load_model(args)
        # predict(args, model)
    elif args.command == 'predict':
        model = load_model(args)
        predict(args, model)
    elif args.command == 'interpret':
        model = load_model(args)
        interpret(args, model)
    elif args.command == 'finetune':
        finetune(args)
        model = load_model(args)
        predict(args, model)
    else:
        raise ValueError(f"Unknown command: {args.command}")

    logger.add_to_log("All done!")

if __name__ == '__main__':
    # get the instance id first so we can print it fast, then continue with the rest
    instance_id = get_instance_id()
    print(f"*** Using instance_id: {instance_id}")

    set_random_seed(seed=42)
    main(instance_id)