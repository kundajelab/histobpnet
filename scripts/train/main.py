import os
import argparse
import lightning as L
import torch
from lightning.pytorch.strategies import DDPStrategy  
import json
from toolbox.utils import get_instance_id, set_random_seed
from toolbox.logger import SimpleLogger
from histobpnet.data_loader.data_config import DataConfig
from histobpnet.model.model_config import ChromBPNetConfig

# TODO do we need to set random seed before these imports?
from histobpnet.model.model_wrappers import create_model_wrapper, load_pretrained_model, adjust_bias_model_logcounts
from histobpnet.data_loader.dataset import DataModule
from histobpnet.utils.metrics import compare_with_observed

# Example usage:
# TODO

def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments shared across train, predict, and interpret commands.
    
    Args:
        parser: ArgumentParser instance to add arguments to
    """
    parser.add_argument('--output_dir', '-o', type=str, required=True,
                        help='Output directory')
    parser.add_argument('--fast_dev_run', action='store_true',
                       help='Run a quick development test')
    # parser.add_argument('--version', '-v', type=str, default=None,
    #                    help='Version identifier for the run')
    # parser.add_argument('--dev', action='store_true',
    #                    help='Run in development mode')
    parser.add_argument('--checkpoint', '-c', type=str, default=None,
                       help='Path to model checkpoint')
    parser.add_argument('--gpu', type=int, nargs='+', default=[0],
                       help='GPU device IDs to use')
    parser.add_argument('--shap', type=str, default='counts',
                       help='Type of SHAP analysis')
    parser.add_argument('--chrom', type=str, default='val',
                       help='Chromosome type to analyze (e.g., train, val, test, all)')
    parser.add_argument('--model_type', type=str, default='chrombpnet',
                       help='Type of model to use')
    parser.add_argument('--alpha', type=float, default=1,
                       help='Weight for count loss (profile loss will be weighted as 1-alpha).')
    # TODO what s bias_scaled as opposed to regular bias model?
    parser.add_argument('--bias_scaled', type=str, required=True,
                       help='Path to bias scaled model')
    parser.add_argument('--adjust_bias', action='store_true',
                       help='Adjust bias model')
    # parser.add_argument('--chrombpnet_wo_bias', type=str, default=None,
    #                    help='ChromBPNet model without bias')
    # parser.add_argument('--verbose', action='store_true',
    #                    help='Verbose output')
    
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
    train_parser.add_argument('--skip_wandb', action='store_true',
                            help='Do not use Weights & Biases logger')
    add_common_args(train_parser)

    # predict sub-command
    predict_parser = subparsers.add_parser('predict', help='Test or predict with the pre-trained histobpnet model.')
    predict_parser.set_defaults(plot=True)
    add_common_args(predict_parser)

    # interpret sub-command
    interpret_parser = subparsers.add_parser('interpret', help='Interpret the pre-trained histobpnet model.')
    add_common_args(interpret_parser)

    return parser

def validate_args(args_d: dict):
    pass

def setup(instance_id: str):
    parser = get_parsers()
    args = parser.parse_args() 

    # convert Namespace to dict
    args_d = vars(args)

    validate_args(args_d)

    # set up instance_id and output directory
    output_dir = os.path.join(args_d["output_dir"], instance_id)
    os.makedirs(output_dir, exist_ok=False)

    # set up logger
    script_name = os.path.basename(__file__).replace(".py", "")
    logger = SimpleLogger(os.path.join(output_dir, f"{script_name}.log"))

    # save configs to disk in output dir
    with open(os.path.join(output_dir, 'args.json'), 'w') as f:
        json.dump(args_d, f, indent=4)

    return args, args_d, output_dir, logger

def train(args, output_dir: str, logger):
    mp = os.path.join(output_dir, 'checkpoints/best_model.ckpt')
    if os.path.exists(mp):
        raise ValueError(f"Model already exists at {mp}. Please delete the existing model or specify a new version.")

    data_config = DataConfig.from_argparse_args(args)
    logger.add_to_log(f"Training with model type: {args.model_type}")
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
    logger.add_to_log(f'alpha: {args.alpha}')

    datamodule = DataModule(data_config)
    # STOPPED HERE
    # what is this for? -> loss weighting, see model_wrappers.py
    args.alpha = datamodule.median_count / 10
    model = create_model_wrapper(args)
    if args.adjust_bias:
        adjust_bias_model_logcounts(model.model.bias, datamodule.negative_dataloader())

    loggers = [L.pytorch.loggers.CSVLogger(output_dir, name=args.model_type, version=f'fold_{args.fold}')]

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
        logger=loggers,
        fast_dev_run=args.fast_dev_run,
        precision=args.precision,
    )
    trainer.fit(model, datamodule)
    if args.model_type == 'chrombpnet' and not args.fast_dev_run:
        torch.save(model.model.state_dict(), os.path.join(output_dir, 'checkpoints/chrombpnet_wo_bias.pt'))

def load_model(args_d, output_dir: str):
    if args_d["checkpoint"] is None:
        checkpoint = os.path.join(output_dir, 'checkpoints/best_model.ckpt')
        if not os.path.exists(checkpoint):
            print(f'No checkpoint found in {output_dir}/checkpoints/best_model.ckpt')
        else:
            args_d["checkpoint"] = checkpoint
            print(f'Loading checkpoint from {checkpoint}')
    model = load_pretrained_model(args_d)
    return model

# TODO review + metrics.py
def predict(args, model, logger, datamodule=None):
    trainer = L.Trainer(logger=False, fast_dev_run=args_d["fast_dev_run"], devices=args_d["gpu"], val_check_interval=None)
    logger.add_to_log(args.out_dir)
    logger.add_to_log(f'{args.model_type}')

    dm = DataModule(DataConfig.from_argparse_args(args)) if datamodule is None else datamodule
    if args.chrom == 'all':
        chroms = dm.chroms
    else:
        chroms = [args.chrom]

    for chrom in chroms:
        dataloader, dataset = dm.chrom_dataloader(chrom)
        regions = dataset.regions
        logger.add_to_log(f"Distribution of peaks/negatives for {chrom}: {regions['is_peak'].value_counts()}")
        compare_with_observed(trainer.predict(model, dataloader), regions, os.path.join(args.out_dir, 'evaluation', chrom))    

# TODO review + interpret.py
def interpret(args, args_d, model, datamodule=None):
    from histobpnet.interpert.interpret import run_modisco_and_shap 
    
    if datamodule is None:
        data_config = DataConfig.from_argparse_args(args)
        datamodule = DataModule(data_config)
    model.to(f'cuda:{args_d["gpu"][0]}')

    tasks = ['profile', 'counts'] if args.shap == 'both' else [args.shap]
    for task in tasks:
        run_modisco_and_shap(
            model.model.model,
            data_config.peaks,
            out_dir=os.path.join(args_d["output_dir"], 'interpret'),
            batch_size=args_d["batch_size"],
            in_window=data_config.in_window,
            out_window=data_config.out_window,
            task=task,
            debug=args_d["debug"]
        )
    # out = model._mutagenesis(dataloader, debug=args.debug)
    # os.makedirs(os.path.join(out_dir, 'interpret'), exist_ok=True)
    # np.save(os.path.join(out_dir, 'interpret', 'mutagenesis.npy'), out)

def main(instance_id: str):
    args, _, output_dir, logger = setup(instance_id)

    # TODO
    # config["run_pid"] = os.getpid()
    # wandb.init(project="expression-models", name=config["run_name"]+"|"+instance_id, config=config)

    if args.command == 'train':
        train(args, output_dir, logger)
        model = load_model(args)
        predict(args, model)
    elif args.command == 'predict':
        model = load_model(args)
        predict(args, model)
    elif args.command == 'interpret':
        model = load_model(args)
        interpret(args, model)
    else:
        raise ValueError(f"Unknown command: {args.command}")

    logger.add_to_log("All done!")

if __name__ == '__main__':
    # get the instance id first so we can print it fast, then continue with the rest
    instance_id = get_instance_id()
    print(f"*** Using instance_id: {instance_id}")

    set_random_seed(seed=42)
    main(instance_id)