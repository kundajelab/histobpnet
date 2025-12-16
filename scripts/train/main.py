import os
import argparse
import sys
from lightning.pytorch.strategies import DDPStrategy
import lightning as L
import torch
import json
import time
import wandb
from toolbox.utils import get_instance_id, set_random_seed
from toolbox.logger import SimpleLogger

from histobpnet.data_loader.data_config import DataConfig
from histobpnet.model.model_config import BPNetModelConfig
from histobpnet.data_loader.dataset import DataModule
from histobpnet.eval.metrics import compare_with_observed, save_predictions, load_output_to_regions
from histobpnet.utils.general_utils import is_histone
from histobpnet.model.bpnet_wrapper import BPNetWrapper, ChromBPNetWrapper
from histobpnet.model.histobpnet_wrapper_v1 import HistoBPNetWrapperV1
from histobpnet.model.histobpnet_wrapper_v2 import HistoBPNetWrapperV2
from histobpnet.model.histobpnet_wrapper_v3 import HistoBPNetWrapperV3

def get_parsers():
    parser = argparse.ArgumentParser()

    parser.add_argument('--output_dir', '-o', type=str, required=True,
                        help='Output directory')
    parser.add_argument('--fast_dev_run', action='store_true',
                       help='Run a quick development test')
    parser.add_argument('--name', type=str, default='',
                       help='Name of the run')
    parser.add_argument('--command', type=str, required=True,
                       help='Command to execute: train, predict, interpret')
    parser.add_argument('--gpu', type=int, nargs='+', default=[0],
                       help='GPU device IDs to use')
    parser.add_argument('--chrom', type=str, default='test',
                       help='Chromosome type to analyze (e.g., train, val, test, all)')
    # what s bias_scaled as opposed to regular bias model? -> see https://github.com/kundajelab/chrombpnet/wiki/Output-format
    parser.add_argument('--bias_scaled', type=str, default=None,
                       help='Path to bias scaled model')
    parser.add_argument('--chrombpnet_wo_bias', type=str, default=None,
                       help='ChromBPNet model without bias')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to model checkpoint (.ckpt) for prediction/interpretation. If chrombpnet, this ckpt must' \
                       'include both the bias model and the accessibility model (bias_scaled and chrombpnet_wo_bias args are' \
                       'ignored if checkpoint is not None). Alternatively, provide bias_scaled and chrombpnet_wo_bias separately' \
                       'and leave checkpoint as None.')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')
    # TODO_later: is this actually used anywhere?
    parser.add_argument('--alpha', type=float, default=1,
                        help='Weight for count loss (profile loss will be weighted as 1-alpha).')
    parser.add_argument("--cvd", dest="cvd", default=None, help="Value to set for CUDA_VISIBLE_DEVICES")
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--optimizer_eps', type=float, default=1e-7,
                        help='Adam optimizer epsilon value')
    parser.add_argument('--skip_wandb', action='store_true',
                        help='Do not use Weights & Biases logger')
    parser.add_argument('--shap', type=str, default='counts',
                        help='Type of SHAP analysis')
    parser.add_argument('--max_epochs', type=int, default=100,
                        help='Maximum number of training epochs')
    parser.add_argument('--precision', type=int, default=32,
                        help='Training precision (16, 32, or 64)')
    parser.add_argument('--gradient_clip', type=float, default=None,
                        help='Gradient clipping value')
    parser.add_argument('--adjust_bias', action='store_true',
                        help='Adjust bias model')

    # Add model-specific arguments
    BPNetModelConfig.add_argparse_args(parser)

    # Add data configuration arguments
    DataConfig.add_argparse_args(parser)

    return parser

def validate_args(args_d: dict):
    # TODO validate that checkpoint ends in .ckpt if provided
    pass

def setup(instance_id: str):
    parser = get_parsers()
    args = parser.parse_args() 

    validate_args(vars(args))

    # set up instance_id and output directory
    output_dir = os.path.join(args.output_dir, instance_id)
    os.makedirs(output_dir, exist_ok=False)

    # set up wandb
    if not args.skip_wandb:
        run_config = vars(args)
        run_config["run_pid"] = os.getpid()
        wandb.init(project="histobpnet", name=args.name+"|"+instance_id, config=run_config)

    # set up logger
    script_name = os.path.basename(__file__).replace(".py", "")
    logger = SimpleLogger(os.path.join(output_dir, f"{script_name}.log"))

    # save configs to disk in output dir
    with open(os.path.join(output_dir, 'args.json'), 'w') as f:
        # convert Namespace to dict
        args_d = vars(args)
        json.dump(args_d, f, indent=4)
    
    # set CUDA_VISIBLE_DEVICES if specified
    if args.cvd is not None:
        print(f"Setting CUDA_VISIBLE_DEVICES to {args.cvd}")
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cvd
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        
    # log the command
    logger.add_to_log("CMD: " + " ".join(sys.argv) + "\n")
    
    return args, output_dir, logger

def train(args, output_dir: str, logger):
    logger.add_to_log(f"Training with model type: {args.model_type}")

    data_config = DataConfig.from_argparse_args(args)
    data_config.set_additional_args(args.output_bins, args.model_type)

    datamodule = DataModule(data_config, len(args.gpu))
    negative_dl = None
    if is_histone(args.model_type):
        args.alpha = 1
    elif args.model_type == 'chrombpnet':
        args.alpha = datamodule.median_count / 10
        negative_dl = datamodule.negative_dataloader() if args.adjust_bias else None
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
    model_wrapper = create_model_wrapper(args, dataloader=negative_dl)

    logger.add_to_log(f"Model config: {args}")

    pt_output_dir = os.path.join(output_dir, "pt_artifacts")
    os.makedirs(pt_output_dir, exist_ok=False)

    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        # valeh: why not 0? TODO_later ask Lei
        # what this does: it reloads the data loaders every n epochs (here every epoch)
        # useful if data augmentation or sampling changes every epoch
        reload_dataloaders_every_n_epochs=1,
        check_val_every_n_epoch=1, # 5
        accelerator='gpu',
        devices=args.gpu,
        # run validation only at the end of each epoch
        val_check_interval=None,
        # https://chatgpt.com/s/t_6930903a25608191b8ebbcb3242ae142
        strategy=DDPStrategy(find_unused_parameters=False),
        # So if early stopping kicks in after (say) epoch 17, then:
        # best_model.ckpt will be from the epoch with lowest val_loss up to epoch 17
        # last.ckpt will be the model as it was at epoch 17, the early-stopped point
        callbacks=[
            L.pytorch.callbacks.EarlyStopping(monitor='val_loss', patience=5),
            L.pytorch.callbacks.ModelCheckpoint(monitor='val_loss', save_top_k=1, mode='min', filename='best_model', save_last=True),
        ],
        logger=[
            L.pytorch.loggers.WandbLogger(save_dir=pt_output_dir),
            L.pytorch.loggers.CSVLogger(pt_output_dir)
        ],
        fast_dev_run=args.fast_dev_run,
        precision=args.precision,
        gradient_clip_val=args.gradient_clip,
    )
    trainer.fit(model_wrapper, datamodule)
    if not args.fast_dev_run:
        # save the raw weights (state_dict) of the underlying model.
        # Even though Lightning’s ModelCheckpoint already saves .ckpt files:
        # .ckpt includes Lightning-specific training state (optimizer, scheduler, etc.)
        # saving just .state_dict() gives a clean PyTorch model that you can load via model.load_state_dict(...)
        # into a plain nn.Module
        ckpt_dir = os.path.join(output_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=False)
        # TODO_later this currently only works for the histobpnet model where I've called the BPNet model bpnet
        # for other models I guess it might make sense to store a list or dict of sub-Modules and then save all
        # those here?
        torch.save(model_wrapper.model.bpnet.state_dict(), os.path.join(ckpt_dir, f'{args.model_type}.pt'))
        # TODO fix
        # torch.save(model.model.model.state_dict(), os.path.join(out_dir, 'checkpoints/chrombpnet_wo_bias.pt'))

    # return the path to the best_model.ckpt
    return os.path.join(trainer.checkpoint_callback.dirpath, 'best_model.ckpt')

def predict(args, output_dir: str, model, logger, mode='predict', chrom: str=None):
    trainer = L.Trainer(logger=False, accelerator='gpu', fast_dev_run=args.fast_dev_run, devices=args.gpu, val_check_interval=None)

    # TODO_later log these
    # print("accelerator:", trainer.accelerator)              # object, e.g. GPUAccelerator(...)
    # print("accelerator type:", trainer.accelerator.__class__.__name__)  # "GPUAccelerator", "CPUAccelerator", ...
    #
    # print("strategy:", trainer.strategy)                    # e.g. DDPStrategy(...)
    # print("strategy name:", trainer.strategy.__class__.__name__)
    #
    # print("num devices:", trainer.num_devices)              # int, e.g. 1, 2, 8
    # print("devices:", trainer.device_ids)                   # list of device indices, e.g. [0], [0, 1, 2, 3]
    #
    # print("global rank:", trainer.global_rank)              # 0 in single-process, 0..world_size-1 in DDP
    # print("local rank:", trainer.local_rank)                # rank within the node
    # print("world size:", trainer.world_size)                # total number of processes
    #
    # print("fast_dev_run:", trainer.fast_dev_run)            # True / False / int
    # print("max_epochs:", trainer.max_epochs)

    data_config = DataConfig.from_argparse_args(args)
    data_config.set_additional_args(output_bins=args.output_bins, model_type=args.model_type)
    dm = DataModule(data_config, len(args.gpu))
    model_config = model.get_model_config()

    chr = chrom if chrom is not None else args.chrom
    dataloader, dataset = dm.chrom_dataloader(chr)
    output = trainer.predict(model, dataloader)
    # the last batch may be smaller than batch_size (drop_last is False by default)
    assert (len(output[:-1]) * data_config.batch_size + output[-1]['pred_count'].shape[0]) == dataset.cur_seqs.shape[0]
    od = os.path.join(output_dir, mode, chr)
    os.makedirs(od, exist_ok=False)
    regions, parsed_output = load_output_to_regions(output, dataset.regions, od)
    skip_profile = is_histone(data_config.model_type)
    compare_with_observed(regions, parsed_output, od, skip_profile=skip_profile)
    # currently only predict counts for histobpnet
    if not skip_profile:
        save_predictions(output, regions, data_config.chrom_sizes, od, seqlen=model_config.out_dim)

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

def create_model_wrapper(args, checkpoint: str = None, dataloader = None):
    """Factory function to create appropriate model wrapper.
    """
    model_type = args.model_type.lower()
    if model_type == 'bpnet':
        assert checkpoint is None
        return BPNetWrapper(args)
    elif model_type == 'chrombpnet':
        if checkpoint is not None:
            assert checkpoint.endswith('.ckpt')
            # TODO do we need to call adjust_bias_model_log_counts here, or can it be already included in the checkpoint?
            return ChromBPNetWrapper.load_from_checkpoint(checkpoint, map_location='cpu')
        model_wrapper = ChromBPNetWrapper(args)
        # TODO do we need to call adjust_bias_model_log_counts here, or can it be already included in the checkpoint?
        model_wrapper.load_pretrained_chrombpnet(args.bias_scaled, args.chrombpnet_wo_bias, dataloader = dataloader)
    elif model_type == 'histobpnet_v1':
        if checkpoint is not None:
            assert checkpoint.endswith('.ckpt')
            return HistoBPNetWrapperV1.load_from_checkpoint(checkpoint, map_location='cpu')
        model_wrapper = HistoBPNetWrapperV1(args)
        model_wrapper.load_pretrained_chrombpnet(args.chrombpnet_wo_bias)
    elif model_type == 'histobpnet_v2':
        if checkpoint is not None:
            assert checkpoint.endswith('.ckpt')
            return HistoBPNetWrapperV2.load_from_checkpoint(checkpoint, map_location='cpu')
        model_wrapper = HistoBPNetWrapperV2(args)
        model_wrapper.load_pretrained_chrombpnet(args.chrombpnet_wo_bias)
    elif model_type == 'histobpnet_v3':
        if checkpoint is not None:
            assert checkpoint.endswith('.ckpt')
            return HistoBPNetWrapperV3.load_from_checkpoint(checkpoint, map_location='cpu')
        model_wrapper = HistoBPNetWrapperV3(args)
        model_wrapper.load_pretrained_chrombpnet(args.chrombpnet_wo_bias)
    else:
        raise ValueError(f"Unknown model type: {model_type}") 

    return model_wrapper

def main(instance_id: str):
    args, output_dir, logger = setup(instance_id)

    if args.command == 'train':
        best_model_ckpt = train(args, output_dir, logger)
        # model = create_model_wrapper(args, checkpoint=best_model_ckpt)
        predict(args, output_dir, model, logger, chrom="test")
    elif args.command == 'predict':
        model = create_model_wrapper(args, checkpoint=args.checkpoint)
        predict(args, output_dir, model, logger)
    elif args.command == 'interpret':
        model = create_model_wrapper(args, checkpoint=args.checkpoint)
        interpret(args, model)
    else:
        raise ValueError(f"Unknown command: {args.command}")

    return logger

if __name__ == '__main__':
    # get the instance id first so we can print it fast, then continue with the rest
    instance_id = get_instance_id()
    print(f"*** Using instance_id: {instance_id}")

    # set_random_seed(seed=42, skip_tf=True)
    # set_random_seed(seed=1234, skip_tf=True)
    L.seed_everything(1234)
    t0 = time.time()
    logger = main(instance_id)
    tt = time.time() - t0
    logger.add_to_log(f"All done! Time taken: {tt//3600:.0f} hrs, {(tt%3600)//60:.0f} mins, {tt%60:.1f} secs")