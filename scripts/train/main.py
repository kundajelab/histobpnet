# Author: Lei Xiong <jsxlei@gmail.com>

"""
HistoBPNet Training Script

This script provides functionality for training, predicting, and interpreting HistoBPNet models.
It supports multiple model types including HistoBPNet, ChromBPNet, BPNet.

Key features:
- Training models with configurable hyperparameters
- Model prediction and evaluation
- Model interpretation and visualization
- Integration with Weights & Biases for experiment tracking
- Support for distributed training

Usage:
    python train.py train [options]  # For training
    python train.py predict [options]  # For prediction
    python train.py interpret [options]  # For interpretation
"""

import os
import argparse
import lightning as L
import torch
from lightning.pytorch.strategies import DDPStrategy  

from histobpnet.utils.general_utils import get_instance_id, set_random_seed
set_random_seed(seed = 42)
from histobpnet.model.model_config import ChromBPNetConfig
from histobpnet.model.model_wrappers import create_model_wrapper, load_pretrained_model, adjust_bias_model_logcounts
from histobpnet.data_loader.dataset import DataModule
from histobpnet.data_loader.data_config import DataConfig, DATA_DIR
from histobpnet.utils.metrics import compare_with_observed
from histobpnet.interpert.interpret import run_modisco_and_shap 
from histobpnet.logging.logger import create_logger

def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments shared across train, predict, and interpret commands.
    
    Args:
        parser: ArgumentParser instance to add arguments to
    """
    parser.add_argument('--fast_dev_run', action='store_true',
                       help='Run a quick development test')
    parser.add_argument('--version', '-v', type=str, default=None,
                       help='Version identifier for the run')
    parser.add_argument('--checkpoint', '-c', type=str, default=None,
                       help='Path to model checkpoint')
    parser.add_argument('--gpu', type=int, nargs='+', default=[0],
                       help='GPU device IDs to use')
    parser.add_argument('--shap', type=str, default='counts',
                       help='Type of SHAP analysis')
    parser.add_argument('--dev', action='store_true',
                       help='Run in development mode')
    parser.add_argument('--chrom', type=str, default='val',
                       help='Chromosome to analyze')
    parser.add_argument('--model_type', type=str, default='chrombpnet',
                       help='Type of model to use')
    parser.add_argument('--out_dir', '-o', type=str, default='output',
                       help='Output directory')
    parser.add_argument('--alpha', type=float, default=1,
                       help='Alpha value for the model')
    parser.add_argument('--beta', type=float, default=1,
                       help='Beta value for the model')
    parser.add_argument('--bias_scaled', type=str, default=os.path.join(DATA_DIR, 'bias_scaled.h5'))
    parser.add_argument('--adjust_bias', action='store_true', default=False,
                       help='Adjust bias model')
    parser.add_argument('--chrombpnet_wo_bias', type=str, default=None,
                       help='ChromBPNet model without bias')
    parser.add_argument('--verbose', action='store_true', default=False,
                       help='Verbose output')
    
    # Add model-specific arguments
    ChromBPNetConfig.add_argparse_args(parser)

    # Add data configuration arguments
    DataConfig.add_argparse_args(parser)

def get_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser.
    
    Returns:
        Configured ArgumentParser instance
    """
    parser = argparse.ArgumentParser(description='Train or test ChromBPNetBPNet model.')
    subparsers = parser.add_subparsers(dest='command', required=True)

    # Train sub-command
    train_parser = subparsers.add_parser('train', help='Train the ChromBPNet model.')
    train_parser.add_argument('--max_epochs', type=int, default=100,
                            help='Maximum number of training epochs')
    train_parser.add_argument('--precision', type=int, default=32,
                            help='Training precision (16, 32, or 64)')
    train_parser.add_argument('--use_wandb', action='store_true', default=True,
                            help='Use Weights & Biases logger')
    train_parser.add_argument('--logger', type=str, default='csv',
                            help='Logger type to use')
    add_common_args(train_parser)

    # Predict sub-command
    predict_parser = subparsers.add_parser('predict', help='Test or predict with the ChromBPNet model.')
    predict_parser.set_defaults(plot=True)
    add_common_args(predict_parser)

    # Interpret sub-command
    interpret_parser = subparsers.add_parser('interpret', help='Interpret the ChromBPNet model.')
    add_common_args(interpret_parser)

    return parser

def train(args):
    data_config = DataConfig.from_argparse_args(args)

    if os.path.exists(os.path.join(args.out_dir, 'checkpoints/best_model.ckpt')):
        raise ValueError(f"Model folder {args.out_dir}/checkpoints/best_model.ckpt already exists. Please delete the existing model or specify a new version.")
    log = create_logger(args.model_type, ch=True, fh=os.path.join(args.out_dir, 'train.log'), overwrite=True)
    log.info(args.out_dir)
    log.info(f'bias: {args.bias_scaled}')      
    log.info(f'adjust_bias: {args.adjust_bias}')
    log.info(f'data_type: {data_config.data_type}')
    log.info(f'in_window: {data_config.in_window}') 
    log.info(f'data_dir: {data_config.data_dir}')
    log.info(f'negative_sampling_ratio: {data_config.negative_sampling_ratio}')
    log.info(f'fold: {args.fold}')
    log.info(f'n_filters: {args.n_filters}')
    log.info(f'batch_size: {data_config.batch_size}')
    log.info(f'precision: {args.precision}')

    datamodule = DataModule(data_config)

    # what is this for? -> loss weighting, see model_wrappers.py
    args.alpha = datamodule.median_count / 10
    log.info(f'alpha: {args.alpha}')

    model = create_model_wrapper(args)
    if args.adjust_bias:
        adjust_bias_model_logcounts(model.model.bias, datamodule.negative_dataloader())

    loggers = [L.pytorch.loggers.CSVLogger(args.out_dir, name=args.model_type, version=f'fold_{args.fold}')]

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
        torch.save(model.model.state_dict(), os.path.join(args.out_dir, 'checkpoints/chrombpnet_wo_bias.pt'))

def load_model(args):
    if args.checkpoint is None:
        checkpoint = os.path.join(args.out_dir, 'checkpoints/best_model.ckpt')
        if not os.path.exists(checkpoint):
            print(f'No checkpoint found in {args.out_dir}/checkpoints/best_model.ckpt')
        else:
            args.checkpoint = checkpoint
            print(f'Loading checkpoint from {checkpoint}')
    model = load_pretrained_model(args)
    return model

# TODO review + metrics.py
def predict(args, model, datamodule=None):
    trainer = L.Trainer(logger=False, fast_dev_run=args.fast_dev_run, devices=args.gpu, val_check_interval=None) 
    log = create_logger(args.model_type, ch=True, fh=os.path.join(args.out_dir, f'predict.log'), overwrite=True)
    log.info(args.out_dir)
    log.info(f'{args.model_type}')

    dm = DataModule(DataConfig.from_argparse_args(args)) if datamodule is None else datamodule
    if args.chrom == 'all':
        chroms = dm.chroms
    else:
        chroms = [args.chrom]

    for chrom in chroms:
        dataloader, dataset = dm.chrom_dataloader(chrom)
        regions = dataset.regions
        log.info(f"Distribution of peaks/negatives for {chrom}: {regions['is_peak'].value_counts()}")
        compare_with_observed(trainer.predict(model, dataloader), regions, os.path.join(args.out_dir, 'evaluation', chrom))    

# TODO review + interpret.py
def interpret(args, model, datamodule=None):
    if datamodule is None:
        data_config = DataConfig.from_argparse_args(args)
        datamodule = DataModule(data_config)
    model.to(f'cuda:{args.gpu[0]}')

    tasks = ['profile', 'counts'] if args.shap == 'both' else [args.shap]
    for task in tasks:
        run_modisco_and_shap(model.model.model, data_config.peaks, out_dir=os.path.join(args.out_dir, 'interpret'), batch_size=args.batch_size,
            in_window=data_config.in_window, out_window=data_config.out_window, task=task, debug=args.debug)
    # out = model._mutagenesis(dataloader, debug=args.debug)
    # os.makedirs(os.path.join(out_dir, 'interpret'), exist_ok=True)
    # np.save(os.path.join(out_dir, 'interpret', 'mutagenesis.npy'), out)

def main():
    parser = get_parser()
    args = parser.parse_args()

    # Set up instance_id and output directory
    instance_id = get_instance_id()
    args.out_dir = os.path.join(args.out_dir, instance_id, f'fold_{args.fold}')
    os.makedirs(args.out_dir, exist_ok=False)

    # # TODO save configs
    # # copy config file from args.config to output dir
    # shutil.copy(args.config, os.path.join(output_dir, 'config.yaml'))

    # TODO
    # config["run_pid"] = os.getpid()
    # wandb.init(project="expression-models", name=config["run_name"]+"|"+instance_id, config=config)

    if args.command == 'train':
        train(args)
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

if __name__ == '__main__':
    main()