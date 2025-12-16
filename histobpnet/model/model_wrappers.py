from typing import Dict, Any, Union
import torch
from lightning import LightningModule
from lightning.pytorch.utilities.types import STEP_OUTPUT
import numpy as np
from tqdm import tqdm
from toolbox.plt_utils import density_scatter
from lightning.pytorch.loggers import WandbLogger
import wandb
import matplotlib.pyplot as plt

from histobpnet.model.bpnet import BPNet
from histobpnet.utils.general_utils import pearson_corr

class ModelWrapper(LightningModule):
    """A generic wrapper for different model architectures to be used with PyTorch Lightning.
    
    This wrapper provides a flexible interface for training, validation, and testing different model architectures
    while maintaining consistent logging and optimization strategies.
    """

    def __init__(
        self,
        args,
        **kwargs
    ):
        """Initialize the model wrapper.
        
        Args:
            model: The underlying model architecture
            alpha: Weight for count loss
            **kwargs: Additional arguments to be passed to the model
        """
        
        super().__init__()
        self.alpha = args.alpha
        # where is this set? And why isnt this just 1-alpha?
        # -> chrombpnet sets this to one, see
        # https://github.com/kundajelab/chrombpnet/blob/master/chrombpnet/training/models/bpnet_model.py#L94-L97
        # it doesnt have to sum to 1 either, apparently
        self.beta = 1
        self.verbose = args.verbose
        self.lr = args.lr
        self.optimizer_eps = args.optimizer_eps

        # Initialize metrics storage
        self.metrics = {
            'train': {'preds': [], 'targets': [], 'peak_status': []},
            'val': {'preds': [], 'targets': [], 'peak_status': []},
            'test': {'preds': [], 'targets': [], 'peak_status': []},
            'predict': {'preds': [], 'targets': [], 'peak_status': []}
        }

        for k, v in kwargs.items():
            setattr(self, k, v)

        # Save hyperparameters
        self.save_hyperparameters(ignore=['model'])
    
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass through the model.
        
        Args:
            x: Input tensor
            **kwargs: Additional arguments to be passed to the model's forward method
            
        Returns:
            Model output
        """
        return self.model(x, **kwargs)

    def get_model_config(self):
        """Retrieve the model configuration.
        
        Returns:
            Model configuration object
        """
        return self.model.get_model_config()
    
    def configure_optimizers(self) -> Union[torch.optim.Optimizer, Dict[str, Any]]:
        # TODO_NOW validate that self.model will be the right model...
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, eps=self.optimizer_eps)
        return optimizer
    
    def _step(self, batch, batch_idx, mode='train'):
        raise NotImplementedError("Subclasses must implement this method")

    def init_bias(self, bias: str, dataloader=None, verbose=False, device=1, instance=None):
        print(f"Loading bias model from {bias}")
        bias_model = BPNet.from_keras(bias, name='bias', instance=instance)

        # Freeze the sub-model
        bias_model.eval()
        for param in bias_model.parameters():
            param.requires_grad = False

        if dataloader is not None:
            bias_model = adjust_bias_model_logcounts(bias_model, dataloader, verbose=verbose, device=device)

        return bias_model

    def init_chrombpnet_wo_bias(self, chrombpnet_wo_bias: str, freeze=True, instance=None):
        print(f"Loading chrombpnet_wo_bias model from {chrombpnet_wo_bias}")
        if chrombpnet_wo_bias.endswith('.h5'):
            # note: this can only load a chrombpnet_wo_bias, not a full chrombpnet...
            model = BPNet.from_keras(chrombpnet_wo_bias, instance=instance)
        else:
            raise ValueError("File format not recognized for chrombpnet_wo_bias")

        if freeze:
            model.eval()
            for param in model.parameters():
                param.requires_grad = False

        return model

    def _predict_on_dataloader(self, dataloader, func, **kwargs):
        raise ValueError("Putting this here for now so I know when it's called!")

        outs = []
        for batch in dataloader:
            out = func(self, batch, **kwargs)
            outs.append(out.detach().cpu())
        return torch.cat(outs, dim=0)

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> STEP_OUTPUT:
        """Training step.
        
        Args:
            batch: Dictionary containing batch data
            batch_idx: Index of the current batch
            
        Returns:
            Loss value
        """
        return self._step(batch, batch_idx, 'train')
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> STEP_OUTPUT:
        """Validation step.
        
        Args:
            batch: Dictionary containing batch data
            batch_idx: Index of the current batch
            
        Returns:
            Loss value
        """
        return self._step(batch, batch_idx, 'val')
    
    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> STEP_OUTPUT:
        """Test step.
        
        Args:
            batch: Dictionary containing batch data
            batch_idx: Index of the current batch
            
        Returns:
            Loss value
        """
        return self._step(batch, batch_idx, 'test')
    
    def predict_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, np.ndarray]:
        """Prediction step.
        
        Args:
            batch: Dictionary containing batch data
            batch_idx: Index of the current batch
            
        Returns:
            Dictionary containing predictions and true values
        """
        return self._step(batch, batch_idx, 'predict')

    def _epoch_end(self, mode: str) -> None:
        """Handle end of epoch operations.
        
        Args:
            mode: Mode of operation ('train', 'val', or 'test')
        """
        # Concatenate predictions and targets across batches
        all_preds = torch.cat(self.metrics[mode]['preds'])
        all_targets = torch.cat(self.metrics[mode]['targets'])
        all_peak_status = torch.cat(self.metrics[mode]['peak_status'])
        
        if self.model_type != "histobpnet_v1":
            # Calculate and log correlation
            pr = pearson_corr(all_preds.reshape(-1), all_targets.reshape(-1), eps=0)
            self.log(f"{mode}_count_pearson", pr, prog_bar=True, logger=True, sync_dist=True)

            print(f"{mode} epoch end: sanity_checking={self.trainer.sanity_checking}", flush=True)
            if (self.trainer.sanity_checking is not None) and (not self.trainer.sanity_checking):
                print(f"{mode} epoch end: here", flush=True)
                at, ap, aps = all_targets.detach().cpu().numpy(), all_preds.detach().cpu().numpy(), all_peak_status.detach().cpu().numpy()
                fig, axes = plt.subplots(1, 3, figsize=(30, 5))
                for i, ax in enumerate(axes):
                    x = at if i == 0 else at[np.where(aps == 1)[0]] if i == 1 else at[np.where(aps == 0)[0]]
                    y = ap if i == 0 else ap[np.where(aps == 1)[0]] if i == 1 else ap[np.where(aps == 0)[0]]
                    suffix = "(All)" if i == 0 else "(Peaks)" if i == 1 else "(Non-peaks)"
                    if len(x) == 0:
                        continue
                    _, _, _, _, _ = density_scatter(
                        x,
                        y,
                        "Log Count Labels " + suffix,
                        "Log Count Predictions " + suffix,
                        s=5,
                        bins=200,
                        incl_stats=True,
                        ax=ax,
                    )
                self._log_plot(fig, name=f"{mode}_scatter")
        else:
            # Calculate and log correlation per bin
            assert all_preds.shape[1] == len(self.model.output_bins), "Mismatch between number of output bins and predictions"
            for i in range(len(self.model.output_bins)):
                pr = pearson_corr(all_preds[:, i].reshape(-1), all_targets[:, i].reshape(-1), eps=0)
                bin_size = str(self.model.output_bins[i])
                self.log(f"{mode}_count_pearson_{bin_size}bp", pr, prog_bar=True, logger=True, sync_dist=True)
                # TODO plot scatterplots..

        # Reset metrics storage
        self.metrics[mode]['preds'] = []
        self.metrics[mode]['targets'] = []
        self.metrics[mode]['peak_status'] = []
    
    def on_train_epoch_end(self) -> None:
        """Handle end of training epoch."""
        self._epoch_end('train')
    
    def on_validation_epoch_end(self) -> None:
        """Handle end of validation epoch."""
        self._epoch_end('val')
    
    def on_test_epoch_end(self) -> None:
        """Handle end of test epoch."""
        self._epoch_end('test')
    
    def _log_plot(self, fig, name: str, close_fig: bool = True):
        # Iterate through loggers to find WandB
        wandb_logger = None
        if isinstance(self.logger, WandbLogger):
            wandb_logger = self.logger
        elif hasattr(self.logger, 'experiment') and hasattr(self.logger.experiment, 'log'):
            # Fallback for single logger that might be WandB
            wandb_logger = self.logger
        elif hasattr(self.logger, 'loggers'):
            # Handle LoggerCollection (multiple loggers)
            for logger in self.logger.loggers:
                if isinstance(logger, WandbLogger):
                    wandb_logger = logger
                    break
        if wandb_logger:
            # I passed step=self.trainer.global_step so I could select epoch as x in wandb
            # but it seems to cause issues with logging val_ and train_scatters, only renders
            # the former, not sure why
            # wandb_logger.experiment.log({name: wandb.Image(fig)}, step=self.trainer.global_step)
            wandb_logger.experiment.log({name: wandb.Image(fig)})
            if close_fig:
                plt.close(fig)

# valeh: ?? TODO
def adjust_bias_model_logcounts(bias_model, dataloader, verbose=False, device=1):
    """
    Given a bias model, sequences and associated counts, the function adds a 
    constant to the output of the bias_model's logcounts that minimizes squared
    error between predicted logcounts and observed logcounts (inferred from 
    cts). This simply reduces to adding the average difference between observed 
    and predicted to the "bias" (constant additive term) of the Dense layer.
    Typically the seqs and counts would correspond to training nonpeak regions.
    ASSUMES model_bias's last layer is a dense layer that outputs logcounts. 
    This would change if you change the model.
    """
    print("Adjusting bias model counts")
    bias_model.eval()
    delta = []
    with torch.no_grad():
        bias_model.to('cuda')
        for batch in tqdm(dataloader):
            batch = {k: v.to('cuda') if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            _, pred_counts = bias_model(batch['onehot_seq'])
            true_counts = batch['profile'].sum(dim=-1).log1p()
            # _delta = out['true_count'].mean(-1) - out['pred_count'].mean(-1)
            _delta = true_counts.mean(-1) - pred_counts.mean(-1)
            delta.append(_delta)
        delta = torch.cat(delta, dim=0).mean()
        bias_model.linear.bias += torch.Tensor(delta).to(bias_model.linear.bias.device)

    if verbose:
        print('### delta', delta, flush=True)
    return bias_model
