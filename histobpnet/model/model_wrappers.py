from typing import Dict, Any, Union
import torch
import torch.nn.functional as F
import lightning as L
from lightning import LightningModule
from lightning.pytorch.utilities.types import STEP_OUTPUT
import numpy as np
from tqdm import tqdm
import os

from histobpnet.model.chrombpnet import BPNet, ChromBPNet
from histobpnet.model.model_config import ChromBPNetConfig
from histobpnet.utils.general_utils import to_numpy, multinomial_nll, pearson_corr

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
        # bpnet_wrapper = BPNetWrapper(args)
        # bpnet_wrapper.model = bias_model
        # output = L.Trainer(logger=False, devices=device).predict(bpnet_wrapper, dataloader)
        # parsed_output = {key: np.concatenate([batch[key] for batch in output]) for key in output[0]}
        # delta = parsed_output['true_count'].mean(-1) - parsed_output['pred_count'].mean(-1)
        # delta = torch.cat([predictions['delta'] for predictions in predictions], dim=0).mean()

    if verbose:
        print('### delta', delta, flush=True)
    return bias_model

def init_bias(bias, dataloader=None, verbose=False, device=1):
    print(f"Loading bias model from {bias}")
    bias_model = BPNet.from_keras(bias, name='bias')
    # Freeze the sub-model
    bias_model.eval()
    for param in bias_model.parameters():
        param.requires_grad = False

    if dataloader is not None:
        bias_model = adjust_bias_model_logcounts(bias_model, dataloader, verbose=verbose, device=device)
    return bias_model

def init_chrombpnet_wo_bias(chrombpnet_wo_bias, freeze=True):
    print(f"Loading chrombpnet_wo_bias model from {chrombpnet_wo_bias}")
    if chrombpnet_wo_bias.endswith('.h5'):
        model = BPNet.from_keras(chrombpnet_wo_bias)
    elif chrombpnet_wo_bias.endswith('.pt'):
        # TODO why n_filters 512 and why map_location cpu?
        model = BPNet(n_filters=512, n_layers=8)
        model.load_state_dict(torch.load(chrombpnet_wo_bias, map_location='cpu'))
    elif chrombpnet_wo_bias.endswith('.ckpt'):
        model = BPNet.load_from_checkpoint(chrombpnet_wo_bias)

    if freeze:
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

    return model

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
        
        # Initialize metrics storage
        self.metrics = {
            'train': {'preds': [], 'targets': []},
            'val': {'preds': [], 'targets': []},
            'test': {'preds': [], 'targets': []},
            'predict': {'preds': [], 'targets': []}
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

    def _step(self, batch, batch_idx, mode='train'):
        raise NotImplementedError("Subclasses must implement this method")

    def init_bias(self, bias, dataloader=None, verbose=False, device=1):
        # print(f"Loading bias model from {bias}")
        return init_bias(bias, dataloader=dataloader, verbose=verbose, device=device)

    def init_chrombpnet_wo_bias(self, chrombpnet_wo_bias, freeze=True):
        # print(f"Initializing chrombpnet_wo_bias model from {chrombpnet_wo_bias}")
        return init_chrombpnet_wo_bias(chrombpnet_wo_bias, freeze=freeze)

    def _predict_on_dataloader(self, dataloader, func, **kwargs):
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
        # Concatenate predictions and targets
        all_preds = torch.cat(self.metrics[mode]['preds']).reshape(-1)
        all_targets = torch.cat(self.metrics[mode]['targets']).reshape(-1)
        
        # Calculate and log correlation
        pr = pearson_corr(all_preds, all_targets, eps=0)
        self.log(f"{mode}_count_pearson", pr, prog_bar=True, logger=True, sync_dist=True)
        
        # Reset metrics storage
        self.metrics[mode]['preds'] = []
        self.metrics[mode]['targets'] = []
    
    def on_train_epoch_end(self) -> None:
        """Handle end of training epoch."""
        self._epoch_end('train')
    
    def on_validation_epoch_end(self) -> None:
        """Handle end of validation epoch."""
        self._epoch_end('val')
    
    def on_test_epoch_end(self) -> None:
        """Handle end of test epoch."""
        self._epoch_end('test')
    
    def configure_optimizers(self) -> Union[torch.optim.Optimizer, Dict[str, Any]]:
        raise NotImplementedError("Subclasses must implement this method")

class BPNetWrapper(ModelWrapper):
    """Wrapper for BPNet model with specific configurations and loss functions.
    
    This wrapper extends the base ModelWrapper to handle BPNet-specific features
    such as profile and count predictions, and appropriate loss calculations.
    """

    def __init__(self, args):
        super().__init__(args)
        self.model = BPNet(
            out_dim=args.out_dim,
            n_filters=args.n_filters, 
            n_layers=args.n_layers, 
            conv1_kernel_size=args.conv1_kernel_size,
            profile_kernel_size=args.profile_kernel_size,
            n_outputs=args.n_outputs, 
            n_control_tracks=args.n_control_tracks, 
            profile_output_bias=args.profile_output_bias, 
            count_output_bias=args.count_output_bias, 
        )

    # TODO walk through this
    def _step(self, batch, batch_idx, mode: str = 'train'):
        assert mode in ['train', 'val', 'test', 'predict'], "Invalid mode. Must be one of ['train', 'val', 'test', 'predict']"

        x = batch['onehot_seq'] # batch_size x 4 x seq_length
        true_profile = batch['profile'] # batch_size x seq_length

        assert x.shape[1] == 4, "Input sequence must be one-hot encoded with 4 channels (A, C, G, T)"
        assert x.shape[0] == true_profile.shape[0], "Batch size of input sequence and profile must match"
        assert x.shape[2] == true_profile.shape[1], "Sequence length of input sequence and profile must match"

        true_counts = torch.log1p(true_profile.sum(dim=-1))

        y_profile, y_count = self(x)
        y_count = y_count.squeeze(-1) # batch_size x 1

        if mode == 'predict':
            return {
                'pred_count': to_numpy(y_count),
                'true_count': to_numpy(true_counts),
                'pred_profile': to_numpy(y_profile),
                'true_profile': to_numpy(true_profile),
            }

        self.metrics[mode]['preds'].append(y_count)
        self.metrics[mode]['targets'].append(true_counts)
        with torch.no_grad():
            profile_pearson = pearson_corr(y_profile.softmax(-1), true_profile).mean()
            self.log_dict(
                {f"{mode}_profile_pearson": profile_pearson},
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=True
            )

        profile_loss = multinomial_nll(y_profile, true_profile)
        count_loss = F.mse_loss(y_count, true_counts)
        loss = self.beta * profile_loss + self.alpha * count_loss

        dict_show = {
            f'{mode}_loss': loss, 
            f'{mode}_profile_loss': profile_loss,
            f'{mode}_count_loss': count_loss,
        }
        self.log_dict(dict_show, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)

        return loss
        
    def configure_optimizers(self):
        # TODO_later config-ify lr and eps
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, eps=1e-7)
        return optimizer

    # TODO review
    def predict(self, x, forward_only=True):
        y_profile, y_count = self(x)
        y_count = torch.exp(y_count)

        if not forward_only:
            y_profile_revcomp, y_count_revcomp = self(x[:, ::-1, ::-1])
            y_count_revcomp = torch.exp(y_count_revcomp)
            y_profile = (y_profile + y_profile_revcomp) / 2
            y_count = (y_count + y_count_revcomp) / 2

        return y_profile.cpu().numpy(), y_count.cpu().numpy()
    
class ChromBPNetWrapper(BPNetWrapper):
    """Wrapper for ChromBPNet model with specific configurations and loss functions.
    
    This wrapper extends the base ModelWrapper to handle ChromBPNet-specific features
    such as chromatin accessibility predictions and appropriate loss calculations.
    """

    def __init__(
        self,
        args,
    ):
        """Initialize ChromBPNet wrapper.
        
        Args:
            model: ChromBPNet model instance
            alpha: Weight for count loss
            bias_scaled: Path to bias model if using scaled bias
            **kwargs: Additional arguments to be passed to the model
        """
        super().__init__(args)

        config = ChromBPNetConfig.from_argparse_args(args)
        self.model = ChromBPNet(config)

def create_model_wrapper(
    args,
) -> ModelWrapper:
    """Factory function to create appropriate model wrapper.
    """
    model_type = args.model_type.lower()
    if model_type == 'bpnet':
        return BPNetWrapper(args)
    elif model_type == 'chrombpnet':
        model_wrapper = ChromBPNetWrapper(args)
        if args.bias_scaled:
            model_wrapper.model.bias = model_wrapper.init_bias(args.bias_scaled)
        if args.chrombpnet_wo_bias:
            model_wrapper.model.model = model_wrapper.init_chrombpnet_wo_bias(args.chrombpnet_wo_bias, freeze=False)
        return model_wrapper
    else:
        raise ValueError(f"Unknown model type: {model_type}") 

def load_pretrained_model(args):
    model_type = args.model_type.lower()
    if model_type != 'chrombpnet':
        raise NotImplementedError("Loading pretrained models is currently only implemented for ChromBPNetWrapper")

    checkpoint = args.checkpoint
    if checkpoint is not None:
        if checkpoint.endswith('.ckpt'):
            model_wrapper = ChromBPNetWrapper.load_from_checkpoint(checkpoint, map_location='cpu')
        elif checkpoint.endswith('.pt'):
            model_wrapper = ChromBPNetWrapper(args)
            # TODO why map location is cpu?
            model_wrapper.model.model.load_state_dict(torch.load(checkpoint, map_location='cpu'))
        elif checkpoint.endswith('.h5'):
            model_wrapper = ChromBPNetWrapper(args)
            # For Keras H5 files, load using the from_keras method
            # note: this can only load a chrombpnet_wo_bias, not a full chrombpnet...
            print(f"Loading chrombpnet_wo_bias model from {checkpoint}")
            model_wrapper.model.model = BPNet.from_keras(checkpoint)
        else:
            raise ValueError("No valid checkpoint found")

        # set bias
        bias_scaled = args.bias_scaled
        if bias_scaled is None and os.path.exists(os.path.join(args.data_dir, 'bias_scaled.h5')):
            bias_scaled = os.path.join(args.data_dir, 'bias_scaled.h5')
        if bias_scaled:
            print(f"Loading bias model from {bias_scaled}")
            model_wrapper.model.bias = model_wrapper.init_bias(bias_scaled)
        else:
            print(f"No bias model found")
    else:
        model_wrapper = ChromBPNetWrapper(args)

    return model_wrapper