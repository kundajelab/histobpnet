from typing import Dict, Any, Union
import torch
import torch.nn.functional as F
import lightning as L
from lightning import LightningModule
from lightning.pytorch.utilities.types import STEP_OUTPUT
import numpy as np

from histobpnet.model.chrombpnet import BPNet, ChromBPNet
from histobpnet.model.model_config import ChromBPNetConfig
from histobpnet.utils.general_utils import to_numpy, multinomial_nll, pearson_corr

# valeh: ??
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
    print("Predicting within adjust counts")
    bias_model.eval()
    with torch.no_grad():
        output = L.Trainer(logger=False, devices=device).predict(bias_model, dataloader)
        parsed_output = {key: np.concatenate([batch[key] for batch in output]) for key in output[0]}
        try:    
            delta = parsed_output['true_count'].mean(-1) - parsed_output['pred_count'].mean(-1)
        except:
            import pdb; pdb.set_trace()
            # delta = parsed_output['true_count'].mean(dim=-1) - parsed_output['pred_count'].mean(dim=-1)
        # delta = torch.cat([predictions['delta'] for predictions in predictions], dim=0)
        bias_model.linear.bias += torch.Tensor(delta).to(bias_model.linear.bias.device)

    if verbose:
        print('### delta', delta.mean(), flush=True)
    return bias_model

# TODO
class _ProfileLogitScaling(torch.nn.Module):
    """This ugly class is necessary because of Captum.

    Captum internally registers classes as linear or non-linear. Because the
    profile wrapper performs some non-linear operations, those operations must
    be registered as such. However, the inputs to the wrapper are not the
    logits that are being modified in a non-linear manner but rather the
    original sequence that is subsequently run through the model. Hence, this
    object will contain all of the operations performed on the logits and
    can be registered.

    Parameters
    ----------
    logits: torch.Tensor, shape=(-1, -1)
        The logits as they come out of a Chrom/BPNet model.
    """

    def __init__(self):
        super(_ProfileLogitScaling, self).__init__()
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, logits):
        y_softmax = self.softmax(logits)
        y = logits * y_softmax
        return y
        #print("a") 
        #y_lsm = torch.nn.functional.log_softmax(logits, dim=-1)
        #return torch.sign(logits) * torch.exp(torch.log(abs(logits)) + y_lsm)
    
# TODO
class ProfileWrapper(torch.nn.Module):
    """A wrapper class that returns transformed profiles.

    This class takes in a trained model and returns the weighted softmaxed
    outputs of the first dimension. Specifically, it takes the predicted
    "logits" and takes the dot product between them and the softmaxed versions
    of those logits. This is for convenience when using captum to calculate
    attribution scores.

    Parameters
    ----------
    model: torch.nn.Module
        A torch model to be wrapped.
    """

    def __init__(self, model):
        super(ProfileWrapper, self).__init__()
        self.model = model
        self.flatten = torch.nn.Flatten()
        self.scaling = _ProfileLogitScaling()

    def forward(self, x, x_ctl=None, **kwargs):
        logits = self.model(x, x_ctl=x_ctl, **kwargs)[0]
        logits = self.flatten(logits)
        logits = logits - torch.mean(logits, dim=-1, keepdims=True)
        return self.scaling(logits).sum(dim=-1, keepdims=True)

# TODO
class CountWrapper(torch.nn.Module):
    """A wrapper class that only returns the predicted counts.

    This class takes in a trained model and returns only the second output.
    For BPNet models, this means that it is only returning the count
    predictions. This is for convenience when using captum to calculate
    attribution scores.

    Parameters
    ----------
    model: torch.nn.Module
        A torch model to be wrapped.
    """

    def __init__(self, model):
            super(CountWrapper, self).__init__()
            self.model = model

    def forward(self, x, x_ctl=None, **kwargs):
        return self.model(x, x_ctl=x_ctl, **kwargs)[1]

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
            beta: Weight for profile loss
            **kwargs: Additional arguments to be passed to the model
        """
        
        super().__init__()
        self.alpha = args.alpha
        # TODO where is this set? And why isnt this just 1-alpha?
        self.beta = args.beta
        self.verbose = args.verbose
        
        if args.model_type == 'chrombpnet':
            config = ChromBPNetConfig.from_argparse_args(args)
            self.model = ChromBPNet(config)
        elif args.model_type == 'bpnet':
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
        else:
            raise ValueError(f"Model type {args.model_type} not supported")
        
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
        # TODO config-ify lr and eps
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, eps=1e-7)
        return optimizer

class ChromBPNetWrapper(BPNetWrapper):
    """Wrapper for ChromBPNet model with specific configurations and loss functions.
    
    This wrapper extends the base ModelWrapper to handle ChromBPNet-specific features
    such as chromatin accessibility predictions and appropriate loss calculations.
    """

    def __init__(
        self,
        args,
    ):
        super().__init__(args)

        if args.bias_scaled:
            self.init_bias(args.bias_scaled)
        if args.chrombpnet_wo_bias is not None:
            self.init_chrombpnet_wo_bias(args.chrombpnet_wo_bias)

    def init_bias(self, bias, dataloader=None, verbose=False, device=1):
        print(f"Loading bias model from {bias}")
        self.model.bias = BPNet.from_keras(bias, name='bias')
        # Freeze the bias model
        self.model.bias.eval() 
        for param in self.model.bias.parameters():
            param.requires_grad = False

        if dataloader is not None:
            self.model.bias = adjust_bias_model_logcounts(self.model.bias, dataloader, verbose=verbose, device=device)

    def init_chrombpnet_wo_bias(self, chrombpnet_wo_bias):
        print(f"Loading chrombpnet_wo_bias model from {chrombpnet_wo_bias}")
        if chrombpnet_wo_bias.endswith('.h5'):
            self.model.conv_tower = BPNet.from_keras(chrombpnet_wo_bias)
        else:
            self.model.conv_tower.load_state_dict(torch.load(chrombpnet_wo_bias, map_location=self.device))

        # TODO I think finetune is not defined?
        if not self.finetune:
            print("Freezing the model")
            self.model.conv_tower.eval()
            for param in self.model.conv_tower.parameters():
                param.requires_grad = False

def create_model_wrapper(
    args,
) -> ModelWrapper:
    """Factory function to create appropriate model wrapper.
    """
    model_type = args.model_type.lower()
    if model_type == 'bpnet':
        return BPNetWrapper(args)
    elif model_type == 'chrombpnet':
        return ChromBPNetWrapper(args)
    else:
        raise ValueError(f"Unknown model type: {model_type}") 

def load_pretrained_model(args):
    checkpoint = args.checkpoint
    if checkpoint is not None:
        if checkpoint.endswith('.ckpt'):
            model_wrapper = ChromBPNetWrapper.load_from_checkpoint(checkpoint)
        elif checkpoint.endswith('.pt'):
            model_wrapper = ChromBPNetWrapper(args)
            # TODO why map location is cpu?
            model_wrapper.model.model.load_state_dict(torch.load(checkpoint, map_location='cpu'))
        elif checkpoint.endswith('.h5'):  
            model_wrapper = ChromBPNetWrapper(args)
            # For Keras H5 files, load using the from_keras method
            # TODO later I think this can only load a chrombpnet_wo_bias, not a full chrombpnet...
            print(f"Loading chrombpnet_wo_bias model from {checkpoint}")
            model_wrapper.model.model = BPNet.from_keras(checkpoint)
    else:
        model_wrapper = ChromBPNetWrapper(args)

    return model_wrapper