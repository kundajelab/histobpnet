import torch
import torch.nn.functional as F

from histobpnet.model.model_wrappers import ModelWrapper
from histobpnet.model.bpnet import BPNet
from histobpnet.model.chrombpnet import ChromBPNet
from histobpnet.model.model_config import BPNetModelConfig
from histobpnet.utils.general_utils import to_numpy, multinomial_nll, pearson_corr

class BaseBPNetWrapper(ModelWrapper):
    """Wrapper for BPNet model with specific configurations and loss functions.
    
    This wrapper extends the base ModelWrapper to handle BPNet-specific features
    such as profile and count predictions, and appropriate loss calculations.
    """

    def __init__(self, args):
        super().__init__(args)

    def _step(self, batch, batch_idx, mode: str = 'train'):
        assert mode in ['train', 'val', 'test', 'predict'], "Invalid mode. Must be one of ['train', 'val', 'test', 'predict']"

        x = batch['onehot_seq'] # batch_size x 4 x seq_length
        true_profile = batch['profile'] # batch_size x seq_length

        assert x.shape[1] == 4, "Input sequence must be one-hot encoded with 4 channels (A, C, G, T)"
        assert x.shape[0] == true_profile.shape[0], "Batch size of input sequence and profile must match"

        # TODO why log1p here but in save_predictions we're not undoing this exact transformation?
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
        self.metrics[mode]['peak_status'].append(batch['peak_status'])
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

    # TODO review
    def predict(self, x, forward_only=True):
        raise ValueError("Putting this here for now so I know when it's called!")

        y_profile, y_count = self(x)
        y_count = torch.exp(y_count)

        if not forward_only:
            y_profile_revcomp, y_count_revcomp = self(x[:, ::-1, ::-1])
            y_count_revcomp = torch.exp(y_count_revcomp)
            y_profile = (y_profile + y_profile_revcomp) / 2
            y_count = (y_count + y_count_revcomp) / 2

        return y_profile.cpu().numpy(), y_count.cpu().numpy()

class BPNetWrapper(BaseBPNetWrapper):
    def __init__(self, args):
        super().__init__(args)

        config = BPNetModelConfig.from_argparse_args(args)
        self.model = BPNet(
            out_dim=config.out_dim,
            n_filters=config.n_filters, 
            n_layers=config.n_layers, 
            conv1_kernel_size=config.conv1_kernel_size,
            profile_kernel_size=config.profile_kernel_size,
            n_outputs=config.n_outputs, 
            n_control_tracks=config.n_control_tracks, 
            profile_output_bias=config.profile_output_bias, 
            count_output_bias=config.count_output_bias, 
        )
        self.model_type = config.model_type

class ChromBPNetWrapper(BaseBPNetWrapper):
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

        config = BPNetModelConfig.from_argparse_args(args)
        self.model = ChromBPNet(config)
        self.model_type = config.model_type

    def load_pretrained_chrombpnet(self, bias_scaled_path: str = None, chrombpnet_wo_bias_path: str = None, instance=None):
        """Load pretrained weights for bias and ChromBPNet without bias.

        Args:
            bias_scaled_path: Path to the pretrained bias model
            chrombpnet_wo_bias_path: Path to the pretrained ChromBPNet model without bias
        """
        if bias_scaled_path is not None:
            self.model.bias = self.init_bias(bias_scaled_path, instance=self.model.bias)
        if chrombpnet_wo_bias_path is not None:
            self.model.model = self.init_chrombpnet_wo_bias(chrombpnet_wo_bias_path, freeze=False, instance=self.model.model)