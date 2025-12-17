import os
import torch

from histobpnet.model.model_wrappers import ModelWrapper
from histobpnet.model.histobpnet_v2 import HistoBPNetV2
from histobpnet.model.model_config import BPNetModelConfig
from histobpnet.utils.general_utils import to_numpy

class HistoBPNetWrapperV2(ModelWrapper):
    def __init__(
        self,
        args,
    ):
        super().__init__(args)

        config = BPNetModelConfig.from_argparse_args(args)
        self.model = HistoBPNetV2(config)
        self.model_type = config.model_type

    def _step(self, batch, batch_idx, mode: str = 'train'):
        assert mode in ['train', 'val', 'test', 'predict'], "Invalid mode. Must be one of ['train', 'val', 'test', 'predict']"

        x = batch['onehot_seq'] # batch_size x 4 x seq_length
        true_profile = batch['profile'] 
        true_profile_ctl = batch['profile_ctrl']
        assert true_profile.shape == true_profile_ctl.shape, "Control profile shape must match chip profile shape"

        # TODO log or log1p?.... bpnet does log1p (see def count_head and def _step)
        if true_profile.dim() == 2:
            true_logsum, true_logsum_ctl = true_profile.sum(dim=-1).log1p(), true_profile_ctl.sum(dim=-1).log1p()
        else:
            true_logsum, true_logsum_ctl = true_profile.log1p(), true_profile_ctl.log1p()

        assert x.shape[1] == 4, "Input sequence must be one-hot encoded with 4 channels (A, C, G, T)"
        assert x.shape[0] == true_logsum.shape[0], "Batch size of input sequence and true_logsum must match"
        assert x.shape[0] == true_logsum_ctl.shape[0], "Batch size of input sequence and true_logsum_ctl must match"
        
        _, y_count = self(x, observed_ctrl=true_logsum_ctl) # batch_size x 1
        y_count = y_count.squeeze(-1)

        if mode == 'predict':
            return {
                'pred_count': to_numpy(y_count),
                'true_count': to_numpy(true_logsum),
            }

        self.metrics[mode]['preds'].append(y_count)
        self.metrics[mode]['targets'].append(true_logsum)
        self.metrics[mode]['peak_status'].append(batch['peak_status'])

        # TODO does mse in log space make sense? that's what bpnet does though (see def _step)
        mse_elements = (y_count - true_logsum) ** 2     # shape: (batch_size, 1)
        count_loss = mse_elements.mean()
        loss = count_loss

        dict_show = {
            f'{mode}_loss': loss, 
            f'{mode}_count_loss': count_loss,
        }
        self.log_dict(dict_show, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)

        return loss

    def load_pretrained_chrombpnet(self, chrombpnet_wo_bias_path: str = None):
        if chrombpnet_wo_bias_path is not None:
            self.model.bpnet = self.init_chrombpnet_wo_bias(
                chrombpnet_wo_bias_path,
                freeze=False, 
                instance=self.model.bpnet,
            )
    
    def save_state_dict(self, save_dir: str):
        print(f"Saving state_dict to {save_dir}...")
        torch.save(self.model.bpnet.state_dict(), os.path.join(save_dir, f'{self.model_type}.pt'))