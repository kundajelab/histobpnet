import torch

from histobpnet.model.histobpnet_v1 import HistoBPNetV1
from histobpnet.model.model_wrappers import ModelWrapper
from histobpnet.model.model_config import BPNetModelConfig
from histobpnet.utils.general_utils import to_numpy

class HistoBPNetWrapperV1(ModelWrapper):
    def __init__(
        self,
        args,
    ):
        super().__init__(args)

        config = BPNetModelConfig.from_argparse_args(args)
        self.model = HistoBPNetV1(config)
        self.model_type = config.model_type

    def _step(self, batch, batch_idx, mode: str = 'train'):
        assert mode in ['train', 'val', 'test', 'predict'], "Invalid mode. Must be one of ['train', 'val', 'test', 'predict']"

        x = batch['onehot_seq'] # batch_size x 4 x seq_length
        # dict of num_bins elements each of shape batch_size x bin_width
        true_bin_profile = batch['per_bin_profile'] 
        true_bin_profile_ctl = batch['per_bin_profile_ctrl']

        true_binned_logsum, true_binned_logsum_ctl = self._make_binned_counts(true_bin_profile, true_bin_profile_ctl, eps=1e-6)

        assert x.shape[1] == 4, "Input sequence must be one-hot encoded with 4 channels (A, C, G, T)"
        assert x.shape[0] == true_binned_logsum.shape[0], "Batch size of input sequence and true_binned_logsum must match"
        assert x.shape[0] == true_binned_logsum_ctl.shape[0], "Batch size of input sequence and true_binned_logsum_ctl must match"

        _, y_count = self(x, observed_ctrl=true_binned_logsum_ctl) # batch_size x num_bins

        if mode == 'predict':
            return {
                'pred_count': to_numpy(y_count),
                'true_count': to_numpy(true_binned_logsum),
            }

        self.metrics[mode]['preds'].append(y_count)
        self.metrics[mode]['targets'].append(true_binned_logsum)
        self.metrics[mode]['peak_status'].append(batch['peak_status'])

        # TODO_NOW does mse in log space make sense? that's what bpnet does though (see def _step)
        mse_elements = (y_count - true_binned_logsum) ** 2     # shape: (batch_size, n_bins)
        count_loss = mse_elements.mean()
        loss = count_loss

        dict_show = {
            f'{mode}_loss': loss, 
            f'{mode}_count_loss': count_loss,
        }
        self.log_dict(dict_show, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)

        return loss
    
    def _make_binned_counts(self, true_bin_profile, true_bin_profile_ctl, eps=1e-6):
        """
        true_bin_profile: dict[int -> tensor], each tensor (batch_size, bin_width)
        true_bin_profile_ctl: same structure as true_bin_profile

        returns:
        true_binned_logsum: (batch_size, num_bins)
        true_binned_lfc:    (batch_size, num_bins)
        """
        # ensure a consistent bin order
        assert list(true_bin_profile.keys()) == self.model.output_bins
        assert list(true_bin_profile_ctl.keys()) == self.model.output_bins

        # sum over bin_width dimension for each bin -> list of (batch_size,)
        true_sums = [true_bin_profile[k].sum(dim=1) for k in self.model.output_bins]
        ctl_sums  = [true_bin_profile_ctl[k].sum(dim=1) for k in self.model.output_bins]

        # stack into (batch_size, num_bins)
        true_sums_mat = torch.stack(true_sums, dim=1)   # (batch_size, num_bins)
        ctl_sums_mat  = torch.stack(ctl_sums, dim=1)    # (batch_size, num_bins)

        # log of summed counts per bin
        # TODO_NOW log or log1p?... here and below. bpnet does log1p (see def count_head and def _step)
        true_binned_logsum = torch.log(true_sums_mat + eps)

        # log fold change per bin: log(sum_t / sum_ctl)
        # true_binned_lfc = torch.log((true_sums_mat + eps) / (ctl_sums_mat + eps))
        true_binned_logsum_ctl = torch.log(ctl_sums_mat + eps)

        return true_binned_logsum, true_binned_logsum_ctl

    def load_from_pretrained(self, chrombpnet_wo_bias_path: str = ""):
        if chrombpnet_wo_bias_path != "":
            self.model.bpnet = self.init_chrombpnet_wo_bias(
                chrombpnet_wo_bias_path,
                freeze=False, 
                instance=self.model.bpnet,
            )