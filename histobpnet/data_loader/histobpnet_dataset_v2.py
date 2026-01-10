import numpy as np
import pandas as pd
from histobpnet.utils.data_utils import (
    load_data,
    crop_revcomp_data,
)
from histobpnet.utils.general_utils import add_peak_id
from histobpnet.data_loader.data_config import DataConfig
from histobpnet.data_loader.chrombpnet_dataset import ChromBPNetDataset, validate_mode

class HistoBPNetDatasetV2(ChromBPNetDataset):
    def __init__(
        self, 
        peak_regions, 
        nonpeak_regions, 
        config: DataConfig,
        inputlen: int, 
        outputlen: int, 
        max_jitter: int, 
        negative_sampling_ratio: float, 
        shuffle_at_epoch_start: bool, 
        rc_frac: float,
        mode: str = "",
        **kwargs
    ):
        assert max_jitter == 0
        assert rc_frac == 0
        assert config.bigwig_ctrl is not None, "bigwig_ctrl must be provided"

        assert config.atac_hgp_map is not None
        atac_hgp_df = pd.read_csv(config.atac_hgp_map, sep="\t", header=None, names=[
            "chrom", "start", "end", "hist_chrom", "hist_start", "hist_end"
        ])
        add_peak_id(atac_hgp_df, chr_key="chrom")

        validate_mode(mode)

        # Load data
        self.peak_seqs, self.peak_cts, self.peak_cts_ctrl, self.peak_coords, \
        self.nonpeak_seqs, self.nonpeak_cts, self.nonpeak_cts_ctrl, self.nonpeak_coords = load_data(
            peak_regions, nonpeak_regions, config.fasta, config.bigwig,
            inputlen, outputlen, max_jitter,
            cts_ctrl_bw_file=config.bigwig_ctrl, atac_hgp_df=atac_hgp_df,
            # TODO_later maybe make get_total_cts an arg
            get_total_cts=True, skip_missing_hist=config.skip_missing_hist,
            mode=mode,
            ctrl_scaling_factor=config.ctrl_scaling_factor,
            outputlen_neg = config.outputlen_neg,
            pass_zero_mode = config.pass_zero_mode,
        )

        # Store parameters
        self.negative_sampling_ratio = negative_sampling_ratio
        self.inputlen = inputlen
        self.outputlen = outputlen
        self.shuffle_at_epoch_start = shuffle_at_epoch_start
        self.rc_frac = rc_frac

        if nonpeak_regions is not None:
            self.regions = pd.concat([peak_regions, nonpeak_regions], ignore_index=True)
        else:
            self.regions = peak_regions

        # Initialize data
        self.crop_revcomp_data()

    def crop_revcomp_data(self):
        self.cur_seqs, self.cur_cts, self.cur_cts_ctrl, self.cur_coords, self.cur_peak_status = crop_revcomp_data(
            self.peak_seqs, self.peak_cts, self.peak_cts_ctrl, self.peak_coords,
            self.nonpeak_seqs, self.nonpeak_cts, self.nonpeak_cts_ctrl, self.nonpeak_coords,
            inputlen=self.inputlen,
            outputlen=self.outputlen,
            negative_sampling_ratio=self.negative_sampling_ratio,
            shuffle=self.shuffle_at_epoch_start,
            do_crop=False,
            rc_frac=self.rc_frac,
        )

    def __getitem__(self, idx):
        return {
            'onehot_seq': self.cur_seqs[idx].astype(np.float32).transpose(),
            'profile': self.cur_cts[idx].astype(np.float32),
            'profile_ctrl': self.cur_cts_ctrl[idx].astype(np.float32),
            'peak_status': self.cur_peak_status[idx].astype(int),
        }
