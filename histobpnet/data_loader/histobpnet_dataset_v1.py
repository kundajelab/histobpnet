import numpy as np
import pandas as pd
from histobpnet.utils.data_utils import (
    load_data,
    crop_revcomp_data,
)
from histobpnet.data_loader.data_config import DataConfig
from histobpnet.data_loader.chrombpnet_dataset import ChromBPNetDataset, validate_mode

class HistoBPNetDatasetV1(ChromBPNetDataset):
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
        assert negative_sampling_ratio == -1
        assert rc_frac == 0
        assert config.bigwig_ctrl is not None, "bigwig_ctrl must be provided"
        if shuffle_at_epoch_start:
            # I just need to change revcomp_shuffle_augment to shuffle labels for all dict elements...
            raise NotImplementedError("shuffle_at_epoch_start must be False for HistoBPNetDatasetV1")
        
        validate_mode(mode)
        output_bins = config.output_bins

        # Load data
        self.peak_seqs, peak_cts, peak_cts_ctrl, self.peak_coords, \
        self.nonpeak_seqs, self.nonpeak_cts, self.nonpeak_cts_ctrl, self.nonpeak_coords = load_data(
            peak_regions, nonpeak_regions, config.fasta, config.bigwig,
            inputlen, outputlen, max_jitter,
            cts_ctrl_bw_file=config.bigwig_ctrl, output_bins=output_bins,
            mode=mode,
            pass_zero_mode=config.pass_zero_mode,
        )

        # peak_cts is an array of shape (num_peaks, max(output_bins))
        # transform it to an array where each row is a list where each
        # element is the array of counts in that bin
        # the mid point of each bin should be aligned to the summit
        # where the summit is at max_output_bin // 2
        output_bins = [int(x) for x in output_bins.split(",")]
        self.per_bin_peak_cts_dict = self._split_counts_into_bins(peak_cts, output_bins, max_jitter=max_jitter)
        self.per_bin_peak_cts_ctrl_dict = self._split_counts_into_bins(peak_cts_ctrl, output_bins, max_jitter=max_jitter)
        if self.nonpeak_cts is not None:
            self.per_bin_nonpeak_cts_dict = self._split_counts_into_bins(self.nonpeak_cts, output_bins, max_jitter=0)
            self.per_bin_nonpeak_cts_ctrl_dict = self._split_counts_into_bins(self.nonpeak_cts_ctrl, output_bins, max_jitter=0)

        # Store parameters
        self.negative_sampling_ratio = negative_sampling_ratio
        self.inputlen = inputlen
        self.outputlen = outputlen
        self.output_bins = output_bins
        self.shuffle_at_epoch_start = shuffle_at_epoch_start
        self.rc_frac = rc_frac

        if nonpeak_regions is not None:
            self.regions = pd.concat([peak_regions, nonpeak_regions], ignore_index=True)
        else:
            self.regions = peak_regions

        # Initialize data
        self.crop_revcomp_data()

    def _split_counts_into_bins(peak_cts, output_bins: list, max_jitter: int = 0):
        """
        peak_cts: array of shape (num_peaks, max_output_bin)
        output_bins: iterable of bin widths (integers)
        max_jitter: int, amount of jitter to consider around the summit

        returns: list of length num_peaks
                each element is a list of 1d arrays, one per bin width
        """
        # peak_cts.shape = (num_peaks, max_output_bin+2*max_jitter)
        _, max_output_bin = peak_cts.shape
        summit_idx = max_output_bin // 2

        bin_mats = {}
        for width in output_bins:
            # for width w, take a symmetric window around summit_idx
            half = width // 2
            start = summit_idx - half
            end = start + width  # ensures length == width

            # slice all peaks at once -> (num_peaks, width)
            bin_mats[width] = peak_cts[:, start-max_jitter:end+max_jitter]

        return bin_mats

    def crop_revcomp_data(self):
        self.cur_seqs, self.cur_cts, self.cur_cts_ctrl, self.cur_coords, self.cur_peak_status = crop_revcomp_data(
            self.peak_seqs, None, None, self.peak_coords,
            self.nonpeak_seqs, None, None, self.nonpeak_coords,
            self.per_bin_peak_cts_dict, self.per_bin_peak_cts_ctrl_dict,
            self.per_bin_nonpeak_cts_dict, self.per_bin_nonpeak_cts_ctrl_dict,
            self.inputlen, self.outputlen, self.output_bins,
            self.negative_sampling_ratio, self.shuffle_at_epoch_start, rc_frac=self.rc_frac
        )
        
    def __getitem__(self, idx):
        return {
            'onehot_seq': self.cur_seqs[idx].astype(np.float32).transpose(),
            'per_bin_profile': {k:v[idx].astype(np.float32) for k,v in self.cur_cts.items()},
            'per_bin_profile_ctrl': {k:v[idx].astype(np.float32) for k,v in self.cur_cts_ctrl.items()},
            'peak_status': self.cur_peak_status[idx].astype(int),
        }
