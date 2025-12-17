import torch
import numpy as np
import pandas as pd
from histobpnet.utils.data_utils import (
    load_data,
    crop_revcomp_data,
    debug_subsample,
)
from histobpnet.data_loader.data_config import DataConfig
from histobpnet.data_loader.chrombpnet_dataset import ChromBPNetDataset, validate_mode

class HistoBPNetDatasetV2(ChromBPNetDataset):
    def __init__(
        self, 
        peak_regions, 
        nonpeak_regions, 
        genome_fasta, 
        inputlen=2114, 
        outputlen=0, 
        max_jitter=0, 
        negative_sampling_ratio=-1, 
        cts_bw_file=None, 
        cts_ctrl_bw_file=None,
        output_bins="",
        atac_hgp_map="",
        skip_missing_hist=False,
        add_revcomp=False, 
        return_coords=False,    
        shuffle_at_epoch_start=False, 
        rc_frac=0.5,
        debug=False,
        mode: str = "train",
        ctrl_scaling_factor: float = 1.0,
        config: DataConfig = None,
        **kwargs
    ):
        assert max_jitter == 0
        # assert negative_sampling_ratio == -1
        assert rc_frac == 0
        # TODO make this mandatory for all datasetclasses probably and replace all the other goo with this
        assert config is not None

        if debug:
            peak_regions = debug_subsample(peak_regions)
            nonpeak_regions = debug_subsample(nonpeak_regions)

        assert atac_hgp_map != ""
        atac_hgp_df = pd.read_csv(atac_hgp_map, sep="\t", header=0)
        add_peak_id(atac_hgp_df, chr_key="chrom")

        validate_mode(mode)

        # Load data
        self.peak_seqs, self.peak_cts, self.peak_cts_ctrl, self.peak_coords, \
        self.nonpeak_seqs, self.nonpeak_cts, self.nonpeak_cts_ctrl, self.nonpeak_coords = load_data(
            peak_regions, nonpeak_regions, genome_fasta, cts_bw_file,
            inputlen, outputlen, max_jitter,
            cts_ctrl_bw_file=cts_ctrl_bw_file, atac_hgp_df=atac_hgp_df,
            # TODO_later maybe make get_total_cts an arg
            get_total_cts=True, skip_missing_hist=skip_missing_hist,
            mode=mode,
            ctrl_scaling_factor=ctrl_scaling_factor,
            outputlen_neg = config.outputlen_neg,
        )

        # Store parameters
        self.negative_sampling_ratio = negative_sampling_ratio
        self.inputlen = inputlen
        self.outputlen = outputlen
        self.output_bins = output_bins
        self.add_revcomp = add_revcomp
        self.return_coords = return_coords
        self.shuffle_at_epoch_start = shuffle_at_epoch_start
        self.rc_frac = rc_frac
        self.max_jitter = max_jitter
        self.genome_fasta = genome_fasta
        self.cts_bw_file = cts_bw_file
        self.cts_ctrl_bw_file = cts_ctrl_bw_file

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
            add_revcomp=self.add_revcomp,
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
