# Author: Lei Xiong <jsxlei@gmail.com>

"""
Data loading and processing module for genomic data.

This module provides classes for loading and processing genomic data for training and evaluation.
It handles the loading of genomic regions, their corresponding sequences, and various data augmentation techniques.
"""

from functools import cached_property
from time import time
import torch
import numpy as np
import pandas as pd
import lightning as L
from histobpnet.utils.data_utils import (
    load_region_df,
    get_cts,
    split_peak_and_nonpeak,
    concat_peaks_and_subsampled_negatives,
)
from histobpnet.utils.general_utils import is_histone
from histobpnet.data_loader.chrombpnet_dataset import ChromBPNetDataset
from histobpnet.data_loader.histobpnet_dataset_v1 import HistoBPNetDatasetV1
from histobpnet.data_loader.histobpnet_dataset_v2 import HistoBPNetDatasetV2
from histobpnet.data_loader.histobpnet_dataset_v3 import HistoBPNetDatasetV3
from histobpnet.data_loader.data_config import DataConfig

class DataModule(L.LightningDataModule):
    """DataModule for loading and processing genomic data for training and evaluation.
    
    This module handles the loading of genomic regions, their corresponding sequences,
    and various data augmentation techniques. It supports the following data types:
    - Profile data: For single-region analysis
    
    The module implements different sampling strategies for training, validation and testing:
    - Train: peaks + negative_sampling_ratio (0.1) of negatives, sampled each epoch
    - Val: peaks + negative_sampling_ratio (1) of negatives, sampled once and fixed
    - Test: peaks + negatives, no sampling
    
    Attributes:
        config: Configuration object containing data loading parameters
        dataset_class: The dataset class to use (eg ChromBPNetDataset)
        peaks: DataFrame containing peak regions
        negatives: DataFrame containing negative regions
        data: Combined DataFrame of peaks and negatives
        train_chroms: List of chromosomes used for training
        val_chroms: List of chromosomes used for validation
        test_chroms: List of chromosomes used for testing
    """
    
    def __init__(self, config: DataConfig, gpu_count: int):
        """Initialize the DataModule.
        
        config:
            config: Configuration object containing data loading parameters
        """
        super().__init__()
        self.config = config

        # Set dataset class based on data type
        if self.config.model_type == 'chrombpnet':
            self.dataset_class = ChromBPNetDataset
        elif self.config.model_type == 'histobpnet_v1':
            self.dataset_class = HistoBPNetDatasetV1
        elif self.config.model_type == 'histobpnet_v2':
            self.dataset_class = HistoBPNetDatasetV2
        elif self.config.model_type == 'histobpnet_v3':
            self.dataset_class = HistoBPNetDatasetV3
        else:
            raise NotImplementedError(f'Unsupported model type: {self.config.model_type}')
        
        # in DDP (eg when accelerator='gpu' and devices>1), each device/process (device = GPU)
        # will use a per_device_batch_size shard of the data (Lightning takes care of the sharding)
        # so the effective (total) batch size will be per_device_batch_size * len(args.gpu).
        # if we want this to be equal to config.batch_size (IOW if we want the total batch size to be config.batch_size),
        # then we need to set per_device_batch_size = config.batch_size // len(args.gpu) as below
        #
        # see https://lightning.ai/docs/pytorch/stable/accelerators/gpu_faq.html#how-should-i-adjust-the-batch-size-when-using-multiple-devices
        # 
        # btw Lightning doesn't use this per_device_batch_size attribute anywhere internally, we use it
        # here when building our dataloaders below (eg see train_dataloader)
        #
        # Beware: if you change the number of devices and dont run the line below, then your effective (global) batch size
        # will change accordingly, which may affect training dynamics. On the other hand, you might under-utilize your GPUs
        # if your config.batch_size is small and you have many GPUs. So pick your poison...
        self.per_device_batch_size = config.batch_size // gpu_count

        # Load and process data
        self._load_regions()
        self._setup_chromosomes()
        self._split_data()

    def _load_regions(self):
        """Load peak and negative regions from files."""
        self.peaks = load_region_df(
            self.config.peaks, 
            chrom_sizes=self.config.chrom_sizes,
            in_window=self.config.in_window,
            shift=self.config.shift,
            is_peak=True,
            skip_missing_hist=self.config.skip_missing_hist,
            atac_hgp_map=self.config.atac_hgp_map,
        )
        
        if self.config.negatives is not None:
            self.negatives = load_region_df(
                self.config.negatives,
                chrom_sizes=self.config.chrom_sizes,
                in_window=self.config.in_window,
                shift=self.config.shift,
                is_peak=False
            )
            self.data = pd.concat([self.peaks, self.negatives], ignore_index=True)
        else:
            self.negatives = None
            self.data = self.peaks

    def _setup_chromosomes(self):
        """Set up chromosome lists for training, validation and testing."""
        self.train_chroms = [i for i in self.config.training_chroms if i not in self.config.exclude_chroms]
        self.val_chroms = [i for i in self.config.validation_chroms if i not in self.config.exclude_chroms]
        self.test_chroms = [i for i in self.config.test_chroms if i not in self.config.exclude_chroms]
        self.chroms = self.train_chroms + self.val_chroms + self.test_chroms

    def _split_data(self):
        """Split data into training, validation and testing sets."""
        self.train_data = self.data[self.data.iloc[:, 0].isin(self.train_chroms)].reset_index(drop=True)
        self.val_data = self.data[self.data.iloc[:, 0].isin(self.val_chroms)].reset_index(drop=True)
        self.train_val_data = self.data[self.data.iloc[:, 0].isin(self.val_chroms+self.train_chroms)].reset_index(drop=True)
        self.test_data = self.data[self.data.iloc[:, 0].isin(self.test_chroms)].reset_index(drop=True)

    def setup(self, stage='fit'):
        print('Setting up data...'); t0 = time()

        config = self.config

        if stage == 'fit':
            train_peaks, train_nonpeaks = split_peak_and_nonpeak(self.train_data)
            val_peaks, val_nonpeaks = split_peak_and_nonpeak(self.val_data)

            self.train_dataset = self.dataset_class(
                peak_regions=train_peaks,
                nonpeak_regions=train_nonpeaks,
                genome_fasta=config.fasta,
                inputlen=config.in_window,                                        
                outputlen=config.out_window,
                max_jitter=config.shift,
                negative_sampling_ratio=config.negative_sampling_ratio,
                cts_bw_file=config.bigwig,
                cts_ctrl_bw_file=config.bigwig_ctrl,
                output_bins=config.output_bins,
                atac_hgp_map=config.atac_hgp_map,
                skip_missing_hist=config.skip_missing_hist,
                add_revcomp=True,
                return_coords=False,
                shuffle_at_epoch_start=False,
                rc_frac=config.rc_frac,
                mode='train',
                ctrl_scaling_factor=config.ctrl_scaling_factor,
                config=config,
            )
            self.val_dataset = self.dataset_class(
                peak_regions=val_peaks,
                nonpeak_regions=val_nonpeaks,
                genome_fasta=config.fasta,
                inputlen=config.in_window,                                        
                outputlen=config.out_window,
                max_jitter=0,
                negative_sampling_ratio=config.negative_sampling_ratio,
                cts_bw_file=config.bigwig,
                cts_ctrl_bw_file=config.bigwig_ctrl,
                output_bins=config.output_bins,
                atac_hgp_map=config.atac_hgp_map,
                skip_missing_hist=config.skip_missing_hist,
                add_revcomp=False,
                return_coords=False,
                shuffle_at_epoch_start=False, 
                rc_frac=config.rc_frac,
                mode='val',
                ctrl_scaling_factor=config.ctrl_scaling_factor,
                config=config,
            )
        elif stage == 'test':
            test_peaks, test_nonpeaks = split_peak_and_nonpeak(self.test_data)
            self.test_dataset = self.dataset_class(
                peak_regions=test_peaks,
                nonpeak_regions=test_nonpeaks,  
                genome_fasta=config.fasta,
                inputlen=config.in_window,                                        
                outputlen=config.out_window,
                max_jitter=0,
                negative_sampling_ratio=-1,
                cts_bw_file=config.bigwig,
                cts_ctrl_bw_file=config.bigwig_ctrl,
                output_bins=config.output_bins,
                atac_hgp_map=config.atac_hgp_map,
                skip_missing_hist=config.skip_missing_hist,
                add_revcomp=False,
                return_coords=False,
                shuffle_at_epoch_start=False, 
                rc_frac=config.rc_frac,
                mode='test',
                ctrl_scaling_factor=config.ctrl_scaling_factor,
                config=config,
            )

        print(f'Data setup complete in {time() - t0:.2f} seconds')

    @cached_property
    def median_count(self):
        if is_histone(self.config.model_type):
            raise NotImplementedError("median_count is not implemented for histone models")
        import pyBigWig
        # Calculate median count to get weight of count loss
        # valeh: I dont fully understand the logic here (ie why use the median to determine the weight)
        self.train_val_subsampled = concat_peaks_and_subsampled_negatives(
            self.train_val_data,
            negative_sampling_ratio=self.config.negative_sampling_ratio
        )
        counts_subsampled = get_cts(self.train_val_subsampled, pyBigWig.open(self.config.bigwig), self.config.out_window).sum(-1)
        return np.median(counts_subsampled)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset, 
            batch_size=self.per_device_batch_size,
            shuffle=True, 
            drop_last=False,
            num_workers=self.config.num_workers, 
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset, 
            batch_size=self.per_device_batch_size, 
            shuffle=False, 
            num_workers=self.config.num_workers, 
        )
    
    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset, 
            batch_size=self.per_device_batch_size, 
            shuffle=False, 
            num_workers=self.config.num_workers, 
        )

    def negative_dataloader(self):
        negative_dataset = self.dataset_class(
            peak_regions=self.negatives,
            nonpeak_regions=None,
            genome_fasta=self.config.fasta,
            inputlen=self.config.in_window,
            outputlen=self.config.out_window,
            max_jitter=0,
            negative_sampling_ratio=-1,
            cts_bw_file=self.config.bigwig,
            cts_ctrl_bw_file=self.config.bigwig_ctrl,
            output_bins=self.config.output_bins,
            atac_hgp_map=self.config.atac_hgp_map,
            skip_missing_hist=self.config.skip_missing_hist,
            add_revcomp=False,
            return_coords=False,
            shuffle_at_epoch_start=False,
            debug=self.config.debug,
            rc_frac=self.config.rc_frac,
            mode='negative',
            ctrl_scaling_factor=self.config.ctrl_scaling_factor,
            config=self.config,
        )
        return torch.utils.data.DataLoader(
            negative_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=False, 
            num_workers=self.config.num_workers, 
        )

    def chrom_dataloader(self, chrom='chr1'):
        dataset = self.chrom_dataset(chrom=chrom)
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
        ), dataset

    def chrom_dataset(self, chrom='chr1'):
        if isinstance(chrom, str):
            if chrom in ['train', 'val', 'test']:
                chrom = getattr(self, f'{chrom}_chroms')
            elif chrom == 'all':
                chrom = self.chroms
            else:
                chrom = [chrom]

        regions = self.data[self.data.iloc[:, 0].isin(chrom)].reset_index(drop=True)
        peaks, nonpeaks = split_peak_and_nonpeak(regions)

        dataset = self.dataset_class(
            peak_regions=peaks,
            nonpeak_regions=nonpeaks,
            genome_fasta=self.config.fasta,
            inputlen=self.config.in_window,
            outputlen=self.config.out_window,
            max_jitter=0,
            negative_sampling_ratio=-1,
            cts_bw_file=self.config.bigwig,
            cts_ctrl_bw_file=self.config.bigwig_ctrl,
            output_bins=self.config.output_bins,
            atac_hgp_map=self.config.atac_hgp_map,
            skip_missing_hist=self.config.skip_missing_hist,
            add_revcomp=False,
            return_coords=False,
            shuffle_at_epoch_start=False,
            debug=self.config.debug,
            rc_frac=self.config.rc_frac,
            mode='chrom',
            ctrl_scaling_factor=self.config.ctrl_scaling_factor,
            config=self.config,
        )
        return dataset
