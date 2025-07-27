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
    load_data,
    random_crop,
    crop_revcomp_augment,
    get_cts,
    split_peak_and_nonpeak,
    concat_peaks_and_subsampled_negatives,
)

class DataModule(L.LightningDataModule):
    """DataModule for loading and processing genomic data for training and evaluation.
    
    This module handles the loading of genomic regions, their corresponding sequences,
    and various data augmentation techniques. It supports different data types:
    - Profile data: For single-region analysis
    - Long-range data: For analyzing interactions between regions
    
    The module implements different sampling strategies for training, validation and testing:
    - Train: peaks + negative_sampling_ratio (0.1) of negatives, sampled each epoch
    - Val: peaks + negatives_sampling_ratio (1) of negatives, sampled once and fixed
    - Test: peaks + negatives, no sampling
    
    Attributes:
        config: Configuration object containing data loading parameters
        dataset_class: The dataset class to use (ChromBPNetBatchGenerator or LongRangeDataset)
        peaks: DataFrame containing peak regions
        negatives: DataFrame containing negative regions
        data: Combined DataFrame of peaks and negatives
        train_chroms: List of chromosomes used for training
        val_chroms: List of chromosomes used for validation
        test_chroms: List of chromosomes used for testing
    """
    
    def __init__(self, config):
        """Initialize the DataModule.
        
        Args:
            config: Configuration object containing data loading parameters
        """
        super().__init__()
        self.config = config

        # Set dataset class based on data type
        if self.config.data_type == 'profile':
            self.dataset_class = ChromBPNetDataset
        else:
            raise NotImplementedError(f'Unsupported data type: {self.config.data_type}')

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
            is_peak=True
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

        if self.config.background is not None:
            self.background = load_region_df(
                self.config.background,
                chrom_sizes=self.config.chrom_sizes,
                in_window=self.config.in_window,
                shift=self.config.shift,
                is_peak=False,
            )
            print(self.background.head())

        if self.config.debug:
            self._debug_subsample()

    def _debug_subsample(self):
        """Subsample data for debugging purposes."""
        # chagpt: you must pass random_state explicitly, or the result will be random every time, even
        # if you called np.random.seed(seed) before (eg during set_random_seed)
        self.peaks = self.peaks.sample(n=int(0.01*len(self.peaks)), random_state=42)
        if self.negatives is not None:
            self.negatives = self.negatives.sample(n=int(0.1*len(self.peaks)), random_state=42)
            self.data = pd.concat([self.peaks, self.negatives], ignore_index=True)
        else:
            self.data = self.peaks

    def _setup_chromosomes(self):
        """Setup chromosome lists for training, validation and testing."""
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

    # TODO go through
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
                batch_size=config.batch_size,
                inputlen=config.in_window,                                        
                outputlen=config.out_window,
                max_jitter=config.shift,
                negative_sampling_ratio=config.negative_sampling_ratio,
                cts_bw_file=config.bigwig,
                add_revcomp=True,
                return_coords=False,
                shuffle_at_epoch_start=False,
            )
            self.val_dataset = self.dataset_class(
                peak_regions=val_peaks,
                nonpeak_regions=val_nonpeaks,
                genome_fasta=config.fasta,
                batch_size=config.batch_size,
                inputlen=config.in_window,                                        
                outputlen=config.out_window,
                max_jitter=0,
                negative_sampling_ratio=-1 if config.data_type == 'profile' else config.negative_sampling_ratio,
                cts_bw_file=config.bigwig,
                add_revcomp=False,
                return_coords=False,
                shuffle_at_epoch_start=False, 
            )
        elif stage == 'test':
            test_peaks, test_nonpeaks = split_peak_and_nonpeak(self.test_data)
            self.test_dataset = self.dataset_class(
                peak_regions=test_peaks,
                nonpeak_regions=test_nonpeaks,  
                genome_fasta=config.fasta,
                batch_size=config.batch_size,
                inputlen=config.in_window,                                        
                outputlen=config.out_window,
                max_jitter=0,
                negative_sampling_ratio=-1 if config.data_type == 'profile' else config.negative_sampling_ratio,
                cts_bw_file=config.bigwig,
                add_revcomp=False,
                return_coords=False,
                shuffle_at_epoch_start=False, 
            )

        print(f'Data setup complete in {time() - t0:.2f} seconds')

    @cached_property
    def median_count(self):
        import pyBigWig
        # Calculate median count to get weight of count loss
        self.train_val_subsampled = concat_peaks_and_subsampled_negatives(self.train_val, negative_sampling_ratio=self.config.negative_sampling_ratio)
        counts_subsampled = get_cts(self.train_val_subsampled, pyBigWig.open(self.config.bigwig), self.config.out_window).sum(-1)
        return np.median(counts_subsampled)

    def train_dataloader(self):
        # TODO: consider removing this since it is already called from the ctor of train_dataset...
        self.train_dataset.crop_revcomp_data()
        
        return torch.utils.data.DataLoader(
            self.train_dataset, 
            batch_size=self.config.batch_size,
            shuffle=True, 
            drop_last=False,
            num_workers=self.config.num_workers, 
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=False, 
            num_workers=self.config.num_workers, 
        )
    
    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=False, 
            num_workers=self.config.num_workers, 
        )

    def negative_dataloader(self):
        self.negative_dataset = self.dataset_class(
            peak_regions=self.negatives,
            nonpeak_regions=None,
            genome_fasta=self.config.fasta,
            batch_size=self.config.batch_size,
            inputlen=self.config.in_window,
            outputlen=self.config.out_window,
            max_jitter=0,
            negative_sampling_ratio=-1,
            cts_bw_file=self.config.bigwig,
            add_revcomp=False,
            return_coords=False,
            shuffle_at_epoch_start=False,
        )
        return torch.utils.data.DataLoader(
            self.negative_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=False, 
            num_workers=self.config.num_workers, 
        )

    def chrom_dataloader(self, chrom='chr1', negative_sampling_ratio=-1):
        dataset = self.chrom_dataset(chrom=chrom, negative_sampling_ratio=negative_sampling_ratio)
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
        ), dataset

    def chrom_dataset(self, chrom='chr1', negative_sampling_ratio=-1):
        if isinstance(chrom, str):
            if chrom in ['train', 'val', 'test']:
                chrom = getattr(self, f'{chrom}_chroms')
            elif chrom == 'all':
                chrom = self.chroms
            else:
                chrom = [chrom]

        regions = self.data[self.data.iloc[:, 0].isin(chrom)].reset_index(drop=True)
        peaks, nonpeaks = split_peak_and_nonpeak(regions)
        if (negative_sampling_ratio > 0) and (len(nonpeaks) > (len(peaks) * negative_sampling_ratio)):
            nonpeaks = nonpeaks.sample(n=int(negative_sampling_ratio * len(peaks)), replace=False)
            regions = pd.concat([peaks, nonpeaks], ignore_index=True)

        dataset = self.dataset_class(
            peak_regions=peaks,
            nonpeak_regions=nonpeaks,
            genome_fasta=self.config.fasta,
            batch_size=self.config.batch_size,
            inputlen=self.config.in_window,
            outputlen=self.config.out_window,
            max_jitter=0,
            negative_sampling_ratio=-1,
            cts_bw_file=self.config.bigwig,
            add_revcomp=False,
            return_coords=False,
            shuffle_at_epoch_start=False,
        )
        return dataset

class ChromBPNetDataset(torch.utils.data.Dataset):
    """Generator for genomic sequence data with random cropping and reverse complement augmentation.
    
    This generator randomly crops (=jitter) and applies reverse complement augmentation to training examples
    for every epoch. It handles both peak and non-peak regions, with configurable sampling ratios.
    """
    
    def __init__(
        self, 
        peak_regions, 
        nonpeak_regions, 
        genome_fasta, 
        batch_size, 
        inputlen, 
        outputlen, 
        max_jitter, 
        negative_sampling_ratio, 
        cts_bw_file, 
        add_revcomp, 
        return_coords, 
        shuffle_at_epoch_start, 
        **kwargs
    ):
        """Initialize the generator.
        
        Args:
            peak_regions: DataFrame containing peak regions
            nonpeak_regions: DataFrame containing non-peak regions
            genome_fasta: Path to genome FASTA file
            batch_size: Size of batches
            inputlen: Length of input sequences
            outputlen: Length of output sequences
            max_jitter: Maximum jitter for random cropping
            negative_sampling_ratio: Ratio of negative samples to use
            cts_bw_file: Path to bigwig file containing counts
            add_revcomp: Whether to add reverse complement augmentation
            return_coords: Whether to return coordinates
            shuffle_at_epoch_start: Whether to shuffle at epoch start
            **kwargs: Additional keyword arguments
        """
        # Load data
        peak_seqs, peak_cts, peak_coords, nonpeak_seqs, nonpeak_cts, nonpeak_coords = load_data(
            peak_regions, nonpeak_regions, genome_fasta, cts_bw_file, inputlen, outputlen, max_jitter
        )

        # Store data
        self.peak_seqs, self.nonpeak_seqs = peak_seqs, nonpeak_seqs
        self.peak_cts, self.nonpeak_cts = peak_cts, nonpeak_cts
        self.peak_coords, self.nonpeak_coords = peak_coords, nonpeak_coords

        # Store parameters
        self.negative_sampling_ratio = negative_sampling_ratio
        self.inputlen = inputlen
        self.outputlen = outputlen
        self.batch_size = batch_size
        self.add_revcomp = add_revcomp
        self.return_coords = return_coords
        self.shuffle_at_epoch_start = shuffle_at_epoch_start

        # Initialize data
        self.crop_revcomp_data()

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.cur_seqs)

    def crop_revcomp_data(self):
        """Apply random cropping and reverse complement augmentation to the data.
        
        This method:
        1. Randomly crops peak data to inputlen and outputlen
        2. Samples negative examples according to negative_sampling_ratio
        3. Applies reverse complement augmentation if enabled
        4. Shuffles data if shuffle_at_epoch_start is True
        """
        if (self.peak_seqs is not None) and (self.nonpeak_seqs is not None):
            # Crop peak data
            cropped_peaks, cropped_cnts, cropped_coords = random_crop(
                self.peak_seqs, self.peak_cts, self.inputlen, self.outputlen, self.peak_coords
            )
            
            # Sample negative examples
            if self.negative_sampling_ratio > 0:
                self.sampled_nonpeak_seqs, self.sampled_nonpeak_cts, self.sampled_nonpeak_coords = subsample_nonpeak_data(
                    self.nonpeak_seqs, self.nonpeak_cts, self.nonpeak_coords,
                    len(self.peak_seqs), self.negative_sampling_ratio
                )
                self.seqs = np.vstack([cropped_peaks, self.sampled_nonpeak_seqs])
                self.cts = np.vstack([cropped_cnts, self.sampled_nonpeak_cts])
                self.coords = np.vstack([cropped_coords, self.sampled_nonpeak_coords])
            else:
                self.seqs = np.vstack([cropped_peaks, self.nonpeak_seqs])
                self.cts = np.vstack([cropped_cnts, self.nonpeak_cts])
                self.coords = np.vstack([cropped_coords, self.nonpeak_coords])
        elif self.peak_seqs is not None:
            # Only peak data
            cropped_peaks, cropped_cnts, cropped_coords = random_crop(
                self.peak_seqs, self.peak_cts, self.inputlen, self.outputlen, self.peak_coords
            )
            self.seqs = cropped_peaks
            self.cts = cropped_cnts
            self.coords = cropped_coords
        elif self.nonpeak_seqs is not None:
            # Only non-peak data
            self.seqs = self.nonpeak_seqs
            self.cts = self.nonpeak_cts
            self.coords = self.nonpeak_coords
        else:
            raise ValueError("Both peak and non-peak arrays are empty")

        # Apply augmentation
        self.cur_seqs, self.cur_cts, self.cur_coords = crop_revcomp_augment(
            self.seqs, self.cts, self.coords, self.add_revcomp, shuffle=self.shuffle_at_epoch_start,
        )
        self.regions = pd.DataFrame(self.cur_coords, columns=['chrom', 'start', 'forward_or_reverse', 'is_peak'])
        # print('Regions', self.regions['is_peak'].value_counts())

    def __getitem__(self, idx):
        """Get a sample from the dataset.
        
        Args:
            idx: Index of the sample to get
            
        Returns:
            Dictionary containing:
                - onehot_seq: One-hot encoded sequence
                - profile: Profile data
        """
        return {
            'onehot_seq': self.cur_seqs[idx].astype(np.float32).transpose(),
            'profile': self.cur_cts[idx].astype(np.float32),
        }
