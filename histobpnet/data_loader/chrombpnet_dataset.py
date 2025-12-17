import torch
import numpy as np
import pandas as pd
from histobpnet.utils.data_utils import (
    load_data,
    crop_revcomp_data,
)
from histobpnet.data_loader.data_config import DataConfig

def validate_mode(mode: str):
    valids = ['train', 'val', 'test', 'chrom', 'negative']
    assert mode in valids, f"Invalid mode: {mode}. Must be one of {valids}"
    
class ChromBPNetDataset(torch.utils.data.Dataset):
    """Generator for genomic sequence data with random cropping and reverse complement augmentation.
    
    This generator randomly crops (=jitter) and applies reverse complement augmentation to training examples
    for every epoch. It handles both peak and non-peak regions, with configurable sampling ratios.
    """
    
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
        """Initialize the generator.
        
        Args:
            peak_regions: DataFrame containing peak regions
            nonpeak_regions: DataFrame containing non-peak regions
            inputlen: Length of input sequences
            outputlen: Length of output sequences
            max_jitter: Maximum jitter for random cropping
            negative_sampling_ratio: Ratio of negative samples to use
            shuffle_at_epoch_start: Whether to shuffle at epoch start
            **kwargs: Additional keyword arguments
        """
        validate_mode(mode)

        # Load data
        self.peak_seqs, self.peak_cts, _, self.peak_coords, \
        self.nonpeak_seqs, self.nonpeak_cts, _, self.nonpeak_coords = load_data(
            peak_regions, nonpeak_regions, config.fasta, config.bigwig,
            inputlen, outputlen, max_jitter, mode=mode,
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
        self.cur_seqs, self.cur_cts, _, self.cur_coords, self.cur_peak_status = crop_revcomp_data(
            self.peak_seqs, self.peak_cts, None, self.peak_coords,
            self.nonpeak_seqs, self.nonpeak_cts, None, self.nonpeak_coords,
            inputlen=self.inputlen,
            outputlen=self.outputlen,
            negative_sampling_ratio=self.negative_sampling_ratio,
            shuffle=self.shuffle_at_epoch_start,
            rc_frac=self.rc_frac,
        )

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
            'peak_status': self.cur_peak_status[idx].astype(int),
        }
