from dataclasses import dataclass
from typing import List, Any, Union
from argparse import ArgumentParser, Namespace
import os
import json
from histobpnet.utils.parse_utils import add_argparse_args, from_argparse_args
from histobpnet.utils.general_utils import is_histone
from .genome import hg38, hg38_datasets, mm10, mm10_datasets

@dataclass
class DataConfig:
    """Base configuration class for data handling.
    
    This class defines the common parameters used across different data types
    and provides validation for these parameters.
    """
    
    def __init__(
        self,
        data_dir: str = None,
        peaks: str = None,
        negatives: str = None,
        bigwig: str = None,
        bigwig_ctrl: str = None,
        negative_sampling_ratio: float = 0.1,
        saved_data: str = None,
        fasta: str = None,
        chrom_sizes: str = None,
        fold: int = 0,
        genome: str = 'hg38',
        in_window: int = 2114,
        out_window: int = 1000,
        output_bins: str = "",
        atac_hgp_map: str = "",
        ctrl_scaling_factor: float = 1.0,
        shift: int = 500,
        rc_frac: float = 0.5,
        outlier_threshold: float = 0.999,
        data_type: str = 'profile',
        exclude_chroms: List = None,
        batch_size: int = 64,
        num_workers: int = 32,
        debug: bool = False,
    ):
        if genome == 'hg38':
            _genome = hg38
            _datasets = hg38_datasets()
        elif genome == 'mm10':
            _genome = mm10
            _datasets = mm10_datasets()
        else:
            raise ValueError(f"Unsupported genome: {genome}")
        
        self.data_dir = data_dir
        self.peaks = peaks if peaks is not None else f'{data_dir}/peaks.bed'
        self.negatives = negatives if negatives is not None else f'{data_dir}/negatives.bed'
        self.bigwig = bigwig if bigwig is not None else f'{data_dir}/unstranded.bw'
        self.bigwig_ctrl = bigwig_ctrl
        
        self.fasta = fasta if fasta is not None else _genome.fasta
        self.chrom_sizes = chrom_sizes if chrom_sizes is not None else _genome.chrom_sizes

        fold_file_path = _datasets.fetch(f'fold_{fold}.json', progressbar=False)
        splits_dict = json.load(open(fold_file_path))
        self.training_chroms = splits_dict['train']
        self.validation_chroms = splits_dict['valid']
        self.test_chroms = splits_dict['test']

        self.exclude_chroms = [] if exclude_chroms is None else exclude_chroms
        self.negative_sampling_ratio = negative_sampling_ratio
        self.saved_data = saved_data
        self.in_window = in_window
        self.out_window = out_window
        self.output_bins = output_bins
        self.atac_hgp_map = atac_hgp_map
        self.ctrl_scaling_factor = ctrl_scaling_factor
        self.shift = shift
        self.rc_frac = rc_frac
        self.outlier_threshold = outlier_threshold
        self.data_type = data_type
        self.batch_size = batch_size    
        self.num_workers = num_workers
        self.debug = debug
        self.fold = fold

        self.__post_init__()
        
    @classmethod
    def add_argparse_args(cls, parent_parser: ArgumentParser, **kwargs: Any):
        return add_argparse_args(cls, parent_parser, **kwargs)

    @classmethod
    def from_argparse_args(
        cls, args: Union[Namespace, ArgumentParser], **kwargs: Any
    ):
        return from_argparse_args(cls, args, **kwargs)
    
    def __post_init__(self):
        """Validate configuration parameters after initialization."""
        self._validate_paths()
        self._validate_windows()
        self._validate_chromosomes()
        self._validate_data_type()

    def _validate_paths(self):
        """Validate that all required files exist."""
        required_files = {
            'FASTA': self.fasta,
            'BigWig': self.bigwig,
            'BigWigCtrl': self.bigwig_ctrl,
            'ATAC_HGP_MAP': self.atac_hgp_map,
            'Peaks': self.peaks
        }
        
        for name, path in required_files.items():
            if name not in ['BigWigCtrl', 'ATAC_HGP_MAP']:
                if not os.path.exists(path):
                    raise FileNotFoundError(f"{name} file not found: {path}")
            else:
                if path is not None and not os.path.exists(path):
                    raise FileNotFoundError(f"{name} file not found: {path}")
    
    def _validate_windows(self):
        """Validate window size parameters."""
        if self.in_window <= 0:
            raise ValueError("Input window size must be positive")
        if self.out_window < 0:
            raise ValueError("Output window size must be positive")
        if self.in_window < self.out_window:
            raise ValueError("Input window must be larger than output window")
        if self.output_bins != "":
            bins = [int(b.strip()) for b in self.output_bins.split(',')]
            # if sum(bins) != self.out_window:
                # raise ValueError("Sum of output bins must equal out_window")
            if any(b <= 0 for b in bins):
                raise ValueError("All output bins must be positive integers")
    
    def _validate_chromosomes(self):
        """Validate chromosome configuration."""
        all_chroms = set(self.training_chroms + self.validation_chroms + self.test_chroms)
        excluded = set(self.exclude_chroms)
        
        if not all_chroms:
            raise ValueError("No chromosomes specified for training, validation, or testing")
        
        if excluded.intersection(all_chroms) != excluded:
            raise ValueError("Some excluded chromosomes are not in the training/validation/test sets")
    
    def _validate_data_type(self):
        """Validate data type parameter."""
        if self.data_type not in ['profile', 'longrange'] and not is_histone(self.data_type):
            raise ValueError("Data type must be either 'profile', 'longrange', or 'histobpnet_v*'")