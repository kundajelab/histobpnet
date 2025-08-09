from dataclasses import dataclass
from typing import List, Any, Union
from argparse import ArgumentParser, Namespace
import os
import json
from histobpnet.utils.parse_utils import add_argparse_args, from_argparse_args

@dataclass
class DataConfig:
    """Base configuration class for data handling.
    
    This class defines the common parameters used across different data types
    and provides validation for these parameters.
    """
    
    def __init__(
        self,
        data_dir: str = "",
        peaks: str = '{}/peaks.bed',
        negatives: str = '{}/negatives.bed', 
        bigwig: str = '{}/unstranded.bw', 
        plus: str = '{}/plus.bw',
        minus: str = '{}/minus.bw',
        background: str = None,
        ctl_plus: str = None, 
        ctl_minus: str = None, 
        negative_sampling_ratio: float = 0.1,
        saved_data: str = None,
        fasta: str = "",
        chrom_sizes: str = "",
        fold: int = 0,
        fold_path: str = "",
        in_window: int = 2114,
        out_window: int = 1000,
        shift: int = 500,
        rc: float = 0.5,
        outlier_threshold: float = 0.999,
        data_type: str = 'profile',
        training_chroms: List = None,
        validation_chroms: List = None,
        test_chroms: List = None,
        exclude_chroms: List = None,
        batch_size: int = 64,
        num_workers: int = 32,
        debug: bool = False,
    ):
        self.data_dir = data_dir
        self.peaks = peaks.format(data_dir)
        self.negatives = negatives.format(data_dir)
        self.bigwig = bigwig.format(data_dir)
        self.plus = plus.format(data_dir)
        self.minus = minus.format(data_dir)
        self.background = background
        self.ctl_plus = ctl_plus
        self.ctl_minus = ctl_minus
        self.negative_sampling_ratio = negative_sampling_ratio
        self.saved_data = saved_data
        self.fasta = fasta
        self.chrom_sizes = chrom_sizes
        self.in_window = in_window
        self.out_window = out_window
        self.shift = shift
        self.rc = rc
        self.outlier_threshold = outlier_threshold
        self.data_type = data_type
        self.training_chroms = training_chroms
        self.validation_chroms = validation_chroms
        self.test_chroms = test_chroms
        self.exclude_chroms = exclude_chroms
        self.batch_size = batch_size    
        self.num_workers = num_workers
        self.debug = debug
        self.fold = fold
        # this should be in validation functions but we need it here to load the chrom, so oh well
        assert os.path.isfile(fold_path) and fold_path.endswith('.json'), f"Fold path must be a valid JSON file: {fold_path}"
        splits_dict = json.load(open(fold_path))
        self.training_chroms = splits_dict['train'] if training_chroms is None else training_chroms
        self.validation_chroms = splits_dict['valid'] if validation_chroms is None else validation_chroms
        self.test_chroms = splits_dict['test'] if test_chroms is None else test_chroms
        self.exclude_chroms = [] if exclude_chroms is None else exclude_chroms
        
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
            # TODO What is this for? I dont know where to find it...
            # 'BigWig': self.bigwig,
            'Peaks': self.peaks
        }
        
        for name, path in required_files.items():
            if not os.path.exists(path):
                raise FileNotFoundError(f"{name} file not found: {path}")
    
    def _validate_windows(self):
        """Validate window size parameters."""
        if self.in_window <= 0:
            raise ValueError("Input window size must be positive")
        if self.out_window <= 0:
            raise ValueError("Output window size must be positive")
        if self.in_window < self.out_window:
            raise ValueError("Input window must be larger than output window")
    
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
        if self.data_type not in ['profile', 'longrange']:
            raise ValueError("Data type must be either 'profile' or 'longrange'")
