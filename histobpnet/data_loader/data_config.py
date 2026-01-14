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
        # catch-all for args that we cannot add as standalone kwargs because
        # they clash with an arg of the same name in BPNetodelConfig
        extra_kwargs: dict = None,
        # these are parameters with common default values
        peaks: str = None,
        negatives: str = None,
        bigwig: str = None,
        bigwig_ctrl: str = None,
        fasta: str = "/large_storage/goodarzilab/valehvpa/refs/hg38/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta",
        chrom_sizes: str = "/large_storage/goodarzilab/valehvpa/refs/hg38/hg38.chrom.sizes",
        saved_data: str = None,
        negative_sampling_ratio: float = 0.1,
        fold: int = 0,
        genome: str = 'hg38',
        in_window: int = 2114,
        atac_hgp_map: str = None,
        outlier_threshold: float = 0.999,
        skip_missing_hist: str = "N/A",
        pass_zero_mode: str = "N/A",
        exclude_chroms: List = None,
        batch_size: int = 64,
        num_workers: int = 16,
        debug: bool = False,
        deep_shap_type='counts', 
        deep_shap_batch_size=64, 
        modisco_max_seqlets=1000_000, 
        modisco_width=500, 
        # these are parameters whose default values can vary based on model_type
        shift: int = None,
        rc_frac: float = None,
        out_window: int = None,
        ctrl_scaling_factor: float = None,
        outputlen_neg: int = None,
    ):
        assert "model_type" in extra_kwargs
        self.model_type = extra_kwargs["model_type"]
    
        assert "output_bins" in extra_kwargs
        self.output_bins = extra_kwargs["output_bins"]
        if self.output_bins is not None:
            bins = [int(b.strip()) for b in self.output_bins.split(',')]
            if any(b <= 0 for b in bins):
                raise ValueError("All output bins must be positive integers")

        if genome == 'hg38':
            _datasets = hg38_datasets()
        elif genome == 'mm10':
            _datasets = mm10_datasets()
        else:
            raise ValueError(f"Unsupported genome: {genome}")
        
        self.fasta = fasta
        self.chrom_sizes = chrom_sizes
        self.peaks = peaks
        self.negatives = negatives
        self.bigwig = bigwig
        self.bigwig_ctrl = bigwig_ctrl

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
        self.atac_hgp_map = atac_hgp_map
        self.skip_missing_hist = skip_missing_hist
        self.pass_zero_mode = pass_zero_mode
        self.ctrl_scaling_factor = ctrl_scaling_factor
        self.outputlen_neg = outputlen_neg
        self.shift = shift
        self.rc_frac = rc_frac
        self.outlier_threshold = outlier_threshold
        self.batch_size = batch_size 
        self.num_workers = num_workers
        self.debug = debug
        self.fold = fold
        self.deep_shap_type = deep_shap_type
        self.deep_shap_batch_size = deep_shap_batch_size
        self.modisco_max_seqlets = modisco_max_seqlets
        self.modisco_width = modisco_width

        if self.shift is None:
            if self.model_type == "chrombpnet":
                self.shift = 500
            elif is_histone(self.model_type):
                self.shift = 0

        if self.rc_frac is None:
            if self.model_type == "chrombpnet":
                self.rc_frac = 0.5
            elif is_histone(self.model_type):
                self.rc_frac = 0

        if self.out_window is None:
            if self.model_type == "chrombpnet":
                self.out_window = 1000
            elif is_histone(self.model_type):
                self.out_window = 0

        if self.ctrl_scaling_factor is None:
            if is_histone(self.model_type):
                self.ctrl_scaling_factor = 1.0

        if self.outputlen_neg is None:
            if is_histone(self.model_type):
                self.outputlen_neg = 1000

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
        self._validate_model_type()
        self._validate_output_bins()
        self._validate_skip_missing_hist()
        self._validate_pass_zero_mode()

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
        # TODO deal witht this
        # if (not is_histone(self.model_type)) and (self.in_window < self.out_window):
        #     raise ValueError("Input window must be larger than output window")
    
    def _validate_chromosomes(self):
        """Validate chromosome configuration."""
        all_chroms = set(self.training_chroms + self.validation_chroms + self.test_chroms)
        excluded = set(self.exclude_chroms)
        
        if not all_chroms:
            raise ValueError("No chromosomes specified for training, validation, or testing")
        
        if excluded.intersection(all_chroms) != excluded:
            raise ValueError("Some excluded chromosomes are not in the training/validation/test sets")
    
    def _validate_model_type(self):
        if self.model_type not in ['chrombpnet'] and not is_histone(self.model_type):
            raise ValueError(f"Unrecognized model_type: {self.model_type}")

    def _validate_output_bins(self):
        if self.output_bins is not None:
            bins = [int(b.strip()) for b in self.output_bins.split(',')]
            if any(b <= 0 for b in bins):
                raise ValueError("All output bins must be positive integers")

    def _validate_skip_missing_hist(self):
        valids = ["Yes", "No", "N/A"]
        assert self.skip_missing_hist in valids

    def _validate_pass_zero_mode(self):
        valids = ["zero_seq", "zero_ctl", "zero_cts", "N/A"]
        assert self.pass_zero_mode in valids