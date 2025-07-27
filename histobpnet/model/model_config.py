from argparse import ArgumentParser, Namespace
from typing import Any, Union
from histobpnet.utils.parse_utils import add_argparse_args, from_argparse_args

class BaseConfig:
    model_type = "base"

    @classmethod
    def add_argparse_args(cls, parent_parser: ArgumentParser, **kwargs: Any):
        return add_argparse_args(cls, parent_parser, **kwargs)

    @classmethod
    def from_argparse_args(
        cls, args: Union[Namespace, ArgumentParser], **kwargs: Any
    ):
        return from_argparse_args(cls, args, **kwargs)
    
class ChromBPNetConfig(BaseConfig):
    model_type = "chrombpnet"

    def __init__(
        self,
        out_dim: int=1000,
        n_filters: int=512, 
        n_layers: int=8, 
        conv1_kernel_size: int=21,
        profile_kernel_size: int=75,
        n_outputs: int=1, 
        n_control_tracks: int=0, 
        profile_output_bias: int=True, 
        count_output_bias: int=True, 
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.out_dim = out_dim
        self.n_filters = n_filters
        self.n_layers = n_layers
        self.conv1_kernel_size = conv1_kernel_size
        self.profile_kernel_size = profile_kernel_size
        self.n_outputs = n_outputs
        self.n_control_tracks = n_control_tracks
        self.profile_output_bias = profile_output_bias
        self.count_output_bias = count_output_bias