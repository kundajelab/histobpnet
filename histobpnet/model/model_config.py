from argparse import ArgumentParser, Namespace
from typing import Any, Union
from histobpnet.utils.parse_utils import add_argparse_args, from_argparse_args
from histobpnet.utils.general_utils import is_histone

class BaseConfig:
    @classmethod
    def add_argparse_args(cls, parent_parser: ArgumentParser, **kwargs: Any):
        return add_argparse_args(cls, parent_parser, **kwargs)

    @classmethod
    def from_argparse_args(
        cls, args: Union[Namespace, ArgumentParser], **kwargs: Any
    ):
        return from_argparse_args(cls, args, **kwargs)

class ChromBPNetConfig(BaseConfig):
    def __init__(
        self,
        model_type: str = "chrombpnet",
        out_dim: int = 1000,
        n_filters: int = 512, 
        n_layers: int = 8, 
        conv1_kernel_size: int = 21,
        profile_kernel_size: int = 75,
        n_outputs: int = 1,
        n_control_tracks: int = 0, 
        profile_output_bias: bool = True, 
        count_output_bias: bool = True, 
        output_bins: str = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.model_type = model_type
        self.out_dim = out_dim
        self.n_filters = n_filters
        self.n_layers = n_layers
        self.conv1_kernel_size = conv1_kernel_size
        self.profile_kernel_size = profile_kernel_size
        self.n_outputs = n_outputs
        self.n_control_tracks = n_control_tracks
        self.profile_output_bias = profile_output_bias
        self.count_output_bias = count_output_bias
        # TODO_later deal witht his
        self.output_bins = output_bins

class HistoBPNetConfigV1(BaseConfig):
    def __init__(
        self,
        model_type: str = "histobpnet_v1",
        out_dim: int = 0,
        n_filters: int = 512, 
        n_layers: int = 8, 
        conv1_kernel_size: int = 21,
        profile_kernel_size: int = 0,
        n_outputs: int = 1,
        profile_output_bias: bool = True, 
        count_output_bias: bool = True, 
        # TODO_NOW fix bins
        output_bins: str = "1000, 2000, 4000, 8000, 16000",
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.model_type = model_type
        self.out_dim = out_dim
        self.n_filters = n_filters
        self.n_layers = n_layers
        self.conv1_kernel_size = conv1_kernel_size
        self.profile_kernel_size = profile_kernel_size
        self.n_outputs = n_outputs
        self.profile_output_bias = profile_output_bias
        self.count_output_bias = count_output_bias
        self.output_bins = [int(x) for x in output_bins.split(",")]
        self.n_control_tracks = len(self.output_bins)
        self.n_count_outputs = len(self.output_bins)

class HistoBPNetConfigV2(BaseConfig):
    def __init__(
        self,
        model_type: str = "histobpnet_v2",
        out_dim: int = 0,
        n_filters: int = 512, 
        n_layers: int = 8, 
        conv1_kernel_size: int = 21,
        profile_kernel_size: int = 0,
        n_outputs: int = 1,
        output_bins: str = None,
        profile_output_bias: bool = True, 
        count_output_bias: bool = True, 
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.model_type = model_type
        self.out_dim = out_dim
        self.n_filters = n_filters
        self.n_layers = n_layers
        self.conv1_kernel_size = conv1_kernel_size
        self.profile_kernel_size = profile_kernel_size
        self.n_outputs = n_outputs
        self.profile_output_bias = profile_output_bias
        self.count_output_bias = count_output_bias
        self.n_control_tracks = 1
        self.n_count_outputs = 1
        # TODO_later deal witht his
        self.output_bins = output_bins

# class BPNetModelConfig(BaseConfig):
#     def __init__(
#         self,
#         model_type: str = "",
#         out_dim: int = None,
#         n_filters: int = 512, 
#         n_layers: int = 8, 
#         conv1_kernel_size: int = 21,
#         profile_kernel_size: int = None,
#         n_outputs: int = 1,
#         n_control_tracks: int = None, 
#         output_bins: str = None,
#         profile_output_bias: bool = True, 
#         count_output_bias: bool = True,
#         n_count_outputs: int = None,
#         **kwargs,
#     ):
#         super().__init__(**kwargs)

#         assert (model_type in "chrombpnet") or is_histone(model_type), f"Unsupported model_type: {model_type}"
#         self.model_type = model_type
#         self.out_dim = out_dim
#         self.n_filters = n_filters
#         self.n_layers = n_layers
#         self.conv1_kernel_size = conv1_kernel_size
#         self.profile_kernel_size = profile_kernel_size
#         self.n_outputs = n_outputs
#         self.n_control_tracks = n_control_tracks
#         self.output_bins = output_bins
#         self.profile_output_bias = profile_output_bias
#         self.count_output_bias = count_output_bias
#         self.n_count_outputs = n_count_outputs

#         if self.out_dim is None:
#             if model_type == "chrombpnet":
#                 self.out_dim = 1000
#             elif is_histone(model_type):
#                 self.out_dim = 0

#         if self.profile_kernel_size is None:
#             if model_type == "chrombpnet":
#                 self.profile_kernel_size = 75
#             elif is_histone(model_type):
#                 self.profile_kernel_size = 0

#         if self.output_bins is None:
#             if model_type == "histobpnet_v1":
#                 # TODO_NOW fix bins
#                 self.output_bins = "1000, 2000, 4000, 8000, 16000"

#         if self.output_bins is not None:
#             self.output_bins = [int(x) for x in self.output_bins.split(",")]

#         if self.n_control_tracks is None:
#             if model_type == "chrombpnet":
#                 self.n_control_tracks = 0
#             elif model_type == "histobpnet_v1":
#                 self.n_control_tracks = len(self.output_bins)
#             elif model_type == "histobpnet_v2":
#                 self.n_control_tracks = 1

#         if n_count_outputs is None:
#             if model_type == "histobpnet_v1":
#                 self.n_count_outputs = len(self.output_bins)
#             elif model_type == "histobpnet_v2":
#                 self.n_count_outputs = 1