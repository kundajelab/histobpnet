import torch.nn as nn
from histobpnet.model.bpnet import BPNet
from histobpnet.model.model_config import BPNetModelConfig
from histobpnet.utils.general_utils import _Log, _Exp

# adapted from BPNet in bpnet-lite, credit goes to Jacob Schreiber <jmschreiber91@gmail.com>

class ChromBPNet(nn.Module):
    """A ChromBPNet model.
    See https://github.com/jmschrei/bpnet-lite/tree/master?tab=readme-ov-file#chrombpnet
    """

    def __init__(
        self, 
        config: BPNetModelConfig,
    ):
        super().__init__()

        # TODO_later call this acc instead of model
        self.model = BPNet(
            n_filters = config.n_filters, 
            n_layers = config.n_layers, 
            out_dim = config.out_dim,
            conv1_kernel_size = config.conv1_kernel_size,
            profile_kernel_size = config.profile_kernel_size,
            n_outputs = config.n_outputs, 
            n_control_tracks = config.n_control_tracks, 
            profile_output_bias = config.profile_output_bias, 
            count_output_bias = config.count_output_bias, 
        )

        self.bias = BPNet(
            n_filters = 128,
            n_layers = 4,
            # the rest is same as self.model
            out_dim = config.out_dim,
            conv1_kernel_size = config.conv1_kernel_size,
            profile_kernel_size = config.profile_kernel_size,
            n_outputs = config.n_outputs, 
            n_control_tracks = config.n_control_tracks, 
            profile_output_bias = config.profile_output_bias, 
            count_output_bias = config.count_output_bias, 
        )

        self._log = _Log()
        self._exp1 = _Exp()
        self._exp2 = _Exp()

        self.n_control_tracks = config.n_control_tracks
        self.config = config

        self.tf_style_reinit()

    def get_model_config(self):
        return self.config
    
    def tf_style_reinit(self):
        """
        Re-initializes model weights for Linear and Conv1d layers using
        TensorFlow's default: Xavier/Glorot uniform for weights, zeros for bias.
        Operates in-place!
        """
        # print("Reinitializing with TF strategy")
        # the initialization is only for accessibility model (self.model), since the bias is often trained separately
        # and loaded directly. If that's not the case, however, we may want to init both. TODO
        for m in self.model.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.xavier_uniform_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """A forward pass through the network.

        Parameters
        ----------
        x: torch.tensor, shape=(batch_size, 4, 2114)
            A one-hot encoded sequence tensor.

        Returns
        -------
        y_profile: torch.tensor, shape=(batch_size, 1000)
            The predicted logit profile for each example. Note that this is not
            a normalized value.
        y_counts: torch.tensor, shape=(batch_size,)
            The predicted log-count for each example.
        """
        acc_profile, acc_counts = self.model(x)
        bias_profile, bias_counts = self.bias(x)

        y_profile = acc_profile + bias_profile
        # combine the two logit outputs via log-sum-exp
        y_counts = self._log(self._exp1(acc_counts) + self._exp2(bias_counts))
        
        # DO NOT SQUEEZE y_counts, as it is needed for running deep_lift_shap
        return y_profile.squeeze(1), y_counts #.squeeze() 