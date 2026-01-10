import torch.nn as nn
from histobpnet.model.bpnet import BPNet
from histobpnet.model.model_config import BPNetModelConfig

class HistoBPNetV3(nn.Module):
    """A HistoPNet model.
    """

    def __init__(
        self, 
        config: BPNetModelConfig,
    ):
        super().__init__()

        n_cc = config.n_control_tracks if config.feed_ctrl else 0
        
        self.bpnet = BPNet(
            n_filters = config.n_filters, 
            n_layers = config.n_layers, 
            out_dim = config.out_dim,
            conv1_kernel_size = config.conv1_kernel_size,
            profile_kernel_size = config.profile_kernel_size,
            n_outputs = config.n_outputs, 
            n_control_tracks = n_cc, 
            profile_output_bias = config.profile_output_bias, 
            count_output_bias = config.count_output_bias, 
            n_count_outputs=config.n_count_outputs,
            for_histone='histobpnet_v3',
            use_linear_w_ctrl=config.use_linear_w_ctrl,
        )

        self.n_control_tracks = n_cc
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
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.xavier_uniform_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, observed_ctrl=None):
        """A forward pass through the network.

        Parameters
        ----------
        x: torch.tensor, shape=(batch_size, 4, 2114)
            A one-hot encoded sequence tensor.

        Returns
        -------
        y_counts: torch.tensor, shape=(batch_size,)
            The predicted log-count for each example.
        observed_ctrl: torch.tensor, shape=(batch_size, n_control_tracks)
            The observed log of the sum of the (scaled) raw input control counts.
        """
        y_counts = self.bpnet(x, x_ctl_hist=observed_ctrl)

        # DO NOT SQUEEZE y_counts (if applicable), as it is needed for running deep_lift_shap
        return y_counts