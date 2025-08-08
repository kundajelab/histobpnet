import torch.nn as nn
from histobpnet.model.bpnet import BPNet

# adapted from BPNet in bpnet-lite, credit goes to Jacob Schreiber <jmschreiber91@gmail.com>
class ChromBPNet(nn.Module):
    """A ChromBPNet model.

    ChromBPNet is an extension of BPNet to handle chromatin accessibility data,
    in contrast to the protein binding data that BPNet handles. The distinction
    between these data types is that an enzyme used in DNase-seq and ATAC-seq
    experiments itself has a soft sequence preference, meaning that the
    strength of the signal is driven by real biology but that the exact read
    mapping locations are driven by the soft sequence bias of the enzyme.

    ChromBPNet handles this by treating the data using two models: a bias
    model that is initially trained on background (non-peak) regions where
    the bias dominates, and an accessibility model that is subsequently trained
    using a frozen version of the bias model. The bias model learns to remove
    the enzyme bias so that the accessibility model can learn real motifs.
    """

    def __init__(
        self, 
        config,
    ):
        super().__init__()

        self.model = BPNet(        
            out_dim = config.out_dim,
            n_filters = config.n_filters, 
            n_layers = config.n_layers, 
            conv1_kernel_size = config.conv1_kernel_size,
            profile_kernel_size = config.profile_kernel_size,
            n_outputs = config.n_outputs, 
            n_control_tracks = config.n_control_tracks, 
            profile_output_bias = config.profile_output_bias, 
            count_output_bias = config.count_output_bias, 
        )

        self.bias = BPNet(
            out_dim = config.out_dim,
            n_layers = 4,
            n_filters = 128
        )

        self._log = _Log()
        self._exp1 = _Exp()
        self._exp2 = _Exp()

    def forward(self, x):
        """A forward pass through the network.

        This function is usually accessed through calling the model, e.g.
        doing `model(x)`. The method defines how inputs are transformed into
        the outputs through interactions with each of the layers.

        Parameters
        ----------
        x: torch.tensor, shape=(batch_size, 4, 2114)
            A one-hot encoded sequence tensor.

        Returns
        -------
        y_profile: torch.tensor, shape=(batch_size, 1000)
            The predicted logit profile for each example. Note that this is not
            a normalized value.
        """
        acc_profile, acc_counts = self.model(x)
        bias_profile, bias_counts = self.bias(x)

        y_profile = acc_profile + bias_profile
        # combine the two logit outputs via log-sum-exp
        y_counts = self._log(self._exp1(acc_counts) + self._exp2(bias_counts))
        
        # DO NOT SQUEEZE y_counts, as it is needed for running deep_lift_shap
        return y_profile.squeeze(1), y_counts #.squeeze() 