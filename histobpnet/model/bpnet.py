import torch
import torch.nn.functional as F

# adapted from BPNet in bpnet-lite, credit goes to Jacob Schreiber <jmschreiber91@gmail.com>

class BPNet(torch.nn.Module):
    """A BPNet model.
    See https://github.com/jmschrei/bpnet-lite/tree/master?tab=readme-ov-file#bpnet

    Parameters
    ----------
    n_filters: int, optional
        The number of filters to use per convolution. Default is 64.

    n_layers: int, optional
        The number of dilated residual layers to include in the model.
        Default is 8.

    n_outputs: int, optional
        The number of profile outputs from the model. Generally either 1 or 2 
        depending on if the data is unstranded or stranded. Default is 1.

    n_control_tracks: int, optional
        The number of control tracks to feed into the model. When predicting
        TFs, this is usually 2. When predicting accessibility, this is usually
        0. When 0, this input is removed from the model. Default is 0.

    profile_output_bias: bool, optional
        Whether to include a bias term in the final profile convolution.
        Removing this term can help with attribution stability and will usually
        not affect performance. Default is True.

    count_output_bias: bool, optional
        Whether to include a bias term in the linear layer used to predict
        counts. Removing this term can help with attribution stability but
        may affect performance. Default is True.

    name: str or None, optional
        The name to save the model to during training.

    verbose: bool, optional
        Whether to display statistics during training. Setting this to False
        will still save the file at the end, but does not print anything to
        screen during training. Default is True.
    """

    def __init__(
        self, 
        out_dim = 1000,
        n_filters: int = 64, 
        n_layers: int = 8, 
        rconvs_kernel_size: int = 3,
        conv1_kernel_size: int = 21,
        profile_kernel_size: int = 75,
        n_outputs: int = 1,
        n_control_tracks: int = 0,
        profile_output_bias: bool = True,
        count_output_bias: bool = True,
        name: str = None,
        verbose: bool = True,
        n_count_outputs: int = 1,
    ):
        super().__init__()

        self.out_dim = out_dim
        self.n_filters = n_filters
        self.n_layers = n_layers
        self.n_outputs = n_outputs
        self.n_control_tracks = n_control_tracks
        self.verbose = verbose
        
        self.name = name or "bpnet.{}.{}".format(n_filters, n_layers)

        # first convolution without dilation
        # args are: # in_channels, out_channels, kernel_size, padding
        # padding='valid' means no padding, so the output will be smaller than input
        self.iconv = torch.nn.Conv1d(4, n_filters, kernel_size=conv1_kernel_size, padding='valid')
        self.irelu = torch.nn.ReLU()

        # residual dilated convolutions
        self.rconvs = torch.nn.ModuleList([
            torch.nn.Conv1d(
                n_filters,
                n_filters,
                kernel_size=rconvs_kernel_size,
                padding='valid', 
                dilation=2**i
            ) for i in range(1, self.n_layers+1)
        ])

        self.rrelus = torch.nn.ModuleList([
            torch.nn.ReLU() for i in range(1, self.n_layers+1)
        ])

        if profile_kernel_size > 0:
            # profile prediction
            # afaiu this is not adding control element-wise but rather concatenating along channel axis
            # and then convolving, which is equivalent to adding element-wise after convolving separately
            # (https://chatgpt.com/c/6922866e-7c0c-8329-91fd-4d3bc1831921).
            # fwiw I think the original bpnet arch is doing the same thing: see profile_bias_module in
            # https://github.com/kundajelab/bpnet/blob/master/bpnet/model/arch.py
            self.fconv = torch.nn.Conv1d(n_filters+n_control_tracks, n_outputs, 
                kernel_size=profile_kernel_size, padding='valid', bias=profile_output_bias)
        else:
            self.fconv = None

        # count prediction
        n_count_control = 1 if n_control_tracks > 0 else 0
        # hack for histobpnet, TODO_later make better
        if n_count_outputs > 1:
            n_count_control = n_control_tracks
        # will be used to pool (average) over sequence length
        self.global_avg_pool = torch.nn.AdaptiveAvgPool1d(1)
        self.linear = torch.nn.Linear(
            n_filters+n_count_control,
            n_count_outputs,
            bias=count_output_bias
        )

    def forward(self, x, x_ctl=None, x_ctl_hist=None):
        """A forward pass of the model.

        Parameters
        ----------
        x: torch.tensor, shape=(batch_size, 4, length)
            The one-hot encoded batch of sequences.

        x_ctl: torch.tensor or None, shape=(batch_size, n_strands, length)
            A value representing the signal of the control at each position in 
            the sequence. If no controls, pass in None. Default is None.

        Returns
        -------
        pred_profile: torch.tensor, shape=(batch_size, n_outputs, out_length)
            The output predictions for each strand trimmed to the output
            length.
        pred_count: torch.tensor, shape=(batch_size, 1)
        """
        if x.shape[1] != 4:
            x = x.permute(0, 2, 1)
        x = self.get_embs_after_crop(x)

        if self.verbose:
            print(f'trunk shape: {x.shape}')

        if x_ctl is not None:
            crop_size = (x_ctl.shape[2] - x.shape[2]) // 2
            if self.verbose:
                print(f'crop_size: {crop_size}')
            assert crop_size > 0
            x_ctl = x_ctl[:, :, crop_size:-crop_size]
            # else:
            #     x_ctl = F.pad(x_ctl, (-crop_size, -crop_size))

        if x_ctl_hist is not None:
            assert x_ctl_hist.shape[0] == x.shape[0], "Batch size of x_ctl_hist must match that of x"

        if self.fconv is None:
            pred_profile = None
        else:
            pred_profile = self.profile_head(x, x_ctl=x_ctl) # before log_softmax
        pred_count = self.count_head(x, x_ctl=x_ctl, x_ctl_hist=x_ctl_hist) #.squeeze(-1) # (batch_size, 1)

        return pred_profile, pred_count

    def get_embs_after_crop(self, x):
        x = self.irelu(self.iconv(x))
        for i in range(self.n_layers):
            conv_x = self.rrelus[i](self.rconvs[i](x))
            crop_len = (x.shape[2] - conv_x.shape[2]) // 2
            assert crop_len > 0
            x = x[:, :, crop_len:-crop_len]
            x = torch.add(x, conv_x)
        
        return x
    
    def profile_head(self, x, x_ctl=None):
        """
        Profile head of the model.
        output: (batch_size, n_outputs, out_window)
        """
        if x_ctl is not None:
            x = torch.cat([x, x_ctl], dim=1)

        pred_profile = self.fconv(x)

        crop_size = (pred_profile.shape[2] - self.out_dim) // 2
        if crop_size > 0:
            pred_profile = pred_profile[:, :, crop_size:-crop_size]
        else:
            pred_profile = F.pad(pred_profile, (-crop_size, -crop_size)) # pad if out_window > in_window
        
        return pred_profile

    def count_head(self, x, x_ctl=None, x_ctl_hist=None):
        # x is of shape (batch_size, n_filters, length)
        # pred_count shape: (batch_size, n_filters)
        pred_count = self.global_avg_pool(x).squeeze(-1)
        if x_ctl is not None:
            # sum over strands and sequence length, then add a trailing dimension
            # output shape: (batch_size, 1)
            x_ctl = torch.sum(x_ctl, dim=(1, 2)).unsqueeze(-1)
            pred_count = torch.cat([pred_count, torch.log1p(x_ctl)], dim=-1)
        # x_ctl_hist is of shape (batch_size, num_bins)
        if x_ctl_hist is not None:
            pred_count = torch.cat([pred_count, x_ctl_hist], dim=-1)
        pred_count = self.linear(pred_count)
        return pred_count
    
    def predict(self, x, forward_only=True):
        raise ValueError("Putting this here for now so I know when it's called!")
    
        y_profile, y_count = self(x)
        y_count = torch.exp(y_count)

        if not forward_only:
            y_profile_revcomp, y_count_revcomp = self(x.flip(dims=[1, 2])) #[:, ::-1, ::-1])
            y_count_revcomp = torch.exp(y_count_revcomp)
            y_profile = (y_profile + y_profile_revcomp) / 2
            y_count = (y_count + y_count_revcomp) / 2

        return y_profile.cpu().numpy().squeeze(1), y_count.cpu().numpy().squeeze(-1)

    @classmethod
    def from_keras(cls, filename, name='chrombpnet', instance=None):
        """Loads a model from ChromBPNet TensorFlow format.
    
        This method will load one of the components of a ChromBPNet model
        from TensorFlow format. Note that a full ChromBPNet model is made up
        of an accessibility model and a bias model and that this will load
        one of the two.

        Parameters
        ----------
        filename: str
            The name of the h5 file that stores the trained model parameters.

        Returns
        -------
        model: BPNet
            A BPNet model compatible with this repository in PyTorch.
        """
        if filename.endswith('.h5'):
            import h5py
            h5 = h5py.File(filename, "r")
            w = h5['model_weights']
        else:
            import os
            os.system('conda activate chrombpnet')
            import tensorflow as tf
            model = tf.keras.models.load_model(filename)
            w = model.get_weights()
            os.system('conda deactivate')

        # we expect 31 keys in the saved model weights:
        #
        # 'add', 'add_1', 'add_2', 'add_3', 'add_4', 'add_5', 'add_6', 'add_7',
        #
        # 'wo_bias_bpnet_1conv', 'wo_bias_bpnet_2conv', 'wo_bias_bpnet_3conv',
        # 'wo_bias_bpnet_4conv', 'wo_bias_bpnet_5conv', 'wo_bias_bpnet_6conv',
        # 'wo_bias_bpnet_7conv', 'wo_bias_bpnet_8conv',
        #
        # 'wo_bias_bpnet_1crop', 'wo_bias_bpnet_2crop', 'wo_bias_bpnet_3crop',
        # 'wo_bias_bpnet_4crop', 'wo_bias_bpnet_5crop', 'wo_bias_bpnet_6crop',
        # 'wo_bias_bpnet_7crop', 'wo_bias_bpnet_8crop',
        #
        # 'wo_bias_bpnet_1st_conv'
        #
        # 'wo_bias_bpnet_logcount_predictions', 'wo_bias_bpnet_logits_profile_predictions;
        # 'wo_bias_bpnet_logitt_before_flatten', 'wo_bias_bpnet_prof_out_precrop'
        # 'gap', 'sequence'
        #
        # ['add', 'add_1', ..., 'add_7'] are from the residual connections (there is n_layers of them)
        # ['wo_bias_bpnet_1conv', ..., 'wo_bias_bpnet_8conv'] are the dilated convolutions (there is n_layers of them)
        # ['wo_bias_bpnet_1crop', ..., 'wo_bias_bpnet_8crop'] are also for the dilated convolutions?
        # 'wo_bias_bpnet_1st_conv' is the first (non-dilated) convolution applied to the input before passing it into the dilated convolutions tower
        # 'wo_bias_bpnet_prof_out_precrop' is the last (non-dilated) convolution applied to the output of the dilated convolution tower
        # 'wo_bias_bpnet_logcount_predictions' is the final Dense layer (after the dilated conv tower) that predicts the logcounts
        # 'gap' is GlobalAveragePooling
        # Idk about these: 'sequence', 'wo_bias_bpnet_logitt_before_flatten', 'wo_bias_bpnet_logits_profile_predictions'

        print(f"Loading {name} model from {filename}", flush=True)
        if 'bpnet_1conv' in w.keys():
            prefix = ""
        else:
            prefix = "wo_bias_"

        namer = lambda prefix, suffix: '{0}{1}/{0}{1}'.format(prefix, suffix)
        k, b = 'kernel:0', 'bias:0'

        n_layers = 0
        for layer_name in w.keys():
            try:
                idx = int(layer_name.split("_")[-1].replace("conv", ""))
                n_layers = max(n_layers, idx)
            except:
                pass

        name = namer(prefix, "bpnet_1conv")
        n_filters = w[name][k].shape[2]

        if instance is None:
            model = BPNet(n_layers=n_layers, n_filters=n_filters, n_outputs=1,
                n_control_tracks=0)
        else:
            model = instance

        convert_w = lambda x: torch.nn.Parameter(torch.tensor(
            x[:]).permute(2, 1, 0))
        convert_b = lambda x: torch.nn.Parameter(torch.tensor(x[:]))

        iname = namer(prefix, 'bpnet_1st_conv')

        model.iconv.weight = convert_w(w[iname][k])
        model.iconv.bias = convert_b(w[iname][b])

        for i in range(1, n_layers+1):
            lname = namer(prefix, 'bpnet_{}conv'.format(i))

            model.rconvs[i-1].weight = convert_w(w[lname][k])
            model.rconvs[i-1].bias = convert_b(w[lname][b])

        prefix = prefix + "bpnet_" if prefix != "" else ""

        if model.fconv is not None:
            fname = namer(prefix, 'prof_out_precrop')
            model.fconv.weight = convert_w(w[fname][k])
            model.fconv.bias = convert_b(w[fname][b])

        if model.fconv is not None:
            # right now model.fconv is None means we're running for histobpnet
            # TODO_later detect this in a cleaner way maybe
            # also TODO do we want to / can we partially initialize the weights or something?
            name = namer(prefix, "logcount_predictions")
            model.linear.weight = torch.nn.Parameter(torch.tensor(w[name][k][:].T))
            model.linear.bias = convert_b(w[name][b])
        return model