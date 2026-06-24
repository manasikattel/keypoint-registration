import torch
import torch.nn as nn

class UNet(nn.Module):
    """UNet architecture"""

    def __init__(self,
                 n_input_channels=1,
                 n_output_channels=1,
                 n_levels=4,
                 n_conv=2,
                 n_feat=32,
                 feat_mult=2,
                 kernel_size=3,
                 activation='relu',
                 last_activation='softmax',
                 batch_norm_after_each_conv=False,
                 residual_blocks=False,
                 encoder_only=False,
                 upsample=False,
                 use_skip_connections=True,
                 rm_top_skip_connection=0,
                 predict_residual=False):
        """
        :param n_input_channels: number of input channels
        :param n_output_channels: number of output channels (i.e. feature maps)
        :param n_levels: number of resolution levels
        :param n_conv: number of convolution per resolution level
        :param n_feat: number of initial feature maps after the first convolution
        :param feat_mult: feature multiplier after each max pooling
        :param kernel_size: size of convolutional kernels
        :param activation: non-linearity to use. Can be 'relu' or 'elu'
        :param last_activation: last non-linearity before softmax. Can be 'relu' or 'elu', or None
        :param batch_norm_after_each_conv: if false, batch norm will be performed at teh end of each resolution level
        :param residual_blocks: whether to use residual connection at the end of each block
        :param encoder_only: do not add a decoder
        :param upsample: if encoder only, whether to upsample the bottleneck to the size of the inputs
        :param use_skip_connections: whether to use skip connections at all
        :param rm_top_skip_connection: whether to remove the top skip connection. Default is 0 where none are removed
        :param predict_residual: whether to add a residual connection between the input and the last layer
        """

        super(UNet, self).__init__()

        # input/output channels
        self.n_input_channels = n_input_channels
        self.n_output_channels = n_output_channels

        # general architecture
        self.encoder_only = encoder_only
        self.upsample = upsample
        self.rm_top_skip_connection = rm_top_skip_connection if use_skip_connections else self.n_levels
        self.predict_residual = predict_residual

        # convolution block parameters
        self.n_levels = n_levels
        self.n_conv = n_conv
        self.feat_mult = feat_mult
        self.feat_list = [n_feat * feat_mult ** i for i in range(self.n_levels)]
        self.kernel_size = kernel_size
        self.activation = activation
        self.batch_norm_after_each_conv = batch_norm_after_each_conv
        self.residual_blocks = residual_blocks

        # define convolutional blocks
        self.list_encoder_blocks = self.get_list_encoder_blocks()  # list of length self.n_levels
        if not self.encoder_only:
            self.list_decoder_blocks = self.get_list_decoder_blocks()  # list of length self.n_levels - 1
            self.last_conv = torch.nn.Conv3d(self.feat_list[0], self.n_output_channels, kernel_size=1)
        else:
            self.list_decoder_blocks = []
            self.last_conv = torch.nn.Conv3d(self.feat_list[-1], self.n_output_channels, kernel_size=1)

        if last_activation == 'relu':
            self.last_activation = torch.nn.ReLU()
        elif last_activation == 'elu':
            self.last_activation = torch.nn.ELU()
        elif last_activation == 'softmax':
            self.last_activation = torch.nn.Softmax(dim=-1)
        elif last_activation == 'tanh':
            self.last_activation = torch.nn.Tanh()
        elif last_activation == 'sigmoid':
            self.last_activation = torch.nn.Sigmoid()
        else:
            self.last_activation = None

    def forward(self, x):
        """takes an input of shape [B, C, H, W, D]"""

        tens = x

        # down-arm
        list_encoders_features = []
        for i, encoder_block in enumerate(self.list_encoder_blocks):
            if i > 0:
                tens = torch.nn.functional.max_pool3d(tens, kernel_size=2)
            tens_out = encoder_block(tens)
            tens = tens + tens_out if self.residual_blocks else tens_out
            list_encoders_features.append(tens)

        # up-arm
        if not self.encoder_only:

            # remove output of last encoder block (i.e. the bottleneck) from the list of features to be concatenated
            list_encoders_features = list_encoders_features[::-1][1:]

            # build conv
            for i in range(len(self.list_decoder_blocks)):
                tens = torch.nn.functional.interpolate(tens, scale_factor=2, mode='trilinear')
                if i < (self.n_levels - 1 - self.rm_top_skip_connection):
                    tens_out = torch.cat((list_encoders_features[i], tens), dim=1)
                else:
                    tens_out = tens
                tens_out = self.list_decoder_blocks[i](tens_out)
                tens = tens + tens_out if self.residual_blocks else tens_out

        # final convolution
        tens = self.last_conv(tens)
        if self.last_activation is not None:
            B, C, *spatial = tens.shape
            tens_flat = tens.view(B, C, -1)
            tens_prob = self.last_activation(tens_flat)
            tens_prob = tens_prob.view(B, C, *spatial)

        if self.upsample:
            if self.encoder_only:
                tens = torch.nn.functional.interpolate(tens, scale_factor=2 ** (self.n_levels - 1), mode='trilinear')
            else:
                raise ValueError('upsample is only supported when encoder_only is set to True')

        # residual
        if self.predict_residual:
            tens = x + tens

        return tens_prob

    def get_list_encoder_blocks(self):
        """Build encoder blocks.

        Returns
        -------
        torch.nn.ModuleList
            Encoder blocks ordered from shallow to deep.
        """

        list_encoder_blocks = []
        for i in range(self.n_levels):

            # number of input/output feature maps for each convolution
            if i == 0:
                n_input_feat = [self.n_input_channels] + [self.feat_list[i]] * (self.n_conv - 1)
            else:
                n_input_feat = [self.feat_list[i - 1]] + [self.feat_list[i]] * (self.n_conv - 1)
            n_output_feat = self.feat_list[i]

            # build conv block
            layers = self.build_block(n_input_feat, n_output_feat)
            list_encoder_blocks.append(torch.nn.Sequential(*layers))

        return nn.ModuleList(list_encoder_blocks)

    def get_list_decoder_blocks(self):
        """Build decoder blocks.

        Returns
        -------
        torch.nn.ModuleList
            Decoder blocks ordered from deep to shallow.
        """

        list_decoder_blocks = []
        for i in range(0, self.n_levels - 1):

            # number of input/output feature maps for each convolution
            if i < (self.n_levels - 1 - self.rm_top_skip_connection):
                n_input_feat = [self.feat_list[::-1][i + 1] * (1 + self.feat_mult)] + \
                               [self.feat_list[::-1][i + 1]] * (self.n_conv - 1)
            else:
                n_input_feat = [self.feat_list[::-1][i]] + \
                               [self.feat_list[::-1][i + 1]] * (self.n_conv - 1)
            n_output_feat = self.feat_list[::-1][i + 1]

            # build conv block
            layers = self.build_block(n_input_feat, n_output_feat)
            list_decoder_blocks.append(torch.nn.Sequential(*layers))

        return nn.ModuleList(list_decoder_blocks)

    def build_block(self, n_input_feat, n_output_feat):
        """Build one convolutional block.

        Parameters
        ----------
        n_input_feat : sequence of int
            Input channel count for each convolution.
        n_output_feat : int
            Output channel count for each convolution.

        Returns
        -------
        list of torch.nn.Module
            Layers forming the convolutional block.
        """

        # convolutions + activations
        layers = list()
        for conv in range(self.n_conv):
            layers.append(torch.nn.Conv3d(n_input_feat[conv], n_output_feat, kernel_size=self.kernel_size,
                                          padding=self.kernel_size // 2))
            if self.activation == 'relu':
                layers.append(torch.nn.ReLU())
            elif self.activation == 'elu':
                layers.append(torch.nn.ELU())
            else:
                raise ValueError('activation should be relu or elu, had: %s' % self.activation)
            if self.batch_norm_after_each_conv:
                layers.append(torch.nn.BatchNorm3d(n_output_feat))

        # batch norm
        if not self.batch_norm_after_each_conv:
            layers.append(torch.nn.BatchNorm3d(n_output_feat))

        return layers

    def to(self, *args, **kwargs):
        """Move the module to a device or dtype.

        Parameters
        ----------
        *args
            Positional arguments forwarded to ``torch.nn.Module.to``.
        **kwargs
            Keyword arguments forwarded to ``torch.nn.Module.to``.

        Returns
        -------
        UNet
            The moved module.
        """
        self = super().to(*args, **kwargs)
        return self
    
