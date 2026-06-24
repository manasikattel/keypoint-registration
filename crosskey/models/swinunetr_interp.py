import inspect

import torch
import torch.nn as nn
import torch.nn.functional as F


def spatial_softmax(logits):
    """Apply spatial softmax independently to each channel.

    Parameters
    ----------
    logits : torch.Tensor
        Logit tensor with shape ``[B, C, D, H, W]``.

    Returns
    -------
    torch.Tensor
        Spatial probability maps with the same shape as ``logits``.
    """
    batch_size, channels, *spatial = logits.shape
    probs = torch.softmax(logits.view(batch_size, channels, -1), dim=-1)
    return probs.view(batch_size, channels, *spatial)


def call_with_supported_kwargs(cls, **kwargs):
    """Instantiate a class with only supported keyword arguments.

    Parameters
    ----------
    cls : type
        Class to instantiate.
    **kwargs
        Candidate keyword arguments.

    Returns
    -------
    object
        Instance of ``cls``.
    """
    signature = inspect.signature(cls)
    supported = {key: value for key, value in kwargs.items() if key in signature.parameters}
    return cls(**supported)


class InterpolateConvUpBlock(nn.Module):
    """SwinUNETR decoder block using resize-convolution instead of transposed convolution."""

    def __init__(
        self,
        spatial_dims=3,
        in_channels=1,
        skip_channels=1,
        out_channels=1,
        kernel_size=3,
        norm_name="instance",
        mode="trilinear",
        align_corners=False,
    ):
        """Initialize an interpolation-convolution upsampling block.

        Parameters
        ----------
        spatial_dims : int, optional
            Number of spatial dimensions.
        in_channels : int, optional
            Number of input channels.
        skip_channels : int, optional
            Number of skip-connection channels.
        out_channels : int, optional
            Number of output channels.
        kernel_size : int, optional
            Convolution kernel size.
        norm_name : str, optional
            MONAI normalization name.
        mode : str, optional
            Interpolation mode.
        align_corners : bool, optional
            ``align_corners`` argument for interpolation.
        """
        super().__init__()
        if spatial_dims != 3:
            raise ValueError("InterpolateConvUpBlock supports spatial_dims=3 only.")

        from monai.networks.blocks import UnetrBasicBlock

        self.mode = mode
        self.align_corners = align_corners
        self.channel_project = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.refine = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=out_channels + skip_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

    def forward(self, x, skip):
        """Upsample, concatenate a skip feature, and refine.

        Parameters
        ----------
        x : torch.Tensor
            Decoder feature tensor.
        skip : torch.Tensor
            Encoder skip feature tensor.

        Returns
        -------
        torch.Tensor
            Refined decoder feature tensor.
        """
        interpolate_kwargs = {"size": skip.shape[2:], "mode": self.mode}
        if self.mode in {"linear", "bilinear", "bicubic", "trilinear"}:
            interpolate_kwargs["align_corners"] = self.align_corners
        x = F.interpolate(x, **interpolate_kwargs)
        x = self.channel_project(x)
        x = torch.cat((skip, x), dim=1)
        return self.refine(x)


class InterpDecoderSwinUNETRBranch(nn.Module):
    """SwinUNETR branch with interpolation + convolution decoder upsampling."""

    def __init__(
        self,
        in_channels=2,
        out_channels=32,
        feature_size=24,
        depths=(2, 2, 2, 2),
        num_heads=(3, 6, 12, 24),
        norm_name="instance",
        drop_rate=0.0,
        attn_drop_rate=0.0,
        dropout_path_rate=0.0,
        use_checkpoint=False,
        upsample_mode="trilinear",
        align_corners=False,
        use_v2=False,
    ):
        """Initialize one SwinUNETR keypoint branch.

        Parameters
        ----------
        in_channels : int, optional
            Number of input channels.
        out_channels : int, optional
            Number of output keypoint heatmaps.
        feature_size : int, optional
            Base SwinUNETR feature size.
        depths : sequence of int, optional
            Number of transformer blocks per stage.
        num_heads : sequence of int, optional
            Number of attention heads per stage.
        norm_name : str, optional
            MONAI normalization name.
        drop_rate : float, optional
            Dropout rate.
        attn_drop_rate : float, optional
            Attention dropout rate.
        dropout_path_rate : float, optional
            Stochastic depth rate.
        use_checkpoint : bool, optional
            Whether to enable gradient checkpointing.
        upsample_mode : str, optional
            Decoder interpolation mode.
        align_corners : bool, optional
            ``align_corners`` argument for interpolation.
        use_v2 : bool, optional
            Whether to enable MONAI SwinUNETR v2 options when available.
        """
        super().__init__()
        from monai.networks.blocks import UnetOutBlock, UnetrBasicBlock
        from monai.networks.nets.swin_unetr import SwinTransformer

        if feature_size % 12 != 0:
            raise ValueError("MONAI SwinUNETR expects feature_size divisible by 12.")

        self.normalize = True
        self.swin_encoder = call_with_supported_kwargs(
            SwinTransformer,
            in_chans=in_channels,
            embed_dim=feature_size,
            window_size=(7, 7, 7),
            patch_size=(2, 2, 2),
            depths=depths,
            num_heads=num_heads,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=dropout_path_rate,
            norm_layer=nn.LayerNorm,
            use_checkpoint=use_checkpoint,
            spatial_dims=3,
            use_v2=use_v2,
        )

        self.encoder1 = UnetrBasicBlock(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )
        self.encoder2 = UnetrBasicBlock(
            spatial_dims=3,
            in_channels=feature_size,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )
        self.encoder3 = UnetrBasicBlock(
            spatial_dims=3,
            in_channels=2 * feature_size,
            out_channels=2 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )
        self.encoder4 = UnetrBasicBlock(
            spatial_dims=3,
            in_channels=4 * feature_size,
            out_channels=4 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )
        self.encoder10 = UnetrBasicBlock(
            spatial_dims=3,
            in_channels=16 * feature_size,
            out_channels=16 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        up_kwargs = dict(
            spatial_dims=3,
            kernel_size=3,
            norm_name=norm_name,
            mode=upsample_mode,
            align_corners=align_corners,
        )
        self.decoder5 = InterpolateConvUpBlock(
            in_channels=16 * feature_size,
            skip_channels=8 * feature_size,
            out_channels=8 * feature_size,
            **up_kwargs,
        )
        self.decoder4 = InterpolateConvUpBlock(
            in_channels=8 * feature_size,
            skip_channels=4 * feature_size,
            out_channels=4 * feature_size,
            **up_kwargs,
        )
        self.decoder3 = InterpolateConvUpBlock(
            in_channels=4 * feature_size,
            skip_channels=2 * feature_size,
            out_channels=2 * feature_size,
            **up_kwargs,
        )
        self.decoder2 = InterpolateConvUpBlock(
            in_channels=2 * feature_size,
            skip_channels=feature_size,
            out_channels=feature_size,
            **up_kwargs,
        )
        self.decoder1 = InterpolateConvUpBlock(
            in_channels=feature_size,
            skip_channels=feature_size,
            out_channels=feature_size,
            **up_kwargs,
        )
        self.out = UnetOutBlock(
            spatial_dims=3,
            in_channels=feature_size,
            out_channels=out_channels,
        )

    def forward(self, image):
        """Compute keypoint heatmaps from an input volume.

        Parameters
        ----------
        image : torch.Tensor
            Input tensor with shape ``[B, C, D, H, W]``.

        Returns
        -------
        torch.Tensor
            Spatial-softmax keypoint heatmaps with shape ``[B, K, D, H, W]``.
        """
        hidden_states = self.swin_encoder(image, self.normalize)
        enc0 = self.encoder1(image)
        enc1 = self.encoder2(hidden_states[0])
        enc2 = self.encoder3(hidden_states[1])
        enc3 = self.encoder4(hidden_states[2])
        dec4 = self.encoder10(hidden_states[4])
        dec3 = self.decoder5(dec4, hidden_states[3])
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        dec0 = self.decoder2(dec1, enc1)
        out = self.decoder1(dec0, enc0)
        return spatial_softmax(self.out(out))
