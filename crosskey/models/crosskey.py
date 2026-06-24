import torch
import torch.nn as nn
from crosskey.evaluation.metrics import rot_error, compute_surface_registration_error
from crosskey.models.equiv_unet import EquivUNet
from crosskey.models.swinunetr_interp import InterpDecoderSwinUNETRBranch
from crosskey.models.unet import UNet
from crosskey.training.losses import get_loss, bbvre_scaled_to_mean, gaussian_kl_loss, frobenius_norm, repulsive_loss
from crosskey.utils.geometry import transform_points, pts_to_xfm_numerical, calculate_barycenter_and_covariance
from lightning.pytorch import LightningModule


def build_unet_extractor(
    in_channels,
    out_channels,
    base_feat,
    feat_mult,
    levels,
    kernel_size,
    n_conv,
    last_activation,
):
    """Build a baseline 3D U-Net keypoint extractor.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of keypoint heatmaps.
    base_feat : int
        Base number of feature channels.
    feat_mult : int
        Multiplicative channel factor between levels.
    levels : int
        Number of U-Net levels.
    kernel_size : int
        Convolution kernel size.
    n_conv : int
        Number of convolutions per block.
    last_activation : str
        Final heatmap activation.

    Returns
    -------
    UNet
        Configured U-Net extractor.
    """
    return UNet(
        n_input_channels=in_channels,
        n_output_channels=out_channels,
        n_levels=levels,
        n_feat=base_feat,
        feat_mult=feat_mult,
        kernel_size=kernel_size,
        activation="relu",
        last_activation=last_activation,
        n_conv=n_conv,
    )


def build_equiv_unet_extractor(
    in_channels,
    out_channels,
    feat_mult,
    levels,
    kernel_size,
    last_activation,
    equiv_steps,
    equiv_lmax,
    equiv_scale,
    equiv_normalization,
):
    """Build an equivariant U-Net keypoint extractor.

    Parameters
    ----------
    in_channels : int
        Number of scalar input channels.
    out_channels : int
        Number of scalar output heatmaps.
    feat_mult : int
        Feature multiplier.
    levels : int
        Number of U-Net levels.
    kernel_size : int
        Equivariant convolution kernel diameter.
    last_activation : str
        Final heatmap activation.
    equiv_steps : sequence of float
        Voxel spacing used by equivariant kernels.
    equiv_lmax : int
        Maximum spherical harmonic order.
    equiv_scale : int
        Downsampling scale.
    equiv_normalization : str
        Normalization mode.

    Returns
    -------
    EquivUNet
        Configured equivariant extractor.
    """
    return EquivUNet(
        irreps_in=f"{in_channels}x0e",
        irreps_out=f"{out_channels}x0e",
        steps=tuple(equiv_steps),
        n_levels=levels,
        feat_mult=feat_mult,
        kernel_size=kernel_size,
        last_activation=last_activation,
        normalization=equiv_normalization,
        lmax=equiv_lmax,
        scale=equiv_scale,
    )


def build_swinunetr_interp_extractor(
    in_channels,
    out_channels,
    swin_feature_size,
    swin_depths,
    swin_num_heads,
    swin_norm_name,
    swin_drop_rate,
    swin_attn_drop_rate,
    swin_dropout_path_rate,
    swin_use_checkpoint,
    swin_upsample_mode,
    swin_align_corners,
    swin_use_v2,
):
    """Build the SwinUNETR interpolation-decoder extractor.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of keypoint heatmaps.
    swin_feature_size : int
        Base SwinUNETR feature size.
    swin_depths : sequence of int
        Transformer block counts per stage.
    swin_num_heads : sequence of int
        Attention head counts per stage.
    swin_norm_name : str
        MONAI normalization name.
    swin_drop_rate : float
        Dropout rate.
    swin_attn_drop_rate : float
        Attention dropout rate.
    swin_dropout_path_rate : float
        Stochastic depth rate.
    swin_use_checkpoint : bool
        Whether to enable gradient checkpointing.
    swin_upsample_mode : str
        Decoder interpolation mode.
    swin_align_corners : bool
        ``align_corners`` flag for interpolation.
    swin_use_v2 : bool
        Whether to enable MONAI SwinUNETR v2 options when available.

    Returns
    -------
    InterpDecoderSwinUNETRBranch
        Configured SwinUNETR branch.
    """
    return InterpDecoderSwinUNETRBranch(
        in_channels=in_channels,
        out_channels=out_channels,
        feature_size=swin_feature_size,
        depths=tuple(swin_depths),
        num_heads=tuple(swin_num_heads),
        norm_name=swin_norm_name,
        drop_rate=swin_drop_rate,
        attn_drop_rate=swin_attn_drop_rate,
        dropout_path_rate=swin_dropout_path_rate,
        use_checkpoint=swin_use_checkpoint,
        upsample_mode=swin_upsample_mode,
        align_corners=swin_align_corners,
        use_v2=swin_use_v2,
    )


class KeyReg(nn.Module):
    """Dual-branch keypoint registration network."""

    def __init__(self,spatial_dims=3,
                 in_channels=1,
                 out_channels=1,
                 base_feat=32,
                 feat_mult=2,
                 levels=4,
                 kernel_size=3,
                 n_conv=2,
                 last_activation="softmax",
                 input_shape=(96,96,96),
                 backbone="unet",
                 equiv_steps=(1.0, 1.0, 1.0),
                 equiv_lmax=2,
                 equiv_scale=2,
                 equiv_normalization="instance",
                 swin_feature_size=24,
                 swin_depths=(2, 2, 2, 2),
                 swin_num_heads=(3, 6, 12, 24),
                 swin_norm_name="instance",
                 swin_drop_rate=0.0,
                 swin_attn_drop_rate=0.0,
                 swin_dropout_path_rate=0.0,
                 swin_use_checkpoint=False,
                 swin_upsample_mode="trilinear",
                 swin_align_corners=False,
                 swin_use_v2=False,
                 ):
        """Initialize source and target keypoint extractors.

        Parameters
        ----------
        spatial_dims : int, optional
            Number of spatial dimensions. Only 3D is supported.
        in_channels : int, optional
            Number of input channels per branch.
        out_channels : int, optional
            Number of keypoint heatmaps.
        input_shape : tuple of int, optional
            Spatial shape used by the rigid registration layer.
        backbone : {"unet", "equiv_unet", "swinunetr_interp"}, optional
            Feature extractor architecture.
        **kwargs
            Backbone-specific architecture options.
        """
        
        super().__init__()

        self.im_shape = input_shape
        if backbone == "unet":
            extractor_factory = lambda: build_unet_extractor(
                in_channels,
                out_channels,
                base_feat,
                feat_mult,
                levels,
                kernel_size,
                n_conv,
                last_activation,
            )
        elif backbone == "equiv_unet":
            extractor_factory = lambda: build_equiv_unet_extractor(
                in_channels,
                out_channels,
                feat_mult,
                levels,
                kernel_size,
                last_activation,
                equiv_steps,
                equiv_lmax,
                equiv_scale,
                equiv_normalization,
            )
        elif backbone == "swinunetr_interp":
            extractor_factory = lambda: build_swinunetr_interp_extractor(
                in_channels,
                out_channels,
                swin_feature_size,
                swin_depths,
                swin_num_heads,
                swin_norm_name,
                swin_drop_rate,
                swin_attn_drop_rate,
                swin_dropout_path_rate,
                swin_use_checkpoint,
                swin_upsample_mode,
                swin_align_corners,
                swin_use_v2,
            )
        else:
            raise ValueError(f"Unsupported backbone for KeyReg: {backbone}")

        self.source_extractor = extractor_factory()
        self.target_extractor = extractor_factory()

    def forward(self, source, target, ):
        """Estimate a rigid transform from source and target inputs.

        Parameters
        ----------
        source : torch.Tensor
            Source tensor with shape ``[B, C, D, H, W]``.
        target : torch.Tensor
            Target tensor with shape ``[B, C, D, H, W]``.

        Returns
        -------
        list
            Predicted transform, source/target heatmaps, source/target
            keypoints, and source/target covariance matrices.
        """

        feat_source = self.source_extractor(source)
        feat_target = self.target_extractor(target)

        source_keypoints, source_cov = calculate_barycenter_and_covariance(feat_source)
        target_keypoints, target_cov = calculate_barycenter_and_covariance(feat_target)
        xfm = pts_to_xfm_numerical(source_keypoints, target_keypoints, self.im_shape)  # [B, 4, 4]

        outputs = [xfm, feat_source, feat_target, source_keypoints, target_keypoints, source_cov, target_cov]
        return outputs

def split_source_target(sample4D):
    """Split packed source/target channels.

    Parameters
    ----------
    sample4D : torch.Tensor
        Tensor with alternating source and target channels.

    Returns
    -------
    tuple of torch.Tensor
        Source tensor and target tensor.
    """
    C = sample4D.shape[1]

    if C == 2:
        source = sample4D[:, 0:1]  # (B, 1, D, H, W)
        target = sample4D[:, 1:2]

    elif C == 4:
        source = sample4D[:, [0, 2]]  # (B, 2, D, H, W)
        target = sample4D[:, [1, 3]]

    elif C == 6:
        source = sample4D[:, [0, 2, 4]]  # (B, 3, D, H, W)
        target = sample4D[:, [1, 3, 5]]

    else:
        raise ValueError(f"Unexpected number of channels in sample4D: {C}")

    return source, target

class KeyRegModule(LightningModule):    
    """Lightning module that trains and validates CrossKey."""

    def __init__(self, 
                 learning_rate=1e-4,
                 loss:str = "dice",
                 scheduler: torch.optim.lr_scheduler=None,
                 chkp_pretrained: str = None,
                 num_keypoints=128,
                 base_feat=32,
                 levels = 4,
                 kernel_size=3,
                 backbone="unet",
                 input_channels=1,
                 input_shape=(96, 96, 96),
                 n_conv=2,
                 feat_mult=2,
                 last_activation="softmax",
                 equiv_steps=(1.0, 1.0, 1.0),
                 equiv_lmax=2,
                 equiv_scale=2,
                 equiv_normalization="instance",
                 swin_feature_size=24,
                 swin_depths=(2, 2, 2, 2),
                 swin_num_heads=(3, 6, 12, 24),
                 swin_norm_name="instance",
                 swin_drop_rate=0.0,
                 swin_attn_drop_rate=0.0,
                 swin_dropout_path_rate=0.0,
                 swin_use_checkpoint=False,
                 swin_upsample_mode="trilinear",
                 swin_align_corners=False,
                 swin_use_v2=False,
                 temperature_repulsive_loss = 0.1,
                 weight_kl_loss = 0,
                 weight_repulsive_loss=0,
                 weight_var_loss = 0,
                 weight_registration_loss=1,
                 weight_kp_dist = 0,
                 device=None):
        """Initialize the CrossKey training module.

        Parameters
        ----------
        learning_rate : float, optional
            Adam learning rate.
        loss : str, optional
            Registration loss selector.
        scheduler : callable, optional
            Optional scheduler factory.
        chkp_pretrained : str, optional
            Optional checkpoint path.
        num_keypoints : int, optional
            Number of learned keypoints.
        backbone : str, optional
            Backbone architecture name.
        input_channels : int, optional
            Number of input channels passed to each branch.
        input_shape : tuple of int, optional
            Spatial shape used by the registration layer.
        temperature_repulsive_loss : float, optional
            Temperature for repulsive regularization.
        weight_kl_loss : float, optional
            Gaussian KL loss weight.
        weight_repulsive_loss : float, optional
            Repulsive loss weight.
        weight_var_loss : float, optional
            Variance regularization weight.
        weight_registration_loss : float, optional
            Registration loss weight.
        weight_kp_dist : float, optional
            Landmark matching loss weight.
        device : torch.device or str, optional
            Device used for module initialization.
        """
        super().__init__()
        self.learning_rate = learning_rate
        self.loss = loss
        self.dice = get_loss("dice")
        self.mse = get_loss("mse")
        self.geodesic = get_loss("geodesic")
        self.chkp_pretrained = chkp_pretrained
        self.num_keypoints = num_keypoints
        self.temperature_repulsive_loss = temperature_repulsive_loss
        self.weight_kl_loss = weight_kl_loss
        self.weight_repulsive_loss = weight_repulsive_loss
        self.weight_var_loss = weight_var_loss
        self.weight_registration_loss = weight_registration_loss
        self.weight_kp_dist = weight_kp_dist

        self.backbone = backbone
        self.input_channels = input_channels

        valid_backbones = {"unet", "equiv_unet", "swinunetr_interp"}
        if backbone not in valid_backbones:
            raise ValueError(
                "backbone must be one of: 'unet', 'equiv_unet', 'swinunetr_interp'"
            )

        keyreg_kwargs = dict(
            in_channels=input_channels,
            out_channels=num_keypoints,
            base_feat=base_feat,
            levels=levels,
            kernel_size=kernel_size,
            n_conv=n_conv,
            feat_mult=feat_mult,
            last_activation=last_activation,
            input_shape=input_shape,
            backbone=backbone,
            equiv_steps=equiv_steps,
            equiv_lmax=equiv_lmax,
            equiv_scale=equiv_scale,
            equiv_normalization=equiv_normalization,
            swin_feature_size=swin_feature_size,
            swin_depths=swin_depths,
            swin_num_heads=swin_num_heads,
            swin_norm_name=swin_norm_name,
            swin_drop_rate=swin_drop_rate,
            swin_attn_drop_rate=swin_attn_drop_rate,
            swin_dropout_path_rate=swin_dropout_path_rate,
            swin_use_checkpoint=swin_use_checkpoint,
            swin_upsample_mode=swin_upsample_mode,
            swin_align_corners=swin_align_corners,
            swin_use_v2=swin_use_v2,
        )

        self.model = KeyReg(**keyreg_kwargs).to(device)

            
        if self.chkp_pretrained is not None and self.chkp_pretrained != 'None':
            self.model = KeyRegModule.load_from_checkpoint(chkp_pretrained).model
                        
        self.save_hyperparameters()
    
    def forward(self, source, target):
        """Run the CrossKey network on a source/target pair.

        Parameters
        ----------
        source : torch.Tensor
            Source tensor with shape ``[B, C, D, H, W]``.
        target : torch.Tensor
            Target tensor with shape ``[B, C, D, H, W]``.

        Returns
        -------
        tuple
            Predicted transform, heatmaps, keypoints, and covariances.
        """
        source, target = self._select_model_inputs(source, target)
        pred_mat, feat_source, feat_target, source_keypoints, target_keypoints, source_cov, target_cov = self.model(source, target)
        return pred_mat, feat_source, feat_target, source_keypoints, target_keypoints, source_cov, target_cov

    def _select_model_inputs(self, source, target):
        """Select the channels consumed by the registration network.

        Parameters
        ----------
        source : torch.Tensor
            Source tensor with shape ``[B, C, D, H, W]``.
        target : torch.Tensor
            Target tensor with shape ``[B, C, D, H, W]``.

        Returns
        -------
        tuple of torch.Tensor
            Source and target tensors with ``self.input_channels`` channels.
        """
        if self.input_channels == source.shape[1]:
            return source, target
        if self.input_channels == 1 and source.shape[1] >= 2:
            return source[:, [1], :, :, :], target[:, [1], :, :, :]
        raise ValueError(
            f"Configured input_channels={self.input_channels}, but source/target tensors have {source.shape[1]} channels."
        )

    def _shared_step(self, batch, stage):
        """Compute losses and metrics for a training or validation batch.

        Parameters
        ----------
        batch : tuple
            Batch returned by ``PairedVolumeDataset``.
        stage : {"train", "val"}
            Logging prefix that identifies the current loop.

        Returns
        -------
        torch.Tensor
            Scalar loss for the current batch.
        """
        labels = batch[1]
        gt_mat = batch[2]
        corners = batch[3]
        batch_size = labels.shape[0]

        source, target = split_source_target(labels)
        source_mask = labels[:,0, :, :, :]   
        target_mask = labels[:,1, :, :, :]
        pred_mat, feat_source, feat_target, source_keypoints, target_keypoints, source_cov, target_cov = self.forward(source, target)
 
        mask_repulsive_loss = torch.triu(torch.ones(
            [batch_size, self.num_keypoints, self.num_keypoints],
            dtype=torch.bool,
            device=labels.device,
        ), diagonal=1)
        kl_loss = (
            gaussian_kl_loss(
                feat_source,
                torch.flip(source_keypoints, dims=[-1]),
                torch.flip(source_cov, dims=[-1, -2]),
            )
            + gaussian_kl_loss(
                feat_target,
                torch.flip(target_keypoints, dims=[-1]),
                torch.flip(target_cov, dims=[-1, -2]),
            )
        ) / 2

        var_loss = (frobenius_norm(source_cov).mean() + frobenius_norm(target_cov).mean()) / 2
        rep_loss = (repulsive_loss(source_keypoints, self.temperature_repulsive_loss, mask_repulsive_loss) +
                                repulsive_loss(target_keypoints, self.temperature_repulsive_loss, mask_repulsive_loss)) / 2
        
        rot_err = rot_error(pred_mat[:,:3,:3], gt_mat[:,:3,:3])
        rot_err = rot_err.mean()
        geodesic = self.geodesic(pred_mat[:,:3,:3],gt_mat[:,:3,:3])
        bbvre_norm = bbvre_scaled_to_mean(corners, pred_mat, gt_mat, source_mask.shape).mean()
        
        source_points_transformed = transform_points(source_keypoints, torch.linalg.inv(gt_mat), labels[:,0, :, :, :].shape)
        keypoint_distance = torch.norm(target_keypoints - source_points_transformed, dim=-1).mean(dim=1).mean()
        sre_atreg_source = compute_surface_registration_error(source_mask, pred_mat,
                        gt_mat).mean()
        sre_atreg_target = compute_surface_registration_error(target_mask, torch.linalg.inv(pred_mat),
                        torch.linalg.inv(gt_mat)).mean()
        sre = sre_atreg_source  + sre_atreg_target

        if self.loss =="sre":
            registration_loss = sre + self.weight_kp_dist * keypoint_distance
        else:
            registration_loss = bbvre_norm + self.weight_kp_dist * keypoint_distance

        loss = (
            self.weight_registration_loss * registration_loss
            + self.weight_repulsive_loss * rep_loss
            + self.weight_kl_loss * kl_loss
            + self.weight_var_loss * var_loss
        )

        self.log_dict({
            f"{stage}_keypoint_distance": keypoint_distance,
            f"{stage}_bbvre_norm": bbvre_norm,
            f"{stage}_loss": loss,
            f"{stage}_roterr": rot_err,
            f"{stage}_geodesic": torch.rad2deg(geodesic),
            f"{stage}_kl_loss": kl_loss,
            f"{stage}_var_loss": var_loss,
            f"{stage}_rep_loss": rep_loss,
            f"{stage}_sre": sre,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=batch_size,
        )
        return loss

    def training_step(self, batch, batch_idx):
        """Compute and log the training loss for one batch.

        Parameters
        ----------
        batch : tuple
            Batch returned by ``PairedVolumeDataset``.
        batch_idx : int
            Batch index supplied by Lightning.

        Returns
        -------
        torch.Tensor
            Scalar training loss.
        """
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        """Compute and log validation metrics for one batch.

        Parameters
        ----------
        batch : tuple
            Batch returned by ``PairedVolumeDataset``.
        batch_idx : int
            Batch index supplied by Lightning.

        Returns
        -------
        torch.Tensor
            Scalar validation loss.
        """
        return self._shared_step(batch, "val")

    def configure_optimizers(self):
        """Configure the optimizer and scheduler.

        Returns
        -------
        dict
            Lightning optimizer configuration.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
        
        #learning rate scheduler
        return {"optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler,
                                 "monitor":"val_loss"}}
