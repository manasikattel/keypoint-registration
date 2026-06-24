import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from crosskey.utils.geometry import make_gaussian


class MSELoss(nn.Module):
    """Mean squared error loss wrapper."""

    def forward(self, pred, target):
        """Compute mean squared error.

        Parameters
        ----------
        pred : torch.Tensor
            Predicted tensor.
        target : torch.Tensor
            Target tensor with the same shape as ``pred``.

        Returns
        -------
        torch.Tensor
            Scalar MSE loss.
        """
        return F.mse_loss(pred, target)


class DiceLoss(nn.Module):
    """Soft Dice loss for volumetric masks."""

    def __init__(self, smooth=1e-6):
        """Initialize the Dice loss.

        Parameters
        ----------
        smooth : float, optional
            Numerical stabilizer added to numerator and denominator.
        """
        super().__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        """Compute soft Dice loss.

        Parameters
        ----------
        inputs : torch.Tensor
            Predicted mask probabilities.
        targets : torch.Tensor
            Target mask values.

        Returns
        -------
        torch.Tensor
            Scalar Dice loss, equal to ``1 - Dice``.
        """
        inputs = inputs.view(inputs.size(0), -1)
        targets = targets.reshape(targets.size(0), -1)
        intersection = (inputs * targets).sum(dim=1)
        dice_coeff = (2.0 * intersection + self.smooth) / (
            inputs.sum(dim=1) + targets.sum(dim=1) + self.smooth
        )
        return 1 - dice_coeff.mean()


class GeodesicLoss(nn.Module):
    """Geodesic distance loss between 3D rotation matrices."""

    def __init__(self, eps: float = 1e-7, reduction: str = "mean") -> None:
        """Initialize the rotation loss.

        Parameters
        ----------
        eps : float, optional
            Clamp margin for numerical stability in ``acos``.
        reduction : {"none", "mean", "sum"}, optional
            Reduction applied to per-sample angular distances.
        """
        super().__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """Compute geodesic distance between rotations.

        Parameters
        ----------
        input : torch.Tensor
            Predicted rotation matrices with shape ``[B, 3, 3]``.
        target : torch.Tensor
            Target rotation matrices with shape ``[B, 3, 3]``.

        Returns
        -------
        torch.Tensor
            Reduced angular distance in radians.
        """
        r_diffs = input @ target.permute(0, 2, 1)
        traces = r_diffs.diagonal(dim1=-2, dim2=-1).sum(-1)
        dists = torch.acos(torch.clamp((traces - 1) / 2, -1 + self.eps, 1 - self.eps))
        if self.reduction == "none":
            return dists
        if self.reduction == "mean":
            return dists.mean()
        if self.reduction == "sum":
            return dists.sum()
        raise ValueError(f"Unsupported reduction: {self.reduction}")


def get_loss(loss):
    """Create a loss object by name.

    Parameters
    ----------
    loss : str
        Loss identifier. Supported values are ``"dice"``, ``"mse"``, and
        ``"geodesic"``.

    Returns
    -------
    torch.nn.Module
        Instantiated loss module.
    """
    if loss == "dice":
        return DiceLoss()
    if loss == "mse":
        return MSELoss()
    if loss == "geodesic":
        return GeodesicLoss()
    raise ValueError(f"Unsupported loss: {loss}")


def transform_points(points, affine_matrices, volume_shape):
    """Apply batched normalized affine transforms to voxel-space points.

    Parameters
    ----------
    points : torch.Tensor
        Points with shape ``[B, N, 3]`` in voxel coordinates.
    affine_matrices : torch.Tensor
        Homogeneous affine transforms with shape ``[B, 4, 4]``.
    volume_shape : tuple
        Shape tuple ``(B, D, H, W)`` used for normalization.

    Returns
    -------
    torch.Tensor
        Transformed points with shape ``[B, N, 3]``.
    """
    batch_size, depth, height, width = volume_shape
    _, num_points, _ = points.shape

    points_normalized = (
        2.0
        * points
        / torch.tensor([depth - 1, height - 1, width - 1], device=points.device)
        - 1.0
    )
    ones = torch.ones((batch_size, num_points, 1), device=points.device)
    points_homogeneous = torch.cat((points_normalized, ones), dim=2)
    transformed_points = torch.einsum(
        "bij,bnj->bni", affine_matrices, points_homogeneous
    )

    return (
        (transformed_points[:, :, :3] + 1.0)
        * torch.tensor([depth - 1, height - 1, width - 1], device=points.device)
        / 2.0
    )


def corner_displacement_distance(corners, gtmat, tmat, image_shape):
    """Compute corner displacement after composing predicted and target transforms.

    Parameters
    ----------
    corners : torch.Tensor
        Bounding-box corners with shape ``[B, 8, 3]``.
    gtmat : torch.Tensor
        Ground-truth transforms with shape ``[B, 4, 4]``.
    tmat : torch.Tensor
        Predicted transforms with shape ``[B, 4, 4]``.
    image_shape : tuple
        Shape tuple ``(B, D, H, W)``.

    Returns
    -------
    torch.Tensor
        Mean corner displacement for each batch item.
    """
    inv_gtmat = torch.linalg.pinv(gtmat)
    corners_gtmat = transform_points(corners, inv_gtmat, image_shape)
    corners_gtmat_tmat = transform_points(corners_gtmat, tmat, image_shape)
    return torch.linalg.norm(corners - corners_gtmat_tmat, dim=-1).mean(dim=-1)


def get_bb_diagonal_length(corners):
    """Compute bounding-box diagonal length.

    Parameters
    ----------
    corners : torch.Tensor
        Bounding-box corners with shape ``[B, 8, 3]``.

    Returns
    -------
    torch.Tensor
        Diagonal length for each batch item.
    """
    min_xyz = torch.min(corners, dim=1).values
    max_xyz = torch.max(corners, dim=1).values
    return torch.linalg.norm(max_xyz - min_xyz, dim=1)


def bbvre_scaled_to_mean(corners, gtmat, tmat, image_shape, mean_diag=79.52):
    """Compute Bounding Box Volume Registration Error.

    Parameters
    ----------
    corners : torch.Tensor
        Bounding-box corners with shape ``[B, 8, 3]``.
    gtmat : torch.Tensor
        Ground-truth transforms with shape ``[B, 4, 4]``.
    tmat : torch.Tensor
        Predicted transforms with shape ``[B, 4, 4]``.
    image_shape : tuple
        Shape tuple ``(B, D, H, W)``.
    mean_diag : float, optional
        Reference bounding-box diagonal used for scaling.

    Returns
    -------
    torch.Tensor
        Scaled BBVRE for each batch item.
    """
    displacement = corner_displacement_distance(corners, gtmat, tmat, image_shape)
    diagonal = get_bb_diagonal_length(corners)
    return displacement / diagonal * mean_diag


def gaussian_kl_loss(
    tensor,
    means,
    covariance_matrices,
    gaussian_type="anisotropic",
    make_probabilistic=True,
):
    """Compare heatmaps to Gaussian distributions with matched moments.

    Parameters
    ----------
    tensor : torch.Tensor
        Heatmaps with shape ``[B, K, D, H, W]``.
    means : torch.Tensor
        Heatmap centers with shape ``[B, K, 3]``.
    covariance_matrices : torch.Tensor
        Heatmap covariance matrices with shape ``[B, K, 3, 3]``.
    gaussian_type : str, optional
        Gaussian parameterization passed to ``make_gaussian``.
    make_probabilistic : bool, optional
        If ``True``, normalize heatmaps spatially before computing KL.

    Returns
    -------
    torch.Tensor
        Scalar KL divergence loss.
    """
    n_dims = means.shape[-1]
    im_shape = tensor.shape[-n_dims:]
    dim_to_sum = list(range(1, 2 + n_dims))

    if make_probabilistic:
        tensor = torch.abs(tensor)
        tensor = tensor / tensor.sum(dim=list(range(2, 2 + n_dims)), keepdim=True)

    with torch.no_grad():
        gaussian = make_gaussian(means, covariance_matrices, im_shape, gaussian_type)

    loss = tensor * (torch.log(tensor + 1e-24) - torch.log(gaussian + 1e-24))
    return torch.sum(loss, dim=dim_to_sum).mean()


def repulsive_loss(points, temperature=1, mask=None):
    """Encourage keypoints to be spatially separated.

    Parameters
    ----------
    points : torch.Tensor
        Keypoints with shape ``[B, K, 3]``.
    temperature : float, optional
        Temperature for the sigmoid distance penalty.
    mask : torch.Tensor, optional
        Boolean mask selecting point pairs to include.

    Returns
    -------
    torch.Tensor
        Scalar repulsive regularization loss.
    """
    dist = torch.cdist(points, points)
    if mask is None:
        mask = torch.triu(torch.ones_like(dist, dtype=torch.bool), diagonal=1)
    mask = mask.to(device=dist.device)
    return (mask * (1 - 1 / (1 + torch.exp(-dist / temperature)))).sum(dim=[1, 2]).mean()


def frobenius_norm(covariance):
    """Compute squared Frobenius norm for covariance matrices.

    Parameters
    ----------
    covariance : torch.Tensor
        Covariance tensor with shape ``[B, K, 3, 3]`` or variance tensor.

    Returns
    -------
    torch.Tensor
        Per-keypoint squared covariance magnitude.
    """
    if len(covariance.shape) > 2:
        return torch.sum(covariance**2, dim=[2, 3])
    return covariance**2
