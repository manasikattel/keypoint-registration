import torch
from kornia.geometry.conversions import normalize_homography3d
import torch.nn.functional as F
import numpy as np


def calculate_barycenter(prob_mask: torch.Tensor):
    """
    Calculate the barycenter (center of mass) for batched, multi-channel probabilistic masks.
    
    Args:
        prob_mask: torch.Tensor [B, C, *spatial_dims]
    
    Returns:
        barycenters: torch.Tensor [B, C, n_dims]
    """
    prob_mask = torch.clamp(prob_mask, min=0.0)
    
    B, C, *spatial_dims = prob_mask.shape
    n_dims = len(spatial_dims)
    
    # create coordinate grids [n_dims, *spatial_dims]
    grids = torch.meshgrid(
        *[torch.arange(s, device=prob_mask.device, dtype=prob_mask.dtype) for s in spatial_dims],
        indexing="ij"
    )
    coords = torch.stack(grids, dim=0)  # [n_dims, *spatial_dims]
    
    # reshape for broadcasting
    coords = coords.unsqueeze(0).unsqueeze(0)  # [1, 1, n_dims, *spatial_dims]
    prob_mask_exp = prob_mask.unsqueeze(2)     # [B, C, 1, *spatial_dims]
    
    # sum over spatial dimensions
    spatial_axes = tuple(range(3, 3 + n_dims))
    total_mass = prob_mask_exp.sum(dim=spatial_axes)  # [B, C, 1]
    
    # weighted sum
    weighted_sum = (prob_mask_exp * coords).sum(dim=spatial_axes)  # [B, C, n_dims]
    
    # divide
    barycenters = weighted_sum / (total_mass + 1e-12)  # [B, C, n_dims]
    
    return torch.flip(barycenters, dims=[-1])


def make_gaussian(mean, covariance_matrix, im_shape, gaussian_type='anisotropic'):
    """This function returns a torch tensor of size [B, C, *im_shape] representing Gaussians of specified means and
    covariance matrices. The covariance matrices can be given as tensors of shape [B, C] (isotropic Gaussians) or
     [B, C, n_dims, n_dims] (anisotropic Gaussians).
    :param mean: means of the Gaussians. size is [B, C, n_dims]
    :param covariance_matrix: covariance matrix of the Gaussians. size is [B, C] (isotropic Gaussians) or
    [B, C, n_dims, n_dims] (anisotropic Gaussians).
    :param im_shape: dimensions of the Gaussians to create, for example [H, W] (2D case) or [H, W, D] (3D case).
    :param gaussian_type: type of the Gaussian to create.
    """
    # prepare coordinate grid
    n_dims = len(im_shape)
    coords_list = [torch.arange(shape, dtype=mean.dtype, device=mean.device) for shape in im_shape]
    coords_grid = add_axis(torch.stack(torch.meshgrid(*coords_list), -1), [0, 1])  # [1, 1, *, n_dims]

    # simplified version for isotropic (no matrix inversion, no reshaping)
    if gaussian_type == 'isotropic':

        mean = add_axis(mean, axis=[2] * n_dims)
        covariance_matrix = add_axis(covariance_matrix, [-1] * n_dims)
        exp_term = - torch.sum((coords_grid - mean) ** 2, dim=-1) / (2 * covariance_matrix)
        gaussian = torch.exp(exp_term) / torch.sqrt((2 * torch.pi * covariance_matrix) ** n_dims)

    elif gaussian_type == 'anisotropic':

        # linear algebra on covariance matrix
        det = torch.linalg.det(covariance_matrix)
        if not ((det.isnan() | det.isinf() | (det < 1e-5)).any()):
            covariance_matrix_inv = torch.linalg.inv(covariance_matrix)

        # In case there's a singular matrix (it happens when the features are a single point, so the covariance matrix
        # is very small everywhere) we split along batch and channel to invert the cov one by one. We replace the
        # singular covariance matrices by 1e-8 * Identity
        else:
            covariance_matrix_inv = list()
            id_3x3 = add_axis(torch.eye(3, device=covariance_matrix.device), [0, 1])
            for cov_batch in torch.split(covariance_matrix, split_size_or_sections=1, dim=0):
                cov_channel_inv = list()

                for cov_channel in torch.split(cov_batch, split_size_or_sections=1, dim=1):
                    det_c = torch.linalg.det(cov_channel)

                    if det_c.isnan() or det_c.isinf() or det_c <= 1e-6:
                        try:
                            cov_channel_inv.append(torch.linalg.inv(torch.clamp(cov_channel, min=1e-6) * id_3x3))
                        except:
                            cov_channel_inv.append(torch.linalg.inv(1e-6 * id_3x3))

                    elif torch.max(cov_channel) < 1e-6:
                        cov_channel_inv.append(torch.linalg.inv(1e-6 * id_3x3))


                    else:
                        cov_channel_inv.append(torch.linalg.inv(cov_channel))

                cov_channel_inv = torch.concat(cov_channel_inv, dim=1)
                covariance_matrix_inv.append(cov_channel_inv)
            covariance_matrix_inv = torch.concat(covariance_matrix_inv, dim=0)
            det = 1 / torch.linalg.det(covariance_matrix_inv)

        # prepare exp term
        coords_grid = coords_grid - add_axis(mean, axis=[2] * n_dims)
        coords_grid = torch.reshape(coords_grid, list(coords_grid.shape[:2]) + [np.prod(im_shape), n_dims])
        exp_term = torch.moveaxis(torch.matmul(covariance_matrix_inv, torch.moveaxis(coords_grid, -2, -1)), -2, -1)
        exp_term = -0.5 * (coords_grid * exp_term).sum(dim=-1)

        # make gaussian (stabilized)
        exp_term = torch.clamp(exp_term, min=-80.0, max=0.0)
        det_term = add_axis(det, axis=-1) * (2 * torch.pi) ** n_dims
        det_term = torch.clamp(det_term, min=1e-24)
        gaussian = torch.exp(exp_term) / torch.sqrt(det_term)
        gaussian = torch.reshape(gaussian, list(coords_grid.shape[:2]) + list(im_shape))
        gaussian = torch.nan_to_num(gaussian, nan=0.0, posinf=0.0, neginf=0.0)

    else:
        raise ValueError('gaussian_type must be anisotropic or anisotropic, had %s' % gaussian_type)

    # make gaussian probabilistic, just in case (we can have funny problems because of field-of-view)
    norm = gaussian.sum(dim=list(range(2, 2 + n_dims)), keepdim=True)
    norm = torch.clamp(norm, min=1e-24)
    return gaussian / norm
    # return gaussian / gaussian.sum(dim=list(range(2, 2 + n_dims)), keepdim=True)  # [B, C, *]



def calculate_barycenter_and_covariance(prob_mask: torch.Tensor,isotropic=False):
    """
    Calculate barycenter and covariance matrices for batched, multi-channel probabilistic masks.

    Args:
        prob_mask: torch.Tensor [B, C, *spatial_dims]
    
    Returns:
        barycenters: [B, C, n_dims]
        covariances: [B, C, n_dims, n_dims]
    """
    prob_mask = torch.clamp(prob_mask, min=0.0)
    B, C, *spatial_dims = prob_mask.shape
    n_dims = len(spatial_dims)

    # --- coordinate grid ---
    grids = torch.meshgrid(
        *[torch.arange(s, device=prob_mask.device, dtype=prob_mask.dtype) for s in spatial_dims],
        indexing="ij"
    )
    coords = torch.stack(grids, dim=0)  # [n_dims, *spatial]
    coords = coords.unsqueeze(0).unsqueeze(0)  # [1,1,n_dims,*spatial]
    coords = coords.expand(B, C, -1, *spatial_dims)  # [B,C,n_dims,*spatial]
    prob_mask_exp = prob_mask.unsqueeze(2)  # [B,C,1,*spatial]
    # --- total mass ---
    spatial_axes = tuple(range(3, 3 + n_dims))
    total_mass = prob_mask_exp.sum(dim=spatial_axes, keepdim=True)  # [B,C,1,1]

    # --- barycenter ---
    weighted_sum = (prob_mask_exp * coords).sum(dim=spatial_axes, keepdim=True)  # [B,C,n_dims,1]
    barycenter = weighted_sum / (total_mass + 1e-12)  # [B,C,n_dims,1]

    # ✅ reshape barycenter to broadcast correctly across spatial dims
    barycenter = barycenter.view(B, C, n_dims, *([1] * n_dims))  # [B,C,n_dims,1,1,1] for 3D

    # --- covariance ---
    diff = coords - barycenter  # [B,C,n_dims,*spatial]
    diff_outer = diff.unsqueeze(3) * diff.unsqueeze(2)  # [B,C,n_dims,n_dims,*spatial]
    weighted_diff_outer = prob_mask_exp.unsqueeze(2) * diff_outer

    cov = weighted_diff_outer.sum(dim=(-1,-2,-3))  / (total_mass.squeeze(2,3) + 1e-12)  # [B,C,n_dims,n_dims]

    # final barycenter shape [B,C,n_dims]
    barycenter = barycenter.squeeze(dim=list(range(3, 3 + n_dims)))
    barycenter = torch.flip(barycenter, dims=[-1])     # flip (z,y,x) -> (x,y,z)
    cov = torch.flip(cov, dims=[-1, -2])               # flip both covariance axes

    if isotropic:
        # isotropic variance = mean of diagonal elements (trace / n_dims)
        var_iso = cov.diagonal(dim1=-2, dim2=-1).mean(dim=-1, keepdim=True)  # [B,C,1]
        cov_result = var_iso
    else:
        cov_result = cov
    return barycenter, cov_result



def transform_points(points, affine_matrices, volume_shape):
    """Transform voxel-space points with normalized affine matrices.

    Parameters
    ----------
    points : torch.Tensor
        Points with shape ``[B, N, 3]``.
    affine_matrices : torch.Tensor
        Homogeneous transforms with shape ``[B, 4, 4]``.
    volume_shape : tuple
        Shape tuple ``(B, D, H, W)`` used for coordinate normalization.

    Returns
    -------
    torch.Tensor
        Transformed points with shape ``[B, N, 3]``.
    """
    B, D, H, W = volume_shape
    B, N, _ = points.shape  # Extract batch size (B) and number of points (N)
    
    # Normalize the point coordinates to the range [-1, 1]
    points_normalized = 2.0 * points / torch.tensor([D-1, H-1, W-1], device=points.device) - 1.0  # Shape: (B, N, 3)
    
    # Add a row of ones to handle affine transformation (homogeneous coordinates)
    ones = torch.ones((B, N, 1), device=points.device)  # Shape: (B, N, 1)
    points_homogeneous = torch.cat((points_normalized, ones), dim=2)  # Shape: (B, N, 4)
    # Apply the affine transformation (batch-wise matrix multiplication)
    transformed_points = torch.einsum('bij,bnj->bni', affine_matrices, points_homogeneous)  # Shape: (B, N, 4)
    
    # Denormalize the transformed points back to the original space
    transformed_points = (transformed_points[:, :, :3] + 1.0) * torch.tensor([D-1, H-1, W-1], device=points.device) / 2.0
    
    return transformed_points


def add_axis(x, axis=None):
    """Add axis to a numpy array or pytorch tensor.
    :param x: input array/tensor
    :param axis: index of the new axis to add. Can also be a list of indices to add several axes at the same time."""
    func = torch.unsqueeze if torch.is_tensor(x) else np.expand_dims
    axis = 0 if axis is None else axis
    if not isinstance(axis, list):
        axis = [axis]
    for ax in axis:
        x = func(x, ax)
    return x


def pts_to_xfm_numerical(means_moving, means_fixed, im_shape, weights=None, scaling_type=None):
    """gets rigid optimal transform to go from means_moving to means_fixed
    :param: means_moving, means_fixed: torch tensors of size [batch, n_channels, 3]
    :param: im_shape: torch tensor of size [3]
    :param: weights: torch tensor of size [batch, n_channels, 1] indicating the weight of each point in computing the
    transform. Doesn't need to be normalised (i.e., sum up to 1), because this is done internally.
    This parameter is interesting when points are virtually outside the image shape. In this case, it's good to
    set their weight to 0, such that they are not taken into account (otherwise they mess up the estimation).
    :param scaling_type: if scaling_type is not None, this function will also try to estimate a scaling between the two
    point-clouds. Chose between isotropic and anisotropic scaling.
    """

    # shift feature coordinates to center of the image
    # half_im_shape = add_axis((im_shape - 1) / 2.0, [0, 0])  # [1, 1, 3]
    # means_moving = means_moving - half_im_shape  # [B, K, 3]
    # means_fixed = means_fixed - half_im_shape  # [B, K, 3]
    # correct for centroids

    if weights is not None:
        weights = weights / weights.sum(dim=1, keepdim=True)
        centroid_moving = (means_moving * weights).sum(dim=1, keepdims=True)  # [B, 1, 3]
        centroid_fixed = (means_fixed * weights).sum(dim=1, keepdims=True)  # [B, 1, 3]
    else:
        centroid_moving = torch.mean(means_moving, dim=1, keepdim=True)  # [B, 1, 3]
        centroid_fixed = torch.mean(means_fixed, dim=1, keepdim=True)  # [B, 1, 3]
    means_moving = means_moving - centroid_moving  # [B, K, 3]
    means_fixed = means_fixed - centroid_fixed  # [B, K, 3]

    if scaling_type is None:

        # SVD decomposition
        if weights is not None:
            H = (means_moving * weights).transpose(1, 2) @ means_fixed  # [B, 3, 3]
        else:
            H = means_moving.transpose(1, 2) @ means_fixed  # [B, 3, 3]
        U, S, V = torch.svd(H, compute_uv=True)  # U, V: [B, 3, 3],  S: [B, 3]

        # special reflection case
        det = torch.det(U) * torch.det(V)  # [B, 1]

        B = V.shape[0]

        corr_det = torch.eye(3, device=V.device, dtype=V.dtype).unsqueeze(0).repeat(B, 1, 1)
        corr_det[:, 2, 2] = det
        xfm = V @ corr_det @ U.transpose(1, 2)  # [B, 3, 3]

    elif scaling_type == 'isotropic':
        assert weights is None, "weights interfere with scaling, so these two parameters cannot be used together"

        # SVD decomposition
        H = means_moving.transpose(1, 2) @ means_fixed  # [B, 3, 3]
        U, S, V = torch.svd(H, compute_uv=True)  # U, V: [B, 3, 3],  S: [B, 3]

        # special reflection case
        dets = add_axis(torch.det(U) * torch.det(V), -1)  # [B, 1]
        dets = torch.stack([torch.ones_like(dets), torch.ones_like(dets), dets], dim=1)  # [B, 3, 1]
        R = V * torch.sign(dets) @ U.transpose(1, 2)  # [B, 3, 3]

        # find scaling
        scaling = S.sum(dim=1) / torch.einsum("...ii", means_moving.transpose(1, 2) @ means_moving)
        scaling = add_axis(scaling, [1, 2]) * add_axis(torch.eye(3,device=means_moving.device, dtype=means_moving.dtype))  # [B, 3, 3]
        xfm = R @ scaling

    elif scaling_type == 'anisotropic':

        assert weights is None, "weights interfere with scaling, so these two parameters cannot be used together"

        # SVD decomposition
        H = means_fixed.transpose(1, 2) @ means_moving @ torch.linalg.inv(means_moving.transpose(1, 2) @ means_moving)
        U, S, V = torch.svd(H.transpose(1, 2), compute_uv=True)

        # special reflection case
        dets = add_axis(torch.det(U) * torch.det(V), -1)  # [B, 1]
        dets = torch.stack([torch.ones_like(dets), torch.ones_like(dets), dets], dim=1)  # [B, 3, 1]
        R = V * torch.sign(dets) @ U.transpose(1, 2)  # [B, 3, 3]

        # find scaling
        scaling = list()
        for i in range(3):
            lambda_i = torch.zeros_like(R)
            lambda_i[:, i, i] = torch.ones(1)
            numerator = torch.einsum("...ii", means_fixed @ R @ lambda_i @ means_moving.transpose(1, 2))
            denominator = torch.einsum("...ii", means_moving @ lambda_i @ means_moving.transpose(1, 2))
            scaling.append(numerator / denominator)
        scaling = torch.diag_embed(torch.stack(scaling, dim=-1))
        xfm = R @ scaling

    else:
        raise ValueError("Scaling type should be None, 'isotropic' or 'anisotropic', had %s" % scaling_type)

    # find translation
    T = centroid_fixed.transpose(1, 2) - xfm @ centroid_moving.transpose(1, 2)   # [B, 3, 1]

    # reconstruct affine matrix
    xfm = torch.cat([xfm, T], dim=-1)  # [B, 3, 4]
    last_row = add_axis(torch.tensor([0] * means_moving.shape[-1] + [1]), [0, 0]).repeat(xfm.shape[0], 1, 1)
    xfm = torch.cat([xfm, last_row.to(device=xfm.device, dtype=xfm.dtype)], dim=1)  # [B, 4, 4]
    mat = normalize_homography3d(xfm, im_shape, im_shape)
    xfm = torch.linalg.pinv(mat)
    return xfm


def apply_translation_list(images, normalized_translations, return_transform=False):
    """
    Apply the given normalized translations to a batch of 3D volumes using grid sampling.
    
    Args:
        volume_batch (torch.Tensor): Input volume (B, C, D, H, W).
        normalized_translations (torch.Tensor): Normalized translations (B, 3).
        
    Returns:
        torch.Tensor: Translated volume batch (B, C, D, H, W).
    """
    combined = torch.stack(images, dim=0)[:, None, :, :, :]
    # if len(combined.shape)==3:
    #     combined = combined[None,None,:,:,:]
    # else:
    #     combined = combined[:, None, :, :, :]  # Shape: (B, 1, D, H, W)
    B, _, D, H, W = combined.shape
    device = images[0].device
    
    affine_matrices = torch.eye(3, 4).repeat(B, 1, 1).to(device)

    affine_matrices[:, 0, 3] = normalized_translations[:, 0]  # X translation
    affine_matrices[:, 1, 3] = normalized_translations[:, 1]  # Y translation
    affine_matrices[:, 2, 3] = normalized_translations[:, 2]  # Z translation
    
    grid = F.affine_grid(affine_matrices, size=(B, 1, D, H, W), align_corners=False).to(device)    
    image_t = F.grid_sample(combined, grid, align_corners=False, padding_mode="zeros").squeeze(1)
    image_t = torch.unbind(image_t, dim=0)

    if return_transform:
        return image_t, affine_matrices
    return image_t

def gaussian_kernel_3d(size=5, sigma=1.0):
    """Creates a 3D Gaussian kernel."""
    x = torch.arange(size).float() - size // 2
    grid = torch.stack(torch.meshgrid(x, x, x, indexing='ij'))  # Create a 3D grid
    kernel = torch.exp(-(grid[0]**2 + grid[1]**2 + grid[2]**2) / (2 * sigma**2))
    kernel /= kernel.sum()
    return kernel

def min_max_normalize(blurred_mask):
    """Normalize a tensor to the ``[0, 1]`` range.

    Parameters
    ----------
    blurred_mask : torch.Tensor
        Input tensor.

    Returns
    -------
    torch.Tensor
        Min-max normalized tensor, or zeros if the input is constant.
    """
    if blurred_mask.max() != blurred_mask.min():
        blurred_mask = (blurred_mask - blurred_mask.min()) / (blurred_mask.max() - blurred_mask.min())
    else:
        blurred_mask = torch.zeros_like(blurred_mask)  # Keep it zero if the input was zero
    return blurred_mask

def blur_mask_3d(mask, kernel_size=5, sigma=1.0):
    """
    Applies Gaussian blur to a 3D binary mask.

    Args:
        mask (torch.Tensor): Binary mask of shape (D, H, W) or (1, D, H, W).
        kernel_size (int): Size of the Gaussian kernel.
        sigma (float): Standard deviation of the Gaussian.

    Returns:
        torch.Tensor: Blurred probability mask.
    """
    # Ensure the mask has shape (1, 1, D, H, W) for convolution
    if mask.dim() == 3:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.dim() == 4:
        mask = mask.unsqueeze(0)

    # Create 3D Gaussian kernel
    kernel = gaussian_kernel_3d(kernel_size, sigma)
    kernel = kernel.view(1, 1, kernel_size, kernel_size, kernel_size)  # Shape: (1, 1, k, k, k)

    # Pad the mask to prevent border effects
    pad = kernel_size // 2
    mask = F.pad(mask, (pad, pad, pad, pad, pad, pad), mode='reflect')

    # Apply Gaussian blur using 3D convolution
    blurred_mask = F.conv3d(mask, kernel, padding=0)

    # Normalize to [0, 1] to form a probability mask
    # blurred_mask = (blurred_mask - blurred_mask.min())/( blurred_mask.max()-blurred_mask.min())
    blurred_mask = min_max_normalize(blurred_mask)
    return blurred_mask.squeeze(0).squeeze(0)  # Remove batch/channel dims
