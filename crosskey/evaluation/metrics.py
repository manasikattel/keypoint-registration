import torch
from crosskey.utils.rotation_conversions import matrix_to_axis_angle
import torch.nn.functional as F

def rot_error(R_gt, R_pred):
    """
    Computes rotation (in degrees) and denormalized translation error (in pixels/voxels) for a batch.

    Args:
        rtvec_pred (torch.Tensor): Predicted 6D vectors [rvec (3), tvec (3)], shape (B, 6)
        rtvec_gt (torch.Tensor): Ground truth 6D vectors [rvec (3), tvec (3)], shape (B, 6)
        volume_shape (tuple): (D, H, W) - size of the image/volume

    Returns:
        rot_err_deg (torch.Tensor): Rotation errors in degrees, shape (B,)
        trans_err (torch.Tensor): Translation errors in voxel space, shape (B,)
    """
    R_rel = torch.matmul(R_gt.transpose(-1, -2), R_pred)
    theta_rad = torch.norm(matrix_to_axis_angle(R_rel), dim=1)
    theta_deg = torch.rad2deg(theta_rad)
    return theta_deg


def extract_surface_voxels(mask: torch.Tensor) -> torch.Tensor:
    """
    Returns [N, 4] tensor of (batch_idx, x, y, z) for surface voxels.
    """
    B, D, H, W = mask.shape
    mask = (mask > 0.3).float().unsqueeze(1)
    kernel = torch.ones((1, 1, 3, 3, 3), device=mask.device)
    kernel[0, 0, 1, 1, 1] = 0

    neighbor_count = F.conv3d(mask, kernel, padding=1)
    surface = (mask == 1) & (neighbor_count < 26)
    coords = torch.nonzero(surface, as_tuple=False)
    coords = coords[:, [0, 4, 3, 2]]  # [B, x, y, z]
    return coords

def transform_points_flat(coords: torch.Tensor, T: torch.Tensor, volume_shape):
    """
    coords: [N, 4] -> (b, x, y, z)
    T: [B, 4, 4]
    volume_shape: (B, D, H, W)
    Returns:
        b_idx: [N]
        transformed_points: [N, 3]
    """
    B, D, H, W = volume_shape
    b = coords[:, 0]
    points = coords[:, 1:].float()
    norm = torch.tensor([D-1, H-1, W-1], device=points.device)
    points_normalized = 2.0 * points / norm - 1.0

    ones = torch.ones((points.shape[0], 1), device=points.device)
    homogeneous = torch.cat([points_normalized, ones], dim=1)

    transformed = torch.empty_like(homogeneous[:, :3])
    for i in range(B):
        idx = b == i
        if idx.any():
            T_batch = T[i]
            pts_i = homogeneous[idx]
            transformed[idx] = (T_batch @ pts_i.T).T[:, :3]

    # Denormalize
    transformed = (transformed + 1.0) * norm / 2.0
    return b, transformed



def compute_surface_registration_error(source_mask: torch.Tensor, T_gt: torch.Tensor, T_pred: torch.Tensor) -> torch.Tensor:
    """
    Compute Surface Registration Error (SRE) in target space when source is the moving image.

    Args:
        source_mask: [B, D, H, W] binary mask of source segmentation
        T_gt: [B, 4, 4] ground truth source → target transforms
        T_pred: [B, 4, 4] predicted source → target transforms

    Returns:
        SRE: [B] SRE value per batch item (mean L2 error over surface points)
    """
    B, D, H, W = source_mask.shape
    device = source_mask.device

    # Step 1: Extract surface points from source mask: returns [N, 4] (b, x, y, z)
    surface_coords = extract_surface_voxels(source_mask)  # custom function from before

    # Step 2: Transform surface points using GT and predicted transforms
    b_idx, coords = surface_coords[:, 0], surface_coords[:, 1:].float()  # [N], [N, 3]

    # Apply GT and Pred transforms to same surface points
    _, coords_gt = transform_points_flat(surface_coords, T_gt, (B, D, H, W))     # [N, 3]
    _, coords_pred = transform_points_flat(surface_coords, T_pred, (B, D, H, W)) # [N, 3]

    # Step 3: Compute per-point L2 error in target space
    errors = (coords_pred - coords_gt).pow(2).sum(dim=1).sqrt()  # [N]

    # Step 4: Compute mean per-batch
    sre_per_batch = torch.zeros(B, device=device)
    for b in range(B):
        mask = (b_idx == b)
        if mask.sum() > 0:
            sre_per_batch[b] = errors[mask].mean()

    return sre_per_batch  # shape: [B]
