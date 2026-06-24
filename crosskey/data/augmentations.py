import torch
import torch.nn.functional as F
import numpy as np
from scipy.stats import special_ortho_group
from scipy.spatial.transform import Rotation as R


class MinMaxNormalize:
    """Min-max normalize a tensor to the ``[0, 1]`` range."""

    def __call__(self, volume):
        """Normalize a volume.

        Parameters
        ----------
        volume : torch.Tensor or numpy.ndarray
            Input 3D or channel-first 4D volume.

        Returns
        -------
        torch.Tensor
            Normalized volume.
        """
        if not isinstance(volume, torch.Tensor):
            volume = torch.from_numpy(volume)

        vmin = volume.min()
        vmax = volume.max()
        if vmax == vmin:
            return torch.zeros_like(volume)
        return (volume - vmin) / (vmax - vmin + 1e-9)


class IntensityAugmenter:
    """MONAI intensity augmentation used for source/target training volumes."""

    def __init__(self, apply_mri_augmentations=False):
        """Initialize intensity augmentation parameters.

        Parameters
        ----------
        apply_mri_augmentations : bool, optional
            If ``True``, include MRI-specific Rician noise, Gibbs ringing, and
            bias-field perturbation in addition to shared noise, smoothing, and
            contrast augmentation.
        """
        from monai.transforms import (
            Compose,
            RandAdjustContrast,
            RandBiasField,
            RandGaussianNoise,
            RandGaussianSmooth,
            RandGibbsNoise,
            RandRicianNoise,
        )

        self.apply_mri_augmentations = apply_mri_augmentations
        shared_transforms = [
            MinMaxNormalize(),
            RandGaussianNoise(prob=0.8, mean=0.0, std=0.1),
            RandGaussianSmooth(
                prob=0.8,
                sigma_x=(0.5, 1.0),
                sigma_y=(0.5, 1.0),
                sigma_z=(0.5, 1.0),
            ),
            RandAdjustContrast(prob=0.8, gamma=(0.7, 1.5)),
            MinMaxNormalize(),
        ]

        if self.apply_mri_augmentations:
            self.pipeline = Compose(
                [
                    MinMaxNormalize(),
                    RandGaussianNoise(prob=0.8, mean=0.0, std=0.1),
                    RandRicianNoise(prob=0.8, mean=0.0, std=0.1),
                    RandGaussianSmooth(
                        prob=0.8,
                        sigma_x=(0.5, 1.0),
                        sigma_y=(0.5, 1.0),
                        sigma_z=(0.5, 1.0),
                    ),
                    RandGibbsNoise(prob=0.8, alpha=(0.5, 0.8)),
                    RandBiasField(prob=0.8, coeff_range=(0, 0.3)),
                    RandAdjustContrast(prob=0.8, gamma=(0.7, 1.5)),
                    MinMaxNormalize(),
                ]
            )
        else:
            self.pipeline = Compose(shared_transforms)

    def __call__(self, image):
        """Apply intensity augmentation.

        Parameters
        ----------
        image : torch.Tensor
            Input image with shape ``[D, H, W]``.

        Returns
        -------
        torch.Tensor
            Augmented image normalized to ``[0, 1]``.
        """
        return self.apply_transformation(image)

    def apply_transformation(self, image):
        """Apply the configured MONAI intensity transform pipeline.

        Parameters
        ----------
        image : torch.Tensor
            Input image with shape ``[D, H, W]``.

        Returns
        -------
        torch.Tensor
            Augmented image.
        """
        return self.pipeline(image)

def get_corrected_rotation_matrix(rotation_matrix, center, image_shape):
    """Create a homogeneous rotation around a specified image center.

    Parameters
    ----------
    rotation_matrix : torch.Tensor
        Rotation matrix with shape ``[3, 3]``.
    center : tuple
        Center of rotation in voxel coordinates.
    image_shape : tuple
        Spatial image shape ``(D, H, W)``.

    Returns
    -------
    torch.Tensor
        Homogeneous affine matrix with shape ``[4, 4]``.
    """
    center = torch.tensor(center, dtype=torch.float32)
    D, H, W = image_shape

    # Convert center to normalized coordinates [-1, 1]
    center_norm = 2 * center / torch.tensor([W-1, H-1, D-1]) - 1  # (z, y, x) in [-1, 1]

    # Create a 4x4 identity matrix
    affine_matrix = torch.eye(4, dtype=torch.float32)

    # Insert rotation matrix (top-left 3x3 block)
    affine_matrix[:3, :3] = rotation_matrix

    # Compute the corrected translation to keep center fixed
    affine_matrix[:3, 3] = center_norm - (rotation_matrix @ center_norm)

    return affine_matrix


def make_homogeneous(matrix):
    """Convert a 3-by-4 affine matrix to homogeneous form.

    Parameters
    ----------
    matrix : torch.Tensor
        Affine matrix with shape ``[3, 4]``.

    Returns
    -------
    torch.Tensor
        Homogeneous matrix with shape ``[4, 4]``.
    """
    assert matrix.shape == (3, 4), "Input matrix must be of shape (3,4)"
    
    bottom_row = torch.tensor([[0, 0, 0, 1]], device=matrix.device)  # Shape: (1, 4)
    
    return torch.cat([matrix, bottom_row], dim=0)  # Shape: (4, 4)

def make_rotation_homogeneous(rotation_matrix):
    """Convert a rotation matrix to a homogeneous transform.

    Parameters
    ----------
    rotation_matrix : torch.Tensor
        Rotation matrix with shape ``[3, 3]``.

    Returns
    -------
    torch.Tensor
        Homogeneous transform with shape ``[4, 4]``.
    """
    assert rotation_matrix.shape == (3, 3), "Input must be a 3x3 rotation matrix"
    
    # Create a 4x4 identity matrix
    homogeneous_matrix = torch.eye(4, device=rotation_matrix.device)
    
    # Insert the rotation matrix in the top-left 3x3 block
    homogeneous_matrix[:3, :3] = rotation_matrix
    
    return homogeneous_matrix  # Shape: (4, 4)


def add_axis(x, axis=None):
    """Insert one or more singleton axes.

    Parameters
    ----------
    x : numpy.ndarray or torch.Tensor
        Input array or tensor.
    axis : int or list of int, optional
        Axis or axes where singleton dimensions are inserted.

    Returns
    -------
    numpy.ndarray or torch.Tensor
        Input with added singleton axes.
    """
    func = torch.unsqueeze if torch.is_tensor(x) else np.expand_dims
    axis = 0 if axis is None else axis
    if not isinstance(axis, list):
        axis = [axis]
    for ax in axis:
        x = func(x, ax)
    return x



def create_transform(rx, ry, rz,
                     tx, ty, tz,
                     ordering='txyz', input_angle_unit='degrees'):
    """Create a homogeneous transform from Euler rotations and translations.

    Parameters
    ----------
    rx, ry, rz : float or torch.Tensor
        Rotation parameters.
    tx, ty, tz : float or torch.Tensor
        Translation parameters.
    ordering : str, optional
        Order in which transform components are composed.
    input_angle_unit : {"degrees", "radians"}, optional
        Unit for the rotation parameters.

    Returns
    -------
    numpy.ndarray or torch.Tensor
        Homogeneous transform matrix.
    """

    if torch.is_tensor(rx):

        if input_angle_unit == 'degrees':
            if len(rx.shape) == 1:
                rx, ry, rz = torch.split(torch.cat([rx, ry, rz], dim=0) / 180 * np.pi, 1, dim=0)
            else:
                rx, ry, rz = torch.split(torch.cat([rx, ry, rz], dim=1) / 180 * np.pi, 1, dim=1)

        one = torch.ones_like(rx)
        zero = torch.zeros_like(rx)

        Rx = torch.cat([torch.stack([one, zero, zero, zero], dim=-1),
                        torch.stack([zero, torch.cos(rx), -torch.sin(rx), zero], dim=-1),
                        torch.stack([zero, torch.sin(rx), torch.cos(rx), zero], dim=-1),
                        torch.stack([zero, zero, zero, one], dim=-1)],
                       dim=-2)

        Ry = torch.cat([torch.stack([torch.cos(ry), zero, torch.sin(ry), zero], dim=-1),
                        torch.stack([zero, one, zero, zero], dim=-1),
                        torch.stack([-torch.sin(ry), zero, torch.cos(ry), zero], dim=-1),
                        torch.stack([zero, zero, zero, one], dim=-1)],
                       dim=-2)

        Rz = torch.cat([torch.stack([torch.cos(rz), -torch.sin(rz), zero, zero], dim=-1),
                        torch.stack([torch.sin(rz), torch.cos(rz), zero, zero], dim=-1),
                        torch.stack([zero, zero, one, zero], dim=-1),
                        torch.stack([zero, zero, zero, one], dim=-1)],
                       dim=-2)

        T = torch.cat([torch.stack([one, zero, zero, tx], dim=-1),
                       torch.stack([zero, one, zero, ty], dim=-1),
                       torch.stack([zero, zero, one, tz], dim=-1),
                       torch.stack([zero, zero, zero, one], dim=-1)],
                      dim=-2)

        transform_matrix = torch.cat([torch.stack([one, zero, zero, zero], dim=-1),
                                      torch.stack([zero, one, zero, zero], dim=-1),
                                      torch.stack([zero, zero, one, zero], dim=-1),
                                      torch.stack([zero, zero, zero, one], dim=-1)],
                                     dim=-2)

    else:

        if input_angle_unit == 'degrees':
            rx, ry, rz = np.array([rx, ry, rz]) * np.pi / 180

        if len(rx.shape) == 0:
            rx, ry, rz, tx, ty, tz = add_axis(np.array([rx, ry, rz, tx, ty, tz]), -1)

        one = np.ones_like(rx)
        zero = np.zeros_like(rx)

        Rx = np.concatenate([np.stack([one, zero, zero, zero], axis=-1),
                             np.stack([zero, np.cos(rx), -np.sin(rx), zero], axis=-1),
                             np.stack([zero, np.sin(rx), np.cos(rx), zero], axis=-1),
                             np.stack([zero, zero, zero, one], axis=-1)],
                            axis=-2)

        Ry = np.concatenate([np.stack([np.cos(ry), zero, np.sin(ry), zero], axis=-1),
                             np.stack([zero, one, zero, zero], axis=-1),
                             np.stack([-np.sin(ry), zero, np.cos(ry), zero], axis=-1),
                             np.stack([zero, zero, zero, one], axis=-1)],
                            axis=-2)

        Rz = np.concatenate([np.stack([np.cos(rz), -np.sin(rz), zero, zero], axis=-1),
                             np.stack([np.sin(rz), np.cos(rz), zero, zero], axis=-1),
                             np.stack([zero, zero, one, zero], axis=-1),
                             np.stack([zero, zero, zero, one], axis=-1)],
                            axis=-2)

        T = np.concatenate([np.stack([one, zero, zero, tx], axis=-1),
                            np.stack([zero, one, zero, ty], axis=-1),
                            np.stack([zero, zero, one, tz], axis=-1),
                            np.stack([zero, zero, zero, one], axis=-1)],
                           axis=-2)

        transform_matrix = np.eye(4)

    dd = {'x': Rx, 'y': Ry, 'z': Rz, 't': T}
    for ll in ordering[::-1]:
        transform_matrix = dd[ll] @ transform_matrix

    return transform_matrix


def random_rotation_matrix(max_angle_degrees=30, exact_angle_degrees=None, tolerance_degrees=1.0):
    """Generate a random 3D rotation matrix.

    Parameters
    ----------
    max_angle_degrees : float, optional
        Maximum rotation angle when ``exact_angle_degrees`` is not set.
    exact_angle_degrees : float, optional
        Target rotation angle.
    tolerance_degrees : float, optional
        Allowed tolerance around ``exact_angle_degrees``.

    Returns
    -------
    torch.Tensor
        Rotation matrix with shape ``[3, 3]``.
    """
    if exact_angle_degrees is not None:
        target_angle_rad = np.deg2rad(exact_angle_degrees)
        tolerance_rad = np.deg2rad(tolerance_degrees)
        
        while True:
            # Generate random axis
            axis = np.random.randn(3)
            axis /= np.linalg.norm(axis)

            # Create rotation vector with target angle
            rotvec = axis * target_angle_rad
            r = R.from_rotvec(rotvec)

            angle = np.linalg.norm(rotvec)
            if abs(angle - target_angle_rad) <= tolerance_rad:
                rot_np = r.as_matrix()
                return torch.tensor(rot_np, dtype=torch.float32)
    else:
        max_angle_radians = np.deg2rad(max_angle_degrees)
        while True:
            # Generate random rotation matrix
            rot_np = special_ortho_group.rvs(3)
            r = R.from_matrix(rot_np)

            # Get rotation vector (axis-angle)
            rotvec = r.as_rotvec()
            angle = np.linalg.norm(rotvec)

            if angle <= max_angle_radians:
                return torch.tensor(rot_np, dtype=torch.float32)


def random_rotation_matrix_euler(rotation_range=30, use_max_values=False):
    """Sample a rotation matrix from Euler angles.

    Parameters
    ----------
    rotation_range : float, optional
        Maximum absolute Euler angle in degrees.
    use_max_values : bool, optional
        If ``True``, use the maximum configured rotation around the z-axis.

    Returns
    -------
    torch.Tensor
        Rotation matrix with shape ``[3, 3]``.
    """
    if use_max_values:
        rx, ry, rz = 0, 0,rotation_range
        # tx, ty, tz = translation_range, translation_range, translation_range
    else:
        rx, ry, rz = np.random.uniform(-rotation_range, rotation_range, 3)
        # tx, ty, tz = np.random.uniform(-translation_range, translation_range, 3)
    transform_matrix = create_transform(rx, ry, rz, 0,0,0, ordering='txyz')
    return torch.from_numpy(transform_matrix[:3,:3])


def get_random_rigid_matrix(image_shape, rotation_range=0.0, 
                    translation_range=0.0, use_max_values=False, 
                    sampling_space="rvs", rotation_center=None):
    """Generate a random 3D affine transform for augmentation.

    Parameters
    ----------
    image_shape : tuple
        Spatial shape ``(D, H, W)``.
    rotation_range : float, optional
        Maximum rotation angle in degrees.
    translation_range : float, optional
        Maximum translation in voxels.
    use_max_values : bool, optional
        Whether to use maximum configured augmentation values.
    sampling_space : {"rvs", "euler"}, optional
        Rotation sampling mode.
    rotation_center : tuple, optional
        Optional center of rotation.

    Returns
    -------
    torch.Tensor
        Affine matrix with shape ``[3, 4]``.
    """
    D, H, W = image_shape
    if sampling_space=="rvs":
        if use_max_values is True:
            rotation_matrix = random_rotation_matrix(exact_angle_degrees=rotation_range, 
                                                        tolerance_degrees=0.5)
        else:
            rotation_matrix = random_rotation_matrix(max_angle_degrees=rotation_range)
    elif sampling_space=="euler":
        if use_max_values is True:
            rotation_matrix = random_rotation_matrix_euler(rotation_range=rotation_range, 
                                                    use_max_values=True)
        else:
            rotation_matrix = random_rotation_matrix_euler(rotation_range=rotation_range)


    if rotation_center is not None:
        affine_matrix = get_corrected_rotation_matrix(rotation_matrix,rotation_center,(D,H,W))
    else: 
        affine_matrix = make_rotation_homogeneous(rotation_matrix)
    # Random translation (in pixels)
    translation = torch.FloatTensor(4).uniform_(-translation_range, translation_range)

    # Normalize translation to [-1, 1] (for affine_grid)
    translation[0] /= W / 2  # Normalize X (Width)
    translation[1] /= H / 2  # Normalize Y (Height)
    translation[2] /= D / 2  # Normalize Z (Depth)
    translation[3] = 0
    # Construct 3x4 affine matrix

    # Apply additional translation by modifying the last column of the affine matrix
    affine_matrix[:, 3] += translation

    return affine_matrix[:3, :]  


class SpatialAugmenter():
    """Apply paired rigid spatial augmentation to image/mask tensors."""

    def __init__(self, rotation_range=0.0, translation_range=0.0, sampling_space="rvs", mode="train"):
        """Initialize the spatial augmenter.

        Parameters
        ----------
        rotation_range : float, optional
            Maximum rotation angle in degrees.
        translation_range : float, optional
            Maximum translation in voxels.
        sampling_space : {"rvs", "euler"}, optional
            Rotation sampling mode.
        mode : str, optional
            Dataset mode controlling deterministic maximum-value behavior.
        """
        self.rotation_range = rotation_range
        self.translation_range = translation_range
        self.mode = mode
        self.sampling_space = sampling_space

    def apply_transformation(self, image_and_masks, rotation_center=None, return_transform=False,rigid_matrix=None):
        """Apply one affine transform to a list of 3D tensors.

        Parameters
        ----------
        image_and_masks : sequence of torch.Tensor
            Tensors with shape ``[D, H, W]`` transformed together.
        rotation_center : tuple, optional
            Optional center of rotation.
        return_transform : bool, optional
            Whether to return the sampled homogeneous transform.
        rigid_matrix : torch.Tensor, optional
            Explicit affine matrix with shape ``[3, 4]``.

        Returns
        -------
        tuple or tuple[list, torch.Tensor]
            Transformed tensors, and optionally the homogeneous transform.
        """
        # Combine the image and all masks into one tensor (batch format)
        combined = torch.stack(image_and_masks, dim=0)[:, None, :, :, :]  # Shape: (B, 1, D, H, W)
        B, _, D, H, W = combined.shape
        device = image_and_masks[0].device
        # Get random affine transformation matrix
        # affine_matrix = self.get_random_affine_matrix(D, H, W).unsqueeze(0)  # Shape: (1, 3, 4)


        if self.mode == "test":
            use_max_values = True
        else:
            use_max_values = False
        if rigid_matrix is None:
            rigid_matrix = get_random_rigid_matrix(image_shape=[D,H,W], 
            rotation_range=self.rotation_range,
            translation_range=self.translation_range,
            use_max_values=use_max_values,
            sampling_space=self.sampling_space).unsqueeze(0).repeat(B, 1, 1)  # Shape: (B, 3, 4)
        else:
            rigid_matrix = rigid_matrix.unsqueeze(0).repeat(B, 1, 1)
        # Create a 3D affine grid
        rigid_matrix = rigid_matrix.to(device=device)
        grid = F.affine_grid(rigid_matrix, size=(B, 1, D, H, W), align_corners=True)

        # Apply trilinear interpolation to the entire batch (image + masks)
        transformed_combined = F.grid_sample(combined, grid, mode='bilinear', padding_mode='zeros', align_corners=True).squeeze()

        # Convert the transformed tensor back to a list of tensors (image + masks)
        transformed_image_and_masks = torch.unbind(transformed_combined, dim=0)
        if return_transform:
            
            return transformed_image_and_masks, make_homogeneous(rigid_matrix[0])
        else:
            return transformed_image_and_masks
