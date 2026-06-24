from crosskey.utils.geometry import calculate_barycenter, blur_mask_3d, apply_translation_list
import os
from pathlib import Path
import SimpleITK as sitk
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from crosskey.data.augmentations import (
    IntensityAugmenter,
    SpatialAugmenter,
    get_random_rigid_matrix,
)
from lightning.pytorch.core import LightningDataModule
import logging
import torch.nn.functional as F

logger = logging.getLogger(__name__)

def read_flip(path):
    """Read a NIfTI image and flip the slice axis.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to an image readable by SimpleITK.

    Returns
    -------
    torch.Tensor
        Image tensor with shape ``[D, H, W]``.
    """
    image = sitk.ReadImage(path)
    image = torch.from_numpy(sitk.GetArrayFromImage(image).astype(np.float32))
    image_flipped  = torch.flip(image,dims=[0])
    return image_flipped.to(torch.float)

def min_max_normalize(image):
    """Normalize an image to the ``[0, 1]`` range.

    Parameters
    ----------
    image : torch.Tensor
        Input image.

    Returns
    -------
    torch.Tensor
        Min-max normalized image.
    """
    image = (image - image.min()) / (image.max() - image.min())
    return image

def save_rotations_to_file(rotations_dict, filename):
    """Save deterministic augmentation transforms.

    Parameters
    ----------
    rotations_dict : dict
        Mapping from sample index to source/target transform tensors.
    filename : str or pathlib.Path
        Output ``.npz`` filename.
    """
    # Convert tensors to numpy arrays and save
    flattened = {
        str(k): np.stack([v[0].cpu().numpy(), v[1].cpu().numpy()])
        for k, v in rotations_dict.items()
    }
    np.savez(filename, **flattened)


def get_cube_corners(mask):
    """Find bounding-box corners for a 3D mask.

    Parameters
    ----------
    mask : torch.Tensor
        Binary mask with shape ``[D, H, W]``.

    Returns
    -------
    torch.Tensor
        Eight bounding-box corners with shape ``[8, 3]`` in ``(x, y, z)`` order.
    """
    assert mask.ndim == 3, "Input mask must be a 3D PyTorch tensor."

    # Get the coordinates of all foreground (mask = 1) voxels
    voxel_coords = torch.nonzero(mask == 1, as_tuple=False)

    # Find the bounding box min and max coordinates
    z_min, y_min, x_min = voxel_coords.min(dim=0).values
    z_max, y_max, x_max = voxel_coords.max(dim=0).values

    # Define the 8 corners of the bounding box
    corners = torch.tensor([
        [x_min, y_min, z_min],  # Corner 0
        [x_max, y_min, z_min],  # Corner 1
        [x_min, y_max, z_min],  # Corner 2
        [x_max, y_max, z_min],  # Corner 3
        [x_min, y_min, z_max],  # Corner 4
        [x_max, y_min, z_max],  # Corner 5
        [x_min, y_max, z_max],  # Corner 6
        [x_max, y_max, z_max],  # Corner 7
    ], dtype=torch.float32)

    return corners
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

def compute_normalized_translation_to_center(barycenter, image_size):
    """Compute normalized translation from mask center to image center.

    Parameters
    ----------
    barycenter : array-like
        Mask barycenter in voxel coordinates.
    image_size : array-like
        Spatial image size.

    Returns
    -------
    torch.Tensor
        Translation vector in ``affine_grid`` normalized coordinates.
    """
    # Convert to torch tensors
    barycenter = torch.tensor(barycenter, dtype=torch.float32)
    image_size = torch.tensor(image_size, dtype=torch.float32)

    # Compute center of image in voxel space
    center_voxel = image_size / 2.0

    # Translation vector in voxel space (z, y, x)
    delta =  barycenter-center_voxel

    # Normalize translation for affine_grid (x, y, z)
    norm_translation = 2.0 * delta / image_size

    # Reorder to (x, y, z) for affine_grid compatibility
    # norm_translation = norm_translation[[2, 1, 0]]  # reorder to (x, y, z)

    return norm_translation

def load_rotations_from_file(filename, device="cpu"):
    """Load deterministic augmentation transforms.

    Parameters
    ----------
    filename : str or pathlib.Path
        Input ``.npz`` filename.
    device : str or torch.device, optional
        Device for returned tensors.

    Returns
    -------
    dict
        Mapping from sample index to source/target transforms.
    """
    loaded = np.load(filename)
    return {
            int(k): (
                torch.tensor(loaded[k][0], device=device),
                torch.tensor(loaded[k][1], device=device),
                    )
            for k in loaded.files
            }



def resize_images(images,output_size=(32,32,32)):
    """Resize a list of 3D tensors with trilinear interpolation.

    Parameters
    ----------
    images : sequence of torch.Tensor
        Images with shape ``[D, H, W]``.
    output_size : tuple of int, optional
        Output spatial size.

    Returns
    -------
    list of torch.Tensor
        Resized image tensors.
    """
    images_resized = [F.interpolate(image[None,None,...],
                            size=output_size, 
                            mode="trilinear", 
                            align_corners=False).squeeze()
                for image in images]
    return images_resized

class PairedVolumeDataset(Dataset):
    """Dataset for paired source/target 3D volumes and rigid transforms."""

    def __init__(self, data_dir, data_range,
                    source_mask_file, target_mask_file,
                    source_image_file, target_image_file, transform_file,
                    source_intensity_augmentation, target_intensity_augmentation,
                    output_size=(96, 96, 96),
                    rotation_range=5,translation_range=0,sampling_space="rvs",
                    transforms=False, mode="train",return_corners=False,augment_prob=0.5):
        """Initialize the paired volume dataset.

        Parameters
        ----------
        data_dir : str or pathlib.Path
            Root directory containing one subdirectory per sample.
        data_range : sequence of int
            Start and end indices selecting samples after sorting.
        source_mask_file, target_mask_file : str
            Mask filenames inside each sample directory.
        source_image_file, target_image_file : str
            Image filenames inside each sample directory.
        transform_file : str
            Filename containing the 4-by-4 reference transform.
        source_intensity_augmentation, target_intensity_augmentation : dict
            Keyword arguments passed to ``IntensityAugmenter``.
        output_size : tuple of int, optional
            Spatial size after resizing.
        rotation_range : float, optional
            Maximum spatial augmentation rotation in degrees.
        translation_range : float, optional
            Maximum spatial augmentation translation in voxels.
        sampling_space : str, optional
            Rotation sampling mode.
        transforms : bool, optional
            Whether to apply spatial augmentation.
        mode : str, optional
            Dataset mode, usually ``"train"`` or ``"val"``.
        return_corners : bool, optional
            Whether to return source mask bounding-box corners.
        augment_prob : float, optional
            Probability of applying spatial augmentation.
        """
        self.data_dir = Path(data_dir)
        self.file_names = {
            "source_mask": source_mask_file,
            "target_mask": target_mask_file,
            "source_image": source_image_file,
            "target_image": target_image_file,
            "transform": transform_file,
        }
        self.output_size = tuple(output_size)

        self.sample_list = [item for item
                            in self.data_dir.iterdir() if item.is_dir()]
        self.sample_list = sorted(self.sample_list, key=lambda item: item.name)[data_range[0]:data_range[1]]
        self.transforms = transforms
        self.return_corners = return_corners
        self.mode = mode
        self.rotation_range = rotation_range
        self.translation_range = translation_range
        self.spatial_augmenter =  SpatialAugmenter(rotation_range=self.rotation_range, translation_range=0)
        self.sampling_space=sampling_space
        self.augment_prob = augment_prob
        self.source_intensity_augmenter = IntensityAugmenter(
            **source_intensity_augmentation
        )
        self.target_intensity_augmenter = IntensityAugmenter(
            **target_intensity_augmentation
        )

        if mode =="val" and transforms:
            self.val_rotations = self.init_val_rotation()
            self.spatial_augmenter =  SpatialAugmenter(
                                rotation_range=rotation_range, 
                                    translation_range=translation_range, 
                                    sampling_space=sampling_space,
                                    mode=mode)
    def __len__(self):
        """Return the number of selected samples.

        Returns
        -------
        int
            Dataset length.
        """
        return len(self.sample_list)

    def init_val_rotation(self, device='cpu'):
        """Create or load deterministic validation transforms.

        Parameters
        ----------
        device : str or torch.device, optional
            Device for returned transforms.

        Returns
        -------
        dict
            Mapping from sample index to source/target transforms.
        """
        # Generate file name based on rotation range
        filename = f"paired_val_rotations_range_{self.rotation_range}_trange{self.translation_range}_{self.sampling_space}.npz"
        
        if os.path.exists(filename):
            return load_rotations_from_file(filename, device=device)
        else:
            seed = 42
            torch.manual_seed(seed)
            val_rotations = {}
        
            for i in range(len(self.sample_list)):


                val_rotations_source = get_random_rigid_matrix(image_shape=[102,102,102], 
                                            rotation_range=self.rotation_range,
                                            translation_range=self.translation_range,
                                            use_max_values=False,
                                            sampling_space=self.sampling_space)
                
                val_rotations_target = get_random_rigid_matrix(image_shape=[102,102,102], 
                                            rotation_range=self.rotation_range,
                                            translation_range=self.translation_range,
                                            use_max_values=False,
                                            sampling_space=self.sampling_space)
                
                val_rotations[i] = (val_rotations_source, val_rotations_target)
            save_rotations_to_file(val_rotations, filename)
            return val_rotations

    def _read_image_mask(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.sample_list[idx]
        gt_transform_path = sample / self.file_names["transform"]
        gt_transform = torch.from_numpy(np.loadtxt(gt_transform_path)).to(torch.float32)

        source_mask = read_flip(sample / self.file_names["source_mask"])
        target_mask = read_flip(sample / self.file_names["target_mask"])
        source_image = read_flip(sample / self.file_names["source_image"])
        target_image = read_flip(sample / self.file_names["target_image"])
        

        source_image = min_max_normalize(source_image)
        target_image = min_max_normalize(target_image)
        if self.mode == "train":
            source_image = self.source_intensity_augmenter(source_image)
            target_image = self.target_intensity_augmenter(target_image)
        source_mask = blur_mask_3d(source_mask, kernel_size=7, sigma=2)
        target_mask = blur_mask_3d(target_mask, kernel_size=7, sigma=2)

        return sample, gt_transform, source_mask, target_mask,\
                     source_image, target_image

    def __getitem__(self, idx):
        """Load and preprocess one paired sample.

        Parameters
        ----------
        idx : int
            Sample index.

        Returns
        -------
        tuple
            Sample identifier, packed source/target tensor, reference transform,
            and optionally source bounding-box corners.
        """
        sample_dir, gt_transform,  source_mask_mask, target_mask_mask, source_mask_image, target_mask_image = self._read_image_mask(idx)
        source_images = [source_mask_mask, source_mask_image]
        target_images = [target_mask_mask, target_mask_image]

        source_center = calculate_barycenter(source_images[0].unsqueeze(0).unsqueeze(0)).squeeze()
        target_center = calculate_barycenter(target_images[0].unsqueeze(0).unsqueeze(0)).squeeze()
        
        target_t_center = compute_normalized_translation_to_center(target_center,target_images[0].shape)
        source_t_center = compute_normalized_translation_to_center(source_center,source_images[0].shape)
        source_images,t_pre_source = apply_translation_list(source_images, source_t_center.unsqueeze(0),return_transform=True)
        target_images,t_pre_target = apply_translation_list(target_images, target_t_center.unsqueeze(0), return_transform=True)
        t_pre_source = make_homogeneous(t_pre_source[0])
        t_pre_target = make_homogeneous(t_pre_target[0])

        gt_transform = torch.linalg.inv(t_pre_source)@gt_transform@t_pre_target
        source_images = resize_images(source_images, output_size=self.output_size)
        target_images = resize_images(target_images, output_size=self.output_size)
        new_gt_transform = gt_transform

        if self.transforms == True and torch.rand(1).item() < self.augment_prob:
            source_center = calculate_barycenter(source_images[0].unsqueeze(0).unsqueeze(0)).squeeze()
            target_center = calculate_barycenter(target_images[0].unsqueeze(0).unsqueeze(0)).squeeze()
            if self.mode=="train":
                source_images, source_transform = self.spatial_augmenter.apply_transformation(source_images, return_transform=True)
                target_images, target_transform = self.spatial_augmenter.apply_transformation(target_images, return_transform=True)


                new_gt_transform = torch.linalg.pinv(source_transform)@new_gt_transform.to(torch.float)@target_transform
            
            else: 

                source_images, source_transform = self.spatial_augmenter.apply_transformation(source_images, 
                                    return_transform=True, 
                                    rigid_matrix=self.val_rotations[idx][0])
                target_images, target_transform = self.spatial_augmenter.apply_transformation(target_images, 
                                        return_transform=True, 
                                        rigid_matrix=self.val_rotations[idx][1])

                new_gt_transform = torch.linalg.pinv(source_transform)@new_gt_transform.to(torch.float)@target_transform

        sample4D = torch.zeros((4, *self.output_size))
        sample4D[0, :, :, :] = source_images[0]
        sample4D[1, :, :, :] = target_images[0]
        sample4D[2, :, :, :] = source_images[1]
        sample4D[3, :, :, :] = target_images[1]

        
        if self.return_corners: 
            corners = get_cube_corners(source_images[0])

            return (sample_dir.stem,sample4D,new_gt_transform, corners)
        else:
            return (sample_dir.stem,sample4D,new_gt_transform)



class PairedVolumeDataModule(LightningDataModule):
    """Lightning data module for CrossKey paired-volume training."""

    def __init__(self, cfg):
        """Initialize train and validation datasets from a config object.

        Parameters
        ----------
        cfg : omegaconf.DictConfig
            Data configuration.
        """
        super().__init__()
        self.cfg = cfg

        self.DataSet = PairedVolumeDataset
            
        common = dict(
            data_dir=cfg.data_dir,
            source_mask_file=cfg.files.source_mask,
            target_mask_file=cfg.files.target_mask,
            source_image_file=cfg.files.source_image,
            target_image_file=cfg.files.target_image,
            transform_file=cfg.files.transform,
            output_size=cfg.output_size,
            source_intensity_augmentation=cfg.intensity_augmentation.source,
            target_intensity_augmentation=cfg.intensity_augmentation.target,
            rotation_range=cfg.rotation_range,
            translation_range=cfg.translation_range,
            sampling_space=cfg.sampling_space,
        )
        if cfg.train:
            self.train_dataset = self.DataSet(
                data_range=cfg.train_range,
                transforms=cfg.transforms,
                return_corners=True,
                augment_prob=cfg.augment_prob,
                mode="train",
                **common,
            )

            self.val_dataset = self.DataSet(
                data_range=cfg.val_range,
                transforms=False,
                return_corners=True,
                mode="val",
                **common,
            )
            logger.info(
                f'len of train examples {len(self.train_dataset)}, len of val examples {len(self.val_dataset)}'
            )
    def train_dataloader(self):
        """Create the training dataloader.

        Returns
        -------
        torch.utils.data.DataLoader
            Training dataloader.
        """
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.cfg.train_batch_size,
            shuffle=True,
            num_workers=self.cfg.train_num_workers)

        return train_loader

    def val_dataloader(self):
        """Create the validation dataloader.

        Returns
        -------
        torch.utils.data.DataLoader
            Validation dataloader.
        """
        val_loader = DataLoader(
                self.val_dataset,
                batch_size=self.cfg.val_batch_size,
                shuffle=False,
                num_workers=self.cfg.val_num_workers)
        return val_loader
