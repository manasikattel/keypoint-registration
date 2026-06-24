import csv
import json
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from crosskey.data import PairedVolumeDataset
from crosskey.evaluation.metrics import compute_surface_registration_error, rot_error
from crosskey.models.crosskey import KeyRegModule, split_source_target
from crosskey.training.losses import bbvre_scaled_to_mean
from crosskey.utils.geometry import transform_points


def get_split_range(cfg, split):
    """Return the configured index range for a split.

    Parameters
    ----------
    cfg : omegaconf.DictConfig
        Experiment configuration.
    split : {"train", "val", "test"}
        Dataset split name.

    Returns
    -------
    list
        Start and end indices for the split.
    """
    if split == "train":
        return cfg.data.cfg.train_range
    if split == "val":
        return cfg.data.cfg.val_range
    if split == "test":
        return cfg.data.cfg.test_range
    raise ValueError("prediction.split must be one of: train, val, test")


def get_batch_size(cfg, split):
    """Return the configured batch size for a split.

    Parameters
    ----------
    cfg : omegaconf.DictConfig
        Experiment configuration.
    split : {"train", "val", "test"}
        Dataset split name.

    Returns
    -------
    int
        Batch size.
    """
    if split == "train":
        return cfg.data.cfg.train_batch_size
    if split == "val":
        return cfg.data.cfg.val_batch_size
    if split == "test":
        return cfg.data.cfg.test_batch_size
    raise ValueError("prediction.split must be one of: train, val, test")


def get_num_workers(cfg, split):
    """Return the configured number of dataloader workers for a split.

    Parameters
    ----------
    cfg : omegaconf.DictConfig
        Experiment configuration.
    split : {"train", "val", "test"}
        Dataset split name.

    Returns
    -------
    int
        Number of workers.
    """
    if split == "train":
        return cfg.data.cfg.train_num_workers
    if split == "val":
        return cfg.data.cfg.val_num_workers
    if split == "test":
        return cfg.data.cfg.test_num_workers
    raise ValueError("prediction.split must be one of: train, val, test")


def make_dataset(cfg, split):
    """Create a paired-volume dataset for prediction.

    Parameters
    ----------
    cfg : omegaconf.DictConfig
        Experiment configuration.
    split : {"train", "val", "test"}
        Dataset split name.

    Returns
    -------
    PairedVolumeDataset
        Dataset with spatial augmentation disabled.
    """
    data_cfg = cfg.data.cfg
    return PairedVolumeDataset(
        data_dir=data_cfg.data_dir,
        data_range=get_split_range(cfg, split),
        source_mask_file=data_cfg.files.source_mask,
        target_mask_file=data_cfg.files.target_mask,
        source_image_file=data_cfg.files.source_image,
        target_image_file=data_cfg.files.target_image,
        transform_file=data_cfg.files.transform,
        output_size=data_cfg.output_size,
        source_intensity_augmentation=data_cfg.intensity_augmentation.source,
        target_intensity_augmentation=data_cfg.intensity_augmentation.target,
        rotation_range=data_cfg.rotation_range,
        translation_range=data_cfg.translation_range,
        sampling_space=data_cfg.sampling_space,
        transforms=False,
        mode=split,
        return_corners=True,
        augment_prob=0.0,
    )


def affine_warp(moving, transform):
    """Warp a 3D tensor with a batched affine transform.

    Parameters
    ----------
    moving : torch.Tensor
        Moving image or mask with shape ``[B, C, D, H, W]``.
    transform : torch.Tensor
        Homogeneous transform with shape ``[B, 4, 4]``.

    Returns
    -------
    torch.Tensor
        Warped tensor with the same shape as ``moving``.
    """
    theta = transform[:, :3, :4]
    grid = F.affine_grid(theta, size=moving.shape, align_corners=False)
    return F.grid_sample(moving, grid, align_corners=False)


def soft_dice(pred, target, epsilon=1e-6):
    """Compute per-sample soft Dice scores.

    Parameters
    ----------
    pred : torch.Tensor
        Predicted masks with shape ``[B, D, H, W]``.
    target : torch.Tensor
        Target masks with shape ``[B, D, H, W]``.
    epsilon : float, optional
        Numerical stabilizer.

    Returns
    -------
    torch.Tensor
        Dice score for each batch item.
    """
    pred = pred.reshape(pred.shape[0], -1)
    target = target.reshape(target.shape[0], -1)
    intersection = (pred * target).sum(dim=1)
    denominator = pred.sum(dim=1) + target.sum(dim=1)
    return (2.0 * intersection + epsilon) / (denominator + epsilon)


def translation_error_voxels(pred_mat, gt_mat, volume_shape):
    """Compute translation error in voxel units.

    Parameters
    ----------
    pred_mat : torch.Tensor
        Predicted transforms with shape ``[B, 4, 4]``.
    gt_mat : torch.Tensor
        Ground-truth transforms with shape ``[B, 4, 4]``.
    volume_shape : tuple
        Shape tuple ``(B, D, H, W)``.

    Returns
    -------
    torch.Tensor
        Translation error for each batch item.
    """
    _, depth, height, width = volume_shape
    scale = torch.tensor([depth - 1, height - 1, width - 1], device=pred_mat.device) / 2.0
    return torch.linalg.vector_norm((pred_mat[:, :3, 3] - gt_mat[:, :3, 3]) * scale, dim=1)


def summarize(rows, metric_names):
    """Summarize per-case metrics.

    Parameters
    ----------
    rows : list of dict
        Per-case metric rows.
    metric_names : sequence of str
        Metric fields to summarize.

    Returns
    -------
    dict
        Mean, standard deviation, median, and count for each metric.
    """
    summary = {}
    for metric in metric_names:
        values = np.asarray(
            [row[metric] for row in rows if row.get(metric) is not None],
            dtype=np.float64,
        )
        if values.size == 0:
            continue
        summary[metric] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "median": float(np.median(values)),
            "count": int(values.size),
        }
    return summary


def write_outputs(output_dir, rows, summary, cfg):
    """Write prediction and summary files.

    Parameters
    ----------
    output_dir : pathlib.Path
        Directory where output files are written.
    rows : list of dict
        Per-case prediction rows.
    summary : dict
        Summary metric dictionary.
    cfg : omegaconf.DictConfig
        Resolved experiment configuration.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    metric_names = list(summary.keys())
    fieldnames = [
        "sample_id",
        "rotation_deg",
        "translation_mm",
        "bbvre_mm",
        "sre_mm",
        "mask_dice",
        "keypoint_distance_mm",
        "pred_transform",
    ]
    with (output_dir / "predictions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            csv_row = dict(row)
            csv_row["pred_transform"] = json.dumps(csv_row["pred_transform"])
            writer.writerow(csv_row)

    with (output_dir / "metrics_summary.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "mean", "std", "median", "count"])
        for metric in metric_names:
            values = summary[metric]
            writer.writerow(
                [metric, values["mean"], values["std"], values["median"], values["count"]]
            )

    payload = {
        "config": OmegaConf.to_container(cfg, resolve=True),
        "num_cases": len(rows),
        "summary": summary,
    }
    with (output_dir / "metrics_summary.json").open("w") as handle:
        json.dump(payload, handle, indent=2)


@hydra.main(version_base="1.3", config_path=".", config_name="config")
def main(cfg: DictConfig):
    """Run checkpoint prediction and score registration metrics.

    Parameters
    ----------
    cfg : omegaconf.DictConfig
        Hydra configuration.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split = cfg.prediction.split
    voxel_spacing_mm = float(cfg.prediction.voxel_spacing_mm)
    if cfg.prediction.checkpoint is None:
        raise ValueError("Set prediction.checkpoint=/path/to/checkpoint.ckpt")

    dataset = make_dataset(cfg, split)
    loader = DataLoader(
        dataset,
        batch_size=get_batch_size(cfg, split),
        shuffle=False,
        num_workers=get_num_workers(cfg, split),
    )

    model = KeyRegModule.load_from_checkpoint(
        cfg.prediction.checkpoint,
        map_location=device,
    ).to(device)
    model.eval()

    rows = []
    with torch.no_grad():
        for batch in loader:
            sample_ids = list(batch[0])
            labels = batch[1].to(device)
            gt_mat = batch[2].to(device)
            corners = batch[3].to(device)
            source, target = split_source_target(labels)
            source_mask = labels[:, 0]
            target_mask = labels[:, 1]

            pred_mat, _, _, source_keypoints, target_keypoints, _, _ = model(source, target)

            rotation = rot_error(pred_mat[:, :3, :3], gt_mat[:, :3, :3])
            translation = translation_error_voxels(pred_mat, gt_mat, source_mask.shape)
            bbvre = bbvre_scaled_to_mean(corners, pred_mat, gt_mat, source_mask.shape)
            sre = compute_surface_registration_error(source_mask, pred_mat, gt_mat)
            warped_source = affine_warp(source_mask.unsqueeze(1), pred_mat).squeeze(1)
            dice = soft_dice((warped_source > 0.3).float(), (target_mask > 0.3).float())
            source_points_gt = transform_points(
                source_keypoints,
                torch.linalg.inv(gt_mat),
                source_mask.shape,
            )
            keypoint_distance = torch.linalg.vector_norm(
                target_keypoints - source_points_gt, dim=-1
            ).mean(dim=1)

            for index, sample_id in enumerate(sample_ids):
                row = {
                    "sample_id": sample_id,
                    "rotation_deg": float(rotation[index].detach().cpu()),
                    "translation_mm": float(translation[index].detach().cpu()) * voxel_spacing_mm,
                    "bbvre_mm": float(bbvre[index].detach().cpu()) * voxel_spacing_mm,
                    "sre_mm": float(sre[index].detach().cpu()) * voxel_spacing_mm,
                    "mask_dice": float(dice[index].detach().cpu()),
                    "keypoint_distance_mm": float(keypoint_distance[index].detach().cpu())
                    * voxel_spacing_mm,
                    "pred_transform": pred_mat[index].detach().cpu().flatten().tolist(),
                }
                rows.append(row)

    metric_names = [
        "rotation_deg",
        "translation_mm",
        "bbvre_mm",
        "sre_mm",
        "mask_dice",
        "keypoint_distance_mm",
    ]
    summary = summarize(rows, metric_names)
    write_outputs(Path(cfg.prediction.output_dir), rows, summary, cfg)


if __name__ == "__main__":
    main()
