# CrossKey: Keypoint-Based 3D Registration

This repository contains the training and scoring code for an unsupervised keypoint-based registration model on paired 3D volumes. The default configuration matches the proposed method: a dual-branch SwinUNETR encoder with interpolation-convolution decoder, mask+image input, probabilistic keypoint heatmaps, rigid transform estimation from learned keypoints, and BBVRE/keypoint/regularization losses. Baseline U-Net and equivariant U-Net backbones are also included. Dataset paths, file names, and split ranges are configured in `config.yaml`.
Code layout:

```text
train.py                         # training entry point
predict.py                       # checkpoint scoring/evaluation
config.yaml                      # default CrossKey configuration
crosskey/
  data/                          # dataset, preprocessing, augmentations
  models/                        # CrossKey module and backbone networks
  training/                      # losses used for optimization
  evaluation/                    # metrics used during validation/prediction
  utils/                         # geometry and rotation utilities
```

Each sample is a directory containing two images, two masks, and a 4-by-4 voxel-space transform. The default names are placeholders and can be changed in the config.

```text
data_root/
  sample_0001/
    source_image.nii.gz
    target_image.nii.gz
    source_mask.nii.gz
    target_mask.nii.gz
    transform.txt
```

Install and train:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train.py data.cfg.data_dir=/path/to/data
```

Metrics are written locally as CSV files and checkpoints go under `outputs/`

Logging can be changed through the Hydra `logger` section. The default logger is
Lightning's `CSVLogger`, which keeps the repository self-contained. To use
Weights & Biases instead, install `wandb` and override the logger:

```bash
pip install wandb
python train.py data.cfg.data_dir=/path/to/data \
  logger._target_=lightning.pytorch.loggers.WandbLogger \
  +logger.project=crosskey \
  logger.name=my-run
```

The same pattern can be used for any other PyTorch Lightning logger by changing
`logger._target_` and adding the arguments expected by that logger.

Prediction and scoring:

```bash
python predict.py \
  data.cfg.data_dir=/path/to/data \
  prediction.checkpoint=/path/to/checkpoint.ckpt \
  prediction.split=test \
  prediction.voxel_spacing_mm=1.0
```

This writes `predictions.csv`, `metrics_summary.csv`, and `metrics_summary.json` with rotation error, translation error, BBVRE, source/target/symmetric SRE, mask Dice, keypoint consistency distance, predicted transforms, and optional landmark TRE if landmark filenames are provided in `data.cfg.files`.
