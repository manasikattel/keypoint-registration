# CrossKey: Keypoint-Based 3D Registration

CrossKey trains an unsupervised keypoint-based model for registering MRI TRUS 3D volumes. It learns keypoints from image and mask inputs, estimates a rigid transform from those keypoints, and reports registration quality on held-out data.

The default setup follows the main CrossKey model. Dataset paths, file names, split ranges, and model choices live in `config.yaml`.

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

Each sample should be a directory with source and target images, masks, and a 4-by-4 voxel-space transform. The default filenames are placeholders and can be changed in `config.yaml`.

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

Metrics are written as CSV files, and checkpoints are saved under `outputs/`.

The default logger is Lightning's `CSVLogger`, so training works without any external service. To use Weights & Biases instead, install `wandb` and override the Hydra logger:

```bash
pip install wandb
python train.py data.cfg.data_dir=/path/to/data \
  logger._target_=lightning.pytorch.loggers.WandbLogger \
  +logger.project=crosskey \
  logger.name=my-run
```

The same pattern works for other PyTorch Lightning loggers by changing `logger._target_` and adding the arguments expected by that logger.

Prediction and scoring:

```bash
python predict.py \
  data.cfg.data_dir=/path/to/data \
  prediction.checkpoint=/path/to/checkpoint.ckpt \
  prediction.split=test \
  prediction.voxel_spacing_mm=1.0
```

This writes `predictions.csv`, `metrics_summary.csv`, and `metrics_summary.json` with the predicted transforms and the main registration metrics. Optional landmark TRE is included when landmark filenames are provided in `data.cfg.files`.
