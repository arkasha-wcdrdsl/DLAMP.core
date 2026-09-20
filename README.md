# DLAMP.core
DLAMP.core: a Swin-Transformer-based deep learning weather prediction model for regional (limited-area) mesoscale forecasting, developed for high-resolution atmospheric prediction over Taiwan.

## Environment

The reference GPU environment is defined in [`conda_py11_min_cuda.yaml`](conda_py11_min_cuda.yaml).
It creates the `dlamp` environment with Python 3.11 and a CUDA-enabled PyTorch stack.
The YAML is the dependency baseline, but `torch`, `torchvision`, `torchaudio`, and `lightning`
are not version-pinned, so a clean install does not resolve to one reproducible package set.

```bash
conda env create -f conda_py11_min_cuda.yaml
conda activate dlamp
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

The training entry point is [`train.py`](train.py). It requires a CUDA-capable GPU.

## Data and assets

The current code expects RWRF data at the path defined by `DATA_PATH` in [`src/const.py`](src/const.py).

The active 224×224 training path also requires the standardization file, land/sea mask, topography mask,
blacklist, and the corresponding RWRF files under [`assets/`](assets/).

## Supported losses

The public training contract contains exactly these three losses:

| Public name | `lightning.loss` value | Meaning |
|---|---|---|
| MAE loss | `L1` | Mean absolute error. |
| CRPS-like loss | `MAECRPS` | Current implementation combines an MAE term with the CRPS term. |
| Spectral loss | `SPECTRUM` | Spatial spectrum-aware loss combined with MAE terms. |

For the current Pangu configuration, the CRPS-like loss and Spectral loss expect the active 224×224 data layout with
13 upper-air levels and the configured surface channels.

## Training

Edit [`config/finetune_multistep_pangu.yaml`](config/finetune_multistep_pangu.yaml) before each run.
This is the default configuration loaded by `train.py`; no command-line overrides are required.

For from-scratch training, keep both checkpoint settings as `null`:

```yaml
lightning:
  loss: L1
  finetune_with_checkpoint: null
  resume_from_checkpoint: null
```

The committed configuration is a single-step run:

```yaml
lightning:
  sequence_mode: false
  multi_step_forecast: 1
  output_itv:
    hours: 1
  model_native_interval:
    hours: 1
```

Choose one of the three supported losses by changing `lightning.loss` in the same file:

```yaml
loss: L1        # MAE loss
# loss: MAECRPS # CRPS-like loss
# loss: SPECTRUM # Spectral loss
```

After saving the configuration, run:

```bash
conda run --live-stream -n dlamp python train.py
```

`sampling_rate`, data dates, batch size, workers, and checkpoint settings should be changed directly in
`config/finetune_multistep_pangu.yaml` according to the experiment.


## License

DLAMP was developed by XXX.
The source code and trained model parameters in this repository are made available under the terms of the CC BY-NC-SA 4.0 license. You can find details [here](https://creativecommons.org/licenses/by-nc-sa/4.0/).

**The commercial use of this code and these models is forbidden.**

Please also note that the model was trained using RWRF dataset, provided by CWA. Please follow their data usage policy accordingly.
