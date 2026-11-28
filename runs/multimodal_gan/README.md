# SAR-guided GAN for cloud reconstruction

The generator predicts four optical reflectance bands inside a cloud mask.
It learns optical and coarse radar features in separate streams, fuses them
on the optical grid, and predicts a correction to the visible-pixel mean.
It uses masked L1, an interior gradient loss, and a conditional PatchGAN
loss after two reconstruction-only warm-up epochs. The optical-only GAN uses
the same architecture, seed, crop schedule, and loss with the radar inputs
set to zero. Clear output pixels are copied from the cloudy input.

The design is inspired by Xu et al.'s multiresolution SAR-optical fusion
problem. It uses geospatial SAR reprojection and a lightweight fusion
network. It does not implement Align-CR's learned alignment architecture.
[Multimodal and Multiresolution Data Fusion for High-Resolution Cloud Removal](https://arxiv.org/abs/2301.03432)

## What the percentage means

For this pilot, a masked pixel is counted as within tolerance if **each of the
four predicted reflectance bands differs from the clear reference by at most
0.02** on the normalized 0–1 scale. The denominator contains only masked
pixels with valid reference values. Copied clear pixels contribute nothing
to this percentage. This is a pilot criterion and does not replace MAE,
PSNR, spectral angle, or NDVI error. The audit also measures the alignment
of reflectance gradients within the masked interior.

## Data and split

- **M3R-CR.** One unique public training patch and one test patch from a
  separate site. Sentinel-1 is reprojected to the 3 m PlanetScope grid.
  The train and test sites stay separate. With no public validation patch,
  the checkpoint is the last epoch of the fixed training schedule.
  [Public examples](https://github.com/zhu-xlab/M3R-CR/tree/372d34bbae6ad5703ad6ee5b050699be618c13e3)
- **SEN12MS-CR.** A bounded subset from the published train, validation,
  and test scene lists in the Parquet mirror. The [manifest](sen12_manifest.json)
  lists the exact downloaded row groups and every retained patch. Cloud masks
  come from s2cloudless applied to cloudy Sentinel-2 imagery only, using the
  upstream loader's settings. The subset keeps valid, partially cloudy
  patches with 2–90% detected cloud cover. The pilot scores B2, B3, B4, and
  B8, so it does not reproduce the full 13-band benchmark.
  [Dataset](https://patricktum.github.io/cloud_removal/sen12mscr/) ·
  [Pinned Parquet mirror](https://huggingface.co/datasets/Hermanni/sen12mscr/tree/67cc2d43724a15167c050dd3060ede0e52d71e55) ·
  [Upstream mask settings](https://github.com/PatrickTUM/SEN12MS-CR-TS/blob/2af933422f58fdfd6fc826ba3a827b288aad36fb/data/dataLoader.py#L427)

## Held-out results

The **97% target was not reached** under the fixed four-band tolerance.
SEN12MS-CR used 59 training, 28 validation, and 20 test patches. Two
training scenes contributed usable patches. One validation scene selected
the checkpoint by masked MAE, and two different scenes supplied the test
patches. Both SEN12MS-CR GAN checkpoints came from epoch 8 after 354
adversarial updates. The M3R-CR GANs used 120 adversarial updates and were
scored on the single public test patch.

| Dataset | Method | Masked MAE ↓ | PSNR (dB) ↑ | Angle (°) ↓ | NDVI MAE ↓ | All 4 bands within 0.02 ↑ |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| M3R-CR | Mean fill | 0.04306 | 26.89 | 7.94 | 0.08438 | 0.000% |
| M3R-CR | Optical GAN | 0.03782 | 27.55 | 16.13 | 0.20218 | 0.019% |
| M3R-CR | SAR GAN | 0.03775 | 27.56 | 16.06 | 0.20109 | 0.015% |
| SEN12MS-CR | Mean fill | 0.01864 | 31.25 | 5.89 | 0.11410 | 18.62% |
| SEN12MS-CR | Optical GAN | 0.02398 | 28.72 | 9.34 | 0.13980 | 6.63% |
| SEN12MS-CR | SAR GAN | 0.02345 | 28.86 | 8.93 | 0.13837 | 7.82% |

Radar helped the GAN relative to its optical counterpart on SEN12MS-CR,
but mean fill had lower overall error. A validation sweep of five weights
between pure GAN and either mean fill or spatial interpolation selected
the unblended GAN. The result did not transfer to the test scenes, where
mean fill performed better. The SAR GAN also showed weak alignment with
the reference gradients in the masked interior. These reconstructions are
estimates. For disaster mapping, the project provides a separate workflow
based on radar measured during the event.

[M3R-CR preview](m3r_sar/comparison.png) ·
[SEN12MS-CR preview](sen12_sar/comparison.png) ·
[SEN12MS-CR scores including both test scenes](sen12_sar/scores.json) ·
[Disaster radar workflow](../radar_porto_alegre/README.md)

## Run the public M3R-CR example

From the repo root, with the main README's Python environment activated:

```bash
git clone https://github.com/zhu-xlab/M3R-CR.git ../M3R-CR
git -C ../M3R-CR checkout --detach 372d34bbae6ad5703ad6ee5b050699be618c13e3

python multimodal_gan.py evaluate --dataset m3r --data ../M3R-CR/M3M-CR \
  --weights runs/multimodal_gan/m3r_sar/best.pt --output outputs/m3r_gan
open outputs/m3r_gan/comparison.png
```

For inference on your own patch, use `predict` in place of `evaluate` and
supply `--cloudy` (a georeferenced B/G/R/NIR DN GeoTIFF), `--sar` (VV and VH
in dB), and `--mask` (a binary NPY array on the optical grid) in place of
`--data`. This mode writes reflectance values and a mask of estimated pixels.
The predictor does not read a clear reference.

## Reproduce SEN12MS-CR

Preprocessing needs three additional packages. It downloads approximately
690 MB of bounded Parquet ranges over HTTPS, rather than the full dataset.
Training uses 96 × 96 crops, two CPU threads, and one sample at a time.
Training and evaluation were run on Linux CPU. Mac runtime and peak memory
remain unmeasured.

```bash
python -m pip install pyarrow requests 's2cloudless==1.7.3'
python examples/prepare_sen12_subset.py --parquet-cache outputs/sen12_parquet \
  --output outputs/sen12_subset

python multimodal_gan.py train --dataset sen12 --data outputs/sen12_subset \
  --epochs 8 --output outputs/sen12_gan_train

python multimodal_gan.py evaluate --dataset sen12 --data outputs/sen12_subset \
  --weights runs/multimodal_gan/sen12_sar/best.pt --output outputs/sen12_gan
open outputs/sen12_gan/comparison.png
```

Add `--optical-only` when training to reproduce the matched optical model.
For M3R-CR training, replace `--dataset sen12 --data outputs/sen12_subset`
with `--dataset m3r --data ../M3R-CR/M3M-CR` and use the default six epochs.
