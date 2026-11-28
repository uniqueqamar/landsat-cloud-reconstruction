# Landsat cloud reconstruction

This project fills cloud and shadow areas in small Landsat satellite images. It uses a quality mask to find the covered pixels and keeps the visible pixels unchanged. The filled areas are estimates of what might be underneath. Start with the included 128 × 128 pixel example.

## Map flooding through clouds with radar

For disaster monitoring, this project now compares two Sentinel-1 radar
acquisitions and maps measured darkening. The Porto Alegre example uses a radar
pass during the May 2024 flood and compares the result against a published
flood extent. The outputs include a dated change map, a stricter open-water
candidate layer, and explicit unknown pixels for gaps in radar coverage.
[See the example and run instructions](runs/radar_porto_alegre/README.md).

## Try it in your browser

[Open Landsat Cloud Lab](https://landsat-cloud-lab-qam.snowywren64.chatgpt.site) to run the V2 model on an included example. Turn on the mask overlay to see which pixels the model filled. The hosted demo requires owner access. If you can’t open it, use the Mac steps below.

## Run it on your Mac

These steps use a pretrained model, so you can try one patch at a time on your 8 GB M2 without training anything.

**1. Open the project in Terminal.** If you already have a Git checkout, open Terminal in that folder and run `git pull`. For a fresh checkout, run:

```bash
git clone https://github.com/uniqueqamar/landsat-cloud-reconstruction.git
cd landsat-cloud-reconstruction
```

**2. Install the dependencies once.** Use Python 3.11 or 3.12. Check with `python3 --version`.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

In a new Terminal session, return to the project folder and run `source .venv/bin/activate` again.

**3. Reconstruct the included cloudy patch with V2.**

```bash
python predict.py \
  --weights runs/context_residual_v2/best.pt \
  --input data/cloudy/val/r04608_c02176.tif \
  --output outputs/v2_prediction.tif

open outputs/v2_prediction.png
```

The PNG opens in Preview. The same run saves a four-band GeoTIFF for GIS and a `v2_prediction.mask.npy` file showing which pixels were filled. To try another included example, change `--input` to another TIFF in `data/cloudy/val/` and give it a different output name. Inputs must be prepared Landsat TIFFs with blue, green, red, near-infrared, QA_PIXEL and QA_RADSAT bands.

## See the generative model and V2 side by side

```bash
python compare_models.py
open outputs/v2_comparison.png
```

The image shows the same cloudy input, the original **conditional GAN** (the generative AI model), **V2**, and a clearer satellite image taken eight days later.

![Cloudy Landsat input, original GAN estimate, V2 estimate, and later reference](runs/context_residual_v2/comparison.png)

V2 starts by filling the cloudy area with average colors from visible pixels, then applies learned corrections. It scored better than the GAN and simple mean fill on nine validation patches from one scene. Lower masked MAE means the filled pixels were closer to the later image on average.

| Method | Masked MAE ↓ | Masked PSNR ↑ |
| --- | ---: | ---: |
| Original conditional GAN | 0.02170 | 29.06 dB |
| Simple mean fill | 0.01754 | 30.69 dB |
| V2 | **0.01667** | **31.12 dB** |

To save a separate GAN reconstruction, repeat step 3 with `--weights runs/temporal_cgan/best.pt` and `--output outputs/gan_prediction.tif`. Then open `outputs/gan_prediction.png`.

## Try the SRGAN-style model

This version uses residual blocks and subpixel upsampling, inspired by
[SRGAN](https://doi.org/10.1109/CVPR.2017.19). It upsamples internal features
back to the original 30 m Landsat grid and preserves observed pixels.
Training uses masked four-band L1 loss and an image-only adversarial loss.
The input and output use the same spatial resolution.

```bash
python predict.py \
  --weights runs/temporal_srgan/best.pt \
  --input data/cloudy/val/r04608_c02176.tif \
  --output outputs/srgan_prediction.tif
open outputs/srgan_prediction.png
```

On the same ten Landsat test patches, using the same cloud and shadow masks
and the same eight-day-later reference, the scores were:

| Method | Masked MAE ↓ | PSNR ↑ | Spectral angle ↓ | NDVI MAE ↓ |
| --- | ---: | ---: | ---: | ---: |
| Original conditional GAN | 0.02380 | 28.40 dB | 6.09° | 0.10877 |
| Mean fill | 0.01824 | 30.53 dB | 4.18° | 0.08115 |
| SRGAN-style | 0.01687 | 30.97 dB | 4.03° | 0.08123 |
| V2 | **0.01666** | **31.04 dB** | **3.99°** | **0.08059** |

SRGAN-style training used the original cGAN's 22 training pairs, with batch
size 1 and two reconstruction-only warm-up epochs. Epoch 9 had the lowest
masked MAE on the nine validation patches among adversarial epochs. V2 used
118 training pairs, so the comparison with V2 also reflects different
training set sizes. The SRGAN-style run used Linux CPU. Mac runtime and memory
were not measured.

[Validation preview](runs/temporal_srgan/comparison.png) ·
[Test scores by coverage](runs/temporal_srgan/test_metrics.json) ·
[Training history](runs/temporal_srgan/history.json)

To retrain and score the same split:

```bash
python train.py --task temporal --architecture srgan --epochs 10 \
  --gan-weight 0.005 --warmup 2 --output outputs/srgan_run
python spectral_audit.py --weights outputs/srgan_run/best.pt --split test \
  --coverage-breakdown --output outputs/srgan_test_metrics.json
```

## SRGAN training by cloud type

We retrained two SRGAN variants on the same 118 Landsat pairs with the same
augmentations, seed, and six-epoch schedule. The cloud-balanced run gives
extra weight to underrepresented QA groups by averaging the pixel-weighted
reconstruction loss with the mean loss across the types present in each batch.
Both checkpoints were selected by overall validation MAE.

| Test group | Pixels | Original SRGAN MAE ↓ | Expanded control MAE ↓ | Cloud-balanced MAE ↓ |
| --- | ---: | ---: | ---: | ---: |
| All masked pixels | 60,269 | 0.016874 | 0.016685 | 0.016689 |
| Cloud | 23,076 | 0.018132 | 0.017835 | 0.017832 |
| Cloud shadow | 26,253 | 0.015944 | 0.015876 | 0.015890 |
| Dilated cloud buffer only | 10,940 | 0.016454 | 0.016200 | 0.016195 |

When QA flags overlap, cloud takes priority, followed by cirrus, shadow, and
the dilated cloud buffer. Cirrus had no separately labeled pixels in these
splits. The buffer represents proximity to a cloud and is not a distinct
physical cloud type.

Both expanded runs cut masked MAE by about 1.1% versus the original 22-pair
SRGAN run. Cloud balancing made almost no difference versus the matched
expanded control. It reduced MAE slightly on cloud and buffer pixels and
increased it slightly on shadows. The result covers ten patches from one
scene against a later reference.

To run the expanded control, use `runs/srgan_cloud_types/control.pt` in the
prediction command above. Use `runs/srgan_cloud_types/balanced.pt` for the
cloud-balanced variant. Both use the existing input format.

```bash
python train.py --task temporal --architecture srgan --expanded-train --augment \
  --epochs 6 --gan-weight 0.005 --warmup 2 --cloud-balance --output outputs/cloud_srgan
python spectral_audit.py --weights outputs/cloud_srgan/best.pt --split test \
  --coverage-breakdown --cloud-breakdown --output outputs/cloud_srgan_test.json
```

Omit `--cloud-balance` to train the matched control.
[Weights, full scores, and preview](runs/srgan_cloud_types).

## Reduce visible patch edges

Add `--blend-edges` to a prediction command to soften the transition.
Near the cloud boundary, the prediction is blended with reflectance
interpolated from visible pixels. The blend fades over 16 pixels measured
inward from the boundary, with at most 50% interpolation weight. Every
observed clear pixel is copied exactly. The filter uses the cloudy input,
its mask, and the saved model prediction, so it needs no new weights.

```bash
python predict.py --weights runs/srgan_cloud_types/control.pt \
  --input data/cloudy/val/r04608_c02176.tif \
  --output outputs/srgan_blended.tif --blend-edges
open outputs/srgan_blended.png
```

We selected the setting with the lowest aggregate reflectance MAE for SRGAN
among eight blends on the nine validation patches. We then tested the same
fixed setting on ten Landsat patches, using the same reference grids and masks.
Boundary error measures how much the reflectance change across the mask edge
differs from the change on that same edge in the later clear reference.

| Expanded SRGAN | Boundary-gradient MAE ↓ | Masked MAE ↓ | Masked NDVI MAE ↓ |
| --- | ---: | ---: | ---: |
| Plain prediction | 0.02174 | 0.01668 | 0.08066 |
| With edge blend | 0.01488 | 0.01647 | 0.07673 |

The reference was taken eight days later and all test patches come from one
scene. The blend softens transitions using available context. Large cloud
interiors still lack observed ground detail. The reference is used to score
the outputs and the filter can run on the Mac without a reference image.

`python evaluate_edge_blend.py --output outputs/edge_audit` reproduces the
validation sweep and paired test scores on CPU.
[Before and after preview](runs/edge_blend/comparison.png) ·
[Full test scores](runs/edge_blend/test.json)

## Does the model recover hidden ground detail?

We evaluated 27,425 masked interior pixels across the ten Landsat test patches.
Each scored pixel has a fully masked 9 × 9 neighborhood. We ran two checks:
the real cloudy input against the eight-day-later reference, and a controlled
test where we applied the same cloud mask to the clear reference itself.
In the controlled test, the model only sees observed values outside the mask.
The hidden values from that same image provide the answer for scoring.

| Comparison | Mean fill MAE ↓ | Spatial fill MAE ↓ | SRGAN with edge blend MAE ↓ |
| --- | ---: | ---: | ---: |
| Real cloud interior vs. later reference | 0.01867 | 0.01878 | **0.01698** |
| Known ground hidden in the clear image | 0.01719 | **0.01634** | 0.01756 |

The model produced gradients with only about 3% of the reference RMS amplitude
in the controlled test. Its signed gradient similarity was 0.004 on a scale where 1
means perfect alignment and 0 means no alignment. This shows that the smooth
interiors lack the reference's texture, even when average reflectance is close.
Treat masked output values as estimates and use the exported `.mask.npy` file
to distinguish them from observed ground. The two checks use the same test
locations, so the controlled test does not add independent sites.

[Controlled-test preview](runs/ground_detail/comparison.png) ·
[Full interior scores](runs/ground_detail/scores.json)

Reproduce the CPU audit with
`python evaluate_ground_detail.py --output outputs/ground_detail`.

## Why the filled areas can look wrong

Thick clouds hide the optical signal from the ground. These models see the surrounding clear pixels and have to guess what continues under the cloud. V2 often produces smooth patches, and the GAN can add colored edges or plausible-looking details in the wrong place. Small features are also hard to see at Landsat’s 30 m pixel size. The clearer reference was taken eight days later, so some differences may come from real changes on the ground.

For better reconstruction, the next model would benefit from another observation of the same location, such as a nearby clear optical image or aligned Sentinel-1 radar. Radar can observe the ground through clouds and give the model additional evidence about hidden structures. The [M3R-CR comparison guide](M3R_COMPARISON.md) includes two small pretrained models you can run on the public M3R-CR test example. A larger comparison needs additional held-out patches.

## More details

- [Training commands, data preparation, and source details](https://github.com/uniqueqamar/landsat-cloud-reconstruction/blob/a37193ffbeea52b45f6dd60043e0f2355c56592d/README.md)
- [V2 validation scores by cloud coverage](runs/context_residual_v2/validation_spectral_coverage.json)
- [Browser demo setup](web/README.md)
