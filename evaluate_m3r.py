"""Score paired prediction GeoTIFFs on a small, frozen M3R-CR test list."""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import rasterio


def grid(image):
    return image.crs, image.transform, image.height, image.width


def read_optical(path, expected_grid=None):
    with rasterio.open(path) as image:
        if image.crs is None or image.count != 4 or max(image.shape) > 512:
            raise ValueError(f"{path}: expected a georeferenced four-band patch up to 512×512")
        actual_grid = grid(image)
        if expected_grid is not None and actual_grid != expected_grid:
            raise ValueError(f"{path}: CRS, affine transform or shape differs from the reference")
        bands = image.read(masked=True, out_dtype="float32").filled(np.nan)
    return bands, actual_grid


def score(prediction, reference, selected):
    pred = prediction[:, selected].T.astype(np.float64)
    target = reference[:, selected].T.astype(np.float64)
    if not np.isfinite(pred).all() or np.any((pred < 0) | (pred > 1)):
        raise ValueError("Prediction contains missing pixels or values outside [0, 1] on the scoring mask")
    error = pred - target
    pred_norm = np.linalg.norm(pred, axis=1)
    target_norm = np.linalg.norm(target, axis=1)
    product = pred_norm * target_norm
    cosine = np.divide(
        np.sum(pred * target, axis=1), product,
        out=np.zeros_like(product), where=product > 0,
    )
    cosine[(pred_norm == 0) & (target_norm == 0)] = 1
    angle = np.degrees(np.arccos(np.clip(cosine, -1, 1)))

    def ndvi(values):
        red, nir = values[:, 2], values[:, 3]
        denom = nir + red
        return np.divide(nir - red, denom, out=np.zeros_like(denom), where=denom > 0)

    return {
        "absolute_error": float(np.abs(error).sum()),
        "squared_error": float(np.square(error).sum()),
        "angle_deg": float(angle.sum()),
        "ndvi_error": float(np.abs(ndvi(pred) - ndvi(target)).sum()),
    }


def summarize(rows, methods):
    pixels = sum(row["scored_pixels"] for row in rows)
    summary = {"patches": len(rows), "scored_pixels": pixels}
    for method in methods:
        if not pixels:
            summary[method] = None
            continue
        totals = {
            key: sum(row["sums"][method][key] for row in rows)
            for key in rows[0]["sums"][method]
        }
        mse = totals["squared_error"] / (4 * pixels)
        summary[method] = {
            "masked_mae": totals["absolute_error"] / (4 * pixels),
            "masked_psnr_db": -10 * math.log10(mse) if mse else "inf",
            "masked_sam_deg": totals["angle_deg"] / pixels,
            "masked_ndvi_mae": totals["ndvi_error"] / pixels,
        }
    return summary


def evaluate(args):
    with args.test_list.open(newline="", encoding="utf-8-sig") as source:
        entries = list(csv.reader(source))
    if not 1 <= len(entries) <= 12 or any(len(entry) != 5 for entry in entries):
        raise ValueError("Expected a frozen subset of 1–12 rows in the upstream five-column CSV format")
    ids = [Path(entry[0]).stem for entry in entries]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate reference patch IDs in test list")
    methods = {"mean_fill_smoke": None} if args.smoke_mean_fill else {
        "optical_only": args.optical_predictions, "sar_guided": args.sar_predictions,
    }
    groups = {"(0,30%]": [], "(30%,50%]": [], "(50%,100%)": []}
    rows, skipped = [], []
    for patch_id, entry in zip(ids, entries):
        reference_path, cloudy_path, sar_path, _, mask_path = [
            args.data_root / name for name in entry
        ]
        reference, reference_grid = read_optical(reference_path)
        cloudy, _ = read_optical(cloudy_path, reference_grid)
        with rasterio.open(sar_path) as sar:
            if sar.crs is None or sar.count != 2:
                raise ValueError(f"{sar_path}: expected a georeferenced two-band Sentinel-1 patch")
        mask = np.load(mask_path, allow_pickle=False)
        if mask.shape != reference.shape[1:] or not np.isin(mask, (0, 1)).all():
            raise ValueError(f"{mask_path}: expected an upstream binary mask on the optical grid")
        mask = mask.astype(bool)
        valid_ref = np.isfinite(reference).all(axis=0)
        selected = mask & valid_ref
        visible = ~mask & np.isfinite(cloudy).all(axis=0)
        if not selected.any() or not visible.any():
            skipped.append({
                "patch_id": patch_id,
                "reason": "no valid masked reference pixels" if not selected.any() else "no valid visible input pixels",
            })
            continue
        coverage = mask.mean()
        band = "(0,30%]" if coverage <= .3 else "(30%,50%]" if coverage <= .5 else "(50%,100%)"
        reference = np.clip(reference, 0, 10000) / 10000
        row = {
            "patch_id": patch_id, "cloud_fraction": coverage,
            "scored_pixels": int(selected.sum()), "sums": {},
        }
        for method, directory in methods.items():
            if directory is None:
                clear = np.clip(cloudy[:, visible], 0, 10000) / 10000
                prediction = np.broadcast_to(clear.mean(axis=1)[:, None, None], reference.shape)
            else:
                prediction, _ = read_optical(directory / f"{patch_id}.tif", reference_grid)
            row["sums"][method] = score(prediction, reference, selected)
        rows.append(row)
        groups[band].append(row)
    return {
        "mode": "mean_fill_smoke" if args.smoke_mean_fill else "paired_model_comparison",
        "test_list": str(args.test_list),
        "overall": summarize(rows, methods),
        "by_cloud_coverage": {key: summarize(group, methods) for key, group in groups.items()},
        "patches": [
            {"patch_id": row["patch_id"], "cloud_fraction": row["cloud_fraction"],
             **summarize([row], methods)}
            for row in rows
        ],
        "skipped": skipped,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--test-list", type=Path, required=True)
    parser.add_argument("--optical-predictions", type=Path)
    parser.add_argument("--sar-predictions", type=Path)
    parser.add_argument("--smoke-mean-fill", action="store_true")
    args = parser.parse_args()
    if args.smoke_mean_fill:
        if args.optical_predictions is not None or args.sar_predictions is not None:
            parser.error("The smoke test takes no model predictions")
    elif args.optical_predictions is None or args.sar_predictions is None:
        parser.error("Both prediction directories are required for paired scoring")
    elif args.optical_predictions.resolve() == args.sar_predictions.resolve():
        parser.error("Supply separate prediction directories for the two models")
    print(json.dumps(evaluate(args), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
