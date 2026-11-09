"""Streaming masked spectral diagnostics for four-band Landsat reflectance.

Inputs use B2/B3/B4/B5 ordering. Metrics describe proxy-reference agreement,
not correctness of unobserved ground beneath clouds.
"""
import numpy as np


class MaskedQuality:
    def __init__(self):
        self.absolute = self.squared = self.sam = self.ndvi = 0.0
        self.band_absolute = np.zeros(4, dtype=np.float64)
        self.pixels = 0
        self.spectral_pixels = 0
        self.ndvi_pixels = 0

    def add(self, prediction, reference, mask):
        p = np.asarray(prediction, dtype=np.float64)
        r = np.asarray(reference, dtype=np.float64)
        m = np.asarray(mask, dtype=bool)
        if p.shape != r.shape or p.ndim != 3 or p.shape[0] != 4 or m.shape not in (p.shape[1:], (1, *p.shape[1:])):
            raise ValueError('Expected matching (4,H,W) reflectance and (H,W) mask')
        selected = m.reshape(p.shape[1:])
        if not np.any(selected):
            return
        a, b = p[:, selected], r[:, selected]
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError('Nonfinite reflectance inside mask')
        diff = a - b
        self.absolute += np.abs(diff).sum()
        self.squared += np.square(diff).sum()
        self.band_absolute += np.abs(diff).sum(axis=1)
        self.pixels += a.shape[1]
        norms = np.linalg.norm(a, axis=0) * np.linalg.norm(b, axis=0)
        good = norms > 1e-12
        if np.any(good):
            cos = np.clip(np.sum(a[:, good] * b[:, good], axis=0) / norms[good], -1, 1)
            self.sam += np.arccos(cos).sum()
            self.spectral_pixels += int(good.sum())
        # NDVI uses red (B4, index 2) and NIR (B5, index 3).
        dp, dr = a[3] + a[2], b[3] + b[2]
        good = (np.abs(dp) > 1e-8) & (np.abs(dr) > 1e-8)
        if np.any(good):
            self.ndvi += np.abs((a[3, good] - a[2, good]) / dp[good] - (b[3, good] - b[2, good]) / dr[good]).sum()
            self.ndvi_pixels += int(good.sum())

    def result(self):
        if not self.pixels:
            raise ValueError('No masked pixels available for quality metrics')
        n = self.pixels * 4
        return dict(masked_mae=self.absolute / n,
                    masked_rmse=float(np.sqrt(self.squared / n)),
                    masked_psnr_unit_range_db=float(-10 * np.log10(max(self.squared / n, 1e-12))),
                    per_band_masked_mae=dict(zip(('B2', 'B3', 'B4', 'B5'), (self.band_absolute / self.pixels).tolist())),
                    masked_sam_degrees=float(np.degrees(self.sam / self.spectral_pixels)) if self.spectral_pixels else None,
                    masked_ndvi_mae=self.ndvi / self.ndvi_pixels if self.ndvi_pixels else None,
                    masked_pixel_count=self.pixels,
                    sam_valid_pixel_count=self.spectral_pixels,
                    ndvi_valid_pixel_count=self.ndvi_pixels)
