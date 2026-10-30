"""Coarse alignment and boundary audit; no target is adjusted or used as input."""
import csv
import json
from pathlib import Path
import numpy as np
import rasterio


def dilate(mask, radius=1):
    p = np.pad(mask, radius)
    return np.logical_or.reduce([p[y:y+mask.shape[0], x:x+mask.shape[1]]
                                for y in range(2*radius+1) for x in range(2*radius+1)])


def main():
    root = Path(__file__).resolve().parent
    records = []
    # Validation only: test data is not consulted for diagnosing model changes.
    rows = [r for r in csv.DictReader((root/'pairs.csv').open()) if r['split']=='val']
    for row in rows:
        with rasterio.open(root/row['input']) as s:
            source = s.read([1,2,3,4]).astype('float32')*.0000275-.2
            mask = (s.read(5)&30)!=0
        with rasterio.open(root/row['target']) as s:
            target = s.read([1,2,3,4]).astype('float32')*.0000275-.2
        valid = ~dilate(mask, 3)
        scores = {}
        for dy in range(-2,3):
            for dx in range(-2,3):
                y0,y1 = max(0,-dy), min(128,128-dy)
                x0,x1 = max(0,-dx), min(128,128-dx)
                use=valid[y0:y1,x0:x1]
                a=source[2,y0:y1,x0:x1][use]
                b=target[2,y0+dy:y1+dy,x0+dx:x1+dx][use]
                scores[(dy,dx)] = float(np.corrcoef(a,b)[0,1]) if len(a)>100 and a.std()>1e-6 and b.std()>1e-6 else -1.
        best=max(scores,key=scores.get)
        ring=dilate(mask,2)&~mask
        difference=np.abs(source-target).mean(0)
        records.append(dict(input=row['input'], zero_shift_correlation=scores[(0,0)],
                            best_integer_shift=list(best), best_correlation=scores[best],
                            correlation_gain=scores[best]-scores[(0,0)],
                            mask_boundary_ring_mae=float(difference[ring].mean()),
                            far_clear_mae=float(difference[valid].mean())))
    report=dict(split='val', patches=len(records), search_radius_pixels=2,
                method='Pearson correlation of red-band values outside cloud/shadow masks plus a 3-pixel guard',
                caveat='Coarse diagnostic only. Different acquisition dates and sensors affect correlations; this does not prove subpixel alignment.',
                reference_used_as_model_input=False, records=records)
    out=root/'runs/context_residual_v2/alignment_audit.json'
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2))
    print('Median zero-shift correlation:',float(np.median([r['zero_shift_correlation'] for r in records])))
    print('Max improvement from shifting up to 2 pixels:',max(r['correlation_gain'] for r in records))


if __name__=='__main__':
    main()
