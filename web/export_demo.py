"""Export the v2 model and validation patches for real browser-side inference."""
import argparse
import csv
import json
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import rasterio
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from model import make_generator


def export(output):
    assets=output/'assets';assets.mkdir(parents=True,exist_ok=True)
    checkpoint=torch.load(ROOT/'runs/context_residual_v2/best.pt',map_location='cpu',weights_only=True)
    model=make_generator(checkpoint).eval();model.load_state_dict(checkpoint['model'])
    torch.set_num_threads(2)
    image=torch.zeros(1,4,128,128);mask=torch.zeros(1,1,128,128)
    torch.onnx.export(model,(image,mask),assets/'reconstruction.onnx',
                      input_names=['image','mask'],output_names=['prediction'],
                      opset_version=17,dynamo=False)
    rows=[r for r in csv.DictReader((ROOT/'pairs.csv').open()) if r['split']=='val']
    rows.sort(key=lambda r: (not r['input'].endswith('r04608_c02176.tif'), r['input']))
    samples=[]
    for i,row in enumerate(rows):
        with rasterio.open(ROOT/row['input']) as src:
            observed=src.read([1,2,3,4]).astype('float32')*.0000275-.2
            mask=((src.read(5)&30)!=0).astype('float32')[None]
        with rasterio.open(ROOT/row['target']) as src:
            reference=src.read([1,2,3,4]).astype('float32')*.0000275-.2
        name=f'patch-{i+1:02d}'
        np.concatenate([observed,mask]).astype('<f4').tofile(assets/(name+'.bin'))
        reference.astype('<f4').tofile(assets/(name+'-reference.bin'))
        rgb=np.moveaxis(observed[[2,1,0]],0,-1)
        Image.fromarray((np.clip(rgb/.35,0,1)**(1/1.8)*255).astype('uint8')).save(assets/(name+'.png'))
        with torch.no_grad():prediction=model(torch.from_numpy(observed[None]),torch.from_numpy(mask[None])).numpy()[0]
        if i==0:prediction.astype('<f4').tofile(assets/'verification-expected.bin')
        samples.append(dict(id=name,name=f'Patch {i+1:02d}',coordinates=Path(row['input']).stem,
                            mask_fraction=float(mask.mean()),cloud_fraction=float(row['cloud_fraction']),
                            input=f'assets/{name}.bin',reference=f'assets/{name}-reference.bin',
                            thumbnail=f'assets/{name}.png',source=row['input']))
    metadata=dict(model='context_residual_v2',checkpoint_epoch=checkpoint['epoch'],
                  source_commit='7a8d1599fe730ff9e78abde3ad6e3ff50b2855aa',
                  shape=[1,4,128,128],bands=['blue','green','red','nir'],
                  sample_set='nine validation patches',reference_gap_days=8,samples=samples)
    (assets/'samples.json').write_text(json.dumps(metadata,indent=2))
    print(f'Exported model and {len(samples)} validation samples to {output}')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    export(p.parse_args().output)
