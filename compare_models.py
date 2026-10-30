"""Fixed validation example: original model, revised model, and temporal reference."""
import argparse
import csv
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import rasterio
import torch
from model import make_generator
from train import choose_device


def display(array):
    rgb=np.moveaxis(array[[2,1,0]],0,-1)
    return Image.fromarray((np.clip(rgb/.35,0,1)**(1/1.8)*255).astype('uint8')).resize((224,224))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--weights',type=Path,default=Path('runs/context_residual_v2/best.pt'))
    p.add_argument('--output',type=Path,default=Path('outputs/v2_comparison.png'))
    a=p.parse_args()
    root=Path(__file__).resolve().parent
    sample='data/cloudy/val/r04608_c02176.tif'
    row=next(r for r in csv.DictReader((root/'pairs.csv').open()) if r['input']==sample)
    with rasterio.open(root/sample) as src:
        image=src.read([1,2,3,4]).astype('float32')*.0000275-.2
        mask=((src.read(5)&30)!=0).astype('float32')
    with rasterio.open(root/row['target']) as src:
        reference=src.read([1,2,3,4]).astype('float32')*.0000275-.2
    device=choose_device();torch.set_num_threads(2)
    predictions=[]
    for weights in [root/'runs/temporal_cgan/best.pt',a.weights]:
        c=torch.load(weights,weights_only=True,map_location='cpu')
        model=make_generator(c).to(device);model.load_state_dict(c['model']);model.eval()
        with torch.no_grad():
            predictions.append(model(torch.from_numpy(image[None]).to(device),
                               torch.from_numpy(mask[None,None]).to(device)).cpu().numpy()[0])
    canvas=Image.new('RGB',(988,336),'#f5f5f1');draw=ImageDraw.Draw(canvas)
    draw.text((20,12),'CLOUD RECONSTRUCTION | fixed validation example',fill='#172f30')
    for i,(title,array) in enumerate(zip(['Cloudy input: 27 Sep','Original GAN estimate','Revised residual estimate','Reference: 5 Oct'],[image,*predictions,reference])):
        x=20+i*244
        draw.text((x,40),title,fill='#172f30');canvas.paste(display(array),(x,62))
    draw.text((20,302),'Both predictions are estimates. Reference is eight days later; hidden field details remain uncertain.',fill='#374e50')
    a.output.parent.mkdir(parents=True,exist_ok=True);canvas.save(a.output)
    print(a.output)


if __name__=='__main__':
    main()
