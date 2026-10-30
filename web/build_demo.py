"""Build a self-contained static browser demo using the included checkpoint."""
import argparse
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

VERSION='1.31.1'
HERE=Path(__file__).resolve().parent


def build(output):
    output.mkdir(parents=True,exist_ok=True)
    subprocess.run([sys.executable,str(HERE/'export_demo.py'),'--output',str(output)],check=True)
    for name in ['index.html','style.css','app.js']:
        shutil.copyfile(HERE/name,output/name)
    vendor=output/'vendor';vendor.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        path=Path(tmp)/'runtime.tgz'
        urllib.request.urlretrieve(f'https://registry.npmjs.org/onnxruntime-web/-/onnxruntime-web-{VERSION}.tgz',path)
        with tarfile.open(path) as archive:
            # Extract only named regular assets; never extract arbitrary archive paths.
            for name in ['ort.wasm.min.js','ort-wasm-simd-threaded.mjs','ort-wasm-simd-threaded.wasm']:
                data=archive.extractfile('package/dist/'+name)
                if data is None:raise RuntimeError('Missing runtime asset: '+name)
                (vendor/name).write_bytes(data.read())
        urllib.request.urlretrieve('https://raw.githubusercontent.com/microsoft/onnxruntime/main/LICENSE',vendor/'LICENSE')
    print('Static demo ready:',output)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=HERE/'dist')
    build(p.parse_args().output.resolve())
