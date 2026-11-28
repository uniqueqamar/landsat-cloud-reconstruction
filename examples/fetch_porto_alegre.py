"""Download small public RTC windows and reference polygons for the Porto Alegre example."""
import argparse
import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen


WINDOW = '-51.34,-30.04,-51.18,-29.91'
SCENES = {
    'before': 'S1A_IW_GRDH_1SDV_20231125T084933_20231125T084958_051371_0632FF_rtc',
    'after': 'S1A_IW_GRDH_1SDV_20240511T084943_20240511T085008_053821_068A6E_rtc',
}
SERVICE_PREFIX = (
    'https://arcgis-maps.jrc.ec.europa.eu/hosted_server/rest/services/Hosted/'
    'EMSN194_Vector_Layers_'
)


def download(url, destination):
    limit = 10 * 1024 * 1024
    with urlopen(url, timeout=60) as response:
        data = response.read(limit + 1)
    if len(data) > limit:
        raise ValueError(f'{destination.name}: response exceeds 10 MiB')
    if destination.suffix == '.geojson':
        document = json.loads(data)
        if (document.get('type') != 'FeatureCollection' or not document.get('features')
                or document.get('exceededTransferLimit')):
            raise ValueError(f'{destination.name}: expected a complete vector response')
    elif data[:4] not in (b'II*\x00', b'MM\x00*', b'II+\x00', b'MM\x00+'):
        raise ValueError(f'{destination.name}: expected a TIFF response')
    destination.write_bytes(data)
    print(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--both-bands', action='store_true', help='Also download the VH crops')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for name, item in SCENES.items():
        endpoint = f'https://planetarycomputer.microsoft.com/api/data/v1/item/bbox/{WINDOW}/520x480.tif'
        polarizations = ('vv', 'vh') if args.both_bands else ('vv',)
        for band in polarizations:
            filename = f'{name}.tif' if band == 'vv' else f'{name}_vh.tif'
            query = urlencode(dict(collection='sentinel-1-rtc', item=item, assets=band, dst_crs='EPSG:32722'))
            download(f'{endpoint}?{query}', args.output / filename)
    query = urlencode(dict(
        where='1=1', outFields='*', outSR=4326, f='geojson', geometry=WINDOW, inSR=4326,
        geometryType='esriGeometryEnvelope', spatialRel='esriSpatialRelIntersects',
    ))
    for name, layer in (
        ('reference', 'Extent_Temporal_Evolution_10_05_2024'),
        ('aoi', 'Area_of_Interest'),
    ):
        url = f'{SERVICE_PREFIX}{layer}/FeatureServer/0/query?{query}'
        download(url, args.output / f'{name}.geojson')


if __name__ == '__main__':
    main()
