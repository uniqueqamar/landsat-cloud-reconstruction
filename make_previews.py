"""Make display-only RGB sheets; scientific data stays in the GeoTIFFs."""
import csv
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import rasterio

ROOT = Path(__file__).resolve().parent


def rgb(path):
    with rasterio.open(path) as src:
        image = src.read([3, 2, 1]).astype('float32') * .0000275 - .2
    image = np.moveaxis(image, 0, -1)
    pixels = (np.clip(image / .35, 0, 1) ** (1 / 1.8) * 255).astype('uint8')
    return Image.fromarray(pixels).resize((192, 192))


def main():
    rows = list(csv.DictReader((ROOT / 'manifest.csv').open()))
    canvas = Image.new('RGB', (896, 794), '#f5f5f1')
    draw = ImageDraw.Draw(canvas)
    draw.text((24, 18), 'LANDSAT CLOUD RECONSTRUCTION | actual extracted data', fill='#172f30')
    draw.text((24, 40), 'Landsat 9 | 27 Sep 2026 | 128 x 128 pixels | 30 m/pixel | RGB display stretch', fill='#374e50')
    for rownum, category in enumerate(['clear', 'cloudy']):
        chosen = sorted([r for r in rows if r['category'] == category], key=lambda r: r['path'])[::5][:4]
        y = 110 + rownum * 330
        draw.text((24, y - 28), 'QA-SCREENED CLEAR PATCHES' if category == 'clear' else 'REAL CLOUDY PATCHES (no generated result shown)', fill='#172f30')
        for i, row in enumerate(chosen):
            x = 24 + i * 220
            canvas.paste(rgb(ROOT / row['path']), (x, y))
            draw.text((x, y + 202), Path(row['path']).stem, fill='#172f30')
            draw.text((x, y + 220), f'{row["split"]} | cloud {float(row["cloud_fraction"]):.0%}', fill='#374e50')
    draw.text((24, 756), 'Images are observations, not predictions. NIR and quality bands are retained in each GeoTIFF.', fill='#374e50')
    canvas.save(ROOT / 'data_preview.png')
    if (ROOT / 'pairs.csv').exists():
        pairs = list(csv.DictReader((ROOT / 'pairs.csv').open()))[:4]
        if pairs:
            sheet = Image.new('RGB', (660, 100 + len(pairs) * 224), '#f5f5f1')
            d = ImageDraw.Draw(sheet)
            d.text((20, 18), 'REAL CLOUDY INPUT / ALIGNED TEMPORAL REFERENCE', fill='#172f30')
            d.text((20, 42), '27 Sep Landsat 9 vs 5 Oct Landsat 8 | 8-day gap; changes may occur', fill='#374e50')
            for i, row in enumerate(pairs):
                y = 90 + 224 * i
                sheet.paste(rgb(ROOT / row['input']), (20, y))
                sheet.paste(rgb(ROOT / row['target']), (232, y))
                d.text((446, y + 20), f'{row["split"]}\n{Path(row["input"]).stem}\nInput cloud: {float(row["cloud_fraction"]):.0%}\nReference: QA screened', fill='#172f30')
            sheet.save(ROOT / 'reference_preview.png')


if __name__ == '__main__':
    main()
