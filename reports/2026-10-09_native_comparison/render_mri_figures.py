#!/usr/bin/env python3
"""Render reviewed native MRI masks; no re-segmentation, resampling or edits."""

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image, ImageDraw, ImageFont

BLUE = np.array([0, 114, 178])
ORANGE = np.array([230, 159, 0])
WHITE = np.array([245, 245, 245])
def font_path(bold=False):
    mac_path = Path('/System/Library/Fonts/Supplemental/Arial Bold.ttf' if bold else
                    '/System/Library/Fonts/Supplemental/Arial.ttf')
    if mac_path.exists():
        return str(mac_path)
    from matplotlib.font_manager import FontProperties, findfont
    return findfont(FontProperties(family='DejaVu Sans', weight='bold' if bold else 'normal'))


def font(size, bold=False):
    return ImageFont.truetype(font_path(bold), size)


def select_slices(target, candidate):
    active = np.flatnonzero(np.any(target | candidate, axis=(0, 1)))
    if not active.size:
        raise ValueError('No lesion-containing slices')
    chosen = {int(active[i]) for i in np.linspace(0, active.size - 1, 4).astype(int)}
    chosen.add(int(np.argmax(np.sum(target | candidate, axis=(0, 1)))))
    chosen.add(int(np.argmax(np.sum(target ^ candidate, axis=(0, 1)))))
    for index in np.linspace(0, active.size - 1, min(20, active.size)).astype(int):
        if len(chosen) >= 6:
            break
        chosen.add(int(active[index]))
    return sorted(chosen)[:6]


def composite(background, target, candidate, kind):
    rgb = np.repeat(background[..., None], 3, axis=-1).astype(float)
    if kind == 'matlab':
        rgb[target] = rgb[target] * 0.35 + BLUE * 0.65
    elif kind == 'python':
        rgb[candidate] = rgb[candidate] * 0.35 + ORANGE * 0.65
    elif kind == 'agreement':
        for selection, color in ((target & ~candidate, BLUE),
                                  (candidate & ~target, ORANGE),
                                  (target & candidate, WHITE)):
            rgb[selection] = rgb[selection] * 0.15 + color * 0.85
    return np.clip(rgb, 0, 255).astype(np.uint8)


def panel(array, box, width):
    # Inputs are native RAS. Display R at viewer left and A at top.
    a = array[box[0]:box[1], box[2]:box[3]]
    a = np.swapaxes(a, 0, 1)[::-1, ::-1]
    image = Image.fromarray(a)
    height = round(width * image.height / image.width)
    return image.resize((width, height), Image.Resampling.NEAREST)


def draw_sheet(case, background, target, candidate, slices, box, destination, title):
    width = 420
    gap = 14
    margin = 30
    example = panel(composite(background[:, :, slices[0]], target[:, :, slices[0]],
                              candidate[:, :, slices[0]], 'flair'), box, width)
    image_height = example.height
    row_height = image_height + 66
    sheet = Image.new('RGB', (margin * 2 + 4 * width + 3 * gap,
                             176 + len(slices) * row_height + 60), '#101318')
    draw = ImageDraw.Draw(sheet)
    draw.text((margin, 22), f'{case} | {title}', font=font(32, True), fill='white')
    draw.text((margin, 68), 'Difference panel: blue = MATLAB only; orange = Python only; white = overlap. Threshold > 0.5',
              font=font(23), fill='#e0e5ea')
    names = ('FLAIR background', 'MATLAB LST', 'Current Python', 'Agreement / difference')
    kinds = ('flair', 'matlab', 'python', 'agreement')
    for col, name in enumerate(names):
        draw.text((margin + col * (width + gap), 122), name, font=font(25, True), fill='white')
    for row, z in enumerate(slices):
        y = 176 + row * row_height
        m, p = target[:, :, z], candidate[:, :, z]
        denominator = int(m.sum() + p.sum())
        dice = 2 * int((m & p).sum()) / denominator if denominator else None
        for col, kind in enumerate(kinds):
            image = panel(composite(background[:, :, z], m, p, kind), box, width)
            x = margin + col * (width + gap)
            sheet.paste(image, (x, y))
            draw.text((x + 12, y + 10), f'z={z}', font=font(25, True), fill='white', stroke_width=2, stroke_fill='black')
            if col == 0:
                draw.text((x + 12, y + image_height // 2), 'R', font=font(25, True), fill='white')
                draw.text((x + width - 30, y + image_height // 2), 'L', font=font(25, True), fill='white')
        stats = f'Whole slice z={z}   MATLAB vox={int(m.sum())}   Python vox={int(p.sum())}   '
        stats += f'MATLAB-only={int((m & ~p).sum())}   Python-only={int((p & ~m).sum())}   Dice={dice:.3f}'
        draw.text((margin, y + image_height + 16), stats, font=font(24), fill='#d7dfe8')
    draw.text((margin, sheet.height - 42), 'Same native voxel grid; one FLAIR window for both masks. Display only; no mask changes.',
              font=font(23), fill='#b8c1ce')
    sheet.save(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metrics', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pipeline-root', type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    evidence = json.loads(args.metrics.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    selections = []
    def resolve(path):
        value = Path(path)
        return value if value.is_absolute() else args.pipeline_root / value
    for record in evidence['cases']:
        reference = nib.load(resolve(record['matlab_map'])).get_fdata() > 0.5
        python = nib.load(resolve(record['python_map'])).get_fdata() > 0.5
        flair_img = nib.load(resolve(record['flair_background']))
        if nib.aff2axcodes(flair_img.affine) != ('R', 'A', 'S'):
            raise ValueError('This renderer expects the verified native RAS grid')
        flair = flair_img.get_fdata()
        brain = nib.load(resolve(record['brain_mask'])).get_fdata() > 0
        valid = flair[brain & (flair > 0)]
        low, high = np.percentile(valid, [2, 98.5])
        background = (np.clip((flair - low) / (high - low), 0, 1) * 255).astype(np.uint8)
        xy = np.any(brain, axis=2)
        xx, yy = np.nonzero(xy)
        box = (max(0, int(xx.min()) - 6), min(flair.shape[0], int(xx.max()) + 7),
               max(0, int(yy.min()) - 6), min(flair.shape[1], int(yy.max()) + 7))
        slices = select_slices(reference, python)
        for part in range(2):
            selected = slices[part * 3:(part + 1) * 3]
            path = args.output / f'{record["case"]}_overview_{part + 1}.png'
            draw_sheet(record['case'], background, reference, python, selected, box, path,
                       f'Native axial comparison {part + 1}/2')
            selections.append({'case': record['case'], 'figure': path.name, 'slices': selected,
                'selection_rule': 'Four evenly spaced lesion-containing slices, peak union burden and peak disagreement; fill duplicates with evenly spaced remaining slices',
                'window_low': float(low), 'window_high': float(high), 'crop_xy': list(box)})
        # A reproducible zoom at the largest disagreement slice and location.
        z = int(np.argmax(np.sum(reference ^ python, axis=(0, 1))))
        difference = reference[:, :, z] ^ python[:, :, z]
        # Find the highest local disagreement count in a fixed 40x40 voxel window.
        from scipy.ndimage import uniform_filter
        score = uniform_filter(difference.astype(float), size=40, mode='constant')
        center = tuple(map(int, np.unravel_index(np.argmax(score), score.shape)))
        zoom = (max(0, center[0] - 24), min(flair.shape[0], center[0] + 24),
                max(0, center[1] - 24), min(flair.shape[1], center[1] + 24))
        path = args.output / f'{record["case"]}_detail.png'
        draw_sheet(record['case'], background, reference, python, [z], zoom, path,
                   'Largest disagreement slice: automatic detail crop')
        selections.append({'case': record['case'], 'figure': path.name, 'slices': [z],
            'selection_rule': 'Largest disagreement slice; 48x48 crop centered on highest 40x40 local disagreement density',
            'window_low': float(low), 'window_high': float(high), 'crop_xy': list(zoom)})
    (args.output / 'figure_selections.json').write_text(json.dumps(selections, indent=2))
    print(json.dumps(selections, indent=2))


if __name__ == '__main__':
    main()
