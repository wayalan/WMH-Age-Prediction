#!/usr/bin/env python3
"""Native WMH comparison, with each pipeline's own preprocessing inputs.

No MATLAB tissue cache, reference label or reference lesion is an input to
Python segmentation. The reference map is read after the new map is computed.
All authored outputs stay under --output. Existing subject outputs are read-only.
"""

import argparse
import csv
import hashlib
import json
import os
import platform
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import nibabel as nib
import numpy as np
import scipy
from scipy import ndimage


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def finite_image(path):
    image = nib.load(path)
    data = image.get_fdata()
    if data.ndim != 3 or not np.all(np.isfinite(data)):
        raise ValueError(f'Invalid 3D finite image: {path}')
    return image, data


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pipeline-root', type=Path, required=True)
    parser.add_argument('--cases', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.pipeline_root.resolve()))
    from pywmh.fsl_engine import tissue_segment_fast
    from pywmh.lst_lga import ALGORITHM_VERSION, segment_wmh_lga

    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'metrics.json').exists():
        raise ValueError('Completed metrics already exist; use a new output directory')
    config = json.loads(args.cases.read_text())
    code_path = args.pipeline_root / 'pywmh/lst_lga.py'
    shutil.copyfile(code_path, out / 'lst_lga_tested_snapshot.py')
    summary = {
        'tested_at_utc': datetime.now(timezone.utc).isoformat(),
        'algorithm_version': ALGORITHM_VERSION, 'code_sha256': digest(code_path),
        'environment': {'python': platform.python_version(), 'numpy': np.__version__,
                        'scipy': scipy.__version__, 'nibabel': nib.__version__},
        'scope': 'Native WMH segmentation. No registration estimation or evaluation.',
        'reference_status': 'Existing original MATLAB LST native probability maps, not newly rerun SPM preprocessing',
        'parameters': {'kappa': 0.3, 'phi': 1, 'max_iter': 50, 'binary_threshold': 0.5},
        'path_base': 'pipeline_root',
        'cases': [],
    }
    if Path('/usr/local/fsl/etc/fslversion').exists():
        summary['environment']['fsl'] = Path('/usr/local/fsl/etc/fslversion').read_text().strip()
    all_slices = []
    for spec in config['cases']:
        case_id = spec['id']
        case_out = out / 'native' / case_id
        case_out.mkdir(parents=True, exist_ok=True)
        print(f'[{case_id}] Running FAST with corrected T1 output; keeping existing native geometry', flush=True)
        native = {k: args.pipeline_root / v for k, v in spec['native'].items()}
        portable = lambda path: os.path.relpath(Path(path).resolve(), args.pipeline_root.resolve())
        own = tissue_segment_fast(str(native['t1_brain']), str(case_out / 'fast' / 'T1'))
        flair_img, flair = finite_image(native['flair'])
        load = lambda path: finite_image(path)[1]
        probability, binary = segment_wmh_lga(
            flair=flair, pve_gm=load(own['pve_gm']), pve_wm=load(own['pve_wm']),
            pve_csf=load(own['pve_csf']), t1=load(own['t1_corrected']),
            brain_mask=load(native['brain_mask']) > 0,
            atlas_wm=load(native['atlas_wm']), noles_mask=load(native['noles']) > 0,
            kappa=0.3, phi=1, max_iter=50,
        )
        python_path = case_out / 'python_ples_native.nii.gz'
        image = nib.Nifti1Image(probability, flair_img.affine, flair_img.header)
        image.set_data_dtype(np.float32)
        nib.save(image, python_path)
        image = nib.Nifti1Image(binary, flair_img.affine, flair_img.header)
        image.set_data_dtype(np.uint8)
        nib.save(image, case_out / 'python_bles_native.nii.gz')

        # Reference lesions are introduced only after Python segmentation.
        reference_path = args.pipeline_root / spec['matlab_map']
        reference_img, reference = finite_image(reference_path)
        if reference.shape != probability.shape or not np.allclose(
                reference_img.affine, flair_img.affine, rtol=0, atol=1e-4):
            raise ValueError('Reference is not on the same native voxel grid; no automatic resampling is performed')
        python_voxel = float(np.prod(flair_img.header.get_zooms()[:3])) / 1000
        matlab_voxel = float(np.prod(reference_img.header.get_zooms()[:3])) / 1000
        target = reference > 0.5
        candidate = binary > 0
        overlap = target & candidate
        matlab_only = target & ~candidate
        python_only = candidate & ~target
        union = target | candidate
        intersection = int(overlap.sum())
        n_matlab, n_python = int(target.sum()), int(candidate.sum())
        if not n_matlab or not n_python:
            raise ValueError('Empty comparison map; report needs explicit handling')
        pm = float(probability.astype(np.float64).sum()) * python_voxel
        mm = float(reference.sum()) * matlab_voxel
        components = ndimage.generate_binary_structure(3, 1)
        record = {
            'case': case_id, 'source_directory': spec['source_directory'],
            'matlab_probability_ml': mm, 'python_probability_ml': pm,
            'probability_difference_ml': pm - mm,
            'probability_relative_difference_pct': (pm - mm) / mm * 100,
            'matlab_binary_ml': n_matlab * matlab_voxel,
            'python_binary_ml': n_python * python_voxel,
            'dice': 2 * intersection / (n_matlab + n_python),
            'jaccard': intersection / int(union.sum()),
            'matlab_coverage': intersection / n_matlab,
            'python_overlap_fraction': intersection / n_python,
            'intersection_voxels': intersection, 'matlab_only_voxels': int(matlab_only.sum()),
            'python_only_voxels': int(python_only.sum()),
            'matlab_binary_voxels': n_matlab, 'python_binary_voxels': n_python,
            'matlab_components_6_connected': ndimage.label(target, components)[1],
            'python_components_6_connected': ndimage.label(candidate, components)[1],
            'shape': list(flair.shape), 'native_spacing_mm': list(map(float, flair_img.header.get_zooms()[:3])),
            'same_native_grid': True,
            'flair_background': portable(native['flair']), 'python_map': portable(python_path),
            'matlab_map': portable(reference_path), 'brain_mask': portable(native['brain_mask']),
            'input_identity': {}, 'sources': [],
        }
        for modality in ('t1', 'flair'):
            paths = [args.pipeline_root / spec['raw_inputs'][f'{modality}_{method}'] for method in ('matlab', 'python')]
            raw = [nib.load(p) for p in paths]
            a, b = [im.get_fdata() for im in raw]
            record['input_identity'][modality] = {
                'identical_raw_voxels': bool(a.shape == b.shape and np.array_equal(a, b)),
                'shape': list(a.shape), 'spacing_mm': list(map(float, raw[0].header.get_zooms()[:3]))}
            for method, path in zip(('matlab', 'python'), paths):
                record['sources'].append({'role': f'raw_{modality}_{method}', 'path': portable(path), 'sha256': digest(path)})
        for role, path in {**native, **own, 'matlab_reference': reference_path, 'python_result': python_path}.items():
            record['sources'].append({'role': role, 'path': portable(path), 'sha256': digest(path)})
        assert record['input_identity']['t1']['identical_raw_voxels']
        assert record['input_identity']['flair']['identical_raw_voxels']
        for z in range(flair.shape[2]):
            m, p = target[:, :, z], candidate[:, :, z]
            denominator = int(m.sum() + p.sum())
            all_slices.append({
                'case': case_id, 'z': z,
                'matlab_probability_ml': float(reference[:, :, z].sum()) * matlab_voxel,
                'python_probability_ml': float(probability[:, :, z].astype(float).sum()) * python_voxel,
                'matlab_binary_ml': int(m.sum()) * matlab_voxel,
                'python_binary_ml': int(p.sum()) * python_voxel,
                'intersection_voxels': int((m & p).sum()),
                'matlab_only_voxels': int((m & ~p).sum()),
                'python_only_voxels': int((p & ~m).sum()),
                'dice': 2 * int((m & p).sum()) / denominator if denominator else None,
            })
        summary['cases'].append(record)
        print(json.dumps({k: record[k] for k in ('case', 'matlab_probability_ml', 'python_probability_ml', 'dice')}, ensure_ascii=False), flush=True)
    (out / 'metrics.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    compact = [{k: c[k] for k in ('case', 'matlab_probability_ml', 'python_probability_ml',
        'probability_difference_ml', 'probability_relative_difference_pct', 'matlab_binary_ml',
        'python_binary_ml', 'dice', 'jaccard', 'matlab_coverage', 'python_overlap_fraction',
        'intersection_voxels', 'matlab_only_voxels', 'python_only_voxels')} for c in summary['cases']]
    write_csv(out / 'metrics.csv', compact)
    write_csv(out / 'per_slice_metrics.csv', all_slices)
    print(f'Finished native comparison: {out}', flush=True)


if __name__ == '__main__':
    main()
