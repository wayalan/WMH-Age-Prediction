#!/usr/bin/env python3
"""Validate the Python growth core with one arbitrary MATLAB LST cache.

Reference lesions are read only after Python segmentation, for measurement.
This test adapter is not imported by the production pipeline.
Requires h5py for MATLAB v7.3 files (validation only).
"""

import argparse
import json
import sys
from pathlib import Path

import h5py
import nibabel as nib
import numpy as np
from scipy.io import loadmat

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pywmh.lst_lga import segment_wmh_lga


def load_inputs(path):
    if not h5py.is_hdf5(path):
        lga = loadmat(path, simplify_cells=True)['lga']
        shape = tuple(np.asarray(lga['dim']).ravel().astype(int))
        indices = np.asarray(lga['indx_brain']).ravel().astype(np.int64) - 1
        result = {'flair': np.asarray(lga['f2_vec']).reshape(shape, order='F')}
        for name, field in (('p0', 'p0_vec'), ('atlas_wm', 'atlas_wm_vec'), ('noles_mask', 'noles_vec')):
            values = np.zeros(np.prod(shape), dtype=np.float64)
            values[indices] = np.asarray(lga[field]).ravel()
            result[name] = values.reshape(shape, order='F')
        return result
    with h5py.File(path) as file:
        lga = file['lga']
        shape = tuple(lga['dim'][()].ravel().astype(int))
        indices = lga['indx_brain'][()].ravel().astype(np.int64) - 1
        flair = lga['f2_vec'][()].ravel().reshape(shape, order='F')
        result = {'flair': flair}
        for name, field in (('p0', 'p0_vec'), ('atlas_wm', 'atlas_wm_vec'), ('noles_mask', 'noles_vec')):
            values = np.zeros(np.prod(shape), dtype=np.float64)
            values[indices] = lga[field][()].ravel()
            result[name] = values.reshape(shape, order='F')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--kappa', type=float, default=0.3)
    parser.add_argument('--max-iter', type=int, default=50)
    parser.add_argument('--phi', type=float, default=1)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Choose a new output file; existing results are not overwritten')
    probability, binary = segment_wmh_lga(**load_inputs(args.cache), kappa=args.kappa,
                                        max_iter=args.max_iter, phi=args.phi)
    reference_img = nib.load(args.reference)
    reference = reference_img.get_fdata()
    if reference.shape != probability.shape:
        raise ValueError('Reference and cache must have the identical native voxel grid')
    target = reference > 0.5
    count = int(binary.sum())
    target_count = int(target.sum())
    denominator = count + target_count
    voxel_ml = float(np.prod(reference_img.header.get_zooms()[:3])) / 1000
    result = {
        'scope': 'Identical upstream inputs, native LGA core; no registration evaluation',
        'cache': str(args.cache), 'reference': str(args.reference),
        'kappa': args.kappa, 'phi': args.phi, 'max_iter': args.max_iter,
        'python_probability_ml': float(probability.astype(np.float64).sum()) * voxel_ml,
        'matlab_probability_ml': float(reference.sum()) * voxel_ml,
        'python_binary_ml': count * voxel_ml,
        'matlab_binary_ml': target_count * voxel_ml,
        'dice': float(2 * ((binary > 0) & target).sum() / denominator) if denominator else 1,
        'binary_disagreement_voxels': int(((binary > 0) != target).sum()),
        'maximum_absolute_probability_error': float(np.abs(probability - reference).max()),
        'mean_absolute_probability_error': float(np.abs(probability - reference).mean()),
        'matches_reference_at_float32_precision': bool(np.array_equal(probability, reference.astype(np.float32))),
    }
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
