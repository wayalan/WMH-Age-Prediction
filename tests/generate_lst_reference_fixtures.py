#!/usr/bin/env python3
"""Generate independent synthetic native inputs for the installed original LST.

No clinical example files are used. MATLAB runs only for reference testing.
Run the generated run_reference.m with MATLAB, then compare each resulting
probability image with docs/validate_native_lga.py using the listed parameters.
"""

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.io import savemat

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pywmh.lst_lga import build_pve_label

PROFILES = (((20, 22, 24), 0.3, 50, 1),
            ((21, 23, 25), 0.17, 12, 0.65),
            ((18, 20, 22), 0.4, 25, 1.2))


def matlab_string(path):
    return str(path).replace("'", "''")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spm-root', type=Path, required=True)
    parser.add_argument('--outdir', type=Path, required=True)
    args = parser.parse_args()
    root = args.outdir.resolve()
    spm = args.spm_root.resolve()
    source_path = spm / 'toolbox/LST/ps_LST_lga.m'
    source = source_path.read_text(errors='replace')
    # Use the original initialization statements as the independent oracle.
    start = source.index('        m = [mean(t1(seg == 1))')
    end = source.index('        indx_brain = find(p0 > 0);', start)
    block = source[start:end]
    # Test inputs already include native FLAIR, so no disk reslice is needed.
    block = '\n'.join(line for line in block.splitlines() if 'spm_read_vols' not in line)
    root.mkdir(parents=True, exist_ok=False)
    for index, (shape, kappa, max_iter, phi) in enumerate(PROFILES):
        rng = np.random.default_rng(314159 + index)
        x, y, z = np.indices(shape)
        inside = ((x > 1) & (x < shape[0] - 2) & (y > 1) & (y < shape[1] - 2)
                  & (z > 1) & (z < shape[2] - 2))
        tissue = np.where(x < shape[0] // 3, 0, np.where(x < shape[0] * 2 // 3, 1, 2))
        pves = rng.uniform(0.02, 0.1, (*shape, 3))
        pves[x, y, z, tissue] = rng.uniform(0.65, 0.95, shape)
        pves /= pves.sum(axis=-1, keepdims=True)
        pves[~inside] = 0
        t1 = np.take([35, 85, 125], tissue) + rng.normal(0, 8, shape)
        flair = np.take([25, 105, 90], tissue) + rng.normal(0, 5, shape)
        core = ((x > shape[0] // 2 - 2) & (x < shape[0] // 2 + 3)
                & (y > 6) & (y < 13) & (z > 6) & (z < 14))
        halo = ((x > shape[0] // 2 - 3) & (x < shape[0] // 2 + 6)
                & (y > 4) & (y < 15) & (z > 4) & (z < 16))
        flair[halo] += rng.uniform(35, 75, halo.sum())
        flair[core] += rng.uniform(70, 120, core.sum())
        flair[~inside] = 0
        t1[~inside] = 0
        csf, gm, wm = np.moveaxis(pves, -1, 0)
        p0 = build_pve_label(t1, flair, gm, wm, csf)
        atlas = rng.uniform(0.4, 1, shape) * inside
        exclusion = (rng.uniform(0, 1, shape) < 0.01) & inside
        indices = np.flatnonzero(p0.ravel(order='F') > 0)
        case = root / f'case{index}'
        case.mkdir()
        for name, data in (('T1', t1), ('FLAIR', flair), ('rmFLAIR', flair)):
            nib.save(nib.Nifti1Image(data, np.eye(4)), case / f'{name}.nii')
        # MATLAB find() returns columns: getNeighborhood2 depends on this.
        lga = {'dim': np.array(shape, dtype=float),
               'indx_brain': (indices + 1).astype(float).reshape(-1, 1),
               'f2_vec': flair.ravel(order='F').reshape(-1, 1),
               'or': np.array([1, 2, 3]), 'fl': 0}
        for name, data in (('p0_vec', p0), ('atlas_wm_vec', atlas), ('noles_vec', exclusion)):
            lga[name] = data.astype(float).ravel(order='F')[indices].reshape(-1, 1)
        savemat(case / 'LST_lga_rmFLAIR.mat', {'lga': lga})
        savemat(case / 'label_inputs.mat', {'t1': t1, 'f2': flair, 'p1': gm,
                 'p2': wm, 'p3': csf, 'p0_python': p0})
        print(f'case{index}: kappa={kappa}, max_iter={max_iter}, phi={phi}')
    params = ';'.join(f'{k},{m},{p}' for _, k, m, p in PROFILES)
    script = f"""addpath('{matlab_string(spm)}');
addpath('{matlab_string(spm / 'toolbox/LST')}');
spm('defaults','FMRI'); spm_get_defaults('cmdline',true);
params=[{params}];
for ci=0:2
 cd(fullfile('{matlab_string(root)}',sprintf('case%d',ci)));
 load('label_inputs.mat');
 prob=[p3(:),p1(:),p2(:)]; prob=bsxfun(@times,prob,1./sum(prob,2));
 ib=find(sum(prob,2)>0); seg=zeros(size(t1)); [~,I]=max(prob(ib,:),[],2); seg(ib)=I;
 seg(:,:,1)=0;seg(:,:,end)=0;seg(:,1,:)=0;seg(:,end,:)=0;seg(1,:,:)=0;seg(end,:,:)=0;
{block}
 label_error=max(abs(p0(:)-p0_python(:)));
 fprintf('P0_MAX_ERROR case%d = %.17g\\n',ci,label_error); assert(label_error<1e-12);
 save('label_reference.mat','p0','-v7');
 ps_LST_lga(fullfile(pwd,'T1.nii'),fullfile(pwd,'FLAIR.nii'),params(ci+1,1),params(ci+1,2),params(ci+1,3),0);
end
"""
    script_path = root / 'run_reference.m'
    script_path.write_text(script)
    print(f'MATLAB reference script: {script_path}')


if __name__ == '__main__':
    main()
