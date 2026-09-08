#!/usr/bin/env python3
"""
wmh_age_calc.py - Pure Python Standalone WMH Volume & Age Calculator

Calculates Periventricular WMH (PVWMH), Deep WMH (DWMH), and Predicted Age
from an MNI-registered modulated WMH lesion map (e.g. m0wples*.nii),
completely independent of MATLAB.

Reference:
    Huang CC, Chou KH, Lee WJ, et al. Brain white matter hyperintensities-predicted
    age reflects neurovascular health in middle-to-old aged subjects.
    Age Ageing. 2022 May 1;51(5):afac106.

Usage:
    python3 wmh_age_calc.py --input /path/to/m0wples_rmT2f.nii --res 1.5 --out results.csv
"""

import os
import sys
import argparse
import numpy as np
import nibabel as nib


def compute_wmh_age(
    wmh_path: str,
    atlas_dir: str = None,
    voxel_size: float = 1.5,
    epsilon: float = 1e-4
):
    """
    Computes PVWMH, DWMH, Total WMH volume (in cc/ml) and predicted WMH-Age.
    """
    if not os.path.exists(wmh_path):
        raise FileNotFoundError(f"WMH file not found: {wmh_path}")

    # Determine atlas directory
    if atlas_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        atlas_dir = os.path.join(base_dir, "WMH_Atlas")

    res_str = f"{voxel_size}mm" if f"{voxel_size}mm" in os.listdir(atlas_dir) else f"{int(voxel_size)}mm"
    res_dir = os.path.join(atlas_dir, res_str)

    pv_path = os.path.join(res_dir, "1_PVWMH_1-10mm.nii.gz")
    dw_path = os.path.join(res_dir, "2_DWMH_10mmup.nii.gz")

    if not os.path.exists(pv_path) or not os.path.exists(dw_path):
        raise FileNotFoundError(f"Atlas files missing in {res_dir}")

    # Load NIfTI data
    wmh_img = nib.load(wmh_path)
    pv_img = nib.load(pv_path)
    dw_img = nib.load(dw_path)

    wmh_data = np.nan_to_num(wmh_img.get_fdata())
    pv_mask = pv_img.get_fdata() == 1
    dw_mask = dw_img.get_fdata() == 1

    if wmh_data.shape != pv_mask.shape:
        raise ValueError(
            f"Dimension mismatch: WMH shape {wmh_data.shape} != Atlas shape {pv_mask.shape}."
        )

    # Voxel volume in cc (ml)
    # voxel_size^3 / 1000.0
    voxel_vol_cc = (voxel_size ** 3) / 1000.0

    pv_vol = float(np.sum(wmh_data[pv_mask])) * voxel_vol_cc
    dw_vol = float(np.sum(wmh_data[dw_mask])) * voxel_vol_cc
    total_vol = float(np.sum(wmh_data[wmh_data > 0])) * voxel_vol_cc

    # Numerical floor protection
    pv_safe = max(pv_vol, epsilon)
    dw_safe = max(dw_vol, epsilon)

    # Regression formula: Huang et al., Age Ageing 2022
    pred_age = 11.069 * np.log10(pv_safe) + 1.624 * np.log10(dw_safe) + 64.159

    return {
        "Total_WMH_cc": total_vol,
        "PVWMH_cc": pv_vol,
        "DWMH_cc": dw_vol,
        "Predicted_Age": pred_age,
        "Voxel_Size_mm": voxel_size
    }


def main():
    parser = argparse.ArgumentParser(description="Standalone WMH-Age Estimator")
    parser.add_argument("-i", "--input", required=True, help="Path to MNI modulated WMH map (m0wples*.nii)")
    parser.add_argument("-r", "--res", type=float, default=1.5, choices=[1.0, 1.5], help="Resolution (1.0 or 1.5 mm)")
    parser.add_argument("-a", "--atlas", default=None, help="Path to WMH_Atlas folder")
    parser.add_argument("-o", "--out", default=None, help="Output CSV path")

    args = parser.parse_args()

    results = compute_wmh_age(
        wmh_path=args.input,
        atlas_dir=args.atlas,
        voxel_size=args.res
    )

    print("================ Results ================")
    print(f"Total WMH Volume  : {results['Total_WMH_cc']:.3f} cc")
    print(f"PVWMH Volume      : {results['PVWMH_cc']:.3f} cc")
    print(f"DWMH Volume       : {results['DWMH_cc']:.3f} cc")
    print(f"Predicted WMH Age : {results['Predicted_Age']:.2f} years old")
    print("=========================================")

    if args.out:
        import pandas as pd
        df = pd.DataFrame([results])
        df.to_csv(args.out, index=False)
        print(f"Saved results to {args.out}")


if __name__ == "__main__":
    main()
