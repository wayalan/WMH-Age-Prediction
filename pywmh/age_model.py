"""
pywmh.age_model - PVWMH / DWMH Parcellation and WMH-Predicted Age Model

Based on:
    Huang CC, Chou KH, Lee WJ, Yang AC, Tsai SJ, Chen LK, Chung CP, Lin CP.
    Brain white matter hyperintensities-predicted age reflects neurovascular
    health in middle-to-old aged subjects. Age Ageing. 2022 May 1;51(5):afac106.
"""

import os
from typing import Dict, Any, Optional
import numpy as np
import nibabel as nib
import pandas as pd


def get_voxel_volume_cc(nii_img: nib.Nifti1Image) -> float:
    """Computes voxel volume in cubic centimeters (cc / ml)."""
    zooms = nii_img.header.get_zooms()[:3]
    voxel_mm3 = float(np.prod(zooms))
    return voxel_mm3 / 1000.0


def calculate_wmh_volumes_and_age(
    lesion_img: nib.Nifti1Image,
    pv_mask_img: nib.Nifti1Image,
    dw_mask_img: nib.Nifti1Image,
    subject_id: str = "Subject",
    epsilon: float = 1e-4
) -> Dict[str, Any]:
    """
    Computes PVWMH, DWMH, and Total WMH volumes (in cc) and estimates WMH-predicted age.

    Parameters:
    -----------
    lesion_img : nib.Nifti1Image
        Lesion probability map or binary mask (either in native space or modulated MNI space).
    pv_mask_img : nib.Nifti1Image
        Periventricular WMH mask (value == 1).
    dw_mask_img : nib.Nifti1Image
        Deep WMH mask (value == 1).
    subject_id : str
        Subject identifier.
    epsilon : float
        Numerical floor protection against log10(0).

    Returns:
    --------
    results : dict
        Calculated volumes, predicted age, and metadata.
    """
    lesion_data = np.nan_to_num(lesion_img.get_fdata())
    pv_data = np.nan_to_num(pv_mask_img.get_fdata())
    dw_data = np.nan_to_num(dw_mask_img.get_fdata())

    if lesion_data.shape != pv_data.shape or lesion_data.shape != dw_data.shape:
        raise ValueError(
            f"Shape mismatch: Lesion {lesion_data.shape} vs PV mask {pv_data.shape} vs DW mask {dw_data.shape}"
        )

    voxel_vol_cc = get_voxel_volume_cc(lesion_img)

    pv_mask = pv_data == 1
    dw_mask = dw_data == 1

    # Sum lesion volume
    pv_vol = float(np.sum(lesion_data[pv_mask])) * voxel_vol_cc
    dw_vol = float(np.sum(lesion_data[dw_mask])) * voxel_vol_cc
    total_vol = float(np.sum(lesion_data[lesion_data > 0])) * voxel_vol_cc

    # Numerical floor protection for log10
    pv_safe = max(pv_vol, epsilon)
    dw_safe = max(dw_vol, epsilon)

    # Regression formula: Huang et al., Age Ageing 2022
    pred_age = 11.069 * np.log10(pv_safe) + 1.624 * np.log10(dw_safe) + 64.159

    results = {
        "SubjectID": subject_id,
        "Total_WMH_cc": round(total_vol, 4),
        "PVWMH_cc": round(pv_vol, 4),
        "DWMH_cc": round(dw_vol, 4),
        "Predicted_Age": round(float(pred_age), 2),
        "Voxel_Volume_cc": voxel_vol_cc
    }
    return results


def save_results(results: Dict[str, Any], out_csv: str, out_json: Optional[str] = None):
    """Saves quantification results to CSV and optional JSON."""
    df = pd.DataFrame([results])
    df.to_csv(out_csv, index=False)
    
    if out_json:
        import json
        with open(out_json, "w") as f:
            json.dump(results, f, indent=4)
