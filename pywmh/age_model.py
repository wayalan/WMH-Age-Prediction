"""
pywmh.age_model - PVWMH / DWMH Parcellation, Regional Breakdown, and WMH-Predicted Age Model

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
    epsilon: float = 1e-4,
    chronological_age: Optional[float] = None,
    tiv_cc: Optional[float] = None
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
    chronological_age : float, optional
        Actual chronological age of subject. If provided, calculates Brain Age Gap (BAG).
    tiv_cc : float, optional
        Total Intracranial Volume in cc. If provided, calculates WMH % of TIV.

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

    results: Dict[str, Any] = {
        "SubjectID": subject_id,
        "Total_WMH_cc": round(total_vol, 4),
        "PVWMH_cc": round(pv_vol, 4),
        "DWMH_cc": round(dw_vol, 4),
        "Predicted_Age": round(float(pred_age), 2),
        "Voxel_Volume_cc": voxel_vol_cc
    }

    if chronological_age is not None:
        c_age = float(chronological_age)
        results["Chronological_Age"] = round(c_age, 2)
        results["Brain_Age_Gap"] = round(float(pred_age - c_age), 2)

    if tiv_cc is not None and tiv_cc > 0:
        tiv = float(tiv_cc)
        results["TIV_cc"] = round(tiv, 2)
        results["Total_WMH_pct_TIV"] = round((total_vol / tiv) * 100.0, 4)
        results["PVWMH_pct_TIV"] = round((pv_vol / tiv) * 100.0, 4)
        results["DWMH_pct_TIV"] = round((dw_vol / tiv) * 100.0, 4)

    return results


def calculate_regional_breakdown(
    lesion_img: nib.Nifti1Image,
    lobar_mask_img: Optional[nib.Nifti1Image] = None,
    arterial_mask_img: Optional[nib.Nifti1Image] = None
) -> Dict[str, Any]:
    """
    Computes regional WMH volume distributions across cerebral lobes and arterial vascular territories.

    Lobar Atlas labels:
        1: Left Frontal, 2: Right Frontal
        3: Left Parietal, 4: Right Parietal
        5: Left Temporal, 6: Right Temporal
        7: Left Occipital, 8: Right Occipital
        9: Left Subcortical/Insula, 10: Right Subcortical/Insula
        11: Infratentorial / Cerebellum

    Arterial Atlas labels:
        1: ACAL, 2: ACAR (Anterior Cerebral Artery)
        3: MCAL, 4: MCAR (Middle Cerebral Artery)
        5: PCAL, 6: PCAR (Posterior Cerebral Artery)
        7: VBL, 8: VBR (Vertebrobasilar)
        9: LVL, 10: LVR (Lateral Ventricles)
    """
    lesion_data = np.nan_to_num(lesion_img.get_fdata())
    voxel_vol_cc = get_voxel_volume_cc(lesion_img)
    total_vol = float(np.sum(lesion_data[lesion_data > 0])) * voxel_vol_cc
    eps = 1e-9

    breakdown: Dict[str, Any] = {}

    # 1. Lobar Parcellation
    if lobar_mask_img is not None:
        lobar_data = np.round(np.nan_to_num(lobar_mask_img.get_fdata())).astype(np.int32)
        lobes = {
            "Frontal": (1, 2),
            "Parietal": (3, 4),
            "Temporal": (5, 6),
            "Occipital": (7, 8),
            "Subcortical": (9, 10)
        }
        for lobe_name, (l_idx, r_idx) in lobes.items():
            l_vol = float(np.sum(lesion_data[lobar_data == l_idx])) * voxel_vol_cc
            r_vol = float(np.sum(lesion_data[lobar_data == r_idx])) * voxel_vol_cc
            t_vol = l_vol + r_vol
            pct = (t_vol / max(total_vol, eps)) * 100.0

            breakdown[f"Lobar_{lobe_name}_Total_cc"] = round(t_vol, 4)
            breakdown[f"Lobar_{lobe_name}_Total_pct"] = round(pct, 2)
            breakdown[f"Lobar_{lobe_name}_Left_cc"] = round(l_vol, 4)
            breakdown[f"Lobar_{lobe_name}_Right_cc"] = round(r_vol, 4)

    # 2. Arterial Vascular Territory Parcellation
    if arterial_mask_img is not None:
        arterial_data = np.round(np.nan_to_num(arterial_mask_img.get_fdata())).astype(np.int32)
        territories = {
            "ACA": (1, 2),
            "MCA": (3, 4),
            "PCA": (5, 6),
            "VB": (7, 8)
        }
        for terr_name, (l_idx, r_idx) in territories.items():
            l_vol = float(np.sum(lesion_data[arterial_data == l_idx])) * voxel_vol_cc
            r_vol = float(np.sum(lesion_data[arterial_data == r_idx])) * voxel_vol_cc
            t_vol = l_vol + r_vol
            pct = (t_vol / max(total_vol, eps)) * 100.0

            breakdown[f"Arterial_{terr_name}_Total_cc"] = round(t_vol, 4)
            breakdown[f"Arterial_{terr_name}_Total_pct"] = round(pct, 2)
            breakdown[f"Arterial_{terr_name}_Left_cc"] = round(l_vol, 4)
            breakdown[f"Arterial_{terr_name}_Right_cc"] = round(r_vol, 4)

    return breakdown


def save_results(results: Dict[str, Any], out_csv: str, out_json: Optional[str] = None):
    """Saves quantification results to CSV and optional JSON."""
    df = pd.DataFrame([results])
    df.to_csv(out_csv, index=False)

    if out_json:
        import json
        with open(out_json, "w") as f:
            json.dump(results, f, indent=4)
