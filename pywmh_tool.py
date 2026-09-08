#!/usr/bin/env python3
"""
pywmh_tool.py - Complete Lightweight WMH-Age Estimation Pipeline (Pure Python + FSL)

A MATLAB-free, SPM-free, fully automated tool for:
1. FLAIR to T1 rigid coregistration (FLIRT)
2. T1 brain extraction (BET) and tissue segmentation (FAST)
3. Lesion Growth Algorithm (LST-LGA) in pure Python
4. Spatial alignment with Periventricular & Deep WMH Atlas (WMH_Atlas)
5. WMH Volume quantification & Age prediction (Huang et al., Age Ageing 2022)

Usage Examples:
    # 1. From NIfTI files:
    python3 pywmh_tool.py --t1 sub-01_T1w.nii.gz --flair sub-01_FLAIR.nii.gz --outdir ./output_sub01

    # 2. From DICOM folder:
    python3 pywmh_tool.py --dcm /path/to/dicoms --outdir ./output_sub01

    # 3. Using existing lesion map:
    python3 pywmh_tool.py --lesion /path/to/m0wples_T2f.nii --outdir ./output_sub01
"""

import os
import sys
import argparse
import time
import nibabel as nib
import numpy as np

# Internal package imports
from pywmh.fsl_engine import (
    check_fsl_installed,
    dcm2nii,
    coregister_flair_to_t1,
    brain_extract,
    tissue_segment_fast,
    register_t1_to_mni,
    warp_atlas_to_native,
    warp_lesion_to_mni_modulated
)
from pywmh.lst_lga import segment_wmh_lga
from pywmh.age_model import calculate_wmh_volumes_and_age, save_results


def find_dicom_subfolders(dcm_root: str):
    """Finds T1 and T2/FLAIR subdirectories in DICOM folder."""
    t1_dir = None
    flair_dir = None

    for root, dirs, files in os.walk(dcm_root):
        folder_name = os.path.basename(root).lower()
        if "t1" in folder_name and t1_dir is None:
            t1_dir = root
        elif ("flair" in folder_name or "t2" in folder_name) and flair_dir is None:
            flair_dir = root

    if not t1_dir or not flair_dir:
        raise ValueError(
            f"Could not automatically identify T1 and FLAIR folders in {dcm_root}. "
            "Please ensure directory names contain 'T1' and 'FLAIR' or 'T2'."
        )
    return t1_dir, flair_dir


def run_pipeline(
    t1_path: str = None,
    flair_path: str = None,
    dcm_dir: str = None,
    lesion_path: str = None,
    outdir: str = "./wmh_age_output",
    kappa: float = 0.3,
    voxel_size: float = 1.0,
    nonlinear: bool = False,
    save_mni_lesion: bool = False,
    subject_id: str = None
):
    start_time = time.time()
    os.makedirs(outdir, exist_ok=True)

    print("================================================================")
    print("      PyWMH: Lightweight WMH-Age Pipeline (Python + FSL)      ")
    print("================================================================")

    # 1. Handle Pre-existing MNI Lesion Map (Fast Path)
    if lesion_path:
        print(f"Direct evaluation from lesion map: {lesion_path}")
        script_dir = os.path.dirname(os.path.abspath(__file__))
        atlas_res = f"{voxel_size}mm" if f"{voxel_size}mm" in ["1mm", "1.5mm"] else "1mm"
        atlas_dir = os.path.join(script_dir, "WMH_Atlas", atlas_res)
        pv_atlas = os.path.join(atlas_dir, "1_PVWMH_1-10mm.nii.gz")
        dw_atlas = os.path.join(atlas_dir, "2_DWMH_10mmup.nii.gz")

        lesion_img = nib.load(lesion_path)
        pv_img = nib.load(pv_atlas)
        dw_img = nib.load(dw_atlas)

        subj = subject_id or os.path.basename(lesion_path).split(".")[0]
        res = calculate_wmh_volumes_and_age(lesion_img, pv_img, dw_img, subject_id=subj)

        csv_out = os.path.join(outdir, "WMH_Age_Results.csv")
        save_results(res, csv_out, os.path.join(outdir, "WMH_Age_Results.json"))

        print("\n----------------- Results -----------------")
        print(f"Total WMH Volume  : {res['Total_WMH_cc']:.3f} cc")
        print(f"PVWMH Volume      : {res['PVWMH_cc']:.3f} cc")
        print(f"DWMH Volume       : {res['DWMH_cc']:.3f} cc")
        print(f"Predicted WMH Age : {res['Predicted_Age']:.2f} years old")
        print(f"Saved results to  : {csv_out}")
        print("-------------------------------------------\n")
        return res

    # Verify FSL environment
    check_fsl_installed()

    # 2. DICOM to NIfTI conversion if needed
    if dcm_dir:
        print(f"Converting DICOMs from: {dcm_dir}")
        t1_dcm, flair_dcm = find_dicom_subfolders(dcm_dir)
        t1_conv_dir = os.path.join(outdir, "dcm_t1")
        flair_conv_dir = os.path.join(outdir, "dcm_flair")
        t1_path = dcm2nii(t1_dcm, t1_conv_dir, prefix="T1w")
        flair_path = dcm2nii(flair_dcm, flair_conv_dir, prefix="T2f")

    if not t1_path or not flair_path:
        raise ValueError("Both T1 and FLAIR image paths must be provided.")

    if not subject_id:
        subject_id = os.path.basename(t1_path).split(".")[0]

    # Clean copy to outdir to avoid FSL basename collisions (e.g. .nii vs .nii.gz)
    import shutil
    clean_t1 = os.path.join(outdir, "T1_raw.nii.gz")
    clean_flair = os.path.join(outdir, "FLAIR_raw.nii.gz")

    if not os.path.exists(clean_t1):
        t1_img = nib.load(t1_path)
        nib.save(t1_img, clean_t1)
    if not os.path.exists(clean_flair):
        flair_img = nib.load(flair_path)
        nib.save(flair_img, clean_flair)

    t1_path = clean_t1
    flair_path = clean_flair

    # 3. Coregistration: FLAIR to T1 (FLIRT 6 DOF)
    print("\n[Step 1/5] Coregistering FLAIR to T1 space...")
    flair_coreg = os.path.join(outdir, "rmFLAIR.nii.gz")
    flair2t1_mat = os.path.join(outdir, "flair2t1.mat")
    coregister_flair_to_t1(flair_path, t1_path, flair_coreg, flair2t1_mat)

    # 4. Brain Extraction (BET)
    print("\n[Step 2/5] Performing T1 Skull-stripping (BET)...")
    t1_brain = os.path.join(outdir, "T1_brain.nii.gz")
    brain_extract(t1_path, t1_brain)

    # 5. Tissue Segmentation (FAST)
    print("\n[Step 3/5] Segmenting T1 brain tissues (FAST)...")
    fast_base = os.path.join(outdir, "fast", "T1")
    pve_maps = tissue_segment_fast(t1_brain, fast_base)

    # 6. Spatial Alignment with MNI Space & Warp LST Priors
    print("\n[Step 4/5] Aligning with MNI standard space to project LST White Matter & Exclusion Priors...")
    reg_dir = os.path.join(outdir, "registration")
    reg_info = register_t1_to_mni(
        t1_path=t1_path,
        t1_brain_path=t1_brain,
        out_dir=reg_dir,
        mni_resolution=1.0,
        non_linear=nonlinear
    )

    script_dir = os.path.dirname(os.path.abspath(__file__))
    atlas_wm_mni = os.path.join(script_dir, "WMH_Atlas", "atlas_wm.nii.gz")
    noles_mni = os.path.join(script_dir, "WMH_Atlas", "noles.nii.gz")
    atlas_mni_pv = os.path.join(script_dir, "WMH_Atlas", "1mm", "1_PVWMH_1-10mm.nii.gz")
    atlas_mni_dw = os.path.join(script_dir, "WMH_Atlas", "1mm", "2_DWMH_10mmup.nii.gz")

    if nonlinear and "inverse_warp" in reg_info:
        warp_arg = reg_info["inverse_warp"]
        is_nl = True
    else:
        from pywmh.fsl_engine import run_command
        inv_affine = os.path.join(reg_dir, "mni_to_t1_affine.mat")
        run_command(["convert_xfm", "-omat", inv_affine, "-inverse", reg_info["affine_mat"]], "Inverting affine matrix")
        warp_arg = inv_affine
        is_nl = False

    native_atlas_wm_path = os.path.join(outdir, "watlas_wm.nii.gz")
    native_noles_path = os.path.join(outdir, "wnoles.nii.gz")
    native_pv_path = os.path.join(outdir, "native_PVWMH.nii.gz")
    native_dw_path = os.path.join(outdir, "native_DWMH.nii.gz")

    warp_atlas_to_native(atlas_wm_mni, t1_path, warp_arg, native_atlas_wm_path, is_nonlinear=is_nl)
    warp_atlas_to_native(noles_mni, t1_path, warp_arg, native_noles_path, is_nonlinear=is_nl)
    warp_atlas_to_native(atlas_mni_pv, t1_path, warp_arg, native_pv_path, is_nonlinear=is_nl)
    warp_atlas_to_native(atlas_mni_dw, t1_path, warp_arg, native_dw_path, is_nonlinear=is_nl)

    # 7. WMH Segmentation via Faithful Python LST-LGA
    print(f"\n[Step 5/5] Running Official Python LST-LGA WMH Segmentation (kappa={kappa:.2f})...")
    flair_img = nib.load(flair_coreg)
    flair_data = flair_img.get_fdata()

    gm_data = nib.load(pve_maps["pve_gm"]).get_fdata()
    wm_data = nib.load(pve_maps["pve_wm"]).get_fdata()
    csf_data = nib.load(pve_maps["pve_csf"]).get_fdata()
    brain_mask_data = nib.load(os.path.join(outdir, "T1_brain_mask.nii.gz")).get_fdata() > 0

    watlas_wm_data = nib.load(native_atlas_wm_path).get_fdata()
    wnoles_data = nib.load(native_noles_path).get_fdata() > 0

    prob_map, binary_mask = segment_wmh_lga(
        flair=flair_data,
        pve_gm=gm_data,
        pve_wm=wm_data,
        pve_csf=csf_data,
        brain_mask=brain_mask_data,
        atlas_wm=watlas_wm_data,
        noles_mask=wnoles_data,
        kappa=kappa,
        verbose=True
    )

    # Save native lesion maps
    ples_native_path = os.path.join(outdir, f"ples_lga_k{int(kappa*100):02d}.nii.gz")
    bles_native_path = os.path.join(outdir, f"bles_lga_k{int(kappa*100):02d}.nii.gz")

    nib.save(nib.Nifti1Image(prob_map, flair_img.affine, flair_img.header), ples_native_path)
    nib.save(nib.Nifti1Image(binary_mask, flair_img.affine, flair_img.header), bles_native_path)
    print(f"Saved native lesion probability map: {ples_native_path}")

    # Optional: Also produce modulated MNI lesion map m0wples (for SPM equivalence)
    if save_mni_lesion and nonlinear and "forward_warp" in reg_info:
        m0wples_path = os.path.join(outdir, "m0wples.nii.gz")
        warp_lesion_to_mni_modulated(ples_native_path, reg_info["forward_warp"], reg_info["mni_head"], m0wples_path)
        print(f"Saved MNI-modulated lesion map: {m0wples_path}")

    # Calculate PVWMH, DWMH, and WMH-Age in native space
    lesion_img = nib.load(ples_native_path)
    pv_mask_img = nib.load(native_pv_path)
    dw_mask_img = nib.load(native_dw_path)

    results = calculate_wmh_volumes_and_age(
        lesion_img=lesion_img,
        pv_mask_img=pv_mask_img,
        dw_mask_img=dw_mask_img,
        subject_id=subject_id
    )

    csv_path = os.path.join(outdir, "WMH_Age_Results.csv")
    json_path = os.path.join(outdir, "WMH_Age_Results.json")
    save_results(results, csv_path, json_path)

    elapsed = time.time() - start_time

    print("\n==================== SUMMARY RESULTS ====================")
    print(f" Subject ID       : {results['SubjectID']}")
    print(f" Total WMH Volume : {results['Total_WMH_cc']:.3f} cc")
    print(f" PVWMH Volume     : {results['PVWMH_cc']:.3f} cc")
    print(f" DWMH Volume      : {results['DWMH_cc']:.3f} cc")
    print(f" Predicted WMH-Age: {results['Predicted_Age']:.2f} years old")
    print(f" Output Directory : {outdir}")
    print(f" Total Time       : {elapsed:.1f} seconds")
    print("=========================================================\n")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="PyWMH: Automated Python Pipeline for WMH Segmentation and Age Estimation"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--t1", help="Path to T1-weighted NIfTI image (requires --flair)")
    group.add_argument("--dcm", help="Path to DICOM directory containing T1 and FLAIR")
    group.add_argument("--lesion", help="Path to existing MNI lesion map (m0wples*.nii)")

    parser.add_argument("--flair", help="Path to FLAIR NIfTI image (required if --t1 is given)")
    parser.add_argument("-o", "--outdir", default="./wmh_age_output", help="Output directory")
    parser.add_argument("-k", "--kappa", type=float, default=0.3, help="LGA kappa threshold parameter [default=0.3]")
    parser.add_argument("-r", "--res", type=float, default=1.0, choices=[1.0, 1.5], help="Atlas resolution [default=1.0]")
    parser.add_argument("--nonlinear", action="store_true", help="Use FNIRT non-linear registration instead of linear FLIRT")
    parser.add_argument("--save-mni-lesion", action="store_true", help="Save modulated MNI lesion map (m0wples)")
    parser.add_argument("--id", help="Subject ID")

    args = parser.parse_args()

    if args.t1 and not args.flair:
        parser.error("--flair is required when --t1 is specified.")

    run_pipeline(
        t1_path=args.t1,
        flair_path=args.flair,
        dcm_dir=args.dcm,
        lesion_path=args.lesion,
        outdir=args.outdir,
        kappa=args.kappa,
        voxel_size=args.res,
        nonlinear=args.nonlinear,
        save_mni_lesion=args.save_mni_lesion,
        subject_id=args.id
    )


if __name__ == "__main__":
    main()
