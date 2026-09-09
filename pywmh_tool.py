#!/usr/bin/env python3
"""
pywmh_tool.py - Complete Lightweight WMH-Age Estimation Pipeline (Pure Python + FSL)

A MATLAB-free, SPM-free, fully automated tool for:
1. FLAIR to T1 rigid coregistration (FLIRT)
2. T1 brain extraction (BET) and tissue segmentation (FAST)
3. Lesion Growth Algorithm (LST-LGA) in pure Python
4. Spatial alignment with Periventricular, Deep, Lobar, and Arterial Vascular Atlases
5. WMH Volume quantification & Age prediction (Huang et al., Age Ageing 2022)
6. Automated Visual Quality Control Report (HTML/PNG)

Usage Examples:
    # 1. From NIfTI files with patient's actual age:
    python3 pywmh_tool.py --t1 sub-01_T1w.nii.gz --flair sub-01_FLAIR.nii.gz --age 65.0 --outdir ./output_sub01

    # 2. From DICOM folder:
    python3 pywmh_tool.py --dcm /path/to/dicoms --outdir ./output_sub01

    # 3. Using existing lesion map:
    python3 pywmh_tool.py --lesion /path/to/m0wples_T2f.nii --age 68.5 --outdir ./output_sub01
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
from pywmh.age_model import (
    calculate_wmh_volumes_and_age,
    calculate_regional_breakdown,
    save_results
)
from pywmh.qc_report import generate_qc_report


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
    subject_id: str = None,
    chronological_age: float = None,
    skip_qc: bool = False,
    force: bool = False
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
        lobar_atlas = os.path.join(script_dir, "WMH_Atlas", "lobar_atlas.nii.gz")
        arterial_atlas = os.path.join(script_dir, "WMH_Atlas", "arterial_atlas_level2.nii.gz")

        lesion_img = nib.load(lesion_path)
        pv_img = nib.load(pv_atlas)
        dw_img = nib.load(dw_atlas)

        subj = subject_id or os.path.basename(lesion_path).split(".")[0]
        res = calculate_wmh_volumes_and_age(
            lesion_img=lesion_img,
            pv_mask_img=pv_img,
            dw_mask_img=dw_img,
            subject_id=subj,
            chronological_age=chronological_age
        )

        lobar_img = nib.load(lobar_atlas) if os.path.exists(lobar_atlas) else None
        art_img = nib.load(arterial_atlas) if os.path.exists(arterial_atlas) else None
        regional_res = calculate_regional_breakdown(lesion_img, lobar_img, art_img)

        # Merge for export
        combined_res = {**res, **regional_res}
        csv_out = os.path.join(outdir, "WMH_Age_Results.csv")
        save_results(combined_res, csv_out, os.path.join(outdir, "WMH_Age_Results.json"))

        print("\n----------------- Results -----------------")
        print(f"Total WMH Volume  : {res['Total_WMH_cc']:.3f} cc")
        print(f"PVWMH Volume      : {res['PVWMH_cc']:.3f} cc")
        print(f"DWMH Volume       : {res['DWMH_cc']:.3f} cc")
        print(f"Predicted WMH Age : {res['Predicted_Age']:.2f} years old")
        if chronological_age is not None:
            print(f"Chronological Age : {res['Chronological_Age']:.2f} years old")
            print(f"Brain Age Gap     : {res['Brain_Age_Gap']:+.2f} years")
        print(f"Saved results to  : {csv_out}")
        print("-------------------------------------------\n")
        return combined_res

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
    if not force and os.path.exists(flair_coreg) and os.path.exists(flair2t1_mat):
        print("  -> Existing coregistration found. Skipping FLIRT.")
    else:
        coregister_flair_to_t1(flair_path, t1_path, flair_coreg, flair2t1_mat)

    # 4. Brain Extraction (BET)
    print("\n[Step 2/5] Performing T1 Skull-stripping (BET)...")
    t1_brain = os.path.join(outdir, "T1_brain.nii.gz")
    t1_brain_mask = os.path.join(outdir, "T1_brain_mask.nii.gz")
    if not force and os.path.exists(t1_brain) and os.path.exists(t1_brain_mask):
        print("  -> Existing brain extraction found. Skipping BET.")
    else:
        brain_extract(t1_path, t1_brain)

    # 5. Tissue Segmentation (FAST)
    print("\n[Step 3/5] Segmenting T1 brain tissues (FAST)...")
    fast_base = os.path.join(outdir, "fast", "T1")
    pve_files = {
        "pve_csf": f"{fast_base}_pve_0.nii.gz",
        "pve_gm": f"{fast_base}_pve_1.nii.gz",
        "pve_wm": f"{fast_base}_pve_2.nii.gz"
    }
    if not force and all(os.path.exists(p) for p in pve_files.values()):
        print("  -> Existing tissue segmentation found. Skipping FAST.")
        pve_maps = pve_files
    else:
        pve_maps = tissue_segment_fast(t1_brain, fast_base)

    # 6. Spatial Alignment with MNI Space & Warp Spatial Priors
    print("\n[Step 4/5] Aligning with MNI standard space to project Spatial Priors...")
    reg_dir = os.path.join(outdir, "registration")
    affine_mat = os.path.join(reg_dir, "t1_to_mni_affine.mat")

    if not force and os.path.exists(affine_mat):
        print("  -> Existing MNI registration found. Skipping FLIRT/FNIRT.")
        reg_info = {"affine_mat": affine_mat}
    else:
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
    atlas_mni_lobar = os.path.join(script_dir, "WMH_Atlas", "lobar_atlas.nii.gz")
    atlas_mni_arterial = os.path.join(script_dir, "WMH_Atlas", "arterial_atlas_level2.nii.gz")

    if nonlinear and "inverse_warp" in reg_info:
        warp_arg = reg_info["inverse_warp"]
        is_nl = True
    else:
        from pywmh.fsl_engine import run_command
        inv_affine = os.path.join(reg_dir, "mni_to_t1_affine.mat")
        if not os.path.exists(inv_affine):
            run_command(["convert_xfm", "-omat", inv_affine, "-inverse", reg_info["affine_mat"]], "Inverting affine matrix")
        warp_arg = inv_affine
        is_nl = False

    native_atlas_wm_path = os.path.join(outdir, "watlas_wm.nii.gz")
    native_noles_path = os.path.join(outdir, "wnoles.nii.gz")
    native_pv_path = os.path.join(outdir, "native_PVWMH.nii.gz")
    native_dw_path = os.path.join(outdir, "native_DWMH.nii.gz")
    native_lobar_path = os.path.join(outdir, "native_lobar.nii.gz")
    native_arterial_path = os.path.join(outdir, "native_arterial.nii.gz")

    for src_atlas, dst_native in [
        (atlas_wm_mni, native_atlas_wm_path),
        (noles_mni, native_noles_path),
        (atlas_mni_pv, native_pv_path),
        (atlas_mni_dw, native_dw_path),
        (atlas_mni_lobar, native_lobar_path),
        (atlas_mni_arterial, native_arterial_path)
    ]:
        if os.path.exists(src_atlas):
            if not force and os.path.exists(dst_native):
                continue
            warp_atlas_to_native(src_atlas, t1_path, warp_arg, dst_native, is_nonlinear=is_nl)

    # 7. WMH Segmentation via Faithful Python LST-LGA
    print(f"\n[Step 5/5] Running Official Python LST-LGA WMH Segmentation (kappa={kappa:.2f})...")
    flair_img = nib.load(flair_coreg)
    flair_data = flair_img.get_fdata()

    gm_img = nib.load(pve_maps["pve_gm"])
    gm_data = gm_img.get_fdata()
    wm_data = nib.load(pve_maps["pve_wm"]).get_fdata()
    csf_data = nib.load(pve_maps["pve_csf"]).get_fdata()
    brain_mask_data = nib.load(t1_brain_mask).get_fdata() > 0

    # Calculate Total Intracranial Volume (TIV) from FAST PVE maps
    voxel_vol_cc = float(np.prod(gm_img.header.get_zooms()[:3])) / 1000.0
    tiv_cc = float(np.sum(gm_data + wm_data + csf_data)) * voxel_vol_cc

    watlas_wm_data = nib.load(native_atlas_wm_path).get_fdata()
    wnoles_data = nib.load(native_noles_path).get_fdata() > 0

    ples_native_path = os.path.join(outdir, f"ples_lga_k{int(kappa*100):02d}.nii.gz")
    bles_native_path = os.path.join(outdir, f"bles_lga_k{int(kappa*100):02d}.nii.gz")

    if not force and os.path.exists(ples_native_path) and os.path.exists(bles_native_path):
        print("  -> Existing lesion maps found. Skipping LGA iterations.")
    else:
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
        subject_id=subject_id,
        chronological_age=chronological_age,
        tiv_cc=tiv_cc
    )

    # Calculate Regional Lobar & Arterial Breakdown
    lobar_img = nib.load(native_lobar_path) if os.path.exists(native_lobar_path) else None
    art_img = nib.load(native_arterial_path) if os.path.exists(native_arterial_path) else None
    regional_results = calculate_regional_breakdown(lesion_img, lobar_img, art_img)

    combined_results = {**results, **regional_results}

    csv_path = os.path.join(outdir, "WMH_Age_Results.csv")
    json_path = os.path.join(outdir, "WMH_Age_Results.json")
    save_results(combined_results, csv_path, json_path)

    # Generate Visual Quality Control (QC) Report
    if not skip_qc:
        print("\n[QC Report] Generating visual inspection report and summary charts...")
        qc_html = os.path.join(outdir, "WMH_Age_QC_Report.html")
        qc_png = os.path.join(outdir, "qc_summary.png")
        generate_qc_report(
            flair_path=flair_coreg,
            t1_path=t1_brain,
            lesion_prob_path=ples_native_path,
            pv_mask_path=native_pv_path,
            dw_mask_path=native_dw_path,
            results=results,
            regional_results=regional_results,
            out_html=qc_html,
            out_png=qc_png
        )

    elapsed = time.time() - start_time

    print("\n==================== SUMMARY RESULTS ====================")
    print(f" Subject ID       : {results['SubjectID']}")
    print(f" Total WMH Volume : {results['Total_WMH_cc']:.3f} cc ({results.get('Total_WMH_pct_TIV', 0):.3f}% TIV)")
    print(f" PVWMH Volume     : {results['PVWMH_cc']:.3f} cc")
    print(f" DWMH Volume      : {results['DWMH_cc']:.3f} cc")
    print(f" Predicted WMH-Age: {results['Predicted_Age']:.2f} years old")
    if chronological_age is not None:
        print(f" Chronological Age: {results['Chronological_Age']:.2f} years old")
        print(f" Brain Age Gap    : {results['Brain_Age_Gap']:+.2f} years")
    if regional_results:
        print(" Lobar Breakdown  :")
        for l in ["Frontal", "Parietal", "Temporal", "Occipital", "Subcortical"]:
            vol = regional_results.get(f"Lobar_{l}_Total_cc", 0.0)
            pct = regional_results.get(f"Lobar_{l}_Total_pct", 0.0)
            print(f"   - {l:<12}: {vol:.3f} cc ({pct:.1f}%)")
        print(" Arterial Breakdown:")
        for a in ["MCA", "ACA", "PCA", "VB"]:
            vol = regional_results.get(f"Arterial_{a}_Total_cc", 0.0)
            pct = regional_results.get(f"Arterial_{a}_Total_pct", 0.0)
            print(f"   - {a:<12}: {vol:.3f} cc ({pct:.1f}%)")
    print(f" Output Directory : {outdir}")
    print(f" Total Time       : {elapsed:.1f} seconds")
    print("=========================================================\n")

    return combined_results


def main():
    parser = argparse.ArgumentParser(
        description="PyWMH: Automated Python Pipeline for WMH Segmentation, Regional Parcellation, and Age Estimation"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--t1", help="Path to T1-weighted NIfTI image (requires --flair)")
    group.add_argument("--dcm", help="Path to DICOM directory containing T1 and FLAIR")
    group.add_argument("--lesion", help="Path to existing MNI lesion map (m0wples*.nii)")

    parser.add_argument("--flair", help="Path to FLAIR NIfTI image (required if --t1 is given)")
    parser.add_argument("-o", "--outdir", default="./wmh_age_output", help="Output directory")
    parser.add_argument("-k", "--kappa", type=float, default=0.3, help="LGA kappa threshold parameter [default=0.3]")
    parser.add_argument("-r", "--res", type=float, default=1.0, choices=[1.0, 1.5], help="Atlas resolution [default=1.0]")
    parser.add_argument("--age", type=float, default=None, help="Subject's actual chronological age in years (calculates Brain Age Gap)")
    parser.add_argument("--skip-qc", action="store_true", help="Skip generating visual QC report (HTML/PNG)")
    parser.add_argument("--force", action="store_true", help="Force re-run even if intermediate files exist")
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
        subject_id=args.id,
        chronological_age=args.age,
        skip_qc=args.skip_qc,
        force=args.force
    )


if __name__ == "__main__":
    main()
