"""
pywmh.fsl_engine - FSL Preprocessing, Registration & Transformation Helpers

Provides Python interfaces to FSL (FLIRT, FNIRT, FAST, BET, INVWARP, APPLYWARP)
and dcm2niix for end-to-end neuroimaging workflows.
"""

import os
import shutil
import subprocess
from typing import Dict, Tuple, Optional


def check_fsl_installed() -> str:
    """Verifies that FSL is available on the system."""
    fsl_dir = os.environ.get("FSLDIR", "/usr/local/fsl")
    flirt_bin = shutil.which("flirt") or os.path.join(fsl_dir, "share", "fsl", "bin", "flirt")
    if not os.path.exists(flirt_bin) and not shutil.which("flirt"):
        raise EnvironmentError(
            f"FSL does not appear to be installed or FSLDIR is not configured properly. Checked: {fsl_dir}"
        )
    return fsl_dir


def run_command(cmd_args, description="Running command"):
    """Runs a shell command and raises an informative error if it fails."""
    cmd_str = " ".join(cmd_args)
    proc = subprocess.run(cmd_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Failed {description}:\nCommand: {cmd_str}\nStderr: {proc.stderr}\nStdout: {proc.stdout}")
    return proc.stdout


def dcm2nii(dcm_dir: str, out_dir: str, prefix: str = "img") -> str:
    """Converts a DICOM directory to NIfTI using dcm2niix."""
    os.makedirs(out_dir, exist_ok=True)
    dcm2niix_bin = shutil.which("dcm2niix") or "/usr/local/fsl/bin/dcm2niix"
    cmd = [dcm2niix_bin, "-z", "y", "-f", f"{prefix}_%p_%s", "-o", out_dir, dcm_dir]
    run_command(cmd, f"dcm2niix conversion for {dcm_dir}")
    
    # Locate generated NIfTI
    nii_files = [os.path.join(out_dir, f) for f in os.listdir(out_dir) if f.endswith(".nii") or f.endswith(".nii.gz")]
    if not nii_files:
        raise FileNotFoundError(f"No NIfTI files generated in {out_dir}")
    return nii_files[0]


def coregister_flair_to_t1(
    flair_path: str,
    t1_path: str,
    out_flair_path: str,
    out_mat_path: str
) -> Tuple[str, str]:
    """
    Rigid-body coregisters FLAIR to T1 space using FSL FLIRT (6 DOF).
    Equivalent to SPM coregistration.
    """
    os.makedirs(os.path.dirname(os.path.abspath(out_flair_path)), exist_ok=True)
    cmd = [
        "flirt",
        "-in", flair_path,
        "-ref", t1_path,
        "-dof", "6",
        "-cost", "corratio",
        "-out", out_flair_path,
        "-omat", out_mat_path,
        "-interp", "spline"
    ]
    run_command(cmd, "FLAIR to T1 rigid coregistration (FLIRT)")
    return out_flair_path, out_mat_path


def brain_extract(
    t1_path: str,
    out_brain_path: str,
    frac: float = 0.45
) -> Tuple[str, str]:
    """
    Performs skull-stripping on T1 using FSL BET.
    """
    out_base = out_brain_path.replace(".nii.gz", "").replace(".nii", "")
    cmd = ["bet", t1_path, out_base, "-f", str(frac), "-m"]
    run_command(cmd, "T1 brain extraction (BET)")
    mask_path = f"{out_base}_mask.nii.gz"
    brain_path = f"{out_base}.nii.gz"
    return brain_path, mask_path


def tissue_segment_fast(
    t1_brain_path: str,
    out_base: str
) -> Dict[str, str]:
    """
    Performs 3-class tissue segmentation (GM, WM, CSF) on skull-stripped T1 using FSL FAST.
    Returns dictionary with paths to PVE (Partial Volume Estimate) maps.
    """
    os.makedirs(os.path.dirname(os.path.abspath(out_base)), exist_ok=True)
    cmd = [
        "fast",
        "-t", "1",      # T1-weighted
        "-n", "3",      # 3 classes: CSF (0), GM (1), WM (2)
        "-H", "0.1",    # Spatial smoothness prior
        "-I", "4",      # Number of main-loop iterations
        "-p",           # Output partial volume images
        "-o", out_base,
        t1_brain_path
    ]
    run_command(cmd, "T1 tissue segmentation (FAST)")

    pve_csf = f"{out_base}_pve_0.nii.gz"
    pve_gm  = f"{out_base}_pve_1.nii.gz"
    pve_wm  = f"{out_base}_pve_2.nii.gz"

    for p in [pve_csf, pve_gm, pve_wm]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Expected FAST output missing: {p}")

    return {
        "pve_csf": pve_csf,
        "pve_gm": pve_gm,
        "pve_wm": pve_wm
    }


def register_t1_to_mni(
    t1_path: str,
    t1_brain_path: str,
    out_dir: str,
    mni_resolution: float = 1.0,
    non_linear: bool = True
) -> Dict[str, str]:
    """
    Registers T1 to MNI152 standard space using FLIRT (affine) and FNIRT (non-linear).
    Generates forward warp and inverse warp.
    """
    os.makedirs(out_dir, exist_ok=True)
    fsl_dir = os.environ.get("FSLDIR", "/usr/local/fsl")
    
    res_str = "1mm" if mni_resolution == 1.0 else "2mm"
    mni_brain = os.path.join(fsl_dir, "data", "standard", f"MNI152_T1_{res_str}_brain.nii.gz")
    mni_head  = os.path.join(fsl_dir, "data", "standard", f"MNI152_T1_{res_str}.nii.gz")

    affine_mat = os.path.join(out_dir, "t1_to_mni_affine.mat")
    cmd_flirt = [
        "flirt",
        "-in", t1_brain_path,
        "-ref", mni_brain,
        "-omat", affine_mat,
        "-dof", "12"
    ]
    run_command(cmd_flirt, "T1 to MNI linear affine registration (FLIRT)")

    results = {
        "affine_mat": affine_mat,
        "mni_head": mni_head,
        "mni_brain": mni_brain
    }

    if non_linear:
        warp_coef = os.path.join(out_dir, "t1_to_mni_warp.nii.gz")
        inv_warp = os.path.join(out_dir, "mni_to_t1_warp.nii.gz")

        # FNIRT non-linear registration
        cmd_fnirt = [
            "fnirt",
            f"--in={t1_path}",
            f"--aff={affine_mat}",
            f"--cout={warp_coef}",
            "--config=T1_2_MNI152_2mm"
        ]
        run_command(cmd_fnirt, "T1 to MNI non-linear warp (FNIRT)")

        # Compute inverse warp (MNI -> Native)
        cmd_inv = [
            "invwarp",
            f"--warp={warp_coef}",
            f"--out={inv_warp}",
            f"--ref={t1_path}"
        ]
        run_command(cmd_inv, "Inverting MNI warp (INVWARP)")

        results["forward_warp"] = warp_coef
        results["inverse_warp"] = inv_warp

    return results


def warp_atlas_to_native(
    atlas_mni_path: str,
    t1_ref_path: str,
    warp_or_mat: str,
    out_native_path: str,
    is_nonlinear: bool = True
) -> str:
    """
    Transforms an MNI Atlas (such as PVWMH / DWMH masks) back into Native T1 space.
    Uses Nearest Neighbour interpolation to preserve integer mask labels.
    """
    os.makedirs(os.path.dirname(os.path.abspath(out_native_path)), exist_ok=True)
    if is_nonlinear:
        cmd = [
            "applywarp",
            f"--in={atlas_mni_path}",
            f"--ref={t1_ref_path}",
            f"--warp={warp_or_mat}",
            "--interp=nn",
            f"--out={out_native_path}"
        ]
    else:
        cmd = [
            "flirt",
            "-in", atlas_mni_path,
            "-ref", t1_ref_path,
            "-applyxfm",
            "-init", warp_or_mat,
            "-interp", "nearestneighbour",
            "-out", out_native_path
        ]
    run_command(cmd, f"Warping {os.path.basename(atlas_mni_path)} to native space")
    return out_native_path


def warp_lesion_to_mni_modulated(
    lesion_native_path: str,
    t1_to_mni_warp: str,
    mni_ref_path: str,
    out_mni_path: str
) -> str:
    """
    Warps a native lesion map to MNI space WITH Jacobian volume modulation.
    Equivalent to SPM/CAT12's m0wples*.nii.
    """
    os.makedirs(os.path.dirname(os.path.abspath(out_mni_path)), exist_ok=True)
    cmd = [
        "applywarp",
        f"--in={lesion_native_path}",
        f"--ref={mni_ref_path}",
        f"--warp={t1_to_mni_warp}",
        "--jacobian",               # Jacobian modulation for volume preservation!
        "--interp=trilinear",
        f"--out={out_mni_path}"
    ]
    run_command(cmd, "Warping native lesion map to MNI with Jacobian modulation")
    return out_mni_path
