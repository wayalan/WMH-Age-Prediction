"""FLAIR intensity correction before native LGA (no transform estimation)."""

import hashlib
import json
import os
from pathlib import Path

import nibabel as nib
import numpy as np

from .fsl_engine import run_command


FLAIR_CORRECTION_VERSION = "fast-flair-bias-1"


def _digest(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _save_float32(data, reference, path):
    image = nib.Nifti1Image(data.astype(np.float32), reference.affine, reference.header)
    image.set_data_dtype(np.float32)
    nib.save(image, path)


def correct_flair_bias(flair_path, t1_path, brain_mask_path, flair2t1_mat, outdir, force=False):
    """Estimate a FAST T2 bias field on acquisition-space FLAIR, then resample.

    The existing rigid transform only projects the T1 brain mask to FLAIR and
    applies the corrected image to T1. Neither rigid nor MNI registration is
    estimated here. Tissue labels and lesion maps are not used as references.
    FAST defaults match the pipeline's three-class settings. This substitutes
    for the previously missing FLAIR correction; it does not reproduce SPM's
    unified segmentation bias estimator exactly.
    """
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    inputs = {"flair": flair_path, "t1": t1_path, "brain_mask": brain_mask_path,
              "rigid_transform": flair2t1_mat}
    fsl_version = Path(os.environ.get("FSLDIR", "/usr/local/fsl")) / "etc/fslversion"
    signature = {"method": FLAIR_CORRECTION_VERSION, "image_type": 2, "classes": 3,
                 "H": 0.1, "I": 4, "interpolation": "spline",
                 "fsl_version": fsl_version.read_text().strip() if fsl_version.exists() else "unknown",
                 "inputs": {name: _digest(path) for name, path in inputs.items()}}
    outputs = {"acquisition": out / "FLAIR_corrected.nii.gz",
               "native": out / "rmFLAIR_corrected.nii.gz"}
    metadata = out / "correction_inputs.json"
    if not force:
        try:
            cached = json.loads(metadata.read_text())
            if cached["signature"] == signature and all(
                    _digest(path) == cached["outputs"][name] for name, path in outputs.items()):
                return str(outputs["native"])
        except (OSError, ValueError, KeyError, TypeError):
            pass

    # Fail early rather than writing a cache that appears to be corrected.
    flair = nib.load(flair_path)
    data = flair.get_fdata()
    if data.ndim != 3 or not np.all(np.isfinite(data)):
        raise ValueError("FLAIR correction requires a finite 3D image.")
    inverse = out / "t1_to_flair.mat"
    mask_path = out / "brain_mask_flair.nii.gz"
    run_command(["convert_xfm", "-omat", str(inverse), "-inverse", str(flair2t1_mat)],
                "inverting existing FLAIR transform")
    run_command(["flirt", "-in", str(brain_mask_path), "-ref", str(flair_path),
                 "-applyxfm", "-init", str(inverse), "-interp", "nearestneighbour",
                 "-out", str(mask_path)], "projecting brain mask to FLAIR")
    mask_image = nib.load(mask_path)
    if mask_image.shape != flair.shape or not np.allclose(mask_image.affine, flair.affine, atol=1e-4, rtol=0):
        raise ValueError("Projected FLAIR brain mask has the wrong voxel grid.")
    mask = mask_image.get_fdata() > 0
    if not np.any(mask & (data > 0)):
        raise ValueError("Projected brain mask contains no positive FLAIR samples.")
    brain_path = out / "FLAIR_brain.nii.gz"
    _save_float32(np.where(mask, data, 0), flair, brain_path)
    prefix = out / "FLAIR"
    run_command(["fast", "-t", "2", "-n", "3", "-H", "0.1", "-I", "4", "-b",
                 "-o", str(prefix), str(brain_path)], "estimating acquisition-space FLAIR bias")
    bias_image = nib.load(out / "FLAIR_bias.nii.gz")
    bias = bias_image.get_fdata()
    if (bias_image.shape != flair.shape
            or not np.allclose(bias_image.affine, flair.affine, atol=1e-4, rtol=0)
            or not np.all(np.isfinite(bias)) or np.any(bias <= 0)):
        raise ValueError("FAST FLAIR bias field must be finite, positive and on the acquisition grid.")
    _save_float32(data / bias, flair, outputs["acquisition"])
    run_command(["flirt", "-in", str(outputs["acquisition"]), "-ref", str(t1_path),
                 "-applyxfm", "-init", str(flair2t1_mat), "-interp", "spline",
                 "-out", str(outputs["native"])], "applying existing transform to corrected FLAIR")
    corrected = nib.load(outputs["native"])
    t1 = nib.load(t1_path)
    if (corrected.shape != t1.shape or not np.allclose(corrected.affine, t1.affine, atol=1e-4, rtol=0)
            or not np.all(np.isfinite(corrected.get_fdata()))):
        raise ValueError("Corrected FLAIR must be finite and on the native T1 grid.")
    metadata.write_text(json.dumps({"signature": signature,
        "outputs": {name: _digest(path) for name, path in outputs.items()}}, indent=2) + "\n")
    return str(outputs["native"])
