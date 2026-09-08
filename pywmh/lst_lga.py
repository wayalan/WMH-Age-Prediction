"""
pywmh.lst_lga - Faithful Pure-Python Implementation of SPM LST-LGA

Directly replicates the mathematical formulation of:
    Schmidt P, Gaser C, Arsic M, et al. An automated tool for detection of
    FLAIR-hyperintense white-matter lesions in Multiple Sclerosis.
    NeuroImage. 2012;59(4):3774-3783.
and the official SPM LST toolbox ps_LST_lga.m.
"""

import os
import numpy as np
from scipy import ndimage
from scipy.stats import gamma, norm


def segment_wmh_lga(
    flair: np.ndarray,
    pve_gm: np.ndarray,
    pve_wm: np.ndarray,
    pve_csf: np.ndarray,
    brain_mask: np.ndarray = None,
    atlas_wm: np.ndarray = None,
    noles_mask: np.ndarray = None,
    kappa: float = 0.3,
    max_iter: int = 35,
    phi: float = 1.0,
    verbose: bool = True
):
    """
    Faithfully replicates the official SPM LST-LGA (Lesion Growth Algorithm).

    Parameters:
    -----------
    flair : np.ndarray (3D float)
        FLAIR image coregistered to T1.
    pve_gm, pve_wm, pve_csf : np.ndarray (3D float)
        Partial volume estimates (CSF, GM, WM) from FAST or SPM.
    brain_mask : np.ndarray (3D bool)
        Brain extraction mask.
    atlas_wm : np.ndarray (3D float)
        Native-space warped LST white matter tract prior (atlas_wm.nii).
    noles_mask : np.ndarray (3D bool)
        Native-space warped non-lesion exclusion mask (cerebellum, brainstem, ventricles).
    kappa : float
        Initial threshold parameter (default 0.3).
    max_iter : int
        Maximum iterations for lesion region growing (default 35).
    phi : float
        MRF weight parameter (default 1.0).

    Returns:
    --------
    prob_map : np.ndarray (3D float)
        Continuous lesion probability map in [0, 1].
    binary_mask : np.ndarray (3D uint8)
        Binary lesion mask.
    """
    if verbose:
        print(f"[PyWMH LST-LGA] Initializing official LST-LGA pipeline (kappa = {kappa:.2f})...")

    if brain_mask is None:
        brain_mask = (pve_gm + pve_wm + pve_csf) > 0.1

    if atlas_wm is None:
        atlas_wm = pve_wm.copy()

    if noles_mask is None:
        noles_mask = np.zeros_like(brain_mask, dtype=bool)

    # 1. Clean spatial priors
    atlas_wm = np.copy(atlas_wm)
    atlas_wm[noles_mask] = 0.0
    atlas_wm[~brain_mask] = 0.0
    atlas_wm = np.clip(atlas_wm, 0.0, 1.0)

    # 2. Hard segmentation & PVE label (p0)
    prob = np.stack([pve_csf, pve_gm, pve_wm], axis=-1)
    sum_prob = np.sum(prob, axis=-1, keepdims=True)
    sum_prob[sum_prob == 0] = 1.0
    prob = prob / sum_prob

    seg = np.argmax(prob, axis=-1) + 1  # 1: CSF, 2: GM, 3: WM
    seg[~brain_mask] = 0

    p0 = np.zeros_like(flair, dtype=np.float32)
    p0[seg == 1] = 1.0 + 0.5 * pve_csf[seg == 1]
    p0[seg == 2] = 1.5 + 1.0 * pve_gm[seg == 2]
    p0[seg == 3] = 2.5 + 0.5 * pve_wm[seg == 3]

    # 3. FLAIR Normalization (by histogram mode)
    hist, bin_edges = np.histogram(flair[brain_mask & (flair > 0)], bins=200)
    xmax = bin_edges[np.argmax(hist)]
    f2_norm = flair / max(xmax, 1.0)

    mean_gm = np.mean(f2_norm[(p0 > 1.5) & (p0 <= 2.5)])
    mean_wm = np.mean(f2_norm[p0 > 2.5])
    mean_csf = np.mean(f2_norm[(p0 <= 1.5) & (p0 > 0)])

    if verbose:
        print(f"[PyWMH LST-LGA] FLAIR Mode: {xmax:.1f} | Means: CSF={mean_csf:.3f}, GM={mean_gm:.3f}, WM={mean_wm:.3f}")

    # 4. Lesion Belief Maps
    B_gm = p0 * (p0 > 1.5) * (p0 <= 2.5) * np.maximum(f2_norm - mean_gm, 0.0) * atlas_wm
    B_gm[noles_mask] = 0.0

    B_wm = p0 * (p0 > 2.5) * np.maximum(f2_norm - mean_wm, 0.0) * atlas_wm
    B_wm[noles_mask] = 0.0

    B_csf = p0 * (p0 < 1.5) * (p0 > 0) * np.maximum(f2_norm - mean_csf, 0.0) * atlas_wm
    B_csf[noles_mask] = 0.0

    B = B_gm + B_wm + B_csf
    B_init = np.copy(B_gm)

    # 5. 6-neighborhood kernel
    k6 = ndimage.generate_binary_structure(3, 1).astype(np.float32)
    k6[1, 1, 1] = 0.0

    # Initial seed map: B_init_mean = B_init .* neighborhood_mean(B_init)
    nh_mean = ndimage.convolve(B_init, k6 / 6.0, mode='constant', cval=0.0)
    B_init_mean = B_init * nh_mean

    Lesion_iter = (B_init_mean > kappa).astype(np.float32)

    # Remove isolated seeds with < 2 neighbors in 6-neighborhood
    nh_count = ndimage.convolve(Lesion_iter, k6, mode='constant', cval=0.0)
    Lesion_iter[nh_count < 2] = 0.0

    num_seeds = int(np.sum(Lesion_iter > 0))
    if verbose:
        print(f"[PyWMH LST-LGA] Initialized {num_seeds} lesion seeds.")

    if num_seeds == 0:
        return np.zeros_like(flair, dtype=np.float32), np.zeros_like(flair, dtype=np.uint8)

    # Precompute B_mean
    B_mean = ndimage.convolve(B, k6 / 6.0, mode='constant', cval=0.0)

    # 6. Iterative Lesion Growing
    for it in range(max_iter):
        lesion_vox = f2_norm[Lesion_iter > 0.5]
        if len(lesion_vox) < 10:
            break

        # Fit Gamma distribution to lesion class
        a, loc, scale = gamma.fit(lesion_vox, floc=0)

        # Calculate normal tissue Gaussian mixture
        csf_v = f2_norm[(p0 < 1.5) & (p0 > 0) & (Lesion_iter < 0.5)]
        gm_v  = f2_norm[(p0 >= 1.5) & (p0 <= 2.5) & (Lesion_iter < 0.5)]
        wm_v  = f2_norm[(p0 >= 2.5) & (Lesion_iter < 0.5)]

        n_tot = len(csf_v) + len(gm_v) + len(wm_v)
        if n_tot == 0:
            break

        pi_csf = len(csf_v) / n_tot
        pi_gm  = len(gm_v)  / n_tot
        pi_wm  = len(wm_v)  / n_tot

        dens_normal = (
            pi_csf * norm.pdf(f2_norm, np.mean(csf_v), max(np.std(csf_v), 1e-3)) +
            pi_gm  * norm.pdf(f2_norm, np.mean(gm_v),  max(np.std(gm_v), 1e-3)) +
            pi_wm  * norm.pdf(f2_norm, np.mean(wm_v),  max(np.std(wm_v), 1e-3))
        )
        dens_normal = np.maximum(dens_normal, 1e-8)

        # 6-neighborhood support
        nh = ndimage.convolve(Lesion_iter, k6, mode='constant', cval=0.0)

        # Gamma likelihood
        norm_les = gamma.pdf(f2_norm, a, loc=0, scale=scale)

        prob = (B_mean * norm_les) / dens_normal
        prob = np.nan_to_num(prob)

        # Spatial MRF weight
        mf_img = np.exp(-phi * (6.0 - nh) + phi * nh)

        # Update frontier
        delta = prob * mf_img * (nh >= 2) * (atlas_wm > 0.05)
        delta[noles_mask] = 0.0

        old_sum = np.sum(Lesion_iter > 0.5)
        Lesion_iter = np.clip(Lesion_iter + delta, 0.0, 1.0)
        new_sum = np.sum(Lesion_iter > 0.5)

        if (new_sum - old_sum) < 10:
            if verbose:
                print(f"[PyWMH LST-LGA] Growth stabilized at iteration {it + 1}.")
            break

    # 7. Post-Processing Cleanup (LST official)
    # Fill voxels surrounded by > 18 lesion voxels in 26-neighborhood
    k26 = ndimage.generate_binary_structure(3, 3).astype(np.float32)
    k26[1, 1, 1] = 0.0
    nh26 = ndimage.convolve((Lesion_iter > 0.01).astype(np.float32), k26, mode='constant', cval=0.0)
    Lesion_iter[(Lesion_iter < 1.0) & (nh26 > 18) & (atlas_wm > 0)] = 1.0

    # Delete voxels with 0 first-order neighbors
    nh1 = ndimage.convolve((Lesion_iter > 0.01).astype(np.float32), k6, mode='constant', cval=0.0)
    Lesion_iter[nh1 == 0] = 0.0
    Lesion_iter[noles_mask] = 0.0

    prob_map = np.clip(Lesion_iter, 0.0, 1.0).astype(np.float32)
    binary_mask = (prob_map > 0.5).astype(np.uint8)

    if verbose:
        total_les = int(np.sum(binary_mask))
        print(f"[PyWMH LST-LGA] Final segmented lesion voxels: {total_les}.")

    return prob_map, binary_mask
