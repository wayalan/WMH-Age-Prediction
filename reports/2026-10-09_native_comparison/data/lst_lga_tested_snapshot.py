"""Native-space LST-LGA equations ported from SPM12/LST.

Matching the growth core requires matching input FLAIR, PVE labels and priors.
FAST inputs are not interchangeable with SPM inputs. Reference routines:
ps_LST_lga, ps_scale, ps_quantile, ps_LST_fitgamma, ps_LST_calc_mixture and
createIndependenceStructure (LST, GPL-3.0).
"""

import numpy as np
from scipy import ndimage
from scipy.special import digamma, polygamma, gammaln

ALGORITHM_VERSION = "lst-lga-native-2"


def _array(value, name, shape=None):
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 3 or (shape is not None and result.shape != shape):
        raise ValueError(f"{name} must be a 3D array on the FLAIR grid")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains non-finite values")
    return result


def _clear_border(image):
    for axis in range(3):
        for index in (0, -1):
            selection = [slice(None)] * 3
            selection[axis] = index
            image[tuple(selection)] = 0


def _mask(value, name, shape):
    result = np.asarray(value)
    if result.shape != shape:
        raise ValueError(f"{name} must be on the FLAIR grid")
    # SPM pulled masks may contain NaNs outside their coverage. The LST
    # condition noles > 0 treats those as false rather than excluding them.
    return result > 0


def _scale(values, low, high, constant):
    """LST class-wise min/max scaling; constant classes use a pure endpoint."""
    if not values.size:
        return values
    extent = np.ptp(values)
    if extent == 0:
        # MATLAB divides by zero for a constant class. Explicitly use its
        # pure-tissue endpoint instead of silently erasing lesions with NaNs.
        return np.full(values.shape, constant, dtype=np.float64)
    return low + (values - values.min()) * (high - low) / extent


def build_pve_label(t1, flair, pve_gm, pve_wm, pve_csf, brain_mask=None):
    """Construct LST p0, including FLAIR-bright CSF core relabeling.

    T1 and FLAIR should be bias corrected and already on one native grid.
    This function does not register or resample images.
    """
    flair = _array(flair, "flair")
    shape = flair.shape
    t1 = _array(t1, "t1", shape)
    gm, wm, csf = [_array(v, n, shape) for v, n in (
        (pve_gm, "pve_gm"), (pve_wm, "pve_wm"), (pve_csf, "pve_csf"))]
    if any(np.any((v < 0) | (v > 1)) for v in (gm, wm, csf)):
        raise ValueError("Tissue probabilities must lie in [0, 1]")
    support = (gm + wm + csf) > 0
    if brain_mask is not None:
        support &= _mask(brain_mask, "brain_mask", shape)
    # Normalizing each triplet by its sum does not change its argmax.
    seg = np.argmax(np.stack((csf, gm, wm)), axis=0) + 1
    seg[~support] = 0
    _clear_border(seg)
    gm_voxels = seg == 2
    if not gm_voxels.any():
        raise ValueError("LST initialization requires a GM tissue class")
    gm_t1_mean = t1[gm_voxels].mean()
    p0 = np.zeros(shape, dtype=np.float64)
    selection = seg == 1
    p0[selection] = -_scale(csf[selection], -1.5, -1, -1)
    selection = gm_voxels & (t1 < gm_t1_mean)
    p0[selection] = _scale(gm[selection], 1.5, 2, 2)
    selection = gm_voxels & (t1 > gm_t1_mean)
    p0[selection] = -_scale(gm[selection], -2.5, -2, -2)
    selection = seg == 3
    p0[selection] = _scale(wm[selection], 2.5, 3, 3)
    # Equality to the GM T1 mean is unassigned in the reference as well.
    core = (p0 < 1.5) & (p0 > 0) & (flair > flair[gm_voxels].mean())
    p0[core] = 2.4
    sorted_gm = np.sort(flair[gm_voxels])
    percentile = sorted_gm[int(np.ceil(sorted_gm.size * 0.95)) - 1]
    neighbors = ndimage.generate_binary_structure(3, 1)
    for _ in range(50):  # Original LST core propagation limit.
        addition = gm_voxels & ~core & (flair > percentile)
        addition &= ndimage.binary_dilation(core, structure=neighbors)
        if not addition.any():
            break
        core |= addition
    p0[core] = 2.4
    _clear_border(p0)
    return p0


def _flair_mode(flair, support):
    """Reproduce histc(values, 0:max(values)), including its final exact edge."""
    values = flair[support & (flair > 0)]
    if not values.size:
        raise ValueError("No positive in-brain FLAIR intensities")
    last_edge = int(np.floor(values.max()))
    selected = values[values <= last_edge]
    indices, counts = np.unique(np.floor(selected), return_counts=True)
    if not counts.size:
        raise ValueError("FLAIR intensities do not support LST's unit-width histogram")
    mode = float(indices[counts == counts.max()].mean())
    if mode <= 0:
        raise ValueError("LST FLAIR histogram mode is zero; check intensity units")
    return mode


def _fit_gamma(values):
    """LST Newton iteration: shape a, rate b, absolute tolerance 1e-4."""
    if not values.size or np.any(values <= 0):
        raise ValueError("Lesion Gamma fitting requires positive FLAIR samples")
    gap = np.log(values.mean()) - np.log(values).mean()
    if gap <= 0:
        raise ValueError("Lesion Gamma fitting is undefined for constant samples")
    a = 0.5 / gap
    for _ in range(1000):
        old = a
        a = 1 / (1 / old + (-gap + np.log(old) - digamma(old)) /
                 (old * old * (1 / old - polygamma(1, old))))
        if not np.isfinite(a) or a <= 0:
            raise ValueError("Lesion Gamma fitting did not converge")
        if abs(a - old) <= 0.0001:
            return a, a / values.mean()
    raise ValueError("Lesion Gamma fitting did not converge")


def _mixture_parameters(p0, flair, lesion):
    # Label < 1.5 deliberately includes label zero, exactly as the reference
    # calc_mixture routine does when FLAIR has positive extracranial samples.
    classes = (p0 < 1.5, (p0 >= 1.5) & (p0 < 2.5), p0 >= 2.5)
    samples = [flair[c & (lesion < 0.5) & (flair > 0)] for c in classes]
    total = sum(v.size for v in samples)
    if not total:
        raise ValueError("No normal tissue samples remain for Gaussian fitting")
    parameters = []
    for values in samples:
        if not values.size:
            # An absent class contributes zero weight, not a NaN density.
            continue
        if values.size < 2 or np.ptp(values) == 0:
            raise ValueError("Normal tissue Gaussian variance is undefined")
        parameters.append((values.size / total, values.mean(), values.std(ddof=1)))
    return parameters


def segment_wmh_lga(
    flair, pve_gm=None, pve_wm=None, pve_csf=None,
    brain_mask=None, atlas_wm=None, noles_mask=None,
    kappa=0.3, max_iter=50, phi=1.0, verbose=True, *, t1=None, p0=None,
):
    """Run native-space LST belief maps and lesion growth.

    Supply bias-corrected T1 and PVE maps, or an already computed LST p0.
    The latter allows numerical validation with identical upstream inputs.
    This algorithm never reads MATLAB files or reference lesion maps.
    Returns the continuous map and its strict >0.5 binary mask.
    """
    flair = _array(flair, "flair")
    if not np.isfinite(kappa) or not 0 < kappa < 1:
        raise ValueError("kappa must lie strictly between 0 and 1")
    if not isinstance(max_iter, (int, np.integer)) or max_iter < 1:
        raise ValueError("max_iter must be a positive integer")
    if not np.isfinite(phi) or phi < 0:
        raise ValueError("phi must be finite and nonnegative")
    if p0 is None:
        if t1 is None or any(v is None for v in (pve_gm, pve_wm, pve_csf)):
            raise ValueError("Supply bias-corrected t1 and PVE maps, or a precomputed p0")
        p0 = build_pve_label(t1, flair, pve_gm, pve_wm, pve_csf, brain_mask)
    else:
        p0 = _array(p0, "p0", flair.shape).copy()
        if np.any((p0 < 0) | (p0 > 3)):
            raise ValueError("p0 labels must lie in [0, 3]")
        if brain_mask is not None:
            p0[~_mask(brain_mask, "brain_mask", flair.shape)] = 0
        _clear_border(p0)
    support = p0 > 0
    if atlas_wm is None:
        raise ValueError("LST requires its native-space atlas_wm prior")
    atlas = _array(atlas_wm, "atlas_wm", flair.shape).copy()
    exclusion = np.zeros(flair.shape, dtype=bool) if noles_mask is None else (
        _mask(noles_mask, "noles_mask", flair.shape))
    atlas[exclusion | ~support | (atlas < 0)] = 0
    mode = _flair_mode(flair, support)
    normalized = flair / mode
    belief = np.zeros(flair.shape)
    initial_belief = np.zeros(flair.shape)
    for selection, mean_selection, seed_class in (
        ((p0 > 1.5) & (p0 < 2.5), (p0 > 1.5) & (p0 < 2.5), True),
        ((p0 < 1.5) & support, (p0 <= 1.5) & support, False),
        (p0 > 2.5, p0 > 2.5, False),
    ):
        if not mean_selection.any():
            continue
        mean = normalized[mean_selection].mean()
        component = p0 * selection * np.maximum(normalized - mean, 0) * atlas
        belief += component
        if seed_class:
            initial_belief = component
    k6 = ndimage.generate_binary_structure(3, 1).astype(np.float64)
    k6[1, 1, 1] = 0
    smooth_initial = ndimage.convolve(initial_belief, k6 / 6, mode="constant")
    lesion = (initial_belief * smooth_initial > kappa).astype(np.float64)
    lesion[ndimage.convolve(lesion, k6, mode="constant") < 2] = 0
    belief_mean = ndimage.convolve(belief, k6 / 6, mode="constant")
    belief_mean[belief <= 0] = 0
    if verbose:
        print(f"[LST-LGA] kappa={kappa:g}, FLAIR mode={mode:g}, seeds={int(lesion.sum())}")
    if lesion.any():
        # LST checkerboard class 1 corresponds to even zero-based parity.
        parity = np.zeros(flair.shape, dtype=np.uint8)
        for axis, length in enumerate(flair.shape):
            dimensions = [1, 1, 1]
            dimensions[axis] = length
            parity ^= (np.arange(length) % 2).astype(np.uint8).reshape(dimensions)
        active = (belief_mean > 0) & (normalized > 0)
        for iteration in range(max_iter):
            a, rate = _fit_gamma(normalized[lesion > 0.5])
            parameters = _mixture_parameters(p0, normalized, lesion)
            neighborhood = ndimage.convolve(lesion, k6, mode="constant")
            frontier = active & (lesion == 0) & (neighborhood > 0)
            if not frontier.any():
                break
            indices = np.flatnonzero(frontier)
            x = normalized.flat[indices]
            density = np.zeros(x.size)
            for weight, mean, sigma in parameters:
                density += weight * np.exp(-0.5 * np.log(2 * np.pi * sigma * sigma)
                    - 0.5 * ((x - mean) / sigma) ** 2)
            gamma_density = np.exp(a * np.log(rate) - gammaln(a)
                                   + (a - 1) * np.log(x) - rate * x)
            with np.errstate(divide="ignore", invalid="ignore"):
                odds = belief_mean.flat[indices] * gamma_density / density
            # Undefined 0/0 matches the reference's final NaN -> 0 behavior.
            odds = np.nan_to_num(odds, nan=0, posinf=np.inf)
            eligible = neighborhood.flat[indices] > 1
            colors = parity.flat[indices]
            for color in (0, 1):
                chosen = (colors == color) & eligible
                if not chosen.any():
                    continue
                # Second half uses the neighbors updated by the first half.
                if color == 1:
                    neighborhood = ndimage.convolve(lesion, k6, mode="constant")
                selected = indices[chosen]
                weight = np.exp(phi * (2 * neighborhood.flat[selected] - 6))
                lesion.flat[selected] = np.minimum(1, odds[chosen] * weight)
            if np.max(lesion.flat[indices]) <= 0.01:
                break
        if verbose:
            print(f"[LST-LGA] growth iterations={iteration + 1}")
        k26 = ndimage.generate_binary_structure(3, 3).astype(np.float64)
        k26[1, 1, 1] = 0
        surrounded = ndimage.convolve((lesion > 0.01).astype(np.float64), k26, mode="constant")
        lesion[support & (lesion < 1) & (surrounded > 18)] = 1
        neighbors = ndimage.convolve((lesion > 0.01).astype(np.float64), k6, mode="constant")
        lesion[(neighbors == 0) | ~support] = 0
    # Reference cleanup can fill an enclosed excluded voxel; do not add a
    # final atlas threshold/exclusion pass that would change that behavior.
    probability = lesion.astype(np.float32)
    return probability, (probability > 0.5).astype(np.uint8)
