# FLAIR preprocessing correction — 2026-10-16

## Confirmed defect

The previous production pipeline passed rigidly resampled, intensity-uncorrected
FLAIR to LGA. Only T1 had bias correction through FAST. Original LST corrects both
images. Thus, matching the native LGA equations does not restrict the remaining
differences to T1 tissue segmentation alone.

On an additional low-burden example, the old production output was reproduced
voxel for voxel, with all segmentation input hashes matching its cache. It had
13.7524 mL of probability-weighted lesions, including 8.4931 mL of initial seeds.
The mean FAST GM fraction at binary-positive voxels was 0.8042. The zero-seed
shortcut cannot correct false positives that already exist in the initial seeds.

## Change

`pywmh/preprocessing.py` estimates a FLAIR bias field with FSL FAST (`-t 2 -n 3
-H 0.1 -I 4 -b`). The existing inverse rigid matrix projects the T1 extraction
mask into acquisition-space FLAIR. FAST estimates the field inside that mask;
the full acquisition image is divided by the field and resampled using the
existing forward matrix. No new transform is estimated by this stage.

The corrected native image feeds both Python LGA and visual QC. GMV, WMV, CSFV
and TIV still come from the T1 FAST maps. LGA equations, kappa=0.3, phi=1,
max_iter=50 and strict binary threshold >0.5 are unchanged. No reference lesion,
SPM tissue map, subject identifier, target volume or subject-specific parameter
enters this production correction. FAST is an existing dependency; no MATLAB
or SimpleITK dependency is added.

The correction cache binds settings, FSL version, source image/mask/transform
hashes and corrected-output hashes. Changed corrected FLAIR invalidates the
existing LGA input signature. Invalid, nonpositive or nonfinite bias fields
raise an error rather than being accepted as corrected inputs.

## Actual comparisons

Probability volumes integrate continuous maps; Dice uses a strict >0.5 threshold.
The two previously published cases retain their original MATLAB references.
The additional low-burden case was run afresh with installed original SPM12/LST,
using copied raw T1/FLAIR and kappa=0.3, max_iter=50, phi=1, HTML disabled.

| Case | MATLAB probability mL | Previous Python mL | Corrected Python mL | Previous Dice | Corrected Dice |
|---|---:|---:|---:|---:|---:|
| Existing example 1 | 16.6560 | 21.1014 | 21.0066 | 0.8119 | 0.8530 |
| Existing example 2 | 2.4241 | 1.4203 | 1.9106 | 0.5587 | 0.5921 |
| Additional low-burden example | 0.1333 | 13.7524 | 0.7874 | 0.0076 | 0.1281 |

The additional reference was small, but **not exactly zero** in this fresh run:
its binary volume was 0.1295 mL. Applying LST's >0.015 mL connected-component
rule to its report statistics gives 0.0660 mL; that report-only rule does not
modify the saved probability image.

Providing the new case's complete SPM-derived native inputs to the current
Python LGA reproduced its MATLAB reference exactly, including float32
probabilities and binary masks. This is a core validation test, not the
production correction. In seed-only input substitution experiments, replacing
FLAIR reduced seeds from 8.4931 to 0.5908 mL; replacing labels alone gave
2.8948 mL. Replacing FLAIR also changes the previously resampled image, so this
experiment alone cannot isolate bias estimation from resampling differences.

## Verification and limits

The new stage ran through the actual production entry point with cached native
upstream inputs. A second run reused the verified lesion/correction caches.
Python registration estimation was explicitly disabled in this audit; the
fresh original LST reference ran its own standard preprocessing. All original
example inputs and outputs remained byte-for-byte unchanged; new results and
QC were written separately. Four new tests exercise inverse field application,
fixed-transform wiring, cache invalidation/output corruption and invalid fields
or masks. The complete 17-test suite passes.

This fixes a missing stage and substantially reduces the observed false-positive
burden. **It does not resolve full MATLAB/Python equivalence**: the additional
case still contains false positives, and the two other cases retain material
differences. FAST tissue labels, intensity correction and spatial-prior inputs
remain distinct from SPM's frontend. These results do not justify claiming that
healthy scans are now universally lesion-free, or that MRI segmentation has
been clinically validated. No experimental method was selected by fitting a
case-specific threshold or volume factor.

The earlier illustrated report records the earlier pipeline and is retained as
a historical result. Its figures should not be read as measurements of this
new preprocessing version.

Source definitions: installed `ps_LST_lga.m` preprocessing calls and
`ps_LST_lga_preproc_default.m`; [FSL FAST documentation](https://fsl.fmrib.ox.ac.uk/fsl/docs/structural/fast.html).
