# Age-bias correction


The original prediction and volume/log preprocessing remain the default. Age
correction is an optional post-processing step; it does not segment images,
change volumes, or retrain the original model. Supply chronological age at scan
and either an existing raw WMH age or raw PAD (exactly one).

```bash
# Existing results: no images, FSL, or MATLAB required
python3 -m pywmh.age_correction --age 65 --raw-wmh-age 70 --out corrected.json
python3 -m pywmh.age_correction --age 65 --raw-pad 5

# Pipeline: raw age is supplied automatically by the original model
python3 pywmh_tool.py --t1 T1w.nii --flair FLAIR.nii --age 65 \
  --age-correction linear-reference-v1 -o output

# Also supported with --lesion, and by the standalone image calculator
python3 wmh_age_calc.py --input m0wples.nii --res 1.5 --age 65 \
  --age-correction linear-reference-v1 --out results.csv
```

```python
from pywmh.age_correction import correct_age
result = correct_age(65, raw_wmh_age=70)  # alternatively raw_pad=5
```

```matlab
r = CorrectWMHAge(65, 'raw_wmh_age', 70); % alternatively 'raw_pad', 5
[vol, rawAge, ageResults] = AutomatedWMHAge('subject', ...
    'chronological_age', 65, 'age_correction', 'linear-reference-v1');
```

The fixed correction is:

```text
Raw_PAD = Raw_WMH_Age - Chronological_Age
Expected_PAD = 17.52549262674548 - 0.30849983973570466 * Chronological_Age
Corrected_PAD = Raw_PAD - Expected_PAD
Corrected_WMH_Age = Chronological_Age + Corrected_PAD
```

For raw age 70 and chronological age 65, raw PAD is +5, corrected PAD is
+7.526997 years and corrected WMH age is 72.526997 years. Calculations use
unrounded predictions; formatting is applied only for display. Existing
`Predicted_Age` and `Brain_Age_Gap` fields keep their raw meaning. Enabling
correction adds `Raw_WMH_Age`, `Raw_PAD`, `Corrected_WMH_Age`, `Corrected_PAD`,
`Chronological_Age`, and `Age_Correction_Method`. Python CSV/JSON and MATLAB
CSV/MAT include these fields. The Python QC HTML and summary PNG display raw
and corrected results. Missing age is an error when correction is requested.

### Reference and interpretation

This is a newly derived reference correction, not a correction reported in the
2022 paper. Parameters were estimated from the original healthy reference
(n=491; 312 TAMI + 179 ILAS), using in-sample raw PAD versus age. Original
reference volumes used additive offsets before log10: 0.00147 mL for TAMI and
0.00098 mL for ILAS. The software's existing volume-floor preprocessing remains
unchanged; applying this correction to a different preprocessing/acquisition
pipeline requires validation. The module accepts existing raw results and does
not determine whether they satisfy those reference conditions.

Nested 10-fold validation in the reference sample reduced PAD-age slope from
-0.31145 to 0.00047. Applying fixed full-reference parameters to the remaining
ILAS sample (n=726) reduced slope from -0.52384 to -0.21534; residual age
dependence remains. No ILAS test data were used to fit correction parameters.
The corrected age explicitly incorporates chronological age: a lower MAE or
closer agreement with chronological age does not establish improved MRI-only
prediction or biological validity. Reference ages were 21–89 years; application
outside the reference population is not validated. Original AA/NA/DA grouping
boundaries are not redefined by this feature.

Individual reference/test records and local analysis files are not distributed.
