# WMH-Age-Prediction

Estimate WMH age from T1-weighted and 3D FLAIR MRI, or apply age-bias correction to an existing WMH age / PAD result.

**Outputs:** WMH volumes, raw WMH age, and—when correction is enabled—raw PAD, corrected PAD and corrected WMH age. The Python pipeline also generates an HTML report showing raw and corrected results together.

## Quick start

### 1. Already have a raw WMH age or PAD?

Run from the repository folder. This correction-only command needs Python, with no MRI images, FSL or MATLAB.

```bash
# Actual age 65; raw WMH age 70
python3 -m pywmh.age_correction --age 65 --raw-wmh-age 70

# Equivalent input: actual age 65; raw PAD +5
python3 -m pywmh.age_correction --age 65 --raw-pad 5
```

Both produce:

| Result | Years |
|---|---:|
| Chronological age | 65.00 |
| Raw WMH age | 70.00 |
| Raw PAD | +5.00 |
| Corrected WMH age | 72.53 |
| Corrected PAD | +7.53 |

Add `--out corrected.json` to save the result. Supply **one** of `--raw-wmh-age` or `--raw-pad`.

### 2. Start from MRI and include correction

Install Python 3.8+, [FSL](https://fsl.fmrib.ox.ac.uk/fsl/docs/) and the Python dependencies. FSL commands must be on your PATH and `FSLDIR` configured.

```bash
pip install -r requirements.txt

python3 pywmh_tool.py \
  --t1 /path/to/T1w.nii.gz \
  --flair /path/to/FLAIR.nii.gz \
  --age 65 \
  --age-correction linear-reference-v1 \
  --outdir /path/to/output
```

The pipeline calculates raw WMH age automatically and then applies correction. **Include both `--age` and `--age-correction linear-reference-v1`.** `--age` alone calculates the raw age gap; it does not enable correction.

For the original raw-only prediction, omit `--age-correction`. Segmentation and the raw prediction formula are unchanged.

### 3. Start from an existing MNI lesion map

```bash
python3 pywmh_tool.py \
  --lesion /path/to/m0wples.nii.gz \
  --res 1.5 \
  --age 65 \
  --age-correction linear-reference-v1 \
  --outdir /path/to/output
```

Use `--res 1.0` or `1.5` to match the lesion map and bundled atlas grid. Lesion-only input produces an age summary rather than MRI registration QC or tissue-volume estimates.

## Results and reports

| File | Contents |
|---|---|
| `WMH_Age_Results.csv` / `.json` | Volumes and age results; raw and corrected fields when enabled |
| `WMH_Age_QC_Report.html` | Full MRI pipeline: image QC, tissue volumes, raw and corrected age/PAD |
| `qc_summary.png` | Full MRI pipeline: slice overview and raw/corrected age summary |
| `WMH_Age_Summary.html` | Lesion-only run with correction: raw/corrected age table |
| `ples_lga_k30.nii.gz` / `bles_lga_k30.nii.gz` | Full MRI pipeline: lesion probabilities / binary mask |

`Predicted_Age` always means raw WMH age; the existing `Brain_Age_Gap` field remains raw PAD. Correction adds `Raw_WMH_Age`, `Raw_PAD`, `Corrected_WMH_Age`, `Corrected_PAD`, `Chronological_Age`, and `Age_Correction_Method`.

## How correction works

`Age` below means chronological age at the MRI scan, in years.

```text
Raw PAD           = Raw WMH Age - Age
Expected PAD      = 17.525493 - 0.308500 × Age
Corrected PAD     = Raw PAD - Expected PAD
Corrected WMH Age = Age + Corrected PAD
```

`linear-reference-v1` identifies these fixed reference parameters. It is post-processing of the raw prediction, and requires chronological age. The corrected WMH age therefore includes actual age; it is not an independent MRI-only age prediction.

The correction was derived from the original healthy reference sample (n=491), and reduces rather than completely removes age dependence in the ILAS test sample. Reference volume preprocessing differed from the current pipeline's volume floor; applicability to other processing/acquisition conditions needs validation. See [correction details and validation](docs/age_correction.md) for full-precision coefficients and limitations.

## MATLAB

Requires MATLAB, SPM12, CAT12 and LST on the MATLAB path.

```matlab
% MRI pipeline with correction
[vol, rawAge, ageResults] = AutomatedWMHAge('/path/to/subject', ...
    'chronological_age', 65, ...
    'age_correction', 'linear-reference-v1');

% Correct an existing raw age or PAD
r = CorrectWMHAge(65, 'raw_wmh_age', 70);
r = CorrectWMHAge(65, 'raw_pad', 5);
```

MATLAB saves volumes and age results to CSV/MAT. The HTML/PNG reports above belong to the Python pipeline.

For the legacy headless raw-only workflow:

```bash
./run_wmh_age.sh /path/to/subject 0.3 1.5 /path/to/output
```

## Options

Run `python3 pywmh_tool.py --help` for all arguments.

| Option | Purpose |
|---|---|
| `--age` | Chronological age at scan |
| `--age-correction linear-reference-v1` | Enable correction; requires `--age` |
| `--kappa 0.3` | Initial lesion seed threshold; default 0.3 |
| `--res 1.0` | Atlas resolution; default 1.0 mm for `pywmh_tool.py` |
| `--skip-qc` | Skip HTML/PNG report generation |
| `--nonlinear` | Use FNIRT rather than affine registration |
| `--force` | Recompute cached processing stages |

The standalone image calculator `wmh_age_calc.py` also supports `--age` and `--age-correction`; see its `--help` for input and CSV options.

## Model and validation

The raw model uses log-transformed periventricular (PVWMH) and deep (DWMH) lesion volumes in mL:

```text
Raw WMH Age = 11.069 × log10(PVWMH) + 1.624 × log10(DWMH) + 64.159
```

Current software uses a minimum volume of `0.0001 mL` before log10. The original study used additive minimum-volume constants; correction does not alter this preprocessing.

The Python segmentation uses a native LST-LGA implementation with FSL preprocessing. It does not reproduce the complete SPM/LST pipeline exactly. See [native LGA validation](docs/native_lga_validation.md), [FLAIR preprocessing validation](docs/flair_preprocessing_validation.md), and [methods and references](docs/methods_and_references.md).

[Example segmentation comparisons and report](reports/2026-10-09_native_comparison/WMH_native_comparison_2026-10-09.pdf)

## Citation

Huang CC, et al. *Brain white matter hyperintensities-predicted age reflects neurovascular health in middle-to-old aged subjects*. Age and Ageing. 2022;51(5):afac106. [DOI: 10.1093/ageing/afac106](https://doi.org/10.1093/ageing/afac106).

Please also cite the segmentation and atlas tools used in your study; [full citations and acknowledgements](docs/methods_and_references.md).

## License

GNU General Public License v3; see [LICENSE](LICENSE).
