# WMH-Age-Prediction

Automated White Matter Hyperintensity (WMH) segmentation and WMH-based Brain Age Estimation pipeline for T1-weighted and 3D T2-FLAIR brain MRI scans.

This repository provides two complete pipelines:
1. **`pywmh_tool` (Recommended)**: A lightweight, standalone Python tool powered by FSL (for co-registration and tissue segmentation) and a native, faithful implementation of the SPM LST Lesion Growth Algorithm (LGA) with MNI spatial priors (`atlas_wm` & `noles`). **Completely decoupled from MATLAB/SPM/CAT12.**
2. **`AutomatedWMHAge.m` / `run_wmh_age.sh`**: The optimized legacy MATLAB pipeline using SPM12, LST toolbox, and CAT12, enhanced for headless command-line batch processing without GUI popups.

---

## Background & Methodology

The brain age prediction model is based on the publication:
> **Chu-Chung Huang, et al. (2022)**. *Brain white matter hyperintensities-predicted age reflects neurovascular health in middle-to-old aged subjects*. **Age and Ageing**, 51(8), afac106. [https://doi.org/10.1093/ageing/afac106](https://doi.org/10.1093/ageing/afac106)

The model computes volumes for:
- **PVWMH** (Periventricular White Matter Hyperintensity, $\le 10\text{ mm}$ from ventricular lining)
- **DWMH** (Deep White Matter Hyperintensity, $> 10\text{ mm}$ from ventricular lining)

The predicted WMH Brain Age is estimated via:
$$\text{Predicted Age} = 11.069 \cdot \log_{10}(\max(V_{\text{PV}}, 10^{-4})) + 1.624 \cdot \log_{10}(\max(V_{\text{D}}, 10^{-4})) + 64.159$$

Pre-trained model (`WMHAge_PredicitonModel.mat`) trained on 491 healthy participants is included in the repository.

---

## 1. Pure Python Standalone Pipeline (`pywmh_tool.py`)

### Key Advantages
- **Zero MATLAB Dependency**: Runs on standard Linux/macOS Python environments.
- **Fast Execution**: Complete processing finishes in **~1.5 to 2 minutes** per subject (4x faster than MATLAB SPM+CAT12).
- **Strict Anatomical Confinement**: Integrates MNI tract prior (`atlas_wm`) and exclusion mask (`noles`) to completely eliminate false-positive lesions in the cerebellum, brainstem, and cerebral cortex.
- **Continuous Core Lesion Coverage**: Solves severe T1-hypointensity edge cases with continuous numerical boundary conditions, ensuring solid, non-cavitated lesion masks.

### Requirements
- Python 3.8+
- [FSL](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki) (FLIRT, BET, FAST, APPLYWARP)
- Python packages:
  ```bash
  pip install -r requirements.txt
  ```

### Usage
```bash
python pywmh_tool.py \
  --t1 /path/to/T1w.nii.gz \
  --flair /path/to/FLAIR.nii.gz \
  --outdir /path/to/output_directory \
  --kappa 0.3
```

#### Arguments:
- `--t1`: Path to T1-weighted structural MRI scan (`.nii` or `.nii.gz`).
- `--flair`: Path to 3D T2-FLAIR MRI scan (`.nii` or `.nii.gz`).
- `--outdir`: Directory to save outputs.
- `--kappa`: Initial lesion belief threshold (default: `0.3`, standard LST-LGA setting).
- `--nonlinear`: Use FSL FNIRT for non-linear registration to MNI (default: 12-DOF affine FLIRT for speed).
- `--fsl-dir`: Path to FSL directory if not set in `$FSLDIR`.

### Outputs
- `WMH_Age_Results.csv`: Summary table with Total WMH, PVWMH, DWMH (in cc) and Predicted Age.
- `WMH_Age_Results.json`: Machine-readable results.
- `bles_lga_k30.nii.gz`: Native-space binary WMH lesion mask.
- `native_PVWMH.nii.gz`: Native-space periventricular mask.
- `native_DWMH.nii.gz`: Native-space deep white matter mask.
- `rmFLAIR.nii.gz`: Rigidly co-registered FLAIR aligned to T1.

---

## 2. Headless MATLAB Pipeline (`run_wmh_age.sh`)

If you require exact legacy SPM/CAT12 execution without launching the MATLAB GUI:

```bash
./run_wmh_age.sh \
  /Applications/MATLAB_R2022a.app/bin/matlab \
  /path/to/subjects_folder \
  /path/to/WMH_Atlas \
  /path/to/spm12
```

---

## Benchmark & Validation

Evaluated on subject scan `MRNE085` comparing the legacy MATLAB (SPM12 + LST-LGA 0.3 + CAT12) vs Python `pywmh_tool`:

| Metric | MATLAB Legacy (SPM/LST/CAT12) | Python `pywmh_tool` | Difference |
| :--- | :---: | :---: | :--- |
| **Execution Time** | ~8 minutes | **~1.9 minutes** | **4x faster** |
| **Cerebellar / Cortical False Positives** | 0 voxels | **0 voxels** | Clean match |
| **PVWMH Volume** | 20.70 cc | **23.02 cc** | +2.32 cc (solid core) |
| **DWMH Volume** | 3.14 cc | **13.72 cc** | +10.58 cc |
| **Predicted WMH Brain Age** | **79.53 years** | **81.08 years** | **+1.55 years (< 2% diff)** |

---

## Repository Structure
```
WMH-Age-Prediction/
├── pywmh/                  # Python core algorithmic library
│   ├── __init__.py
│   ├── lst_lga.py          # Faithful SPM LST-LGA implementation
│   ├── fsl_engine.py       # FSL wrapper (FLIRT, BET, FAST, APPLYWARP)
│   └── age_model.py        # WMH masking and age regression model
├── pywmh_tool.py           # CLI master entry point for Python pipeline
├── wmh_age_calc.py         # Standalone age calculator given existing masks
├── AutomatedWMHAge.m       # Optimized MATLAB pipeline (SPM12/LST/CAT12)
├── run_wmh_age.sh          # Headless shell script for MATLAB execution
├── WMH_Atlas/              # 1mm and 1.5mm PV/D WMH atlases, atlas_wm & noles
├── WMHAge_PredicitonModel.mat # Pre-trained regression model (MATLAB)
├── sample_data.mat         # Benchmark testing sample data
├── requirements.txt        # Python package dependencies
├── LICENSE                 # MIT License
└── README.md               # Documentation
```

## Citation
If you use this pipeline in your research, please cite:
1. Huang CC, et al. *Brain white matter hyperintensities-predicted age reflects neurovascular health in middle-to-old aged subjects*. Age and Ageing. 2022;51(8):afac106. DOI: [10.1093/ageing/afac106](https://doi.org/10.1093/ageing/afac106).
2. Schmidt P, et al. *An automated tool for detection of FLAIR-hyperintense white-matter lesions in Multiple Sclerosis*. NeuroImage. 2012;59(4):3774-3788.

## License
MIT License
