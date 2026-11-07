# Methods and references

## Background & Methodology

The brain age prediction model is based on the publication:
> **Chu-Chung Huang, et al. (2022)**. *Brain white matter hyperintensities-predicted age reflects neurovascular health in middle-to-old aged subjects*. **Age and Ageing**, 51(5), afac106. [https://doi.org/10.1093/ageing/afac106](https://doi.org/10.1093/ageing/afac106)

The model computes volumes for:
- **PVWMH** (Periventricular White Matter Hyperintensity, ≤ 10 mm from ventricular lining)
- **DWMH** (Deep White Matter Hyperintensity, > 10 mm from ventricular lining)

The predicted WMH Brain Age is estimated via:

$$
\text{Predicted Age} = 11.069 \cdot \log_{10}(\max(V_{\text{PV}}, 10^{-4})) + 1.624 \cdot \log_{10}(\max(V_{\text{D}}, 10^{-4})) + 64.159
$$

---

## 1. Pure Python Standalone Pipeline (`pywmh_tool.py`)

### Algorithmic Framework: Native Python LST-LGA
`pywmh` is built directly upon the algorithmic principles of the **Lesion Growth Algorithm (LGA)** from the [LST toolbox](https://www.applied-statistics.de/lst.html) (Schmidt et al., 2012):
1. **Tissue Probability Mapping**: FSL FAST estimates Partial Volume Estimation (PVE) maps for CSF, Grey Matter, and White Matter.
   FLAIR also receives its own FAST T2 bias-field correction in acquisition space before resampling with the existing rigid transform. Correcting T1 alone leaves FLAIR intensity inhomogeneity available to create false lesion seeds.
2. **FLAIR Normalization**: Original LST unit-width histogram mode, including tied modes.
3. **Lesion Belief Calculation**: Initial lesion belief maps ($B_{\text{gm}}, B_{\text{wm}}, B_{\text{csf}}$) combine tissue probabilities, contrast differences, and spatial priors.
4. **Spatial Priors**: Uses the official LST white matter prior (`atlas_wm.nii.gz`) and exclusion mask (`noles.nii.gz`) in native space to restrict lesion belief maps.
5. **Iterative MRF Region Growing**: Uses a Markov Random Field (MRF) model with Gamma-distributed lesion likelihood and normal tissue Gaussian mixture models to iteratively grow lesions from seeds until convergence.
6. **Original Label and Growth Rules**: T1-dependent PVE labels, FLAIR-bright CSF core relabeling, two-color sequential MRF updates and the original frontier stopping rule. No subject-specific corrections or fitted volume multipliers are used.

The segmentation function now requires bias-corrected T1 intensities alongside tissue maps, or a precomputed `p0` label. FAST is run with `-B` to provide its corrected T1. Legacy lesion caches are recomputed unless their algorithm version, parameters and input hashes match. Probability outputs are explicitly float32.

For identical-input native-space validation, install `requirements-validation.txt` and run:

```bash
python docs/validate_native_lga.py \
  --cache /path/to/LST_lga_rmFLAIR.mat \
  --reference /path/to/ples_lga_0.3_rmFLAIR.nii \
  --output /tmp/native_lga_validation.json
```

The adapter reads reference lesions only after segmentation for comparison. It is not imported by the production algorithm. Matching this core does not establish equivalence of the complete FAST/FSL and SPM preprocessing pipelines. See `docs/native_lga_validation.md` for the validation scope and results.

### The Role of $\kappa$ (Kappa)
**Does the algorithm require $\kappa$? Yes.**
As in the original LST-LGA implementation, $\kappa$ (kappa) is the user-definable **initial belief threshold** that governs seed initialization:
- **Default value**: `kappa = 0.3` (the recommended baseline by the LST developers for standard 3T FLAIR scans).
- **Lower $\kappa$ (e.g., `0.1` – `0.2`)**: More liberal; initializes more seeds, capturing subtle or faint punctate hyperintensities.
- **Higher $\kappa$ (e.g., `0.4` – `0.5`)**: More conservative; seeds are placed only on prominent, confluent hyperintensities.


## Acknowledgements & Software Attributions

This project builds upon and integrates foundational tools developed by the neuroimaging community:

- **[LST (Lesion Segmentation Tool)](https://www.applied-statistics.de/lst.html)**: Developed by Paul Schmidt, Christian Gaser, and colleagues at the Technische Universität München. The Python native LGA equations are ported from the original LST implementation; complete pipeline equivalence also requires equivalent upstream inputs.
- **[FSL (FMRIB Software Library)](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki)**: Developed by the Analysis Group, FMRIB, University of Oxford. Used in this pipeline for robust structural brain extraction (`bet`), rigid cross-modal co-registration (`flirt`), automated tissue segmentation (`fast`), and the MNI Structural Atlas.
- **[Digital 3D Brain MRI Arterial Territories Atlas](https://github.com/Chin-Fu-Liu/Arterial_Atlas)**: Developed by Chin-Fu Liu, Andreia Faria, and colleagues at Johns Hopkins University School of Medicine. Used for hierarchical parcellation of major arterial vascular territories (ACA, MCA, PCA, VB).
- **[SPM (Statistical Parametric Mapping)](https://www.fil.ion.ucl.ac.uk/spm/)**: Developed by the Wellcome Centre for Human Neuroimaging, University College London (UCL). Hosts the reference MATLAB implementation of the LST toolbox.

---

## Citation

Please cite the corresponding papers depending on which components you use in your study:

### 1. For White Matter Hyperintensity (WMH) Lesion Segmentation (LST-LGA):
> **Schmidt P**, et al. *An automated tool for detection of FLAIR-hyperintense white-matter lesions in Multiple Sclerosis*. **NeuroImage**. 2012;59(4):3774-3788.  
> DOI: [10.1016/j.neuroimage.2011.11.032](https://doi.org/10.1016/j.neuroimage.2011.11.032)

```bibtex
@article{schmidt2012automated,
  title={An automated tool for detection of FLAIR-hyperintense white-matter lesions in Multiple Sclerosis},
  author={Schmidt, Paul and others},
  journal={NeuroImage},
  volume={59},
  number={4},
  pages={3774--3788},
  year={2012},
  publisher={Elsevier},
  doi={10.1016/j.neuroimage.2011.11.032}
}
```

### 2. For WMH-based Brain Age Prediction & Regional WMH Volume Calculation (PVWMH / DWMH):
> **Huang CC**, et al. *Brain white matter hyperintensities-predicted age reflects neurovascular health in middle-to-old aged subjects*. **Age and Ageing**. 2022;51(5):afac106.  
> DOI: [10.1093/ageing/afac106](https://doi.org/10.1093/ageing/afac106)

```bibtex
@article{huang2022brain,
  title={Brain white matter hyperintensities-predicted age reflects neurovascular health in middle-to-old aged subjects},
  author={Huang, Chu-Chung and others},
  journal={Age and Ageing},
  volume={51},
  number={5},
  pages={afac106},
  year={2022},
  publisher={Oxford University Press},
  doi={10.1093/ageing/afac106}
}
```

### 3. For Arterial Vascular Territory Parcellation:
> **Liu CF**, Hsu J, Xu X, Kim G, Sheppard SM, Meier EL, Miller MI, Hillis AE, Faria AV. *Digital 3D Brain MRI Arterial Territories Atlas*. **Scientific Data**. 2023;10(1):74.  
> DOI: [10.1038/s41597-022-01923-0](https://doi.org/10.1038/s41597-022-01923-0)

```bibtex
@article{liu2023digital,
  title={Digital 3D Brain MRI Arterial Territories Atlas},
  author={Liu, Chin-Fu and Hsu, Jui-Yang and Xu, Xiaoying and Kim, Gina and Sheppard, Shannon M and Meier, Elizabeth L and Miller, Michael I and Hillis, Argye E and Faria, Andreia V},
  journal={Scientific Data},
  volume={10},
  number={1},
  pages={74},
  year={2023},
  publisher={Nature Publishing Group UK London},
  doi={10.1038/s41597-022-01923-0}
}
```

### 4. For Cerebral Lobar Parcellation (MNI Structural Atlas):
> **Mazziotta J**, et al. *A probabilistic atlas and reference system for the human brain: International Consortium for Brain Mapping (ICBM)*. **Philosophical Transactions of the Royal Society of London. Series B: Biological Sciences**. 2001;356(1412):1293-1322.  
> DOI: [10.1098/rstb.2001.0915](https://doi.org/10.1098/rstb.2001.0915)  
> **Collins DL**, Holmes CJ, Peters TM, Evans AC. *Automatic 3-D model-based neuroanatomical segmentation*. **Human Brain Mapping**. 1995;3(3):190-208.  
> DOI: [10.1002/hbm.460030304](https://doi.org/10.1002/hbm.460030304)

```bibtex
@article{mazziotta2001probabilistic,
  title={A probabilistic atlas and reference system for the human brain: International Consortium for Brain Mapping (ICBM)},
  author={Mazziotta, John and others},
  journal={Philosophical Transactions of the Royal Society of London. Series B: Biological Sciences},
  volume={356},
  number={1412},
  pages={1293--1322},
  year={2001},
  publisher={The Royal Society},
  doi={10.1098/rstb.2001.0915}
}
```

### 5. For FSL Processing Tools:
> **Jenkinson M**, Beckmann CF, Behrens TE, Woolrich MW, Smith SM. *FSL*. **NeuroImage**. 2012;62(2):782-790.  
> DOI: [10.1016/j.neuroimage.2011.09.015](https://doi.org/10.1016/j.neuroimage.2011.09.015)

```bibtex
@article{jenkinson2012fsl,
  title={FSL},
  author={Jenkinson, Mark and Beckmann, Christian F and Behrens, Timothy EJ and Woolrich, Mark W and Smith, Stephen M},
  journal={NeuroImage},
  volume={62},
  number={2},
  pages={782--790},
  year={2012},
  publisher={Elsevier},
  doi={10.1016/j.neuroimage.2011.09.015}
}
```

