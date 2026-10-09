"""Input-independent regression checks; no example images or fitted targets."""

import json
import tempfile
import unittest
import contextlib
import io
from pathlib import Path
from unittest.mock import patch

import nibabel as nib
import numpy as np
from scipy.stats import gamma

from pywmh.lst_lga import build_pve_label, segment_wmh_lga, _fit_gamma, _flair_mode
from pywmh_tool import lesion_cache_matches, segmentation_signature, run_pipeline


class NativeLGATests(unittest.TestCase):
    def test_csf_probability_direction_and_gm_labels(self):
        shape = (12, 12, 12)
        x, y, z = np.indices(shape)
        csf_class, gm_class, wm_class = x < 4, (x >= 4) & (x < 8), x >= 8
        dominant = 0.7 + y / 100
        csf = np.where(csf_class, dominant, 0.05)
        gm = np.where(gm_class, dominant, 0.05)
        wm = np.where(wm_class, dominant, 0.05)
        t1 = np.where(csf_class, 30, np.where(gm_class, 80 + z, 120))
        flair = np.where(csf_class, 20, np.where(gm_class, 100, 80))
        p0 = build_pve_label(t1, flair, gm, wm, csf)
        self.assertGreater(p0[2, 2, 2], p0[2, 9, 2])
        self.assertTrue(1.5 <= p0[5, 2, 2] <= 2)
        self.assertTrue(2 <= p0[5, 2, 9] <= 2.5)
        for axis in range(3):
            self.assertFalse(np.take(p0, 0, axis=axis).any())
            self.assertFalse(np.take(p0, -1, axis=axis).any())

    def test_no_seeds_stays_empty(self):
        shape = (11, 13, 15)
        p0 = np.full(shape, 2.0)
        probability, binary = segment_wmh_lga(np.full(shape, 100.0),
            p0=p0, atlas_wm=np.ones(shape), verbose=False)
        self.assertFalse(probability.any())
        self.assertFalse(binary.any())

    def test_histc_final_edge_and_mode_ties(self):
        image = np.array([2.1, 2.4, 3.1, 3.2, 4.9]).reshape(1, 1, 5)
        self.assertEqual(_flair_mode(image, np.ones(image.shape, dtype=bool)), 2.5)

    def test_gamma_mle_agrees_with_independent_scipy_fit(self):
        values = np.random.default_rng(271828).gamma(7.0, 0.4, 5000)
        shape, rate = _fit_gamma(values)
        expected, _, scale = gamma.fit(values, floc=0)
        self.assertAlmostEqual(shape, expected, places=6)
        self.assertAlmostEqual(rate, 1 / scale, places=6)

    def test_degenerate_distribution_and_missing_t1_fail_explicitly(self):
        with self.assertRaises(ValueError):
            _fit_gamma(np.ones(20))
        with self.assertRaisesRegex(ValueError, 't1'):
            segment_wmh_lga(np.ones((5, 5, 5)), verbose=False)

    def test_cache_detects_content_and_kappa_changes(self):
        with tempfile.TemporaryDirectory() as folder:
            native = Path(folder) / 'native.bin'
            metadata = Path(folder) / 'cache.json'
            native.write_bytes(b'first input')
            initial = segmentation_signature({'flair': native}, 0.3)
            metadata.write_text(json.dumps(initial))
            self.assertTrue(lesion_cache_matches(metadata, initial))
            self.assertFalse(lesion_cache_matches(metadata,
                segmentation_signature({'flair': native}, 0.301)))
            native.write_bytes(b'other input')
            self.assertFalse(lesion_cache_matches(metadata,
                segmentation_signature({'flair': native}, 0.3)))

    def test_cli_native_wiring_dtypes_and_repeat_cache(self):
        # All native inputs are synthetic. Pre-existing upstream files mean
        # no FSL operation or registration is run by this integration check.
        shape = (11, 13, 15)
        rng = np.random.default_rng(161803)
        flair = rng.uniform(90, 110, shape)
        flair[4:7, 5:8, 6:9] = rng.uniform(220, 250, (3, 3, 3))
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'fast').mkdir()
            (root / 'registration').mkdir()
            images = {
                'T1_raw': rng.uniform(70, 100, shape), 'FLAIR_raw': flair,
                'rmFLAIR': flair, 'T1_brain': rng.uniform(70, 100, shape),
                'T1_brain_mask': np.ones(shape), 'watlas_wm': np.ones(shape),
                'wnoles': np.zeros(shape), 'native_PVWMH': np.ones(shape),
                'native_DWMH': np.zeros(shape), 'native_lobar': np.zeros(shape),
                'native_arterial': np.zeros(shape),
                'fast/T1_restore': rng.uniform(70, 100, shape),
                'fast/T1_pve_0': np.full(shape, 0.1),
                'fast/T1_pve_1': np.full(shape, 0.8),
                'fast/T1_pve_2': np.full(shape, 0.1),
            }
            for name, data in images.items():
                image = nib.Nifti1Image(data, np.eye(4))
                if name == 'rmFLAIR':
                    image.set_data_dtype(np.int16)
                nib.save(image, root / f'{name}.nii.gz')
            for name in ('t1_to_mni_affine', 'mni_to_t1_affine'):
                (root / 'registration' / f'{name}.mat').write_text('unused cached transform')
            with patch('pywmh_tool.check_fsl_installed', return_value='/unused'), \
                 patch('pywmh_tool.register_t1_to_mni', side_effect=AssertionError('No registration in native test')), \
                 patch('pywmh_tool.coregister_flair_to_t1', side_effect=AssertionError('No coregistration in native test')), \
                 contextlib.redirect_stdout(io.StringIO()):
                # Existing coregistration must include its matrix file.
                (root / 'flair2t1.mat').write_text('unused cached transform')
                run_pipeline(t1_path=str(root / 'T1_raw.nii.gz'),
                    flair_path=str(root / 'FLAIR_raw.nii.gz'), outdir=str(root), skip_qc=True)
                self.assertEqual(nib.load(root / 'ples_lga_k30.nii.gz').header.get_data_dtype(), np.dtype('float32'))
                self.assertEqual(nib.load(root / 'bles_lga_k30.nii.gz').header.get_data_dtype(), np.dtype('uint8'))
                self.assertTrue((root / 'lga_k30_inputs.json').exists())
                with patch('pywmh_tool.segment_wmh_lga', side_effect=AssertionError('Validated cache should be reused')):
                    run_pipeline(t1_path=str(root / 'T1_raw.nii.gz'),
                        flair_path=str(root / 'FLAIR_raw.nii.gz'), outdir=str(root), skip_qc=True)


if __name__ == '__main__':
    unittest.main()
