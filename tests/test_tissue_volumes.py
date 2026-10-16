"""Analytical PVE integration and per-subject report/export checks."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import nibabel as nib
import numpy as np
import pandas as pd

from pywmh.age_model import calculate_tissue_volumes, save_results
from pywmh.qc_report import generate_qc_report
from pywmh_tool import run_pipeline


class TissueVolumeTests(unittest.TestCase):
    def maps(self, units="mm"):
        # Four voxels, each 2 x 3 x 4 mm: total volume = 0.096 mL.
        affine = np.diag([2., 3., 4., 1.])
        if units == "meter":
            affine[:3, :3] /= 1000
        images = []
        for fraction in (.25, .5, .25):
            img = nib.Nifti1Image(np.full((2, 2, 1), fraction), affine)
            img.header.set_xyzt_units(units)
            images.append(img)
        return images

    def test_fractional_anisotropic_volumes_and_unit_conversion(self):
        for units in ("mm", "meter", "unknown"):
            with self.subTest(units=units):
                result = calculate_tissue_volumes(*self.maps(units))
                for key, expected in (("GMV_cc", .024), ("WMV_cc", .048),
                                      ("CSFV_cc", .024), ("TIV_cc", .096)):
                    # NIfTI pixel spacings are stored as float32, including metres.
                    self.assertAlmostEqual(result[key], expected, places=7)
                self.assertEqual(result['TIV_cc'], sum(result[k] for k in ('GMV_cc', 'WMV_cc', 'CSFV_cc')))

    def test_mismatched_grids_rejected(self):
        gm, wm, csf = self.maps()
        shifted = wm.affine.copy()
        shifted[0, 3] = 2
        with self.assertRaisesRegex(ValueError, "grid"):
            calculate_tissue_volumes(gm, nib.Nifti1Image(wm.get_fdata(), shifted), csf)

    def test_invalid_fractions_and_empty_tissue_rejected(self):
        for value in (np.nan, np.inf, -.1, 1.1):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "fractions"):
                gm, wm, csf = self.maps()
                gm.get_fdata()[0, 0, 0] = value
                calculate_tissue_volumes(gm, wm, csf)
        empty = nib.Nifti1Image(np.zeros((2, 2, 1)), np.eye(4))
        with self.assertRaisesRegex(ValueError, "no tissue"):
            calculate_tissue_volumes(empty, empty, empty)

    def test_export_roundtrip(self):
        result = calculate_tissue_volumes(*self.maps())
        with tempfile.TemporaryDirectory() as folder:
            csv_path, json_path = Path(folder) / 'results.csv', Path(folder) / 'results.json'
            save_results(result, str(csv_path), str(json_path))
            row = pd.read_csv(csv_path).iloc[0]
            saved = json.loads(json_path.read_text())
            for key in ('GMV_cc', 'WMV_cc', 'CSFV_cc', 'TIV_cc'):
                self.assertAlmostEqual(row[key], result[key])
                self.assertEqual(saved[key], result[key])

    def test_lesion_only_does_not_invent_tissue_volumes(self):
        image = nib.Nifti1Image(np.ones((3, 3, 3)), np.eye(4))
        with tempfile.TemporaryDirectory() as folder, \
             patch('pywmh_tool.nib.load', return_value=image), \
             contextlib.redirect_stdout(io.StringIO()):
            result = run_pipeline(lesion_path='lesion.nii.gz', outdir=folder)
            saved = json.loads((Path(folder) / 'WMH_Age_Results.json').read_text())
            row = pd.read_csv(Path(folder) / 'WMH_Age_Results.csv').iloc[0]
            for key in ('GMV_cc', 'WMV_cc', 'CSFV_cc', 'TIV_cc'):
                self.assertIsNone(result[key])
                self.assertIsNone(saved[key])
                self.assertTrue(pd.isna(row[key]))

    def test_qc_reports_tissues_and_handles_legacy_partial_results(self):
        shape = (12, 14, 64)
        background = np.full(shape, 100.)
        lesion = np.zeros(shape)
        lesion[4:6, 5:7, 20:40] = .75
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            root = Path(folder)
            for name, data in (('flair', background), ('t1', background), ('lesion', lesion),
                               ('pv', np.ones(shape)), ('dw', np.zeros(shape))):
                nib.save(nib.Nifti1Image(data, np.eye(4)), root / f'{name}.nii.gz')
            for results in (calculate_tissue_volumes(*self.maps()), {'TIV_cc': 1000.}):
                generate_qc_report(*(str(root / f'{name}.nii.gz') for name in ('flair', 't1', 'lesion', 'pv', 'dw')),
                                   results=results, out_html=str(root / 'qc.html'), out_png=str(root / 'qc.png'))
                html = (root / 'qc.html').read_text()
                for label in ('GMV', 'WMV', 'CSFV', 'TIV'):
                    self.assertIn(f'{label} ·', html)
                self.assertIn('TIV = GMV + WMV + CSFV', html)
                if 'GMV_cc' in results:
                    self.assertIn('0.02 <span class="metric-unit">mL</span>', html)
                    self.assertIn('0.10 <span class="metric-unit">mL</span>', html)
                else:
                    self.assertIn('Not computed', html)
                self.assertGreater((root / 'qc.png').stat().st_size, 1000)


if __name__ == '__main__':
    unittest.main()
