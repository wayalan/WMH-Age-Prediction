"""Correction identities, input failures, integration and raw compatibility."""
import unittest
import numpy as np
import nibabel as nib
from pywmh.age_correction import correct_age
from pywmh.age_model import calculate_wmh_volumes_and_age


class AgeCorrectionTests(unittest.TestCase):
    def test_reference_example_and_equivalent_inputs(self):
        a = correct_age(65, raw_wmh_age=70)
        self.assertEqual(a, correct_age(65, raw_pad=5))
        self.assertAlmostEqual(a['Corrected_PAD'], 7.526996956075326)
        self.assertAlmostEqual(a['Corrected_WMH_Age'], 72.52699695607532)
        self.assertAlmostEqual(a['Corrected_WMH_Age']-65, a['Corrected_PAD'])

    def test_invalid_inputs(self):
        for kwargs in ({}, {'raw_pad': 1, 'raw_wmh_age': 70},
                       {'raw_pad': float('nan')}, {'raw_wmh_age': float('inf')},
                       {'raw_pad': 1, 'method': 'unknown'}):
            with self.assertRaises(ValueError):
                correct_age(65, **kwargs)
        for age in (0, -1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                correct_age(age, raw_pad=1)

    def test_pipeline_keeps_raw_prediction_and_corrects_before_rounding(self):
        img = nib.Nifti1Image(np.ones((2,2,2)), np.eye(4))
        raw = calculate_wmh_volumes_and_age(img,img,img,chronological_age=65)
        corrected = calculate_wmh_volumes_and_age(img,img,img,chronological_age=65,
                                                 age_correction='linear-reference-v1')
        self.assertEqual(raw['Predicted_Age'], corrected['Predicted_Age'])
        self.assertEqual(raw['Brain_Age_Gap'], corrected['Brain_Age_Gap'])
        self.assertNotIn('Corrected_PAD', raw)
        exact = 12.693*np.log10(.008)+64.159
        self.assertAlmostEqual(corrected['Raw_WMH_Age'], exact)
        self.assertAlmostEqual(corrected['Raw_PAD'], exact-65)
        with self.assertRaises(ValueError):
            calculate_wmh_volumes_and_age(img,img,img,age_correction='linear-reference-v1')


if __name__ == '__main__':
    unittest.main()
