"""FLAIR field application, cache provenance and fixed-transform wiring."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import nibabel as nib
import numpy as np

from pywmh.preprocessing import correct_flair_bias


class FlairPreprocessingTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.shape = (6, 7, 8)
        self.affine = np.diag([.8, .8, 2., 1.])
        x = np.indices(self.shape)[0]
        self.bias = 1. + x / 10
        self.data = 100 * self.bias
        self.save(self.data, self.root / 'flair.nii.gz')
        self.save(np.full(self.shape, 80.), self.root / 't1.nii.gz')
        self.save(np.ones(self.shape), self.root / 'mask.nii.gz')
        (self.root / 'rigid.mat').write_text('existing rigid transform')
        self.commands = []

    def save(self, data, path):
        nib.save(nib.Nifti1Image(data.astype(np.float32), self.affine), path)

    def fake_fsl(self, args, description):
        self.commands.append(args)
        if args[0] == 'convert_xfm':
            Path(args[args.index('-omat') + 1]).write_text('inverse transform')
        elif args[0] == 'fast':
            self.save(self.bias, args[args.index('-o') + 1] + '_bias.nii.gz')
        elif args[0] == 'flirt':
            nib.save(nib.load(args[args.index('-in') + 1]), args[args.index('-out') + 1])
        else:
            self.fail(f'Unexpected command {args[0]}')

    def correct(self, **options):
        return correct_flair_bias(*(str(self.root / name) for name in
            ('flair.nii.gz', 't1.nii.gz', 'mask.nii.gz', 'rigid.mat')),
            str(self.root / 'correction'), **options)

    def test_bias_division_precedes_resampling_and_uses_existing_transform(self):
        with patch('pywmh.preprocessing.run_command', side_effect=self.fake_fsl):
            result = self.correct()
        np.testing.assert_allclose(nib.load(result).get_fdata(), 100., atol=2e-5)
        self.assertEqual(nib.load(result).get_data_dtype(), np.dtype('float32'))
        flirts = [args for args in self.commands if args[0] == 'flirt']
        self.assertEqual(len(flirts), 2)
        for args in flirts:
            self.assertIn('-applyxfm', args)
            self.assertNotIn('-dof', args)
        self.assertEqual(flirts[-1][flirts[-1].index('-init') + 1], str(self.root / 'rigid.mat'))
        fast = next(args for args in self.commands if args[0] == 'fast')
        self.assertEqual(fast[fast.index('-t') + 1], '2')
        np.testing.assert_allclose(nib.load(self.root / 'flair.nii.gz').get_fdata(), self.data, atol=2e-5)

    def test_cache_rejects_changed_inputs_and_corrupted_outputs(self):
        with patch('pywmh.preprocessing.run_command', side_effect=self.fake_fsl):
            result = self.correct()
            count = len(self.commands)
            self.assertEqual(self.correct(), result)
            self.assertEqual(len(self.commands), count)
            (self.root / 'rigid.mat').write_text('different transform')
            self.correct()
            self.assertGreater(len(self.commands), count)
            count = len(self.commands)
            self.save(np.full(self.shape, 7.), result)
            self.correct()
            self.assertGreater(len(self.commands), count)
            count = len(self.commands)
            self.correct(force=True)
            self.assertGreater(len(self.commands), count)

    def test_invalid_bias_is_not_accepted_or_cached(self):
        self.bias[2, 2, 2] = 0
        with patch('pywmh.preprocessing.run_command', side_effect=self.fake_fsl):
            with self.assertRaisesRegex(ValueError, 'positive'):
                self.correct()
        self.assertFalse((self.root / 'correction/correction_inputs.json').exists())

    def test_empty_mask_is_not_accepted(self):
        self.save(np.zeros(self.shape), self.root / 'mask.nii.gz')
        with patch('pywmh.preprocessing.run_command', side_effect=self.fake_fsl):
            with self.assertRaisesRegex(ValueError, 'no positive'):
                self.correct()


if __name__ == '__main__':
    unittest.main()
