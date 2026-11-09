import random
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image, ImageOps

from augmentations import TrainingAugmentation, numpy_rng_state, restore_numpy_rng_state


class AugmentationTests(unittest.TestCase):
    def setUp(self):
        labels = np.zeros((16, 16), dtype=np.uint8)
        labels[:, 8:] = 1
        labels[8:, :8] = 2
        self.mask = Image.fromarray(labels)
        self.image = Image.fromarray(np.repeat((labels * 60 + 20)[..., None], 3, axis=2))

    def test_horizontal_flip_is_paired(self):
        augment = TrainingAugmentation(16, 'basic', dict(scale_min=1., scale_max=1., hflip_prob=1.))
        image, mask = augment(self.image, self.mask)
        np.testing.assert_array_equal(image, ImageOps.mirror(self.image))
        np.testing.assert_array_equal(mask, ImageOps.mirror(self.mask))

    def test_rotation_is_paired_and_nearest_for_mask(self):
        augment = TrainingAugmentation(16, 'standard', dict(scale_min=1., scale_max=1., hflip_prob=0.,
            rotation_prob=1., rotation_degrees=5., photometric_prob=0., noise_prob=0., blur_prob=0.))
        with patch('augmentations.random.uniform', side_effect=[1., 5.]):
            image, mask = augment(self.image, self.mask)
        expected_image = self.image.rotate(5., resample=Image.Resampling.BILINEAR, fillcolor=(124, 116, 104))
        expected_mask = self.mask.rotate(5., resample=Image.Resampling.NEAREST, fillcolor=255)
        np.testing.assert_array_equal(image, expected_image)
        np.testing.assert_array_equal(mask, expected_mask)

    def test_photometric_does_not_change_labels(self):
        augment = TrainingAugmentation(16, 'standard', dict(scale_min=1., scale_max=1., hflip_prob=0.,
            rotation_prob=0., photometric_prob=1., noise_prob=1., noise_std=5., blur_prob=1.))
        random.seed(12)
        np.random.seed(12)
        image, mask = augment(self.image, self.mask)
        np.testing.assert_array_equal(mask, self.mask)
        self.assertFalse(np.array_equal(image, self.image))

    def test_resize_rotation_padding_do_not_create_label_ids(self):
        augment = TrainingAugmentation(32)
        for seed in range(30):
            random.seed(seed)
            np.random.seed(seed)
            image, mask = augment(self.image, self.mask)
            self.assertEqual(image.size, (32, 32))
            self.assertEqual(mask.size, (32, 32))
            self.assertTrue(set(np.unique(mask)) <= {0, 1, 2, 255})

    def test_padding_uses_ignore_not_background(self):
        augment = TrainingAugmentation(32, 'basic', dict(scale_min=1., scale_max=1., hflip_prob=0.))
        image, mask = augment(self.image, self.mask)
        self.assertEqual(np.asarray(mask)[-1, -1], 255)
        np.testing.assert_array_equal(np.asarray(image)[-1, -1], [124, 116, 104])

    def test_seed_reproduces_geometry_and_noise(self):
        augment = TrainingAugmentation(16, overrides=dict(noise_prob=1.))
        random.seed(91)
        np.random.seed(91)
        first = augment(self.image, self.mask)
        random.seed(91)
        np.random.seed(91)
        second = augment(self.image, self.mask)
        for left, right in zip(first, second):
            np.testing.assert_array_equal(left, right)

    def test_none_mode_is_deterministic_letterbox(self):
        augment = TrainingAugmentation(32, 'none')
        first = augment(self.image, self.mask)
        second = augment(self.image, self.mask)
        for left, right in zip(first, second):
            self.assertEqual(left.size, (32, 32))
            np.testing.assert_array_equal(left, right)

    def test_reject_bad_configuration(self):
        for config in (dict(hflip_prob=2), dict(scale_min=0), dict(gamma_min=-1),
                       dict(rotation_degrees=90), dict(noise_std=-1), dict(typo=True),
                       dict(brightness=float('nan'))):
            with self.subTest(config=config), self.assertRaises(ValueError):
                TrainingAugmentation(16, overrides=config)

    def test_input_label_is_not_modified(self):
        before = self.mask.tobytes()
        TrainingAugmentation(32)(self.image, self.mask)
        self.assertEqual(before, self.mask.tobytes())

    def test_numpy_rng_checkpoint_reproduces_noise(self):
        np.random.seed(47)
        state = numpy_rng_state()
        expected = np.random.normal(size=15)
        restore_numpy_rng_state(state)
        np.testing.assert_array_equal(np.random.normal(size=15), expected)


if __name__ == '__main__':
    unittest.main(verbosity=2)
