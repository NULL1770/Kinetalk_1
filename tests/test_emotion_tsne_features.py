import unittest

import torch

from scripts.evaluate_emotion_tsne import upper_motion_features


class UpperMotionFeatureChecks(unittest.TestCase):
    def setUp(self):
        self.motion = torch.zeros(5, 52)
        self.valid = torch.ones(5, dtype=torch.bool)
        self.channel = torch.ones(52, dtype=torch.bool)

    def features(self, motion=None, valid=None, channel=None):
        return upper_motion_features(self.motion if motion is None else motion,
                                     self.valid if valid is None else valid,
                                     self.channel if channel is None else channel)

    def test_distinct_constant_affect_postures_do_not_collapse(self):
        other = self.motion.clone(); other[:, 43] = .7
        neutral, affect = self.features(), self.features(other)
        self.assertEqual(affect.shape, (54,))
        self.assertFalse(torch.equal(neutral, affect))
        self.assertAlmostEqual(affect[2].item(), .7, places=6)
        self.assertEqual(affect[11].item(), 0.)

    def test_zero_mean_oscillation_retains_amplitude(self):
        self.motion[:, 43] = torch.tensor([-1., 1., -1., 1., 0.])
        small, large = self.features(), self.features(self.motion * 2)
        self.assertEqual(small[2].item(), 0.)
        self.assertGreater(small[11].item(), 0.)
        self.assertAlmostEqual(large[11].item(), 2 * small[11].item(), places=6)

    def test_same_posture_distribution_distinct_dynamics(self):
        self.motion[:, 43] = torch.tensor([0., 1., 2., 3., 4.])
        shuffled = self.motion[[0, 4, 1, 3, 2]]
        ordered, swapped = self.features(), self.features(shuffled)
        torch.testing.assert_close(ordered[:36], swapped[:36], rtol=0, atol=0)
        self.assertGreater(swapped[38].item(), ordered[38].item())

    def test_invalid_nan_and_missing_channel_cannot_separate_clips(self):
        self.valid[2] = False; self.channel[43] = False
        baseline = self.features()
        self.motion[2] = float('nan'); self.motion[:, 43] = float('nan')
        torch.testing.assert_close(self.features(), baseline, rtol=0, atol=0)

    def test_observed_nonfinite_rejected(self):
        self.motion[1, 43] = float('nan')
        with self.assertRaisesRegex(ValueError, 'Nonfinite observed'):
            self.features()

    def test_no_adjacent_frames_rejected(self):
        self.valid = torch.tensor([True, False, True, False, True])
        with self.assertRaisesRegex(ValueError, 'adjacent valid'):
            self.features()


if __name__ == '__main__':
    unittest.main()
