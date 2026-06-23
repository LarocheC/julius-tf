# File under the MIT license, see the LICENSE file for details.
# Original author: Alexandre Défossez (adefossez), 2020
# TensorFlow port: Clément Laroche (LarocheC), 2026

import random
import unittest

import tensorflow as tf

from julius import LowPassFilter, LowPassFilters, lowpass_filter, resample_frac
from julius.core import pure_tone


def delta(a, b, ref, fraction=0.9):
    length = a.shape[-1]
    compare_length = int(length * fraction)
    offset = (length - compare_length) // 2
    a = a[..., offset: offset + length]
    b = b[..., offset: offset + length]
    return float(100 * tf.reduce_mean(tf.abs(a - b)) / tf.math.reduce_std(ref))


TOLERANCE = 1  # Tolerance to errors as percentage of the std of the input signal


class _BaseTest(unittest.TestCase):
    def assertSimilar(self, a, b, ref, msg=None, tol=TOLERANCE):
        self.assertLessEqual(delta(a, b, ref), tol, msg)


class TestLowPassFilters(_BaseTest):
    def setUp(self):
        tf.random.set_seed(1234)
        random.seed(1234)

    def test_keep_or_kill(self):
        for _ in range(10):
            freq = random.uniform(0.01, 0.4)
            sr = 1024
            tone = pure_tone(freq * sr, sr=sr, dur=10)

            # For this test we accept 5% tolerance in amplitude, or -26dB in power.
            tol = 5
            zeros = 16

            # If cutoff frequency is under freq, output should be zero
            y_killed = lowpass_filter(tone, 0.9 * freq, zeros=zeros)
            self.assertSimilar(y_killed, 0 * y_killed, tone, f"freq={freq}, kill", tol=tol)

            # If cutoff frequency is under freq, output should be input
            y_pass = lowpass_filter(tone, 1.1 * freq, zeros=zeros)
            self.assertSimilar(y_pass, tone, tone, f"freq={freq}, pass", tol=tol)

    def test_same_as_downsample(self):
        for _ in range(10):
            x = tf.random.normal((2 * 3 * 4 * 100,))
            rolloff = 0.945
            for old_sr in [2, 3, 4]:
                y_resampled = resample_frac(x, old_sr, 1, rolloff=rolloff, zeros=16)
                y_lowpass = lowpass_filter(x, rolloff / old_sr / 2, stride=old_sr, zeros=16)
                self.assertSimilar(y_resampled, y_lowpass, x, f"old_sr={old_sr}")

    def test_fft_nofft(self):
        for _ in range(10):
            x = tf.random.normal((1024,))
            freq = random.uniform(0.01, 0.5)
            y_fft = lowpass_filter(x, freq, fft=True)
            y_ref = lowpass_filter(x, freq, fft=False)
            self.assertSimilar(y_fft, y_ref, x, f"freq={freq}", tol=0.01)

    def test_tf_function(self):
        x = tf.random.normal((128,))

        for mod in [LowPassFilters([0.1, 0.3]), LowPassFilters([0.1, 0.3], fft=True)]:
            fn = tf.function(mod.__call__)
            self.assertEqual(list(fn(x).shape), [2, 128])

        mod = LowPassFilter(0.2)
        fn = tf.function(mod.__call__)
        self.assertEqual(list(fn(x).shape), [128])

    def test_constant(self):
        x = tf.ones((2048,))
        for zeros in [4, 10]:
            for freq in [0.01, 0.1]:
                y_low = lowpass_filter(x, freq, zeros=zeros)
                # A lowpass leaves a DC (constant) signal essentially untouched. The tiny
                # residual is float32 convolution rounding, which grows with the (long)
                # filter size and varies across TensorFlow versions / CPU backends.
                self.assertLessEqual(float(tf.reduce_mean(tf.abs(y_low - 1))), 1e-3, (zeros, freq))


if __name__ == '__main__':
    unittest.main()
