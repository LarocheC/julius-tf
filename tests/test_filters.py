# File under the MIT license, see https://github.com/LarocheC/julius-tf/blob/main/LICENSE for details.
# Original author: Alexandre Défossez (adefossez), 2020
# TensorFlow port: Clément Laroche (LarocheC), 2026

import random
import unittest

import tensorflow as tf

from julius.core import pure_tone
from julius import filters


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


class TestHighPassFilters(_BaseTest):
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

            # If cutoff frequency is under freq, output should be input
            y_pass = filters.highpass_filter(tone, 0.9 * freq, zeros=zeros)
            self.assertSimilar(y_pass, tone, tone, f"freq={freq}, pass", tol=tol)

            # If cutoff frequency is over freq, output should be zero
            y_killed = filters.highpass_filter(tone, 1.1 * freq, zeros=zeros)
            self.assertSimilar(y_killed, 0 * tone, tone, f"freq={freq}, kill", tol=tol)

    def test_fft_nofft(self):
        for _ in range(10):
            x = tf.random.normal((1024,))
            freq = random.uniform(0.01, 0.5)
            y_fft = filters.highpass_filter(x, freq, fft=True)
            y_ref = filters.highpass_filter(x, freq, fft=False)
            self.assertSimilar(y_fft, y_ref, x, f"freq={freq}", tol=0.01)

    def test_tf_function(self):
        x = tf.random.normal((128,))

        for mod in [filters.HighPassFilters([0.1, 0.3]),
                    filters.HighPassFilters([0.1, 0.3], fft=True)]:
            fn = tf.function(mod.__call__)
            self.assertEqual(list(fn(x).shape), [2, 128])

        mod = filters.HighPassFilter(0.2)
        fn = tf.function(mod.__call__)
        self.assertEqual(list(fn(x).shape), [128])

    def test_constant(self):
        x = tf.ones((2048,))
        for zeros in [4, 10]:
            for freq in [0.01, 0.1]:
                y_high = filters.highpass_filter(x, freq, zeros=zeros)
                # A highpass removes a DC (constant) signal. The tiny residual is float32
                # convolution rounding, which grows with the (long) filter size and varies
                # across TensorFlow versions / CPU backends.
                self.assertLessEqual(float(tf.reduce_mean(tf.abs(y_high))), 1e-3, (zeros, freq))

    def test_stride(self):
        x = tf.random.normal((1024,))

        y = filters.highpass_filters(x, [0.1, 0.2], stride=1)[:, ::3]
        y2 = filters.highpass_filters(x, [0.1, 0.2], stride=3)

        self.assertEqual(y.shape, y2.shape)
        self.assertSimilar(y, y2, x)

        y = filters.highpass_filters(x, [0.1, 0.2], stride=1, pad=False)[:, ::3]
        y2 = filters.highpass_filters(x, [0.1, 0.2], stride=3, pad=False)

        self.assertEqual(y.shape, y2.shape)
        self.assertSimilar(y, y2, x)


class TestBandPassFilters(_BaseTest):
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

            y_pass = filters.bandpass_filter(tone, 0.9 * freq, 1.1 * freq, zeros=zeros)
            self.assertSimilar(y_pass, tone, tone, f"freq={freq}, pass", tol=tol)

            y_killed = filters.bandpass_filter(tone, 1.1 * freq, 1.2 * freq, zeros=zeros)
            self.assertSimilar(y_killed, 0 * tone, tone, f"freq={freq}, kill", tol=tol)

            y_killed = filters.bandpass_filter(tone, 0.8 * freq, 0.9 * freq, zeros=zeros)
            self.assertSimilar(y_killed, 0 * tone, tone, f"freq={freq}, kill", tol=tol)

    def test_fft_nofft(self):
        for _ in range(10):
            x = tf.random.normal((1024,))
            freq = random.uniform(0.01, 0.5)
            freq2 = random.uniform(freq, 0.5)
            y_fft = filters.bandpass_filter(x, freq, freq2, fft=True)
            y_ref = filters.bandpass_filter(x, freq, freq2, fft=False)
            self.assertSimilar(y_fft, y_ref, x, f"freq={freq}", tol=0.01)

    def test_tf_function(self):
        x = tf.random.normal((128,))

        mod = filters.BandPassFilter(0.1, 0.3)
        fn = tf.function(mod.__call__)
        self.assertEqual(list(fn(x).shape), [128])

        mod = filters.BandPassFilter(0.1, 0.3, fft=True)
        fn = tf.function(mod.__call__)
        self.assertEqual(list(fn(x).shape), [128])

    def test_constant(self):
        x = tf.ones((2048,))
        for zeros in [4, 10]:
            for freq in [0.01, 0.1]:
                y = filters.bandpass_filter(x, freq, 1.2 * freq, zeros=zeros)
                # A bandpass removes a DC (constant) signal. The tiny residual is float32
                # convolution rounding, which grows with the (long) filter size and varies
                # across TensorFlow versions / CPU backends.
                self.assertLessEqual(float(tf.reduce_mean(tf.abs(y))), 1e-3, (zeros, freq))

    def test_stride(self):
        x = tf.random.normal((1024,))

        y = filters.bandpass_filter(x, 0.1, 0.2, stride=1)[::3]
        y2 = filters.bandpass_filter(x, 0.1, 0.2, stride=3)

        self.assertEqual(y.shape, y2.shape)
        self.assertSimilar(y, y2, x)

        y = filters.bandpass_filter(x, 0.1, 0.2, stride=1, pad=False)[::3]
        y2 = filters.bandpass_filter(x, 0.1, 0.2, stride=3, pad=False)

        self.assertEqual(y.shape, y2.shape)
        self.assertSimilar(y, y2, x)

    def test_same_as_highpass(self):
        x = tf.random.normal((1024,))

        y_ref = filters.highpass_filter(x, 0.2)
        y = filters.bandpass_filter(x, 0.2, 0.5)
        self.assertSimilar(y, y_ref, x)

    def test_same_as_lowpass(self):
        x = tf.random.normal((1024,))

        y_ref = filters.lowpass_filter(x, 0.2)
        y = filters.bandpass_filter(x, 0., 0.2)
        self.assertSimilar(y, y_ref, x)


if __name__ == '__main__':
    unittest.main()
