# File under the MIT license, see https://github.com/LarocheC/julius-tf/blob/main/LICENSE for details.
# Original author: Alexandre Défossez (adefossez), 2020
# TensorFlow port: Clément Laroche (LarocheC), 2026
import math
import random
import unittest

import resampy
import tensorflow as tf

from julius import resample, ResampleFrac


def pure_tone(freq, sr=128, dur=4):
    time = tf.range(int(sr * dur), dtype=tf.float32) / sr
    return tf.cos(2 * math.pi * freq * time)


def delta(a, b, ref, fraction=0.8):
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


class TestDownsample2(_BaseTest):
    def test_lowfreqs(self):
        # For those freq, downsample2 should be almost decimation
        for freq in [8, 16, 20, 28]:
            x = pure_tone(freq, sr=128)
            y_gt = x[::2]
            y = resample._downsample2(x)
            self.assertSimilar(y, y_gt, x, f"freq={freq}")

    def test_hifreqs(self):
        # For those freq, downsample2 should return zero
        for freq in [36, 40, 56, 64]:
            x = pure_tone(freq, sr=128)
            y = resample._downsample2(x)
            y_gt = 0 * y
            self.assertSimilar(y, y_gt, x, f"freq={freq}")

    def test_mixture(self):
        # Test one mixture
        x_low = pure_tone(16, sr=128)
        x_high = pure_tone(40, sr=128)
        x = x_low + x_high
        y = resample._downsample2(x)
        y_gt = x_low[::2]
        self.assertSimilar(y, y_gt, x, "mixture")


class TestUpsample2(_BaseTest):
    def test_upsample(self):
        # For those freq, _downsample2 should be almost decimation
        for freq in [8, 16, 20, 28, 32]:
            x = pure_tone(freq, sr=64)
            y_gt = pure_tone(freq, sr=128)
            y = resample._upsample2(x)
            self.assertSimilar(y, y_gt, x, f"freq={freq}")


class TestResampleFrac(_BaseTest):
    def test_ref(self):
        # Compare to _upsample2 and _downsample2
        for freq in [8, 16, 20, 28, 32, 36, 40, 56, 64]:
            x = pure_tone(freq, sr=128)
            y_gt_down = resample._downsample2(x)
            y_down = resample.resample_frac(x, 2, 1, rolloff=1)
            self.assertSimilar(y_down, y_gt_down, x, f"freq={freq} down")
            y_gt_up = resample._upsample2(x)
            y_up = resample.resample_frac(x, 1, 2, rolloff=1)
            self.assertSimilar(y_up, y_gt_up, x, f"freq={freq} up")

    def test_resampy(self):
        old_sr = 3
        new_sr = 2
        x = pure_tone(7, sr=128, dur=3) + pure_tone(24, sr=128, dur=3)
        y_re = tf.constant(resampy.resample(x.numpy(), old_sr, new_sr), dtype=tf.float32)
        y = resample.resample_frac(x, old_sr, new_sr)
        self.assertSimilar(y, y_re, x, f"{old_sr} to {new_sr}")

        old_sr = 2
        new_sr = 5
        x = pure_tone(7, sr=128) + pure_tone(48, sr=128)
        y_re = tf.constant(resampy.resample(x.numpy(), old_sr, new_sr), dtype=tf.float32)
        y = resample.resample_frac(x, old_sr, new_sr)
        self.assertSimilar(y, y_re, x, f"{old_sr} to {new_sr}")

        # Match julius to resampy's default `kaiser_best` filter. resampy changed
        # these parameters over versions (e.g. num_zeros 64 -> 50, rolloff ~0.948 -> ~0.917
        # between 0.2.x and 0.4.x), so we read them from resampy itself rather than
        # hardcoding, keeping this test robust across resampy versions.
        from resampy.filters import load_filter
        half_window, precision, rolloff = load_filter('kaiser_best')
        num_zeros = round(len(half_window) / precision)

        random.seed(1234)
        tf.random.set_seed(1234)
        for _ in range(10):
            old_sr = random.randrange(8, 128)
            new_sr = random.randrange(8, 128)
            x = tf.random.normal((1024,))
            y_re = tf.constant(resampy.resample(x.numpy(), old_sr, new_sr), dtype=tf.float32)
            y = resample.resample_frac(x, old_sr, new_sr, zeros=num_zeros, rolloff=rolloff)
            # We allow some relatively high tolerance as we are not using the same window.
            self.assertSimilar(y, y_re, x, f"{old_sr} to {new_sr}", tol=3)

    def test_tf_function(self):
        mod = resample.ResampleFrac(5, 7)
        x = tf.random.normal((5 * 26,))
        fn = tf.function(mod.__call__)
        self.assertEqual(list(fn(x).shape), [7 * 26])

    def test_constant(self):
        x = tf.ones((4096,))
        for zeros in [4, 10]:
            for old_sr in [1, 4, 10]:
                for new_sr in [1, 4, 10]:
                    y_low = resample.resample_frac(x, old_sr, new_sr, zeros=zeros)
                    self.assertLessEqual(
                        float(tf.reduce_mean(tf.abs(y_low - 1))), 1e-6, (zeros, old_sr, new_sr))

    def test_default_output_length(self):
        x = tf.ones((1, 2, 32000))

        resampler = resample.ResampleFrac(old_sr=32000, new_sr=48000)
        y = resampler(x)
        self.assertEqual(tuple(y.shape), (1, 2, 48000))

        # Test functional version as well
        y = resample.resample_frac(x, old_sr=32000, new_sr=48000)
        self.assertEqual(tuple(y.shape), (1, 2, 48000))

    def test_custom_output_length(self):
        x = tf.ones((1, 32001))

        resampler = resample.ResampleFrac(old_sr=32000, new_sr=48000)
        y = resampler(x, output_length=48001)
        self.assertEqual(tuple(y.shape), (1, 48001))

        # Test functional version as well
        y = resample.resample_frac(x, old_sr=32000, new_sr=48000, output_length=47999)
        self.assertEqual(tuple(y.shape), (1, 47999))

    def test_custom_output_length_extreme_resampling(self):
        """
        Resample a signal from 1 hz to 499 hz to check that custom_length works
        correctly without extra internal padding
        """
        x = tf.ones((1, 1))

        resampler = resample.ResampleFrac(old_sr=1, new_sr=499)
        y = resampler(x, output_length=499)
        self.assertEqual(tuple(y.shape), (1, 499))

        # Test functional version as well
        y = resample.resample_frac(x, old_sr=1, new_sr=499, output_length=3)
        self.assertEqual(tuple(y.shape), (1, 3))

    def test_custom_output_length_out_of_range(self):
        x = tf.ones((1, 32000))
        with self.assertRaisesRegex(
            ValueError, "output_length must be between 0 and 48000"
        ):
            resample.resample_frac(x, old_sr=32000, new_sr=48000, output_length=48002)

    def test_full(self):
        x = tf.random.normal((19,))
        y = resample.resample_frac(x, 7, 1, full=True)
        self.assertEqual(len(y), 3)
        z = resample.resample_frac(y, 5, 1, full=True)
        y2 = resample.resample_frac(z, 1, 5, full=True)
        x2 = resample.resample_frac(y2, 1, 7, output_length=len(x))
        self.assertEqual(x.shape, x2.shape)

    def test_dynamic_length_graph(self):
        """
        Trace the resampler with an unknown (dynamic) time dimension inside a `tf.function`
        and check it produces the right output length for several input sizes. This mirrors
        the dynamic-input-length scenario that the original library tested through ONNX.
        """
        resampler = ResampleFrac(old_sr=32_000, new_sr=16_000)
        fn = tf.function(
            lambda t: resampler(t),
            input_signature=[tf.TensorSpec([1, None], tf.float32)])

        out1 = fn(tf.random.uniform((1, 100)))
        self.assertEqual(out1.shape[-1], 50)

        x = tf.random.uniform((1, 124))
        out2 = fn(x)
        self.assertEqual(out2.shape[-1], 62)

        # The graph (traced) output should match the eager one.
        self.assertSimilar(fn(x), resampler(x), x)


if __name__ == '__main__':
    unittest.main()
