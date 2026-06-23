# File under the MIT license, see the LICENSE file for details.
# Original author: Alexandre Défossez (adefossez), 2020
# TensorFlow port: Clément Laroche (LarocheC), 2026

import random
import unittest

import tensorflow as tf

import julius
from julius.core import conv1d

# As relative delta in percentage. FFT and direct convolutions accumulate float32
# rounding differently (more so on older TensorFlow / AVX2 builds), so we allow a
# small relative discrepancy. A real bug would produce an O(1) error, far above this.
TOLERANCE = 1e-2


def ref_conv1d(input, weight, bias=None, stride=1, padding=0):
    """Reference channels-first 1D convolution, backed by `tf.nn.conv1d`."""
    out = conv1d(input, weight, stride=stride, padding=padding)
    if bias is not None:
        out = out + bias[:, tf.newaxis]
    return out


class _BaseTest(unittest.TestCase):
    def setUp(self):
        tf.random.set_seed(1234)
        random.seed(1234)

    def assertSimilar(self, a, b, msg=None, tol=TOLERANCE):
        delta = float(100 * tf.norm(a - b) / tf.norm(b))
        self.assertLessEqual(delta, tol, msg)

    def compare_reference(self, *args, block_ratio=10, msg=None, tol=TOLERANCE, **kwargs):
        y_ref = ref_conv1d(*args, **kwargs)
        y = julius.fft_conv1d(*args, block_ratio=block_ratio, **kwargs)
        self.assertEqual(list(y.shape), list(y_ref.shape), msg)
        self.assertSimilar(y, y_ref, msg, tol)


class TestFFTConv1d(_BaseTest):
    def test_same_as_reference(self):
        for _ in range(5):
            kernel_size = random.randrange(4, 128)
            batch_size = random.randrange(1, 6)
            length = random.randrange(kernel_size, 1024)
            chin = random.randrange(1, 12)
            chout = random.randrange(1, 12)
            block_ratio = random.choice([5, 10, 20])
            bias = random.random() < 0.5
            if random.random() < 0.5:
                padding = 0
            else:
                padding = random.randrange(kernel_size // 2, 2 * kernel_size)
            x = tf.random.normal((batch_size, chin, length))
            w = tf.random.normal((chout, chin, kernel_size))
            keys = ["length", "kernel_size", "chin", "chout", "block_ratio", "bias"]
            loc = dict(locals())
            state = {key: loc[key] for key in keys}
            if bias:
                bias = tf.random.normal((chout,))
            else:
                bias = None
            for stride in [1, 2, 5]:
                state["stride"] = stride
                self.compare_reference(
                    x, w, bias, stride, padding, block_ratio=block_ratio,
                    msg=repr(state))

    def test_small_input(self):
        x = tf.random.normal((1, 5, 19))
        w = tf.random.normal((10, 5, 32))
        with self.assertRaises(RuntimeError):
            julius.fft_conv1d(x, w)

        x = tf.random.normal((1, 5, 19))
        w = tf.random.normal((10, 5, 19))
        self.assertEqual(list(julius.fft_conv1d(x, w).shape), [1, 10, 1])

    def test_block_ratio(self):
        x = tf.random.normal((1, 5, 1024))
        w = tf.random.normal((10, 5, 19))
        ref = julius.fft_conv1d(x, w)
        for block_ratio in [1, 5, 10, 20]:
            y = julius.fft_conv1d(x, w, block_ratio=block_ratio)
            self.assertSimilar(y, ref, msg=str(block_ratio))

        with self.assertRaises(RuntimeError):
            y = julius.fft_conv1d(x, w, block_ratio=0.9)

    def test_module(self):
        x = tf.random.normal((16, 4, 1024))
        mod = julius.FFTConv1d(4, 5, 8, bias=True)
        mod(x)
        mod = julius.FFTConv1d(4, 5, 8, bias=False)
        mod(x)

    def test_tf_function(self):
        x = tf.random.normal((16, 4, 1024))
        mod = julius.FFTConv1d(4, 5, 8, bias=True)
        fn = tf.function(mod.__call__)
        self.assertEqual(list(fn(x).shape), [16, 5, 1024 - 8 + 1])

    def test_repr(self):
        mod = julius.FFTConv1d(4, 5, 8, bias=False)
        self.assertEqual(
            repr(mod),
            "FFTConv1d(in_channels=4,out_channels=5,kernel_size=8,bias=False)")
