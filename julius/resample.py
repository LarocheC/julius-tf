# File under the MIT license, see the LICENSE file for details.
# Original author: Alexandre Défossez (adefossez), 2020
# TensorFlow port: Clément Laroche (LarocheC), 2026
"""
Differentiable, TensorFlow based resampling.
Implementation of Julius O. Smith algorithm for resampling.
See https://ccrma.stanford.edu/~jos/resample/ for details.
This implementation is specially optimized for when new_sr / old_sr is a fraction
with a small numerator and denominator when removing the gcd (e.g. new_sr = 700, old_sr = 500).

Very similar to [bmcfee/resampy](https://github.com/bmcfee/resampy) except this implementation
is optimized for the case mentioned before, while resampy is slower but more general.

"""

import math
from typing import Optional

import tensorflow as tf

from .core import sinc, conv1d, pad_replicate, _pad_last
from .utils import simple_repr


class ResampleFrac(tf.Module):
    """
    Resampling from the sample rate `old_sr` to `new_sr`.
    """
    def __init__(self, old_sr: int, new_sr: int, zeros: int = 24, rolloff: float = 0.945):
        """
        Args:
            old_sr (int): sample rate of the input signal x.
            new_sr (int): sample rate of the output.
            zeros (int): number of zero crossing to keep in the sinc filter.
            rolloff (float): use a lowpass filter that is `rolloff * new_sr / 2`,
                to ensure sufficient margin due to the imperfection of the FIR filter used.
                Lowering this value will reduce anti-aliasing, but will reduce some of the
                highest frequencies.

        Shape:

            - Input: `[*, T]`
            - Output: `[*, T']` with `T' = int(new_sr * T / old_sr)


        .. caution::
            After dividing `old_sr` and `new_sr` by their GCD, both should be small
            for this implementation to be fast.

        >>> import tensorflow as tf
        >>> resample = ResampleFrac(4, 5)
        >>> x = tf.random.normal((1000,))
        >>> print(len(resample(x)))
        1250
        """
        super().__init__()
        if not isinstance(old_sr, int) or not isinstance(new_sr, int):
            raise ValueError("old_sr and new_sr should be integers")
        gcd = math.gcd(old_sr, new_sr)
        self.old_sr = old_sr // gcd
        self.new_sr = new_sr // gcd
        self.zeros = zeros
        self.rolloff = rolloff

        self._width = 0
        self.kernel: Optional[tf.Tensor] = None
        self._init_kernels()

    def _init_kernels(self):
        if self.old_sr == self.new_sr:
            return

        kernels = []
        sr = min(self.new_sr, self.old_sr)
        # rolloff will perform antialiasing filtering by removing the highest frequencies.
        # At first I thought I only needed this when downsampling, but when upsampling
        # you will get edge artifacts without this, the edge is equivalent to zero padding,
        # which will add high freq artifacts.
        sr *= self.rolloff

        # The key idea of the algorithm is that x(t) can be exactly reconstructed from x[i] (tensor)
        # using the sinc interpolation formula:
        #   x(t) = sum_i x[i] sinc(pi * old_sr * (i / old_sr - t))
        # We can then sample the function x(t) with a different sample rate:
        #    y[j] = x(j / new_sr)
        # or,
        #    y[j] = sum_i x[i] sinc(pi * old_sr * (i / old_sr - j / new_sr))

        # We see here that y[j] is the convolution of x[i] with a specific filter, for which
        # we take an FIR approximation, stopping when we see at least `zeros` zeros crossing.
        # But y[j+1] is going to have a different set of weights and so on, until y[j + new_sr].
        # Indeed:
        # y[j + new_sr] = sum_i x[i] sinc(pi * old_sr * ((i / old_sr - (j + new_sr) / new_sr))
        #               = sum_i x[i] sinc(pi * old_sr * ((i - old_sr) / old_sr - j / new_sr))
        #               = sum_i x[i + old_sr] sinc(pi * old_sr * (i / old_sr - j / new_sr))
        # so y[j+new_sr] uses the same filter as y[j], but on a shifted version of x by `old_sr`.
        # This will explain the conv1d after, with a stride of old_sr.
        self._width = math.ceil(self.zeros * self.old_sr / sr)
        # If old_sr is still big after GCD reduction, most filters will be very unbalanced, i.e.,
        # they will have a lot of almost zero values to the left or to the right...
        # There is probably a way to evaluate those filters more efficiently, but this is kept for
        # future work.
        idx = tf.range(-self._width, self._width + self.old_sr, dtype=tf.float32)
        for i in range(self.new_sr):
            t = (-i / self.new_sr + idx / self.old_sr) * sr
            t = tf.clip_by_value(t, -self.zeros, self.zeros)
            t = t * math.pi
            window = tf.cos(t / self.zeros / 2)**2
            kernel = sinc(t) * window
            # Renormalize kernel to ensure a constant signal is preserved.
            kernel = kernel / tf.reduce_sum(kernel)
            kernels.append(kernel)

        self.kernel = tf.reshape(tf.stack(kernels), [self.new_sr, 1, -1])

    def __call__(self, x: tf.Tensor, output_length: Optional[int] = None, full: bool = False):
        """
        Resample x.
        Args:
            x (Tensor): signal to resample, time should be the last dimension
            output_length (None or int): This can be set to the desired output length
                (last dimension). Allowed values are between 0 and
                ceil(length * new_sr / old_sr). When None (default) is specified, the
                floored output length will be used. In order to select the largest possible
                size, use the `full` argument.
            full (bool): return the longest possible output from the input. This can be useful
                if you chain resampling operations, and want to give the `output_length` only
                for the last one, while passing `full=True` to all the other ones.
        """
        if self.old_sr == self.new_sr:
            return x
        shape = tf.shape(x)
        length = shape[-1]
        x = tf.reshape(x, tf.stack([-1, 1, length]))
        x = pad_replicate(x, self._width, self._width + self.old_sr)
        ys = conv1d(x, self.kernel, stride=self.old_sr)  # [N, new_sr, T']
        y = tf.transpose(ys, [0, 2, 1])  # [N, T', new_sr]
        y = tf.reshape(y, tf.concat([shape[:-1], [-1]], axis=0))

        # Cast `length` to float64 *before* multiplying: `length` is an int32 tensor
        # (from tf.shape), so `self.new_sr * length` would overflow int32 for large
        # sample rates (e.g. 30001 * 480024 wraps around) before the cast applied.
        float_output_length = self.new_sr * tf.cast(length, tf.float64) / self.old_sr
        max_output_length = tf.cast(tf.math.ceil(float_output_length), tf.int32)
        default_output_length = tf.cast(tf.math.floor(float_output_length), tf.int32)

        if output_length is None:
            applied_output_length = max_output_length if full else default_output_length
        elif output_length < 0 or output_length > int(max_output_length):
            raise ValueError(f"output_length must be between 0 and {int(max_output_length)}")
        else:
            applied_output_length = output_length
            if full:
                raise ValueError("You cannot pass both full=True and output_length")
        return y[..., :applied_output_length]

    def __repr__(self):
        return simple_repr(self)


def resample_frac(x: tf.Tensor, old_sr: int, new_sr: int,
                  zeros: int = 24, rolloff: float = 0.945,
                  output_length: Optional[int] = None, full: bool = False):
    """
    Functional version of `ResampleFrac`, refer to its documentation for more information.

    ..warning::
        If you call repeatidly this functions with the same sample rates, then the
        resampling kernel will be recomputed everytime. For best performance, you should use
        and cache an instance of `ResampleFrac`.
    """
    return ResampleFrac(old_sr, new_sr, zeros, rolloff)(x, output_length, full)


# Easier implementations for downsampling and upsampling by a factor of 2
# Kept for testing and reference

def _kernel_upsample2_downsample2(zeros):
    # Kernel for upsampling and downsampling by a factor of 2. Interestingly,
    # it is the same kernel used for both.
    win = tf.signal.hann_window(4 * zeros + 1, periodic=False)
    winodd = win[1::2]
    t = tf.linspace(-zeros + 0.5, zeros - 0.5, 2 * zeros)
    t = t * math.pi
    kernel = tf.reshape(sinc(t) * winodd, [1, 1, -1])
    return kernel


def _upsample2(x, zeros=24):
    """
    Upsample x by a factor of two. The output will be exactly twice as long as the input.
    Args:
        x (Tensor): signal to upsample, time should be the last dimension
        zeros (int): number of zero crossing to keep in the sinc filter.

    This function is kept only for reference, you should use the more generic `resample_frac`
    one. This function does not perform anti-aliasing filtering.
    """
    *other, time = x.shape.as_list()
    kernel = _kernel_upsample2_downsample2(zeros)
    out = conv1d(tf.reshape(x, [-1, 1, time]), kernel, padding=zeros)[..., 1:]
    out = tf.reshape(out, other + [time])
    y = tf.stack([x, out], axis=-1)
    return tf.reshape(y, other + [-1])


def _downsample2(x, zeros=24):
    """
    Downsample x by a factor of two. The output length is half of the input, ceiled.
    Args:
        x (Tensor): signal to downsample, time should be the last dimension
        zeros (int): number of zero crossing to keep in the sinc filter.

    This function is kept only for reference, you should use the more generic `resample_frac`
    one. This function does not perform anti-aliasing filtering.
    """
    if x.shape[-1] % 2 != 0:
        x = _pad_last(x, 0, 1, value=0.)
    xeven = x[..., ::2]
    xodd = x[..., 1::2]
    *other, time = xodd.shape.as_list()
    kernel = _kernel_upsample2_downsample2(zeros)
    out = conv1d(tf.reshape(xodd, [-1, 1, time]), kernel, padding=zeros)[..., :-1]
    out = xeven + tf.reshape(out, other + [time])
    return tf.reshape(out, other + [-1]) * 0.5
