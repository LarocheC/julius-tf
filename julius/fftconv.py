# File under the MIT license, see https://github.com/LarocheC/julius-tf/blob/main/LICENSE for details.
# Original author: Alexandre Défossez (adefossez), 2020
# TensorFlow port: Clément Laroche (LarocheC), 2026

"""
Implementation of a FFT based 1D convolution in TensorFlow.
While FFT is used in CUDNN for small kernel sizes, it is not the case for long ones, e.g. 512.
This module implements efficient FFT based convolutions for such convolutions. A typical
application is for evaluationg FIR filters with a long receptive field, typically
evaluated with a stride of 1.
"""
import math
from typing import Optional

import tensorflow as tf

from .core import pad_to, unfold, _pad_last
from .utils import simple_repr


def _compl_mul_conjugate(a: tf.Tensor, b: tf.Tensor):
    """
    Given `a` and `b` two complex tensors, returns `a` multiplied by the conjugate of `b`,
    the multiplication being with respect to the channel dimension.
    """
    return tf.einsum("bcft,dct->bdft", a, tf.math.conj(b))


def fft_conv1d(
        input: tf.Tensor, weight: tf.Tensor,
        bias: Optional[tf.Tensor] = None, stride: int = 1, padding: int = 0,
        block_ratio: float = 5):
    """
    Same as `tf.nn.conv1d` (with the `torch.nn.functional.conv1d` channels-first convention)
    but using FFT for the convolution.

    Args:
        input (Tensor): input signal of shape `[B, C, T]`.
        weight (Tensor): weight of the convolution `[D, C, K]` with `D` the number
            of output channels.
        bias (Tensor or None): if not None, bias term for the convolution.
        stride (int): stride of convolution.
        padding (int): padding to apply to the input.
        block_ratio (float): can be tuned for speed. The input is splitted in chunks
            with a size of `int(block_ratio * kernel_size)`.

    Shape:

        - Inputs: `input` is `[B, C, T]`, `weight` is `[D, C, K]` and bias is `[D]`.
        - Output: `(*, T)`


    ..note::
        This function is faster than a regular convolution only in specific cases.
        Typically, the kernel size should be of the order of 256 to see any real gain,
        for a stride of 1.

    ..Warning::
        Dilation and groups are not supported at the moment. This function might use
        more memory than a regular convolution. It also requires the input length to be
        statically known (e.g. avoid `None` time dimensions inside `tf.function`).
    """
    if padding:
        input = _pad_last(input, padding, padding, value=0.)
    out_channels = int(weight.shape[0])
    kernel_size = int(weight.shape[-1])
    length = input.shape[-1]
    if length is None:
        raise RuntimeError("fft_conv1d requires a statically known input length.")
    length = int(length)

    if length < kernel_size:
        raise RuntimeError(f"Input should be at least as large as the kernel size {kernel_size}, "
                           f"but it is only {length} samples long.")
    if block_ratio < 1:
        raise RuntimeError("Block ratio must be greater than 1.")

    # We are going to process the input blocks by blocks, as for some reason it is faster
    # and less memory intensive (I think the culprit is the einsum).
    block_size: int = min(int(kernel_size * block_ratio), length)
    fold_stride = block_size - kernel_size + 1
    weight = pad_to(weight, block_size)
    weight_z = tf.signal.rfft(weight)

    # We pad the input and get the different frames, on which
    frames = unfold(input, block_size, fold_stride)

    frames_z = tf.signal.rfft(frames)
    out_z = _compl_mul_conjugate(frames_z, weight_z)
    out = tf.signal.irfft(out_z, fft_length=[block_size])
    # The last bit is invalid, because FFT will do a circular convolution.
    out = out[..., :-kernel_size + 1]
    out = tf.reshape(out, tf.stack([tf.shape(out)[0], out_channels, -1]))
    out = out[..., ::stride]
    target_length = (length - kernel_size) // stride + 1
    out = out[..., :target_length]
    if bias is not None:
        out = out + bias[:, tf.newaxis]
    return out


class FFTConv1d(tf.Module):
    """
    Same as `tf.keras.layers.Conv1D` / `torch.nn.Conv1d` but based on `fft_conv1d`.

    Args:
        in_channels (int): number of input channels.
        out_channels (int): number of output channels.
        kernel_size (int): kernel size of convolution.
        stride (int): stride of convolution.
        padding (int): padding to apply to the input.
        bias (bool): if True, use a bias term.

    ..note::
        This module is faster than a regular convolution only in specific cases.
        Typically, `kernel_size` should be of the order of 256 to see any real gain,
        for a stride of 1.

    ..warning::
        Dilation and groups are not supported at the moment. This module might use
        more memory than a regular convolution.

    >>> import tensorflow as tf
    >>> fftconv = FFTConv1d(12, 24, 128, 4)
    >>> x = tf.random.normal((4, 12, 1024))
    >>> print(list(fftconv(x).shape))
    [4, 24, 225]
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0, bias: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Mirror the default initialization of a channels-first 1D convolution.
        bound = 1 / math.sqrt(in_channels * kernel_size)
        initializer = tf.random_uniform_initializer(-bound, bound)
        self.weight = tf.Variable(
            initializer([out_channels, in_channels, kernel_size]), name="weight")
        if bias:
            self.bias: Optional[tf.Variable] = tf.Variable(
                initializer([out_channels]), name="bias")
        else:
            self.bias = None

    def __call__(self, input: tf.Tensor):
        return fft_conv1d(
            input, self.weight, self.bias, self.stride, self.padding)

    def __repr__(self):
        return simple_repr(self, overrides={"bias": self.bias is not None})
