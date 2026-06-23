# File under the MIT license, see https://github.com/adefossez/julius/LICENSE for details.
# Author: adefossez, 2020
"""
Signal processing or TensorFlow related utilities.
"""
import math

import tensorflow as tf


def _log10(x: tf.Tensor) -> tf.Tensor:
    """Base 10 logarithm (`tf.math.log` only provides the natural logarithm)."""
    return tf.math.log(x) / tf.math.log(tf.cast(10.0, x.dtype))


def sinc(x: tf.Tensor) -> tf.Tensor:
    """
    Implementation of sinc, i.e. sin(x) / x

    __Warning__: the input is not multiplied by `pi`!
    """
    return tf.where(tf.equal(x, tf.cast(0.0, x.dtype)), tf.ones_like(x), tf.sin(x) / x)


def _pad_last(x: tf.Tensor, left, right, mode: str = 'CONSTANT', value: float = 0.) -> tf.Tensor:
    """Pad the last dimension of `x` with `left` values on the left and `right` on the right."""
    rank = len(x.shape)
    edges = tf.reshape(tf.stack([tf.cast(left, tf.int32), tf.cast(right, tf.int32)]), [1, 2])
    paddings = tf.concat([tf.zeros([rank - 1, 2], tf.int32), edges], axis=0)
    return tf.pad(x, paddings, mode=mode, constant_values=value)


def pad_to(tensor: tf.Tensor, target_length: int, mode: str = 'constant', value: float = 0.):
    """
    Pad the given tensor to the given length, with 0s on the right.
    """
    return _pad_last(tensor, 0, target_length - tf.shape(tensor)[-1], value=value)


def pad_replicate(x: tf.Tensor, left: int, right: int) -> tf.Tensor:
    """
    Pad the last dimension of `x` by replicating its edge values (equivalent to
    PyTorch's ``mode='replicate'``). This avoids discontinuities at the borders that
    would otherwise create strong artifacts when filtering.
    """
    left_pad = tf.repeat(x[..., :1], left, axis=-1)
    right_pad = tf.repeat(x[..., -1:], right, axis=-1)
    return tf.concat([left_pad, x, right_pad], axis=-1)


def conv1d(input: tf.Tensor, weight: tf.Tensor, stride: int = 1, padding: int = 0) -> tf.Tensor:
    """
    1D convolution (cross-correlation) following the `torch.nn.functional.conv1d` convention,
    implemented on top of `tf.nn.conv1d`.

    Args:
        input (Tensor): input signal of shape `[B, C, T]` (channels first).
        weight (Tensor): convolution weight of shape `[D, C, K]` with `D` the number
            of output channels.
        stride (int): stride of the convolution.
        padding (int): amount of zero padding applied to both sides of the input.
    """
    if padding:
        input = _pad_last(input, padding, padding, value=0.)
    x = tf.transpose(input, [0, 2, 1])  # [B, T, C], i.e. NWC as expected by tf.nn.conv1d
    w = tf.transpose(weight, [2, 1, 0])  # [K, C, D]
    y = tf.nn.conv1d(x, w, stride=stride, padding='VALID')
    return tf.transpose(y, [0, 2, 1])  # back to [B, D, T']


def hz_to_mel(freqs: tf.Tensor):
    """
    Converts a Tensor of frequencies in hertz to the mel scale.
    Uses the simple formula by O'Shaughnessy (1987).

    Args:
        freqs (tf.Tensor): frequencies to convert.

    """
    return 2595 * _log10(1 + freqs / 700)


def mel_to_hz(mels: tf.Tensor):
    """
    Converts a Tensor of mel scaled frequencies to Hertz.
    Uses the simple formula by O'Shaughnessy (1987).

    Args:
        mels (tf.Tensor): mel frequencies to convert.
    """
    return 700 * (tf.pow(tf.cast(10.0, mels.dtype), mels / 2595) - 1)


def mel_frequencies(n_mels: int, fmin: float, fmax: float):
    """
    Return frequencies that are evenly spaced in mel scale.

    Args:
        n_mels (int): number of frequencies to return.
        fmin (float): start from this frequency (in Hz).
        fmax (float): finish at this frequency (in Hz).


    """
    low = float(hz_to_mel(tf.constant(float(fmin), tf.float32)))
    high = float(hz_to_mel(tf.constant(float(fmax), tf.float32)))
    mels = tf.linspace(low, high, n_mels)
    return mel_to_hz(mels)


def volume(x: tf.Tensor, floor=1e-8):
    """
    Return the volume in dBFS.
    """
    return _log10(floor + tf.reduce_mean(x**2, axis=-1)) * 10


def pure_tone(freq: float, sr: float = 128, dur: float = 4):
    """
    Return a pure tone, i.e. cosine.

    Args:
        freq (float): frequency (in Hz)
        sr (float): sample rate (in Hz)
        dur (float): duration (in seconds)
    """
    time = tf.range(int(sr * dur), dtype=tf.float32) / sr
    return tf.cos(2 * math.pi * freq * time)


def unfold(input: tf.Tensor, kernel_size: int, stride: int):
    """1D only unfolding similar to the one from PyTorch.

    Given an input tensor of size `[*, T]` this will return
    a tensor `[*, F, K]` with `K` the kernel size, and `F` the number
    of frames. The i-th frame is `i * stride: i * stride + kernel_size`.
    This will automatically pad the input to cover at least once all entries in `input`.

    Args:
        input (Tensor): tensor for which to return the frames.
        kernel_size (int): size of each frame.
        stride (int): stride between each frame.

    Shape:

        - Inputs: `input` is `[*, T]`
        - Output: `[*, F, kernel_size]` with `F = 1 + ceil((T - kernel_size) / stride)`


    ..Warning:: unlike PyTorch unfold, this will pad the input
        so that any position in `input` is covered by at least one frame.

    Implemented on top of `tf.signal.frame`, the natural TensorFlow primitive for this.
    """
    length = tf.shape(input)[-1]
    covered = tf.maximum(length, kernel_size) - kernel_size
    n_frames = (covered + stride - 1) // stride + 1
    tgt_length = (n_frames - 1) * stride + kernel_size
    padded = _pad_last(input, 0, tgt_length - length, value=0.)
    return tf.signal.frame(padded, kernel_size, stride, axis=-1)
