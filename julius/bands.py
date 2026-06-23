# File under the MIT license, see the LICENSE file for details.
# Original author: Alexandre Défossez (adefossez), 2020
# TensorFlow port: Clément Laroche (LarocheC), 2026
"""
Decomposition of a signal over frequency bands in the waveform domain.
"""
from typing import Optional, Sequence

import tensorflow as tf

from .core import mel_frequencies
from .lowpass import LowPassFilters
from .utils import simple_repr


class SplitBands(tf.Module):
    """
    Decomposes a signal over the given frequency bands in the waveform domain using
    a cascade of low pass filters as implemented by `julius.lowpass.LowPassFilters`.
    You can either specify explicitely the frequency cutoffs, or just the number of bands,
    in which case the frequency cutoffs will be spread out evenly in mel scale.

    Args:
        sample_rate (float): Sample rate of the input signal in Hz.
        n_bands (int or None): number of bands, when not giving them explictely with `cutoffs`.
            In that case, the cutoff frequencies will be evenly spaced in mel-space.
        cutoffs (list[float] or None): list of frequency cutoffs in Hz.
        pad (bool): if True, appropriately pad the input with zero over the edge. If `stride=1`,
            the output will have the same length as the input.
        zeros (float): Number of zero crossings to keep. See `LowPassFilters` for more informations.
        fft (bool or None): See `LowPassFilters` for more info.

    ..note::
        The sum of all the bands will always be the input signal.

    ..warning::
        Unlike `julius.lowpass.LowPassFilters`, the cutoffs frequencies must be provided in Hz along
        with the sample rate.

    Shape:

        - Input: `[*, T]`
        - Output: `[B, *, T']`, with `T'=T` if `pad` is True.
            If `n_bands` was provided, `B = n_bands` otherwise `B = len(cutoffs) + 1`

    >>> import tensorflow as tf
    >>> bands = SplitBands(sample_rate=128, n_bands=10)
    >>> x = tf.random.normal((6, 4, 1024))
    >>> list(bands(x).shape)
    [10, 6, 4, 1024]
    """

    def __init__(self, sample_rate: float, n_bands: Optional[int] = None,
                 cutoffs: Optional[Sequence[float]] = None, pad: bool = True,
                 zeros: float = 8, fft: Optional[bool] = None):
        super().__init__()
        if (cutoffs is None) + (n_bands is None) != 1:
            raise ValueError("You must provide either n_bands, or cutoffs, but not boths.")

        self.sample_rate = sample_rate
        self.n_bands = n_bands
        self._cutoffs = list(cutoffs) if cutoffs is not None else None
        self.pad = pad
        self.zeros = zeros
        self.fft = fft

        if cutoffs is None:
            if n_bands is None:
                raise ValueError("You must provide one of n_bands or cutoffs.")
            if not n_bands >= 1:
                raise ValueError(f"n_bands must be greater than one (got {n_bands})")
            cutoffs = mel_frequencies(n_bands + 1, 0, sample_rate / 2).numpy()[1:-1].tolist()
        else:
            if max(cutoffs) > 0.5 * sample_rate:
                raise ValueError("A cutoff above sample_rate/2 does not make sense.")
        if len(cutoffs) > 0:
            self.lowpass: Optional[LowPassFilters] = LowPassFilters(
                [c / sample_rate for c in cutoffs], pad=pad, zeros=zeros, fft=fft)
        else:
            self.lowpass = None

    def __call__(self, input):
        if self.lowpass is None:
            return input[tf.newaxis]
        lows = self.lowpass(input)
        n = len(self.lowpass.cutoffs)
        low = lows[0]
        bands = [low]
        for i in range(1, n):
            low_and_band = lows[i]
            # Get a bandpass filter by substracting lowpasses
            band = low_and_band - low
            bands.append(band)
            low = low_and_band
        # Last band is whatever is left in the signal
        bands.append(input - low)
        return tf.stack(bands)

    @property
    def cutoffs(self):
        if self._cutoffs is not None:
            return self._cutoffs
        elif self.lowpass is not None:
            return [c * self.sample_rate for c in self.lowpass.cutoffs]
        else:
            return []

    def __repr__(self):
        return simple_repr(self, overrides={"cutoffs": self._cutoffs})


def split_bands(signal: tf.Tensor, sample_rate: float, n_bands: Optional[int] = None,
                cutoffs: Optional[Sequence[float]] = None, pad: bool = True,
                zeros: float = 8, fft: Optional[bool] = None):
    """
    Functional version of `SplitBands`, refer to this class for more information.

    >>> import tensorflow as tf
    >>> x = tf.random.normal((6, 4, 1024))
    >>> list(split_bands(x, sample_rate=64, cutoffs=[12, 24]).shape)
    [3, 6, 4, 1024]
    """
    return SplitBands(sample_rate, n_bands, cutoffs, pad, zeros, fft)(signal)
