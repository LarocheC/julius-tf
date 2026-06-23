# File under the MIT license, see the LICENSE file for details.
# Original author: Alexandre Défossez (adefossez), 2020
# TensorFlow port: Clément Laroche (LarocheC), 2026

# flake8: noqa
"""
.. image:: ../logo.png

Julius contains different Digital Signal Processing algorithms implemented
with TensorFlow, so that they are differentiable and available on GPU.
Note that all the modules implemented here can be used inside a `tf.function`.

For now, I have implemented:

- `julius.resample`: fast sinc resampling.
- `julius.fftconv`: FFT based convolutions.
- `julius.lowpass`: FIR low pass filter banks.
- `julius.filters`: FIR high pass and band pass filters.
- `julius.bands`: Decomposition of a waveform signal over mel-scale frequency bands.

Along that, you might found useful utilities in:

- `julius.core`: DSP related functions.
- `julius.utils`: Generic utilities.


Please checkout [the Github repository](https://github.com/LarocheC/julius-tf) for other informations.
For a verification of the speed and correctness of Julius, check the benchmark module `bench`.


This package is named in this honor of
[Julius O. Smith](https://ccrma.stanford.edu/~jos/),
whose books and website were a gold mine of information for me to learn about DSP. Go checkout his website if you want
to learn more about DSP.
"""

from .bands import SplitBands, split_bands
from .fftconv import fft_conv1d, FFTConv1d
from .filters import bandpass_filter, BandPassFilter
from .filters import highpass_filter, highpass_filters, HighPassFilter, HighPassFilters
from .lowpass import lowpass_filter, lowpass_filters, LowPassFilters, LowPassFilter
from .resample import resample_frac, ResampleFrac
