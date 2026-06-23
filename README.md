# Julius, fast TensorFlow based DSP for audio and 1D signals

![linter badge](https://github.com/LarocheC/julius-tf/workflows/linter/badge.svg)
![tests badge](https://github.com/LarocheC/julius-tf/workflows/tests/badge.svg)
![cov badge](https://github.com/LarocheC/julius-tf/workflows/cov%3E90%25/badge.svg)

Julius contains different Digital Signal Processing algorithms implemented
with TensorFlow, so that they are differentiable and available on GPU.
Note that all the modules implemented here can be used inside a `tf.function`.

> **`julius-tf` is a TensorFlow port of [`julius`](https://github.com/adefossez/julius),
> the PyTorch DSP library by [Alexandre Défossez](https://github.com/adefossez).** The DSP
> algorithms and the public API are his work; this project re-implements them on top of
> TensorFlow. See [Credits](#credits) for full attribution.

For now, I have implemented:

- [julius.resample](https://LarocheC.github.io/julius-tf/julius/resample.html): fast sinc resampling.
- [julius.fftconv](https://LarocheC.github.io/julius-tf/julius/fftconv.html): FFT based convolutions.
- [julius.lowpass](https://LarocheC.github.io/julius-tf/julius/lowpass.html): FIR low pass filter banks.
- [julius.filters](https://LarocheC.github.io/julius-tf/julius/filters.html): FIR high pass and band pass filters.
- [julius.bands](https://LarocheC.github.io/julius-tf/julius/bands.html): Decomposition of a waveform signal over mel-scale frequency bands.

Along that, you might found useful utilities in:

- [julius.core](https://LarocheC.github.io/julius-tf/julius/core.html): DSP related functions.
- [julius.utils](https://LarocheC.github.io/julius-tf/julius/utils.html): Generic utilities.

<p align="center">
<img src="./logo.png" alt="Representation of the convolutions filters used for the efficient resampling."
width="500px"></p>

## News

- `julius-tf` ports the whole library from PyTorch to __TensorFlow__. The public API of the
  original `julius` is preserved: modules are `tf.Module`s, callable just like before, and
  usable inside a `tf.function`. The dated entries below are the upstream `julius` releases
  whose behavior this port reproduces.
- 23/06/2026: __`julius-tf` 0.1.0:__ first release on PyPI — TensorFlow port reproducing
  upstream `julius` 0.2.8. Install with `pip install julius-tf`.
- 03/06/2026: __`julius` 0.2.8 released:__: Switching to pyproject.toml, now requires python >= 3.9. Bug fix with -O flag (thanks @aiknownc)
- 19/09/2022: __`julius` 0.2.7 released:__: fixed ONNX compat (thanks @iver56). I know I missed the 0.2.6 one...
- 28/07/2021: __`julius` 0.2.5 released:__: support for setting a custom output length when resampling.
- 22/06/2021: __`julius` 0.2.4 released:__: adding highpass and band passfilters.
  Extra linting and type checking of the code. New `unfold` implemention, up to
  x6 faster FFT convolutions and more efficient memory usage.
- 26/01/2021: __`julius` 0.2.2 released:__ fixing normalization of filters in lowpass and resample to avoid very low frequencies to be leaked.
  Switch from zero padding to replicate padding (uses first/last value instead of 0) to avoid discontinuities with strong artifacts.
- 20/01/2021: `julius` implementation of resampling is now officially <a href="https://github.com/pytorch/audio/pull/1087">part of Torchaudio.</a>

## Installation

`julius-tf` requires python >= 3.9 and TensorFlow >= 2.11. To install:
```bash
pip3 install -U julius-tf
```
The import name stays `julius` (i.e. `pip install julius-tf` then `import julius`).


## Usage

See the [Julius documentation][docs] for the usage of Julius. Hereafter you will find a few examples
to get you quickly started:

```python3
import julius
import tensorflow as tf

signal = tf.random.normal((6, 4, 1024))
# Resample from a sample rate of 100 to 70. The old and new sample rate must be integers,
# and resampling will be fast if they form an irreductible fraction with small numerator
# and denominator (here 10 and 7). Any shape is supported, last dim is time.
resampled_signal = julius.resample_frac(signal, 100, 70)

# Low pass filter with a `0.1 * sample_rate` cutoff frequency.
low_freqs = julius.lowpass_filter(signal, 0.1)

# Fast convolutions with FFT, useful for large kernels
conv = julius.FFTConv1d(4, 10, 512)
convolved = conv(signal)

# Decomposition over frequency bands in the Waveform domain
bands = julius.split_bands(signal, n_bands=10, sample_rate=100)
# Decomposition with n_bands frequency bands evenly spaced in mel space.
# Input shape can be `[*, T]`, output will be `[n_bands, *, T]`.
random_eq = tf.reduce_sum(tf.random.uniform((10, 1, 1, 1)) * bands, axis=0)
```

## Algorithms

### Resample

This is an implementation of the [sinc resample algorithm][resample] by Julius O. Smith.
It is the same algorithm than the one used in [resampy][resampy] but to run efficiently on GPU it
is limited to fractional changes of the sample rate. It will be fast if the old and new sample rate
are small after dividing them by their GCD. For instance going from a sample rate of 2000 to 3000 (2, 3 after removing the GCD)
will be extremely fast, while going from 20001 to 30001 will not.
Julius resampling is faster than resampy even on CPU, and when running on GPU it makes resampling a completely negligible part of your pipeline
(except of course for weird cases like going from a sample rate of 20001 to 30001).


### FFTConv1d

Computing convolutions with very large kernels (>= 128) and a stride of 1 can be much faster
using FFT. This implements the same API as `tf.keras.layers.Conv1D` / `tf.nn.conv1d`
(using the channels-first `[B, C, T]` convention) but with a FFT backend. Dilation and groups
are not supported.
FFTConv will be faster on CPU even for relatively small tensors (a few dozen channels, kernel size
of 128). On CUDA, due to the higher parallelism, regular convolution can be faster in many cases,
but for kernel sizes above 128, for a large number of channels or batch size, FFTConv1d
will eventually be faster (basically when you no longer have idle cores that can hide
the true complexity of the operation).

### LowPass

Classical Finite Impulse Reponse windowed sinc lowpass filter. It will use FFT convolutions automatically
if the filter size is large enough. This is the basic block from which you can build
high pass and band pass filters (see `julius.filters`).

### Bands

Decomposition of a signal over frequency bands in the waveform domain. This can be useful for
instance to perform parametric EQ (see [Usage](#usage) above).

## Benchmarks

You can find speed tests (and comparisons to reference implementations) on the
[benchmark][bench]. The CPU benchmarks are run on a Mac Book Pro 2020, with a 2.4 GHz
8-core intel CPU i9. The GPUs benchmark are run on Nvidia V100 with 16GB of memory.
We also compare the validity of our implementations, as compared to reference ones like `resampy`
or `tf.nn.conv1d`.



## Running tests

Clone this repository, then
```bash
pip3 install '.[dev]'
python3 -m unittest discover -s tests
```

To run the benchmarks:
```
pip3 install .[dev]'
python3 -m bench.gen
```


## License

`julius-tf` is released under the MIT license, the same license as the original `julius`.
The license retains the original copyright of Alexandre Défossez (2020) alongside the
copyright for the TensorFlow port (2026). See [LICENSE](./LICENSE).

## Credits

This project is a TensorFlow port of [`julius`](https://github.com/adefossez/julius) by
[**Alexandre Défossez**](https://github.com/adefossez). All of the DSP algorithms, the
overall design, and the public API originate from his original PyTorch implementation —
full credit for the underlying work goes to him. This repository only re-implements those
algorithms on top of TensorFlow.

- Original project: https://github.com/adefossez/julius (MIT, © 2020 Alexandre Défossez)
- TensorFlow port: Clément Laroche ([@LarocheC](https://github.com/LarocheC))

## Thanks

This package is named in the honor of
[Julius O. Smith](https://ccrma.stanford.edu/~jos/),
whose books and website were a gold mine of information for learning about DSP. Go checkout his website if you want
to learn more about DSP.


[resample]: https://ccrma.stanford.edu/~jos/resample/resample.html
[resampy]: https://resampy.readthedocs.io/
[docs]:  https://LarocheC.github.io/julius-tf/julius/index.html
[bench]:  ./bench.md
