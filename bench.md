## Benchmarking and verification of Julius

In order to verify the correctness and speed of the implementations in Julius,
we compare ourselves to different reference implementations, comparing speed and
checking how far we are.

### ResampleFrac

We compare `julius.resample` to `resampy`, on an input of size (32, 8 * 44100),
i.e. a batch of size 16 of 8 second of audio at 44.1kHz.
We use the same number of zero crossing as `resampy` for this benchmark.
The reported delta is the mean absolute difference against `resampy`'s default
`kaiser_best` filter; it is dominated by the different anti-aliasing window the
two libraries use. With matched filter parameters the two agree to within ~3%
(see `tests/test_resample.py::test_resampy`).


On CPU we have:

| Old sr | New sr | Julius (ms) | Resampy (ms) | Delta (%) |
|--------|--------|-------------|--------------|-----------|
|      2 |      1 |         240 |         4222 |      6.1% |
|      1 |      2 |         128 |         7702 |      8.6% |
|      4 |      5 |          29 |         4106 |      8.6% |
|     10 |     11 |          29 |         3254 |      8.6% |
|  44100 |  16000 |         102 |         2614 |      5.2% |
|  20001 |  30001 |       78013 |         8017 |      8.6% |


On GPU we have:

_Not benchmarked: no CUDA GPU available in this environment._

### FFTConv1d

We compare to `tf.nn.conv1d`, on a input of size [32, 32, 10240],
for a convolution with 32 input channels, 64 output channels and various kernel sizes.

On CPU we have:

| Kernel size | FFT (ms) | No FFT (ms) |   Delta |
|-------------|----------|-------------|---------|
|           8 |      270 |          87 | 4.0e-06 |
|          32 |       87 |          90 | 1.4e-05 |
|          64 |      112 |         159 | 2.7e-05 |
|         128 |      127 |         321 | 5.3e-05 |
|         256 |      132 |         593 | 1.0e-04 |
|        1024 |      153 |        3766 | 2.3e-04 |
|        2048 |      127 |        7828 | 4.5e-04 |


On GPU we have:

_Not benchmarked: no CUDA GPU available in this environment._

### LowPassFilter

We do not compare to anything, but measure the attenuation in dB of a pure tone
at `0.9 * cutoff`, at the `cutoff`, and at `1.1 * cutoff`.
Note that our implementation automatically choses to use FFTConv1d or not when appropriate.

On CPU we have:

| Freq. | Attn. 0.9 (dB) | Attn 1.0 (dB) | Attn 1.1 (dB) | Time (ms) |
|-------|----------------|---------------|---------------|-----------|
| 0.005 |          -1.41 |         -6.02 |        -16.41 |        23 |
|  0.01 |          -1.41 |         -6.02 |        -16.46 |        17 |
|   0.1 |          -1.41 |         -6.02 |        -16.48 |        31 |
|   0.2 |          -1.41 |         -6.02 |        -16.48 |        11 |
|   0.4 |          -1.41 |         -6.03 |        -16.38 |        11 |


On GPU we have:

_Not benchmarked: no CUDA GPU available in this environment._
