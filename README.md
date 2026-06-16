# analyzeAudio

Measure one or more aspects of one or more audio files.

## Note well: FFmpeg & FFprobe binaries must be in PATH

Some options to [download FFmpeg and FFprobe](https://ffmpeg.org/download.html) at ffmpeg.org.

### Install FFmpeg on Google Colab

```python
from analyzeAudio.ffmpeg import verifyFFmpegColab
verifyFFmpegColab()
```

[![pip install analyzeAudio](https://img.shields.io/badge/pip_install-analyzeAudio-gray.svg?labelColor=blue)](https://pypi.org/project/analyzeAudio/)
[![uv add analyzeAudio](https://img.shields.io/badge/uv_add-analyzeAudio-gray.svg?labelColor=blue)](https://pypi.org/project/analyzeAudio/)

## What is in the package

`analyzeAudio` provides two user-facing kinds of audio analysis.

- Audio aspects measure one audio file.
- Audio contests compare two audio files, waveforms, tensors, or spectrograms.

The main user workflows are:

| What you want                                            | Use                                             |
| -------------------------------------------------------- | ----------------------------------------------- |
| One value for each selected measurement on one file      | `analyzeAudioFile`                              |
| The same selected measurements for many files            | `analyzeAudioListPathFilenames`                 |
| A TSV, CSV, or other delimited output file               | `dataTabularTOpathFilenameDelimited`            |
| One specific measurement or detailed frame data          | Import a direct analyzer function               |
| One comparison score between two files                   | Import a filename contest function              |
| One comparison score between two tensors or spectrograms | Import a tensor or spectrogram contest function |

### One-file measurements

Use these names with `analyzeAudioFile` or `analyzeAudioListPathFilenames`. Names are case-sensitive.

Loudness and true peak:

| Name                      | What it measures                |
| ------------------------- | ------------------------------- |
| `LUFS integrated`         | Whole-file integrated loudness. |
| `LUFS momentary maximum`  | Maximum momentary loudness.     |
| `LUFS short-term maximum` | Maximum short-term loudness.    |
| `LUFS loudness range`     | Loudness range.                 |
| `LUFS low`                | Low loudness range boundary.    |
| `LUFS high`               | High loudness range boundary.   |
| `true_peak maximum`       | Maximum true peak level.        |

Signal level, dynamics, and samples:

| Name                          | What it measures                                 |
| ----------------------------- | ------------------------------------------------ |
| `RMS_level overall`           | Overall RMS level.                               |
| `RMS_peak overall`            | Overall RMS peak.                                |
| `RMS_trough overall`          | Overall RMS trough.                              |
| `RMS_difference overall`      | Overall RMS difference between adjacent samples. |
| `Peak_level overall`          | Overall peak level.                              |
| `Peak_count total`            | Total detected peak count.                       |
| `Abs_Peak_count total`        | Total absolute peak count.                       |
| `Crest_factor mean`           | Mean crest factor.                               |
| `Dynamic_range overall`       | Overall dynamic range.                           |
| `DC_offset mean`              | Mean DC offset.                                  |
| `Bit_depth mean`              | Mean detected bit depth.                         |
| `Entropy mean`                | Mean signal entropy.                             |
| `Flat_factor mean`            | Mean flat factor.                                |
| `Max_difference overall`      | Maximum sample difference.                       |
| `Max_level overall`           | Maximum sample level.                            |
| `Mean_difference mean`        | Mean adjacent-sample difference.                 |
| `Min_difference overall`      | Minimum sample difference.                       |
| `Min_level overall`           | Minimum sample level.                            |
| `Noise_floor overall`         | Overall noise floor.                             |
| `Noise_floor_count total`     | Total noise-floor count.                         |
| `Number_of_samples total`     | Total samples.                                   |
| `Zero_crossings total`        | Total zero crossings.                            |
| `Zero_crossings_rate overall` | Overall zero-crossing rate.                      |

Spectral measurements:

| Name                          | What it measures                                     |
| ----------------------------- | ---------------------------------------------------- |
| `Power spectral density mean` | Mean power spectral density.                         |
| `Spectral centroid mean`      | Mean spectral center of mass from filename analysis. |
| `Spectral crest mean`         | Mean spectral crest.                                 |
| `Spectral decrease mean`      | Mean spectral decrease.                              |
| `Spectral entropy mean`       | Mean spectral entropy.                               |
| `Spectral flatness mean`      | Mean spectral flatness from filename analysis.       |
| `Spectral flux mean`          | Mean spectral flux.                                  |
| `Spectral kurtosis mean`      | Mean spectral kurtosis.                              |
| `Spectral rolloff mean`       | Mean spectral rolloff.                               |
| `Spectral skewness mean`      | Mean spectral skewness.                              |
| `Spectral slope mean`         | Mean spectral slope.                                 |
| `Spectral spread mean`        | Mean spectral spread.                                |
| `Spectral variance mean`      | Mean spectral variance.                              |
| `Spectral Bandwidth mean`     | Mean librosa spectral bandwidth.                     |
| `Spectral Centroid mean`      | Mean librosa spectral centroid.                      |
| `Spectral Contrast mean`      | Mean librosa spectral contrast.                      |
| `Spectral Flatness mean`      | Mean librosa spectral flatness ratio.                |
| `Spectral Flatness dB mean`   | Mean librosa spectral flatness in decibels.          |
| `Chromagram mean`             | Mean chroma energy across pitch classes.             |

Waveform, rhythm, and speech measurements:

| Name                      | What it measures                                      |
| ------------------------- | ----------------------------------------------------- |
| `RMS Waveform mean`       | Mean waveform RMS amplitude.                          |
| `RMS Waveform dB mean`    | Mean waveform RMS level in decibels.                  |
| `Tempogram mean`          | Mean tempogram value.                                 |
| `Tempo mean`              | Mean estimated tempo.                                 |
| `Zero Crossing Rate mean` | Mean waveform zero-crossing rate.                     |
| `SRMR mean`               | Mean speech-to-reverberation modulation energy ratio. |

Some names intentionally differ only by capitalization or wording. For example,
`Spectral flatness mean` and `Spectral Flatness mean` are different
measurements. Use the exact name from the table.

### Measure one file

```python
from analyzeAudio import analyzeAudioFile

listAspectNames = [
    "LUFS integrated",
    "true_peak maximum",
    "RMS_level overall",
    "Spectral Flatness mean",
    "Zero Crossing Rate mean",
]

listValues = analyzeAudioFile("voice.wav", listAspectNames)
measurements = dict(zip(listAspectNames, listValues, strict=True))
print(measurements)
```

`analyzeAudioFile` returns one value for each requested name, in the same order.
If a requested name is unavailable, that value is `"not found"`.

### Measure many files

```python
from pathlib import Path
from analyzeAudio import analyzeAudioListPathFilenames

listPathFilenames = tuple(Path("audio").glob("*.wav"))
listAspectNames = [
    "LUFS integrated",
    "LUFS loudness range",
    "true_peak maximum",
]

rows = analyzeAudioListPathFilenames(
    listPathFilenames,
    listAspectNames,
    CPUlimit=4,
)

for row in rows:
    print(row)
```

Each row starts with the analyzed filename, followed by the requested values.
Rows are returned as files finish, so row order can differ from input order.

### Save measurements

```python
from analyzeAudio import (
    analyzeAudioListPathFilenames,
    dataTabularTOpathFilenameDelimited,
)

listAspectNames = ["LUFS integrated", "true_peak maximum"]
rows = analyzeAudioListPathFilenames(["one.wav", "two.wav"], listAspectNames)

dataTabularTOpathFilenameDelimited(
    "measurements.tsv",
    rows,
    ["pathFilename", *listAspectNames],
)
```

For CSV output, use a comma delimiter:

```python
dataTabularTOpathFilenameDelimited(
    "measurements.csv",
    rows,
    ["pathFilename", *listAspectNames],
    delimiterOutput=",",
)
```

### Get detailed arrays

Summary names usually return one number. Direct analyzer functions without
summary words usually return the per-frame, per-channel, or per-band values.

```python
from analyzeAudio.analyzersUseFilename import (
    analyzeLUFSIntegratedOverall,
    analyzeLUFSMomentary,
)

integrated = analyzeLUFSIntegratedOverall("voice.wav")
momentaryFrames = analyzeLUFSMomentary("voice.wav")
```

### Use audio already loaded in Python

Waveform analyzers accept waveform samples shaped as channels by samples.

```python
import numpy
import soundfile
from analyzeAudio.analyzersUseWaveform import (
    analyzeRMSWaveformMean,
    analyzeTempoMean,
    analyzeZeroCrossingRateMean,
)

with soundfile.SoundFile("voice.wav") as audioFile:
    sampleRate = audioFile.samplerate
    waveform = audioFile.read(dtype="float32", always_2d=True).astype(numpy.float32).T

rms = analyzeRMSWaveformMean(waveform)
tempo = analyzeTempoMean(waveform, sampleRate)
zeroCrossingRate = analyzeZeroCrossingRateMean(waveform)
```

Spectrogram analyzers accept magnitude or power spectrograms.

```python
import librosa
import numpy
from analyzeAudio.analyzersUseSpectrogram import (
    analyzeChromagramMean,
    analyzeSpectralCentroidMean,
)

spectrogram = librosa.stft(waveform)
spectrogramMagnitude = numpy.absolute(spectrogram)
spectrogramPower = spectrogramMagnitude**2

spectralCentroid = analyzeSpectralCentroidMean(spectrogramMagnitude)
chromagram = analyzeChromagramMean(spectrogramPower, sampleRate)
```

### Two-input comparisons

Filename contests compare two audio files:

| Function             | What it compares                                 |
| -------------------- | ------------------------------------------------ |
| `analyzePSNRmean`    | Mean peak signal-to-noise ratio.                 |
| `analyzeSDRmean`     | Mean signal-to-distortion ratio.                 |
| `analyzeSI_SDRmean`  | Mean scale-invariant signal-to-distortion ratio. |
| `analyzeKPSNRmean`   | Bounded score from PSNR.                         |
| `analyzeKSDRmean`    | Bounded score from SDR.                          |
| `analyzeKSI_SDRmean` | Bounded score from SI-SDR.                       |

```python
from analyzeAudio.analyzersUseFilename import (
    analyzePSNRmean,
    analyzeSDRmean,
    analyzeSI_SDRmean,
)

pathReference = "reference.wav"
pathEstimate = "estimate.wav"

psnr = analyzePSNRmean(pathReference, pathEstimate)
sdr = analyzeSDRmean(pathReference, pathEstimate)
si_sdr = analyzeSI_SDRmean(pathReference, pathEstimate)
```

Tensor waveform contests usually compare two PyTorch waveform tensors:

| Function                          | What it compares                                                                        |
| --------------------------------- | --------------------------------------------------------------------------------------- |
| `analyzeL1SNRMean`                | Mean L1 signal-to-noise ratio.                                                          |
| `analyzeL1SNRDBMean`              | Mean L1 signal-to-noise ratio in decibels.                                              |
| `analyzeMultiL1SNRDBMean`         | Multi-source L1 SNR in decibels.                                                        |
| `analyzeSTFTL1SNRDBMean`          | STFT-domain L1 SNR in decibels.                                                         |
| `analyzeLogWMSEMean`              | Mean log weighted MSE audio-quality score for reference, estimate, and mixture tensors. |
| `analyzeDCLoss`                   | DC loss.                                                                                |
| `analyzeESRLoss`                  | Error-to-signal ratio loss.                                                             |
| `analyzeLogCoshLoss`              | Log-cosh loss.                                                                          |
| `analyzeSNRLoss`                  | Signal-to-noise ratio loss.                                                             |
| `analyzeSISDRLoss`                | Scale-invariant SDR loss.                                                               |
| `analyzeSDSDRLoss`                | Scale-dependent SDR loss.                                                               |
| `analyzeSTFTLoss`                 | STFT loss.                                                                              |
| `analyzeMelSTFTLoss`              | Mel-STFT loss.                                                                          |
| `analyzeChromaSTFTLoss`           | Chroma-STFT loss.                                                                       |
| `analyzeMultiResolutionSTFTLoss`  | Multi-resolution STFT loss.                                                             |
| `analyzeRandomResolutionSTFTLoss` | Random-resolution STFT loss.                                                            |
| `analyzeSumAndDifferenceSTFTLoss` | Sum-and-difference STFT loss.                                                           |

```python
from analyzeAudio.contestsTensor import (
    analyzeL1SNRDBMean,
    analyzeMultiResolutionSTFTLoss,
)

l1snrdb = analyzeL1SNRDBMean(tensorReference, tensorEstimate)
mrstft = analyzeMultiResolutionSTFTLoss(tensorReference, tensorEstimate)
```

`analyzeLogWMSEMean` also needs the original mixture and sample rate:

```python
from analyzeAudio.contestsTensor import analyzeLogWMSEMean

logwmse = analyzeLogWMSEMean(
    tensorReference,
    tensorEstimate,
    tensorMixture,
    sampleRate,
)
```

Tensor spectrogram contests compare two PyTorch magnitude spectrogram tensors:

| Function                         | What it compares           |
| -------------------------------- | -------------------------- |
| `analyzeSpectralConvergenceLoss` | Spectral convergence loss. |
| `analyzeSTFTMagnitudeLoss`       | STFT magnitude loss.       |
| `analyzeL1FrequencyLoss`         | L1 frequency score.        |

```python
from analyzeAudio.contestsTensorSpectrogram import (
    analyzeSpectralConvergenceLoss,
    analyzeSTFTMagnitudeLoss,
)

spectralConvergence = analyzeSpectralConvergenceLoss(
    tensorSpectrogramMagnitudeReference,
    tensorSpectrogramMagnitudeEstimate,
)
stftMagnitude = analyzeSTFTMagnitudeLoss(
    tensorSpectrogramMagnitudeReference,
    tensorSpectrogramMagnitudeEstimate,
)
```

NumPy spectrogram helpers compare two magnitude spectrograms:

| Function                    | What it returns                                    |
| --------------------------- | -------------------------------------------------- |
| `analyzeBleedFullMelDB`     | Arrays of added and missing mel-scaled dB content. |
| `analyzeBleedFullMelDBMean` | Two scores: `bleed` and `full`.                    |

```python
from analyzeAudio.contestsSpectrogram import analyzeBleedFullMelDBMean

bleedFull = analyzeBleedFullMelDBMean(
    spectrogramMagnitudeReference,
    spectrogramMagnitudeEstimate,
)
print(bleedFull.bleed, bleedFull.full)
```

### Exact-name checks

The tables above describe what is in the package. These helpers are available
when you want a copyable list from the installed version:

```python
from analyzeAudio import getListAvailableAudioAspects, getListAvailableAudioContests

print(getListAvailableAudioAspects())
print(getListAvailableAudioContests())
```

The terminal commands are:

```powershell
whatAspects
whatContests
```

## API standardization

A top priority for this package is a public API that is as standardized as
possible across filename, waveform, spectrogram, tensor, and contest analyzers.
The package wraps libraries with very different calling conventions, but analyzer
function signatures should model this package's dispatcher inputs, not every
underlying library option.

## Wishlist

- [ ] Overhaul the semiotic system.
- [ ] Install FFmpeg in GitHub Actions for testing.
- [ ] Assist with installing FFmpeg in arbitrary environments.
- [ ] Improve speed
  - [ ] Sophisticated caching of large objects and un-hashable objects.

## Reference materials

### A Spectral-Flatness Measure for Studying the Autocorrelation Method of Linear Prediction of Speech Analysis

- Common name: spectral flatness
- DOI: [10.1109/TASSP.1974.1162572](https://doi.org/10.1109/TASSP.1974.1162572). Download [BibTeX citation.](https://github.com/hunterhogan/analyzeAudio/blob/main/citations/gray1974spectralflatness.bib)
- Implementation: [librosa/librosa](https://github.com/librosa/librosa).feature.spectral_flatness

### Perceptual Effects of Spectral Modifications on Musical Timbres

- DOI: [10.1121/1.381843](https://doi.org/10.1121/1.381843). Download [BibTeX citation.](https://github.com/hunterhogan/analyzeAudio/blob/main/citations/grey1978perceptual.bib)

### Robust Entropy-Based Endpoint Detection for Speech Recognition in Noisy Environments

- Common name: spectral entropy
- DOI: [10.21437/ICSLP.1998-527](https://doi.org/10.21437/ICSLP.1998-527). Download [BibTeX citation.](https://github.com/hunterhogan/analyzeAudio/blob/main/citations/shen1998entropy.bib)

### Realtime Chord Recognition of Musical Sound: A System Using Common Lisp Music

- Common name: chroma features
- Proceedings: [University of Michigan ICMC archive](https://quod.lib.umich.edu/i/icmc/bbp2372.1999.446). Download [BibTeX citation.](https://github.com/hunterhogan/analyzeAudio/blob/main/citations/fujishima1999realtime.bib)
- Implementation: [librosa/librosa](https://github.com/librosa/librosa).feature.chroma_stft

### A Robust Audio Classification and Segmentation Method

- Technical report: [Microsoft Research](https://www.microsoft.com/en-us/research/publication/a-robust-audio-classification-and-segmentation-method/). Download [BibTeX citation.](https://github.com/hunterhogan/analyzeAudio/blob/main/citations/lu2001robust.bib)
- Implementations:
  - [FFmpeg `astats` filter](https://ffmpeg.org/ffmpeg-filters.html#astats); [C file with implementation details for AI agents.](https://ffmpeg.org/doxygen/8.0/af__astats_8c_source.html)
  - [FFmpeg `aspectralstats` filter](https://ffmpeg.org/ffmpeg-filters.html#aspectralstats); [C file reference with implementation functions for AI agents.](https://ffmpeg.org/doxygen/8.0/af__aspectralstats_8c.html)
  - [librosa/librosa](https://github.com/librosa/librosa)

### Music Type Classification by Spectral Contrast Feature

- Common name: spectral contrast
- DOI: [10.1109/ICME.2002.1035731](https://doi.org/10.1109/ICME.2002.1035731). Download [BibTeX citation.](https://github.com/hunterhogan/analyzeAudio/blob/main/citations/jiang2002spectralcontrast.bib)
- Free PDF: [Tsinghua University](https://hcsi.cs.tsinghua.edu.cn/Paper/Paper02/200218.pdf)
- Implementation: [librosa/librosa](https://github.com/librosa/librosa).feature.spectral_contrast

### A Speech/Music Discriminator Based on RMS and Zero-Crossings

- Common names: RMS, zero-crossing rate
- DOI: [10.1109/TMM.2004.840604](https://doi.org/10.1109/TMM.2004.840604). Download [BibTeX citation.](https://github.com/hunterhogan/analyzeAudio/blob/main/citations/panagiotakis2005speechmusic.bib)
- Free author proof: [University of Crete](https://www.csd.uoc.gr/~tziritas/papers/07tmm01-panagiotakis-proof.pdf)
- Implementations:
  - [librosa/librosa](https://github.com/librosa/librosa).feature.rms
  - [librosa/librosa](https://github.com/librosa/librosa).feature.zero_crossing_rate

### Zero-Crossing Rate

- Common name: zero-crossing rate
- Online chapter: [Introduction to Speech Processing](https://speechprocessingbook.aalto.fi/Representations/Zero-crossing_rate.html). Download [BibTeX citation.](https://github.com/hunterhogan/analyzeAudio/blob/main/citations/backstrom2026zerocrossingrate.bib)
- Implementation: [librosa/librosa](https://github.com/librosa/librosa).feature.zero_crossing_rate

### Performance Measurement in Blind Audio Source Separation

- Common name: BSS Eval SDR
- DOI: [10.1109/TSA.2005.858005](https://doi.org/10.1109/TSA.2005.858005). Download [BibTeX citation.](https://github.com/hunterhogan/analyzeAudio/blob/main/citations/vincent2006performance.bib)
- Free author PDF: [IRISA](https://www.irit.fr/~Cedric.Fevotte/publications/journals/ieee_asl_bsseval.pdf)
- Implementations:
  - [sigsep/sigsep-mus-eval](https://github.com/sigsep/sigsep-mus-eval)
  - [mir-evaluation/mir_eval](https://github.com/mir-evaluation/mir_eval)
  - [FFmpeg `asdr` filter](https://ffmpeg.org/ffmpeg-filters.html#asdr); [C source with implementation formulas for AI agents.](https://ffmpeg.org/doxygen/8.0/af__asdr_8c_source.html)

### Automatic Chord Recognition from Audio Using a HMM with Supervised Learning

- Proceedings: [ISMIR 2006](https://ismir.net/conferences/ismir-2006/). Download [BibTeX citation.](https://github.com/hunterhogan/analyzeAudio/blob/main/citations/lee2006automaticchord.bib)
- Implementation: [librosa/librosa](https://github.com/librosa/librosa).feature.chroma_stft

### Cyclic Tempogram: A Mid-Level Tempo Representation for Music Signals

- Common name: tempogram
- DOI: [10.1109/ICASSP.2010.5495219](https://doi.org/10.1109/ICASSP.2010.5495219). Download [BibTeX citation.](https://github.com/hunterhogan/analyzeAudio/blob/main/citations/grosche2010cyclic.bib)
- Free author PDF: [AudioLabs Erlangen](https://www.audiolabs-erlangen.de/content/resources/MIR/tempogramtoolbox/2010_GroscheMuellerKurth_TempogramCyclic_ICASSP.pdf)
- Implementations:
  - [librosa/librosa](https://github.com/librosa/librosa).feature.tempogram
  - [Vamp Tempogram Plugin](https://code.soundsoftware.ac.uk/projects/tempogram)

### A Non-Intrusive Quality and Intelligibility Measure of Reverberant and Dereverberated Speech

- Common name: SRMR
- DOI: [10.1109/TASL.2010.2052247](https://doi.org/10.1109/TASL.2010.2052247). Download [BibTeX citation.](https://github.com/hunterhogan/analyzeAudio/blob/main/citations/falk2010srmr.bib)
- Free author PDF: [MUSEA Lab](https://musaelab.ca/pdfs/J19.pdf)
- Implementation:
  - [Lightning-AI/torchmetrics](https://github.com/Lightning-AI/torchmetrics)
    - [SRMR official documentation.](https://lightning.ai/docs/torchmetrics/stable/audio/speech_reverberation_modulation_energy_ratio.html)
    - [Python source with implementation details for AI agents.](https://github.com/Lightning-AI/torchmetrics/blob/master/src/torchmetrics/functional/audio/srmr.py)

### Signal Processing for Music Analysis

- DOI: [10.1109/JSTSP.2011.2112333](https://doi.org/10.1109/JSTSP.2011.2112333). Download [BibTeX citation.](https://github.com/hunterhogan/analyzeAudio/blob/main/citations/muller2011signal.bib)
- Free author PDF: [Columbia University](https://www.ee.columbia.edu/~dpwe/pubs/MuEKR11-spmus.pdf)
- Implementation: [librosa/librosa](https://github.com/librosa/librosa)

### The Timbre Toolbox: Extracting Audio Descriptors from Musical Signals

- DOI: [10.1121/1.3642604](https://doi.org/10.1121/1.3642604). Download [BibTeX citation.](https://github.com/hunterhogan/analyzeAudio/blob/main/citations/peeters2011timbre.bib)
- Free PDF: [McGill University](https://www.mcgill.ca/mpcl/files/mpcl/peeters_2011_jasa.pdf)
- Implementations:
  - [MPCL-McGill/TimbreToolbox-R2021a](https://github.com/MPCL-McGill/TimbreToolbox-R2021a)
  - [librosa/librosa](https://github.com/librosa/librosa)

### Blind Audio Watermarking Technique Based on Two Dimensional Cellular Automata

- Common name: APSNR reference
- DOI: [10.14257/ijsia.2016.10.9.18](https://doi.org/10.14257/ijsia.2016.10.9.18). Download [BibTeX citation.](https://github.com/hunterhogan/analyzeAudio/blob/main/citations/hiary2016blind.bib)
- Implementation: [FFmpeg `apsnr` filter](https://ffmpeg.org/ffmpeg-filters.html#apsnr); [C source with implementation formulas for AI agents.](https://ffmpeg.org/doxygen/8.0/af__asdr_8c_source.html)

### SDR - Half-Baked or Well Done?

- Common name: SI-SDR
- DOI: [10.1109/ICASSP.2019.8683855](https://doi.org/10.1109/ICASSP.2019.8683855). Download [BibTeX citation.](https://github.com/hunterhogan/analyzeAudio/blob/main/citations/leroux2019sdr.bib) [TeX Source with precise formulas for AI agents.](https://arxiv.org/src/1811.02508)
- Free author PDF: [Jonathan Le Roux](https://www.jonathanleroux.org/pdf/LeRoux2019ICASSP05sdr.pdf)
- Implementations: [Lightning-AI/torchmetrics](https://github.com/Lightning-AI/torchmetrics)

### Loudness Metering: EBU Mode Metering to Supplement Loudness Normalisation

- Common name: momentary LUFS
- Standard: [EBU Tech 3341](https://tech.ebu.ch/publications/tech3341). Download [BibTeX citation.](https://github.com/hunterhogan/analyzeAudio/blob/main/citations/ebu2023tech3341.bib)
- Implementation: [FFmpeg `ebur128` filter](https://ffmpeg.org/ffmpeg-filters.html#ebur128); [C source with implementation details for AI agents](https://ffmpeg.org/doxygen/8.0/f__ebur128_8c_source.html); [C source with loudness calculations for AI agents.](https://ffmpeg.org/doxygen/8.0/ebur128_8c_source.html)

### Loudness Range: A Measure to Supplement EBU R 128 Loudness Normalisation

- Common name: LUFS
- Standard: [EBU Tech 3342](https://tech.ebu.ch/publications/tech3342). Download [BibTeX citation.](https://github.com/hunterhogan/analyzeAudio/blob/main/citations/ebu2023tech3342.bib)
- Implementation: [FFmpeg `ebur128` filter](https://ffmpeg.org/ffmpeg-filters.html#ebur128); [C source with implementation details for AI agents](https://ffmpeg.org/doxygen/8.0/f__ebur128_8c_source.html); [C source with loudness calculations for AI agents.](https://ffmpeg.org/doxygen/8.0/ebur128_8c_source.html)

### Algorithms to Measure Audio Programme Loudness and True-Peak Audio Level

- Common name: True peak
- Standard: [ITU-R BS.1770-5](https://www.itu.int/rec/R-REC-BS.1770-5-202311-I). Download [BibTeX citation.](https://github.com/hunterhogan/analyzeAudio/blob/main/citations/itu2023bs1770.bib)
- Implementation: [FFmpeg `ebur128` filter](https://ffmpeg.org/ffmpeg-filters.html#ebur128); [C source with implementation details for AI agents](https://ffmpeg.org/doxygen/8.0/f__ebur128_8c_source.html); [C source with loudness calculations for AI agents.](https://ffmpeg.org/doxygen/8.0/ebur128_8c_source.html)

### An Overview on Sound Features in Time and Frequency Domain

- DOI: [10.2478/ijasitels-2023-0006](https://doi.org/10.2478/ijasitels-2023-0006). Download [BibTeX citation.](https://github.com/hunterhogan/analyzeAudio/blob/main/citations/constantinescu2023overview.bib)

### Perceptual Loss Function for Neural Modelling of Audio Systems

- Common names: ESR loss, DC loss
- arXiv abstract: [1911.08922](https://arxiv.org/abs/1911.08922). Download [BibTeX citation.](https://github.com/hunterhogan/analyzeAudio/blob/main/citations/wright2019perceptualloss.bib)
- TeX source with formulas for AI agents: [arXiv source](https://arxiv.org/src/1911.08922)

### Log Hyperbolic Cosine Loss Improves Variational Auto-Encoder

- Common name: log-cosh loss
- OpenReview page: [rkglvsC9Ym](https://openreview.net/forum?id=rkglvsC9Ym). Download [BibTeX citation.](https://github.com/hunterhogan/analyzeAudio/blob/main/citations/chen2019logcosh.bib)

### logWMSE Audio Quality Metric and PyTorch Loss Implementation

- Common name: logWMSE
- Implementations:
  - [nomonosound/log-wmse-audio-quality](https://github.com/nomonosound/log-wmse-audio-quality). Download [BibTeX citation.](https://github.com/hunterhogan/analyzeAudio/blob/main/citations/jordal2026logwmse.bib)
  - [crlandsc/torch-log-wmse](https://github.com/crlandsc/torch-log-wmse). Download [BibTeX citation.](https://github.com/hunterhogan/analyzeAudio/blob/main/citations/landschoot2026torchlogwmse.bib)

### Fast Spectrogram Inversion using Multi-head Convolutional Neural Networks

- Common names: spectral convergence, STFT magnitude loss terms
- DOI: [10.48550/arXiv.1808.06719](https://doi.org/10.48550/arXiv.1808.06719). Download [BibTeX citation.](https://github.com/hunterhogan/analyzeAudio/blob/main/citations/arik2018fastspectrogram.bib)
- arXiv abstract: [1808.06719](https://arxiv.org/abs/1808.06719); TeX source with formulas for AI agents: [arXiv source](https://arxiv.org/src/1808.06719)

### Probability density distillation with generative adversarial networks for high-quality parallel waveform generation

- DOI: [10.48550/arXiv.1904.04472](https://doi.org/10.48550/arXiv.1904.04472). Download [BibTeX citation.](https://github.com/hunterhogan/analyzeAudio/blob/main/citations/yamamoto2019pdd.bib) TeX source with formulas for AI agents: [arXiv source](https://arxiv.org/src/1904.04472)

### Parallel WaveGAN: A fast waveform generation model based on generative adversarial networks with multi-resolution spectrogram

- Common name: multi-resolution STFT
- DOI: [10.48550/arXiv.1910.11480](https://doi.org/10.48550/arXiv.1910.11480). Download [BibTeX citation.](https://github.com/hunterhogan/analyzeAudio/blob/main/citations/yamamoto2019parallelwavegan.bib) TeX source with formulas for AI agents: [arXiv source](https://arxiv.org/src/1910.11480)

### auraloss: Audio focused loss functions in PyTorch

- Common names: random-resolution STFT loss implementation source
- Workshop paper PDF: [DMRN+15 PDF](https://www.christiansteinmetz.com/s/DMRN15__auraloss__Audio_focused_loss_functions_in_PyTorch.pdf). Download [BibTeX citation.](https://github.com/hunterhogan/analyzeAudio/blob/main/citations/steinmetz2020auraloss.bib)
- Implementation: [csteinmetz1/auraloss](https://github.com/csteinmetz1/auraloss). [BibTeX citation for the source repository.](https://github.com/hunterhogan/analyzeAudio/blob/main/citations/steinmetz2020auralosssoftware.bib)

### Automatic multitrack mixing with a differentiable mixing console of neural audio effects

- Common names: sum-and-difference STFT loss in neural mixing
- DOI: [10.48550/arXiv.2010.10291](https://doi.org/10.48550/arXiv.2010.10291). Download [BibTeX citation.](https://github.com/hunterhogan/analyzeAudio/blob/main/citations/steinmetz2020multitrackmixing.bib) TeX source with formulas for AI agents: [arXiv source](https://arxiv.org/src/2010.10291)

### Neural source-filter waveform models for statistical parametric speech synthesis

- Related in auraloss docs for multi-resolution spectral training context
- DOI: [10.48550/arXiv.1904.12088](https://doi.org/10.48550/arXiv.1904.12088). Download [BibTeX citation.](https://github.com/hunterhogan/analyzeAudio/blob/main/citations/wang2019neuralsourcefilter.bib) TeX source with formulas for AI agents: [arXiv source](https://arxiv.org/src/1904.12088)

### DDSP: Differentiable Digital Signal Processing

- DOI: [10.48550/arXiv.2001.04643](https://doi.org/10.48550/arXiv.2001.04643). Download [BibTeX citation.](https://github.com/hunterhogan/analyzeAudio/blob/main/citations/engel2020ddsp.bib) TeX source with formulas for AI agents: [arXiv source](https://arxiv.org/src/2001.04643)

### A Generalized Bandsplit Neural Network for Cinematic Audio Source Separation

- Common name: L1SNR reference
- DOI: [10.1109/OJSP.2023.3339428](https://doi.org/10.1109/OJSP.2023.3339428). Download [BibTeX citation.](https://github.com/hunterhogan/analyzeAudio/blob/main/citations/watcharasupat2024generalizedbandsplit.bib) TeX source with formulas for AI agents: [arXiv source](https://arxiv.org/src/2309.02539)

### A Stem-Agnostic Single-Decoder System for Music Source Separation Beyond Four Stems

- Common name: L1SNR reference
- DOI: [10.48550/arXiv.2406.18747](https://doi.org/10.48550/arXiv.2406.18747). Download [BibTeX citation.](https://github.com/hunterhogan/analyzeAudio/blob/main/citations/watcharasupat2024stemagnostic.bib) TeX source with formulas for AI agents: [arXiv source](https://arxiv.org/src/2406.18747)

### Separate This, and All of these Things Around It: Music Source Separation via Hyperellipsoidal Queries

- Common name: L1SNRDB reference
- DOI: [10.48550/arXiv.2501.16171](https://doi.org/10.48550/arXiv.2501.16171). Download [BibTeX citation.](https://github.com/hunterhogan/analyzeAudio/blob/main/citations/watcharasupat2025hyperellipsoidal.bib) TeX source with formulas for AI agents: [arXiv source](https://arxiv.org/src/2501.16171)

### torch-l1-snr: L1 Signal-to-Noise Ratio Loss Functions for Audio Source Separation in PyTorch

- Common name: torch-l1-snr
- Source: [crlandsc/torch-l1-snr](https://github.com/crlandsc/torch-l1-snr). Download [BibTeX citation.](https://github.com/hunterhogan/analyzeAudio/blob/main/citations/landschoot2026torchl1snr.bib)

### Packages and documentation

- [analyzeAudio](https://github.com/hunterhogan/analyzeAudio)
  - [Context7 reference for AI agents](https://context7.com/hunterhogan/analyzeaudio)
- [FFmpeg documentation](https://ffmpeg.org/documentation.html)
  - [FFmpeg filter documentation](https://ffmpeg.org/ffmpeg-filters.html#Audio-Filters)
  - [FFmpeg source tree](https://github.com/FFmpeg/FFmpeg)
  - [FFmpeg Doxygen source browser](https://ffmpeg.org/doxygen/8.0/)
- [librosa/librosa](https://github.com/librosa/librosa)
  - [official documentation](https://librosa.org/doc/latest/index.html)
  - [Context7 documentation](https://context7.com/librosa/librosa)
- [Lightning-AI/torchmetrics](https://github.com/Lightning-AI/torchmetrics)
  - [official documentation](https://lightning.ai/docs/torchmetrics/stable/)
  - [Context7 documentation](https://context7.com/lightning-ai/torchmetrics)
- [PyTorch `torch.nn.Module`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html)
  - [Context7 documentation](https://context7.com/pytorch/pytorch)
  - [Python source with implementation details for AI agents.](https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/module.py)
- [sigsep/sigsep-mus-eval](https://github.com/sigsep/sigsep-mus-eval)
- [mir-evaluation/mir_eval](https://github.com/mir-evaluation/mir_eval)

## My recovery

[![Static Badge](https://img.shields.io/badge/2011_August-Homeless_since-blue?style=flat)](https://HunterThinks.com/support)
[![YouTube Channel Subscribers](https://img.shields.io/youtube/channel/subscribers/UC3Gx7kz61009NbhpRtPP7tw)](https://www.youtube.com/@HunterHogan)

[![CC-BY-NC-4.0](https://raw.githubusercontent.com/hunterhogan/analyzeAudio/refs/heads/main/.github/CC-BY-NC-4.0.png)](https://creativecommons.org/licenses/by-nc/4.0/)
