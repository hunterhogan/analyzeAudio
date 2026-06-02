# analyzeAudio

Measure one or more aspects of one or more audio files.

## Note well: FFmpeg & FFprobe binaries must be in PATH

Some options to [download FFmpeg and FFprobe](https://www.ffmpeg.org/download.html) at ffmpeg.org.

[![pip install analyzeAudio](https://img.shields.io/badge/pip_install-analyzeAudio-gray.svg?labelColor=blue)](https://pypi.org/project/analyzeAudio/)
[![uv add analyzeAudio](https://img.shields.io/badge/uv_add-analyzeAudio-gray.svg?labelColor=blue)](https://pypi.org/project/analyzeAudio/)

## Some ways to use this package

`analyzeAudio` works at five practical levels: audio path, waveform array, spectrogram array,
waveform tensor, and magnitude-spectrogram tensor. The top-level API is the quickest way to ask
for named measurements from audio paths. The lower-level modules are there when you already have
decoded audio in memory, want the full array or tensor instead of one summary float, or want to
call direct comparison and loss analyzers yourself.

### Top-level exports you will probably reach for first

| Export                                                                             | Purpose                                                                            |
| ---------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| `analyzeAudioFile(pathFilename, listAspectNames)`                                  | Analyze one audio path and return one result per requested registered aspect name. |
| `analyzeAudioListPathFilenames(listPathFilenames, listAspectNames, CPUlimit=None)` | Analyze many audio paths in parallel and return one completed row per audio path.  |
| `getListAvailableAudioAspects()`                                                   | Return the sorted list of registered aspect names.                                 |
| `audioAspects`                                                                     | Registry of `aspect name -> analyzer callable + required parameter names`.         |
| `truncateTensors(listTensors)`                                                     | Trim multiple tensors to the same trailing length before direct comparison.        |
| `dataTabularTOpathFilenameDelimited(...)`                                          | Write batch results to a delimited text file.                                      |

The package also re-exports the type aliases `Audio`, `Spectrogram`, `SpectrogramMagnitude`, and
`SpectrogramPower` so downstream code can annotate the same representations the analyzers expect.

### Choose the module that matches the representation you already have

| If you already have...                      | Reach for...                            | What you get                                                                                     |
| ------------------------------------------- | --------------------------------------- | ------------------------------------------------------------------------------------------------ |
| one or more audio paths                     | top-level API or `analyzersUseFilename` | named single-path measurements, paired-path comparisons, FFprobe/FFmpeg-derived arrays, loudness |
| waveform `numpy.ndarray` + `sampleRate`     | `analyzersUseWaveform`                  | tempogram, RMS, tempo, and zero-crossing arrays or their mean summaries                          |
| spectrogram magnitude/power `numpy.ndarray` | `analyzersUseSpectrogram`               | chroma and spectral-descriptor arrays or their mean summaries                                    |
| waveform `torch.Tensor`                     | `analyzersUseTensor`                    | SRMR, logWMSE, source-separation scores, and waveform/STFT-domain loss analyzers                 |
| magnitude spectrogram `torch.Tensor`        | `analyzersUseTensorSpectrogram`         | spectrogram-magnitude comparison losses such as spectral convergence and STFT-magnitude distance |

The registry spans more than single-path measurements. Some registered names are convenient
single-audio measurements, some are paired-path comparisons, some expect waveform tensors, and some
expect magnitude spectrogram tensors.

### Use `analyzeAudioFile` to measure registered single-audio aspects from one path

```python
from analyzeAudio import analyzeAudioFile

listAspectNames = [
    'LUFS integrated',
    'RMS peak',
    'SRMR mean',
    'Spectral Flatness mean',
]

listMeasurements = analyzeAudioFile(pathFilename, listAspectNames)
dictionaryMeasurements = dict(zip(listAspectNames, listMeasurements, strict=True))
```

`analyzeAudioFile` reads one audio path, prepares shared intermediate representations, and lets
registered analyzers reuse those representations.

`analyzeAudioFile` preserves the order of `listAspectNames`. If a requested aspect name is not
registered, the matching return entry is `'not found'`.

Registered names are case-sensitive, and some intentionally similar names come from different
analysis routes. For example, `Spectral Flatness mean` and `Spectral flatness mean` are different
registered names, and so are `Zero-crossing rate mean` and `Zero-crossings rate`.

Some registered names require inputs that cannot come from a single audio path alone. Examples
include `SI-SDR mean`, `LogWMSE`, `L1SNRDB`, and `SpectralConvergenceLoss`. For those, inspect the
registry and call the analyzer directly.

### Use `analyzeAudioListPathFilenames` to batch single-audio measurements across many paths

```python
from analyzeAudio import analyzeAudioListPathFilenames, dataTabularTOpathFilenameDelimited

listAspectNames = ['LUFS integrated', 'Spectral Flatness mean']
rowsListFilenameAspectValues = analyzeAudioListPathFilenames(listPathFilenames, listAspectNames)

dataTabularTOpathFilenameDelimited(
    pathFilenameOutput,
    rowsListFilenameAspectValues,
    ['pathFilename', *listAspectNames],
)
```

Each returned row starts with the audio path converted to POSIX text, followed by the requested
values. The rows are returned in worker-completion order rather than the original input order.
Use `CPUlimit` when you want to cap the worker count explicitly.

### Use `getListAvailableAudioAspects` and `audioAspects` to inspect the registry or call an analyzer directly

```python
from analyzeAudio import audioAspects, getListAvailableAudioAspects

print(getListAvailableAudioAspects())
print(audioAspects['Chromagram mean']['analyzerParameters'])
print(audioAspects['SI-SDR mean']['analyzerParameters'])
print(audioAspects['LogWMSE']['analyzerParameters'])

SI_SDR_channelsMean = audioAspects['SI-SDR mean']['analyzer'](
    pathFilenameAudioFile,
    pathFilenameDifferentAudioFile,
)
```

Use `audioAspects[name]['analyzerParameters']` first. It tells you whether the registered name
expects one audio path, two audio paths, waveform tensors, a reference-estimate-mixture triple, or
spectrogram magnitudes.

That is the quickest way to discover whether a name is meant for the high-level single-audio API or
for direct invocation.

### Use the lower-level modules when you want the actual analyzer instead of one registry float

These are the actual analyzers, organized by the representation they consume.

- `analyzeAudio.analyzersUseFilename`
  - paired-path comparison metrics: `getPSNRmean`, `getSDRmean`, `getSI_SDRmean`
  - framewise spectral arrays with matching mean summaries: `analyzeSpectralCentroid`,
    `analyzeSpectralCrest`, `analyzeSpectralDecrease`, `analyzeSpectralEntropy`,
    `analyzeSpectralFlatness`, `analyzeSpectralFlux`, `analyzeSpectralKurtosis`,
    `analyzeSpectralMean`, `analyzeSpectralRolloff`, `analyzeSpectralSkewness`,
    `analyzeSpectralSlope`, `analyzeSpectralSpread`, `analyzeSpectralVariance`
  - file-level FFprobe `astats` scalars: `analyzeZero_crossings`, `analyzeZero_crossings_rate`,
    `analyzeDCoffset`, `analyzeDynamicRange`, `analyzeSignalEntropy`, `analyzeNumber_of_samples`,
    `analyzePeak_level`, `analyzeRMS_level`, `analyzeCrest_factor`, `analyzeRMS_peak`,
    `analyzeAbs_Peak_count`, `analyzeBit_depth`, `analyzeFlat_factor`, `analyzeMax_difference`,
    `analyzeMax_level`, `analyzeMean_difference`, `analyzeMin_difference`, `analyzeMin_level`,
    `analyzeNoise_floor`, `analyzeNoise_floor_count`, `analyzePeak_count`,
    `analyzeRMS_difference`, `analyzeRMS_trough`
  - loudness and true-peak arrays plus scalar summaries: `analyzeTruePeak`,
    `analyzeLUFSMomentary`, `analyzeLUFSShortTerm`, `analyzeLUFSIntegrated`, `analyzeLRA`,
    `analyzeLUFSlow`, `analyzeLUFShigh`, plus the matching `...Overall` scalar functions
- `analyzeAudio.analyzersUseWaveform`
  - raw arrays: `analyzeTempogram`, `analyzeRMS`, `analyzeTempo`, `analyzeZeroCrossingRate`
  - mean summaries: `analyzeTempogramMean`, `analyzeRMSMean`, `analyzeTempoMean`,
    `analyzeZeroCrossingRateMean`
- `analyzeAudio.analyzersUseSpectrogram`
  - raw arrays: `analyzeChromagram`, `analyzeSpectralContrast`, `analyzeSpectralBandwidth`,
    `analyzeSpectralCentroid`, `analyzeSpectralFlatness`
  - mean summaries: `analyzeChromagramMean`, `analyzeSpectralContrastMean`,
    `analyzeSpectralBandwidthMean`, `analyzeSpectralCentroidMean`,
    `analyzeSpectralFlatnessMean`
- `analyzeAudio.analyzersUseTensor`
  - reverberation and intelligibility: `analyzeSRMR`, `analyzeSRMRMean`
  - reference-estimate-mixture scoring: `analyzeLogWMSEMean`
  - source-separation scores: `analyzeL1SNRMean`, `analyzeL1SNRDBMean`,
    `analyzeMultiL1SNRDBMean`, `analyzeSTFTL1SNRDBMean`
  - waveform-domain and STFT-domain loss analyzers: `analyzeDCLoss`, `analyzeESRLoss`,
    `analyzeLogCoshLoss`, `analyzeSNRLoss`, `analyzeSISDRLoss`, `analyzeSDSDRLoss`,
    `analyzeSTFTLoss`, `analyzeMelSTFTLoss`, `analyzeChromaSTFTLoss`,
    `analyzeMultiResolutionSTFTLoss`, `analyzeRandomResolutionSTFTLoss`,
    `analyzeSumAndDifferenceSTFTLoss`
- `analyzeAudio.analyzersUseTensorSpectrogram`
  - magnitude-spectrogram comparison analyzers: `analyzeSpectralConvergenceLoss`,
    `analyzeSTFTMagnitudeLoss`, `analyzeL1FrequencyLoss`
- `analyzeAudio.ffmpeg`
  - environment check for Colab-style sessions: `verifyFFmpegColab`

Several concept names exist in more than one module. That is intentional. For example,
`Spectral flatness mean` comes from the filename-based FFprobe route, while `Spectral Flatness mean`
comes from the spectrogram route. Similar names do not necessarily mean duplicate implementations.

```python
import numpy
import soundfile

from analyzeAudio.analyzersUseWaveform import analyzeTempogram

with soundfile.SoundFile(pathFilename) as readSoundFile:
    sampleRate = readSoundFile.samplerate
    waveform = readSoundFile.read(dtype='float32').astype(numpy.float32).T

tempogram = analyzeTempogram(waveform, sampleRate)
```

```python
from analyzeAudio.analyzersUseTensor import analyzeL1SNRDBMean
from analyzeAudio.analyzersUseTensorSpectrogram import analyzeSpectralConvergenceLoss

valueScore = analyzeL1SNRDBMean(tensorAudioReference, tensorAudioEstimate)
valueLoss = analyzeSpectralConvergenceLoss(tensorMagnitudeReference, tensorMagnitudeEstimate)
```

### Use `truncateTensors` when you want the aligned tensors yourself

Most tensor comparison analyzers already trim inputs. `truncateTensors` is there for the
times when you want aligned tensors before reusing them across several metrics.

```python
from analyzeAudio import truncateTensors

tensorAudioReference, tensorAudioEstimate = truncateTensors([
    tensorAudioReference,
    tensorAudioEstimate,
])
```

### Use `whatMeasurements` to list registered measurements from the command line

```sh
whatMeasurements
```

This prints the same sorted registry names returned by `getListAvailableAudioAspects()`.

## Reference materials

### A Spectral-Flatness Measure for Studying the Autocorrelation Method of Linear Prediction of Speech Analysis

- Common name: spectral flatness
- [BibTeX citation.](https://github.com/hunterhogan/analyzeAudio/blob/main/citations/gray1974spectralflatness.bib)
- DOI: [10.1109/TASSP.1974.1162572](https://doi.org/10.1109/TASSP.1974.1162572)
- IEEE Xplore: [document 1162647](https://ieeexplore.ieee.org/document/1162647)
- Implementation:
  - [librosa/librosa](https://github.com/librosa/librosa).feature.spectral_flatness

### Perceptual Effects of Spectral Modifications on Musical Timbres

- [BibTeX citation.](https://github.com/hunterhogan/analyzeAudio/blob/main/citations/grey1978perceptual.bib)
- DOI: [10.1121/1.381843](https://doi.org/10.1121/1.381843)

### Robust Entropy-Based Endpoint Detection for Speech Recognition in Noisy Environments

- Common name: spectral entropy
- [BibTeX citation.](https://github.com/hunterhogan/analyzeAudio/blob/main/citations/shen1998entropy.bib)
- DOI: [10.21437/ICSLP.1998-527](https://doi.org/10.21437/ICSLP.1998-527)
- Proceedings: [ISCA Archive](https://www.isca-archive.org/icslp_1998/shen98b_icslp.html)
- Free author PDF: [Columbia University](https://www.ee.columbia.edu/~dpwe/papers/ShenHL98-endpoint.pdf)

### Realtime Chord Recognition of Musical Sound: A System Using Common Lisp Music

- Common name: chroma features
- [BibTeX citation.](https://github.com/hunterhogan/analyzeAudio/blob/main/citations/fujishima1999realtime.bib)
- Proceedings: [University of Michigan ICMC archive](https://quod.lib.umich.edu/i/icmc/bbp2372.1999.446)
- CCRMA HTML copy: [Stanford CCRMA](https://ccrma.stanford.edu/~jos/mus423h/Real_Time_Chord_Recognition_Musical.html)
- Implementation:
  - [librosa/librosa](https://github.com/librosa/librosa).feature.chroma_stft

### A Robust Audio Classification and Segmentation Method

- [BibTeX citation.](https://github.com/hunterhogan/analyzeAudio/blob/main/citations/lu2001robust.bib)
- Technical report: [Microsoft Research](https://www.microsoft.com/en-us/research/publication/a-robust-audio-classification-and-segmentation-method/)
- Free PDF: [Microsoft Research](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr-2001-79.pdf)
- Implementations:
  - [FFprobe `astats` filter](https://ffmpeg.org/ffprobe-all.html#astats)
    - [C file with implementation details for AI agents.](https://www.ffmpeg.org/doxygen/8.0/af__astats_8c_source.html)
  - [FFprobe `aspectralstats` filter](https://ffmpeg.org/ffprobe-all.html#aspectralstats)
    - [C file reference with implementation functions for AI agents.](https://www.ffmpeg.org/doxygen/8.0/af__aspectralstats_8c.html)
  - [librosa/librosa](https://github.com/librosa/librosa)

### Music Type Classification by Spectral Contrast Feature

- Common name: spectral contrast
- [BibTeX citation.](https://github.com/hunterhogan/analyzeAudio/blob/main/citations/jiang2002spectralcontrast.bib)
- DOI: [10.1109/ICME.2002.1035731](https://doi.org/10.1109/ICME.2002.1035731)
- Free PDF: [Tsinghua University](https://hcsi.cs.tsinghua.edu.cn/Paper/Paper02/200218.pdf)
- Implementation:
  - [librosa/librosa](https://github.com/librosa/librosa).feature.spectral_contrast

### A Speech/Music Discriminator Based on RMS and Zero-Crossings

- Common names: RMS, zero-crossing rate
- [BibTeX citation.](https://github.com/hunterhogan/analyzeAudio/blob/main/citations/panagiotakis2005speechmusic.bib)
- DOI: [10.1109/TMM.2004.840604](https://doi.org/10.1109/TMM.2004.840604)
- Free author proof: [University of Crete](https://www.csd.uoc.gr/~tziritas/papers/07tmm01-panagiotakis-proof.pdf)
- Implementation:
  - [librosa/librosa](https://github.com/librosa/librosa).feature.rms
  - [librosa/librosa](https://github.com/librosa/librosa).feature.zero_crossing_rate

### Zero-Crossing Rate

- Common name: zero-crossing rate
- [BibTeX citation.](https://github.com/hunterhogan/analyzeAudio/blob/main/citations/backstrom2026zerocrossingrate.bib)
- Online chapter: [Introduction to Speech Processing](https://speechprocessingbook.aalto.fi/Representations/Zero-crossing_rate.html)
- Implementation:
  - [librosa/librosa](https://github.com/librosa/librosa).feature.zero_crossing_rate

### Performance Measurement in Blind Audio Source Separation

- Common name: BSS Eval SDR
- [BibTeX citation.](https://github.com/hunterhogan/analyzeAudio/blob/main/citations/vincent2006performance.bib)
- DOI: [10.1109/TSA.2005.858005](https://doi.org/10.1109/TSA.2005.858005)
- Free author PDF: [IRISA](https://www.irit.fr/~Cedric.Fevotte/publications/journals/ieee_asl_bsseval.pdf)
- Implementations:
  - [sigsep/sigsep-mus-eval](https://github.com/sigsep/sigsep-mus-eval)
  - [mir-evaluation/mir_eval](https://github.com/mir-evaluation/mir_eval)
  - [FFprobe `asdr` filter](https://ffmpeg.org/ffprobe-all.html#asdr)
    - [C source with implementation formulas for AI agents.](https://www.ffmpeg.org/doxygen/8.0/af__asdr_8c_source.html)

### Automatic Chord Recognition from Audio Using a HMM with Supervised Learning

- [BibTeX citation.](https://github.com/hunterhogan/analyzeAudio/blob/main/citations/lee2006automaticchord.bib)
- Proceedings: [ISMIR 2006](https://ismir.net/conferences/ismir-2006/)
- Free PDF: [Stanford CCRMA](https://ccrma.stanford.edu/~kglee/pubs/klee-ismir06.pdf)
- Implementation:
  - [librosa/librosa](https://github.com/librosa/librosa).feature.chroma_stft

### Cyclic Tempogram: A Mid-Level Tempo Representation for Music Signals

- Common name: tempogram
- [BibTeX citation.](https://github.com/hunterhogan/analyzeAudio/blob/main/citations/grosche2010cyclic.bib)
- DOI: [10.1109/ICASSP.2010.5495219](https://doi.org/10.1109/ICASSP.2010.5495219)
- Free author PDF: [AudioLabs Erlangen](https://www.audiolabs-erlangen.de/content/resources/MIR/tempogramtoolbox/2010_GroscheMuellerKurth_TempogramCyclic_ICASSP.pdf)
- Implementations:
  - [librosa/librosa](https://github.com/librosa/librosa).feature.tempogram
  - [Vamp Tempogram Plugin](https://code.soundsoftware.ac.uk/projects/tempogram)

### A Non-Intrusive Quality and Intelligibility Measure of Reverberant and Dereverberated Speech

- Common name: SRMR
- [BibTeX citation.](https://github.com/hunterhogan/analyzeAudio/blob/main/citations/falk2010srmr.bib)
- DOI: [10.1109/TASL.2010.2052247](https://doi.org/10.1109/TASL.2010.2052247)
- Free author PDF: [MUSEA Lab](https://musaelab.ca/pdfs/J19.pdf)
- Implementation:
  - [Lightning-AI/torchmetrics](https://github.com/Lightning-AI/torchmetrics)
    - [SRMR official documentation.](https://lightning.ai/docs/torchmetrics/stable/audio/speech_reverberation_modulation_energy_ratio.html)
    - [Python source with implementation details for AI agents.](https://github.com/Lightning-AI/torchmetrics/blob/master/src/torchmetrics/functional/audio/srmr.py)

### Signal Processing for Music Analysis

- [BibTeX citation.](https://github.com/hunterhogan/analyzeAudio/blob/main/citations/muller2011signal.bib)
- DOI: [10.1109/JSTSP.2011.2112333](https://doi.org/10.1109/JSTSP.2011.2112333)
- Free author PDF: [Columbia University](https://www.ee.columbia.edu/~dpwe/pubs/MuEKR11-spmus.pdf)
- Implementation:
  - [librosa/librosa](https://github.com/librosa/librosa)

### The Timbre Toolbox: Extracting Audio Descriptors from Musical Signals

- [BibTeX citation.](https://github.com/hunterhogan/analyzeAudio/blob/main/citations/peeters2011timbre.bib)
- DOI: [10.1121/1.3642604](https://doi.org/10.1121/1.3642604)
- Free PDF: [McGill University](https://www.mcgill.ca/mpcl/files/mpcl/peeters_2011_jasa.pdf)
- Implementations:
  - [MPCL-McGill/TimbreToolbox-R2021a](https://github.com/MPCL-McGill/TimbreToolbox-R2021a)
  - [librosa/librosa](https://github.com/librosa/librosa)

### Blind Audio Watermarking Technique Based on Two Dimensional Cellular Automata

- Common name: APSNR reference
- [BibTeX citation.](https://github.com/hunterhogan/analyzeAudio/blob/main/citations/hiary2016blind.bib)
- DOI: [10.14257/ijsia.2016.10.9.18](https://doi.org/10.14257/ijsia.2016.10.9.18)
- Free repository copy: [Universidad Autonoma de Madrid](https://repositorio.uam.es/handle/10486/676356)
- Implementation:
  - [FFprobe `apsnr` filter](https://ffmpeg.org/ffprobe-all.html#apsnr)
    - [C source with implementation formulas for AI agents.](https://www.ffmpeg.org/doxygen/8.0/af__asdr_8c_source.html)

### SDR - Half-Baked or Well Done?

- Common name: SI-SDR
- [BibTeX citation.](https://github.com/hunterhogan/analyzeAudio/blob/main/citations/leroux2019sdr.bib) [TeX Source with precise formulas for AI agents.](https://arxiv.org/src/1811.02508)
- DOI: [10.1109/ICASSP.2019.8683855](https://doi.org/10.1109/ICASSP.2019.8683855)
- Free author PDF: [Jonathan Le Roux](https://www.jonathanleroux.org/pdf/LeRoux2019ICASSP05sdr.pdf)
- Implementations:
  - [Lightning-AI/torchmetrics](https://github.com/Lightning-AI/torchmetrics)
  - [FFprobe `asisdr` filter](https://ffmpeg.org/ffprobe-all.html#asisdr)

### Loudness Metering: EBU Mode Metering to Supplement Loudness Normalisation

- Common name: momentary LUFS
- [BibTeX citation.](https://github.com/hunterhogan/analyzeAudio/blob/main/citations/ebu2023tech3341.bib)
- Standard: [EBU Tech 3341](https://tech.ebu.ch/publications/tech3341)
- Free PDF: [EBU](https://tech.ebu.ch/docs/tech/tech3341.pdf)
- Implementation:
  - [FFprobe `ebur128` filter](https://ffmpeg.org/ffprobe-all.html#ebur128)
    - [C source with implementation details for AI agents.](https://ffmpeg.org/doxygen/8.0/f__ebur128_8c_source.html)
    - [C source with loudness calculations for AI agents.](https://www.ffmpeg.org/doxygen/8.0/ebur128_8c_source.html)

### Loudness Range: A Measure to Supplement EBU R 128 Loudness Normalisation

- Common name: LUFS
- [BibTeX citation.](https://github.com/hunterhogan/analyzeAudio/blob/main/citations/ebu2023tech3342.bib)
- Standard: [EBU Tech 3342](https://tech.ebu.ch/publications/tech3342)
- Free PDF: [EBU](https://tech.ebu.ch/docs/tech/tech3342.pdf)
- Implementation:
  - [FFprobe `ebur128` filter](https://ffmpeg.org/ffprobe-all.html#ebur128)
    - [C source with implementation details for AI agents.](https://ffmpeg.org/doxygen/8.0/f__ebur128_8c_source.html)
    - [C source with loudness calculations for AI agents.](https://www.ffmpeg.org/doxygen/8.0/ebur128_8c_source.html)

### Algorithms to Measure Audio Programme Loudness and True-Peak Audio Level

- Common name: True peak
- [BibTeX citation.](https://github.com/hunterhogan/analyzeAudio/blob/main/citations/itu2023bs1770.bib)
- Standard: [ITU-R BS.1770-5](https://www.itu.int/rec/R-REC-BS.1770-5-202311-I)
- Free PDF: [ITU](https://www.itu.int/dms_pubrec/itu-r/rec/bs/R-REC-BS.1770-5-202311-I!!PDF-E.pdf)
- Implementation:
  - [FFprobe `ebur128` filter](https://ffmpeg.org/ffprobe-all.html#ebur128)
    - [C source with implementation details for AI agents.](https://ffmpeg.org/doxygen/8.0/f__ebur128_8c_source.html)
    - [C source with loudness calculations for AI agents.](https://www.ffmpeg.org/doxygen/8.0/ebur128_8c_source.html)

### An Overview on Sound Features in Time and Frequency Domain

- [BibTeX citation.](https://github.com/hunterhogan/analyzeAudio/blob/main/citations/constantinescu2023overview.bib)
- DOI: [10.2478/ijasitels-2023-0006](https://doi.org/10.2478/ijasitels-2023-0006)
- Open access article: [Reference Global](https://reference-global.com/fr/article/10.2478/ijasitels-2023-0006)
- PDF: [Reference Global](https://reference-global.com/download/article/10.2478/ijasitels-2023-0006.pdf)

### Perceptual Loss Function for Neural Modelling of Audio Systems

- Common names: ESR loss, DC loss
- Used by: `analyzeESRLoss`, `analyzeDCLoss`
- [BibTeX citation.](https://github.com/hunterhogan/analyzeAudio/blob/main/citations/wright2019perceptualloss.bib)
- arXiv abstract: [1911.08922](https://arxiv.org/abs/1911.08922)
- TeX source with formulas for AI agents: [arXiv source](https://arxiv.org/src/1911.08922)
- PDF: [arXiv PDF](https://arxiv.org/pdf/1911.08922)

### Log Hyperbolic Cosine Loss Improves Variational Auto-Encoder

- Common name: log-cosh loss
- Used by: `analyzeLogCoshLoss`
- [BibTeX citation.](https://github.com/hunterhogan/analyzeAudio/blob/main/citations/chen2019logcosh.bib)
- OpenReview page: [rkglvsC9Ym](https://openreview.net/forum?id=rkglvsC9Ym)
- PDF: [OpenReview PDF](https://openreview.net/pdf?id=rkglvsC9Ym)

### logWMSE Audio Quality Metric and PyTorch Loss Implementation

- Common name: logWMSE
- Used by: `analyzeLogWMSEMean`
- Original implementation:
  - [BibTeX citation.](https://github.com/hunterhogan/analyzeAudio/blob/main/citations/jordal2026logwmse.bib)
  - [nomonosound/log-wmse-audio-quality](https://github.com/nomonosound/log-wmse-audio-quality)
- PyTorch implementation:
  - [BibTeX citation.](https://github.com/hunterhogan/analyzeAudio/blob/main/citations/landschoot2026torchlogwmse.bib)
  - [crlandsc/torch-log-wmse](https://github.com/crlandsc/torch-log-wmse)

### Fast Spectrogram Inversion using Multi-head Convolutional Neural Networks

- Common names: spectral convergence, STFT magnitude loss terms
- Used by: `analyzeSpectralConvergenceLoss`, `analyzeSTFTMagnitudeLoss`, `analyzeSTFTLoss`
- [BibTeX citation.](https://github.com/hunterhogan/analyzeAudio/blob/main/citations/arik2018fastspectrogram.bib)
- DOI: [10.48550/arXiv.1808.06719](https://doi.org/10.48550/arXiv.1808.06719)
- arXiv abstract: [1808.06719](https://arxiv.org/abs/1808.06719)
- TeX source with formulas for AI agents: [arXiv source](https://arxiv.org/src/1808.06719)

### Probability density distillation with generative adversarial networks for high-quality parallel waveform generation

- Used by: `analyzeSTFTLoss`
- [BibTeX citation.](https://github.com/hunterhogan/analyzeAudio/blob/main/citations/yamamoto2019pdd.bib)
- DOI: [10.48550/arXiv.1904.04472](https://doi.org/10.48550/arXiv.1904.04472)
- arXiv abstract: [1904.04472](https://arxiv.org/abs/1904.04472)
- TeX source with formulas for AI agents: [arXiv source](https://arxiv.org/src/1904.04472)

### Parallel WaveGAN: A fast waveform generation model based on generative adversarial networks with multi-resolution spectrogram

- Common name: multi-resolution STFT
- Used by: `analyzeMultiResolutionSTFTLoss`
- [BibTeX citation.](https://github.com/hunterhogan/analyzeAudio/blob/main/citations/yamamoto2019parallelwavegan.bib)
- DOI: [10.48550/arXiv.1910.11480](https://doi.org/10.48550/arXiv.1910.11480)
- arXiv abstract: [1910.11480](https://arxiv.org/abs/1910.11480)
- TeX source with formulas for AI agents: [arXiv source](https://arxiv.org/src/1910.11480)

### auraloss: Audio focused loss functions in PyTorch

- Common names: random-resolution STFT loss implementation source
- Used by: `analyzeRandomResolutionSTFTLoss`
- [BibTeX citation.](https://github.com/hunterhogan/analyzeAudio/blob/main/citations/steinmetz2020auraloss.bib)
- Workshop paper PDF: [DMRN+15 PDF](https://www.christiansteinmetz.com/s/DMRN15__auraloss__Audio_focused_loss_functions_in_PyTorch.pdf)
- Source:
  - [BibTeX citation for the source repository.](https://github.com/hunterhogan/analyzeAudio/blob/main/citations/steinmetz2020auralosssoftware.bib)
  - [csteinmetz1/auraloss](https://github.com/csteinmetz1/auraloss)

### Automatic multitrack mixing with a differentiable mixing console of neural audio effects

- Common names: sum-and-difference STFT loss in neural mixing
- Used by: `analyzeSumAndDifferenceSTFTLoss`
- [BibTeX citation.](https://github.com/hunterhogan/analyzeAudio/blob/main/citations/steinmetz2020multitrackmixing.bib)
- DOI: [10.48550/arXiv.2010.10291](https://doi.org/10.48550/arXiv.2010.10291)
- arXiv abstract: [2010.10291](https://arxiv.org/abs/2010.10291)
- TeX source with formulas for AI agents: [arXiv source](https://arxiv.org/src/2010.10291)

### Neural source-filter waveform models for statistical parametric speech synthesis

- Related in auraloss docs for multi-resolution spectral training context
- [BibTeX citation.](https://github.com/hunterhogan/analyzeAudio/blob/main/citations/wang2019neuralsourcefilter.bib)
- DOI: [10.48550/arXiv.1904.12088](https://doi.org/10.48550/arXiv.1904.12088)
- arXiv abstract: [1904.12088](https://arxiv.org/abs/1904.12088)
- TeX source with formulas for AI agents: [arXiv source](https://arxiv.org/src/1904.12088)

### DDSP: Differentiable Digital Signal Processing

- Related in auraloss docs for STFT-magnitude formulation context
- [BibTeX citation.](https://github.com/hunterhogan/analyzeAudio/blob/main/citations/engel2020ddsp.bib)
- DOI: [10.48550/arXiv.2001.04643](https://doi.org/10.48550/arXiv.2001.04643)
- arXiv abstract: [2001.04643](https://arxiv.org/abs/2001.04643)
- TeX source with formulas for AI agents: [arXiv source](https://arxiv.org/src/2001.04643)

### A Generalized Bandsplit Neural Network for Cinematic Audio Source Separation

- Common name: L1SNR reference
- Used by: `analyzeL1SNRMean`
- [BibTeX citation.](https://github.com/hunterhogan/analyzeAudio/blob/main/citations/watcharasupat2024generalizedbandsplit.bib)
- DOI: [10.1109/OJSP.2023.3339428](https://doi.org/10.1109/OJSP.2023.3339428)
- arXiv abstract: [2309.02539](https://arxiv.org/abs/2309.02539)
- TeX source with formulas for AI agents: [arXiv source](https://arxiv.org/src/2309.02539)

### A Stem-Agnostic Single-Decoder System for Music Source Separation Beyond Four Stems

- Common name: L1SNR reference
- Used by: `analyzeL1SNRMean`, `analyzeL1SNRDBMean`, `analyzeMultiL1SNRDBMean`, `analyzeSTFTL1SNRDBMean`
- [BibTeX citation.](https://github.com/hunterhogan/analyzeAudio/blob/main/citations/watcharasupat2024stemagnostic.bib)
- DOI: [10.48550/arXiv.2406.18747](https://doi.org/10.48550/arXiv.2406.18747)
- arXiv abstract: [2406.18747](https://arxiv.org/abs/2406.18747)
- arXiv HTML used by docstrings: [2406.18747v2](https://arxiv.org/html/2406.18747v2)
- TeX source with formulas for AI agents: [arXiv source](https://arxiv.org/src/2406.18747)

### Separate This, and All of these Things Around It: Music Source Separation via Hyperellipsoidal Queries

- Common name: L1SNRDB reference
- Used by: `analyzeL1SNRDBMean`, `analyzeMultiL1SNRDBMean`, `analyzeSTFTL1SNRDBMean`
- [BibTeX citation.](https://github.com/hunterhogan/analyzeAudio/blob/main/citations/watcharasupat2025hyperellipsoidal.bib)
- DOI: [10.48550/arXiv.2501.16171](https://doi.org/10.48550/arXiv.2501.16171)
- arXiv abstract: [2501.16171](https://arxiv.org/abs/2501.16171)
- arXiv HTML used by docstrings: [2501.16171v1](https://arxiv.org/html/2501.16171v1)
- TeX source with formulas for AI agents: [arXiv source](https://arxiv.org/src/2501.16171)

### torch-l1-snr: L1 Signal-to-Noise Ratio Loss Functions for Audio Source Separation in PyTorch

- Common name: torch-l1-snr
- Used by: `analyzeL1SNRMean`, `analyzeL1SNRDBMean`, `analyzeMultiL1SNRDBMean`, `analyzeSTFTL1SNRDBMean`
- [BibTeX citation.](https://github.com/hunterhogan/analyzeAudio/blob/main/citations/landschoot2026torchl1snr.bib)
- Source: [crlandsc/torch-l1-snr](https://github.com/crlandsc/torch-l1-snr)

### Packages and documentation

- [analyzeAudio](https://github.com/hunterhogan/analyzeAudio)
  - [Context7 reference for AI agents](https://context7.com/hunterhogan/analyzeaudio)
- [FFmpeg documentation](https://ffmpeg.org/documentation.html)
  - [FFprobe filter documentation](https://ffmpeg.org/ffprobe-all.html#Audio-Filters)
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
