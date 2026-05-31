# analyzeAudio

Measure one or more aspects of one or more audio files.

## Note well: FFmpeg & FFprobe binaries must be in PATH

Some options to [download FFmpeg and FFprobe](https://www.ffmpeg.org/download.html) at ffmpeg.org.

[![pip install analyzeAudio](https://img.shields.io/badge/pip_install-analyzeAudio-gray.svg?labelColor=blue)](https://pypi.org/project/analyzeAudio/)
[![uv add analyzeAudio](https://img.shields.io/badge/uv_add-analyzeAudio-gray.svg?labelColor=blue)](https://pypi.org/project/analyzeAudio/)

## Some ways to use this package

The top-level package re-exports a small high-level API:

| API                                                                                | Purpose                                                                      |
| ---------------------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| `analyzeAudioFile(pathFilename, listAspectNames)`                                  | Analyze one file and return one result per requested registered aspect name. |
| `analyzeAudioListPathFilenames(listPathFilenames, listAspectNames, CPUlimit=None)` | Analyze many files in parallel and return one row per completed file.        |
| `getListAvailableAudioAspects()`                                                   | Return the sorted list of registered aspect names.                           |
| `audioAspects`                                                                     | Registry of `aspect name -> analyzer callable + required parameter names`.   |
| `dataTabularTOpathFilenameDelimited(...)`                                          | Write batch results to a delimited text file.                                |

### Use `analyzeAudioFile` to measure one or more registered aspects of a single audio file

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

`analyzeAudioFile` preserves the order of `listAspectNames`. If a requested aspect name is not registered, the matching return entry is `'not found'`.

The registered names are case-sensitive and sometimes very similar names refer to different measurements. For example, `Spectral Flatness mean` and `Spectral flatness mean` are different entries, and so are `Zero-crossing rate mean` and `Zero-crossings rate`.

### Use `analyzeAudioListPathFilenames` to measure one or more aspects for many audio files

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

Each returned row starts with the file path converted to POSIX text, followed by the requested values. The rows are returned in worker-completion order rather than the original input order.

### Use `getListAvailableAudioAspects` and `audioAspects` to inspect the registry or call an analyzer directly

```python
from analyzeAudio import audioAspects, getListAvailableAudioAspects

print(getListAvailableAudioAspects())
print(audioAspects['Chromagram mean']['analyzerParameters'])

SI_SDR_channelsMean = audioAspects['SI-SDR mean']['analyzer'](
    pathFilenameAudioFile,
    pathFilenameDifferentAudioFile,
)
```

Use `audioAspects[name]['analyzerParameters']` to see what inputs a registered analyzer expects. This is especially useful when a registered analyzer needs more than the single `pathFilename` accepted by `analyzeAudioFile`, such as a comparison between two files or a metric that expects tensors.

### Use the lower-level analyzer modules when you want data arrays or tensors instead of one float

Most registered names ending in `mean` are scalar summaries. If you want the full feature array or tensor instead, import the lower-level analyzer function directly:

- `analyzeAudio.analyzersUseFilename`
  - filename-based scalar analyzers, including comparisons such as `SI-SDR mean`
- `analyzeAudio.analyzersUseWaveform`
  - `analyzeTempogram` -> full tempogram array
  - `analyzeRMS` -> framewise RMS-in-dB array
  - `analyzeTempo` -> tempo array
  - `analyzeZeroCrossingRate` -> framewise zero-crossing-rate array
- `analyzeAudio.analyzersUseSpectrogram`
  - `analyzeChromagram` -> chromagram matrix
  - `analyzeSpectralContrast` -> spectral-contrast array
  - `analyzeSpectralBandwidth` -> spectral-bandwidth array
  - `analyzeSpectralCentroid` -> spectral-centroid array
  - `analyzeSpectralFlatness` -> spectral-flatness-in-dB array
- `analyzeAudio.analyzersUseTensor`
  - `analyzeSRMR` -> `torch.Tensor` of SRMR values

The matching `...Mean` functions return one float summary instead. `getListAvailableAudioAspects()` lists the registered public aspect names, not every lower-level helper function.

```python
import numpy
import soundfile

from analyzeAudio.analyzersUseWaveform import analyzeTempogram

with soundfile.SoundFile(pathFilename) as readSoundFile:
    sampleRate = readSoundFile.samplerate
    waveform = readSoundFile.read(dtype='float32').astype(numpy.float32).T

tempogram = analyzeTempogram(waveform, sampleRate)
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

### Realtime Chord Recognition of Musical Sound: A System Using Common Lisp Music

- Common name: chroma features
- [BibTeX citation.](https://github.com/hunterhogan/analyzeAudio/blob/main/citations/fujishima1999realtime.bib)
- Proceedings: [University of Michigan ICMC archive](https://quod.lib.umich.edu/i/icmc/bbp2372.1999.446)
- Implementation:
  - [librosa/librosa](https://github.com/librosa/librosa).feature.chroma_stft

### A Robust Audio Classification and Segmentation Method

- [BibTeX citation.](https://github.com/hunterhogan/analyzeAudio/blob/main/citations/lu2001robust.bib)
- Technical report: [Microsoft Research](https://www.microsoft.com/en-us/research/publication/a-robust-audio-classification-and-segmentation-method/)
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

### Packages and documentation

- [FFmpeg documentation](https://ffmpeg.org/documentation.html)
  - [FFprobe filter documentation](https://ffmpeg.org/ffprobe-all.html#Audio-Filters)
- [librosa/librosa](https://github.com/librosa/librosa)
  - [official documentation](https://librosa.org/doc/latest/index.html)
- [Lightning-AI/torchmetrics](https://github.com/Lightning-AI/torchmetrics)
  - [official documentation](https://lightning.ai/docs/torchmetrics/stable/)
- [sigsep/sigsep-mus-eval](https://github.com/sigsep/sigsep-mus-eval)
- [mir-evaluation/mir_eval](https://github.com/mir-evaluation/mir_eval)

## My recovery

[![Static Badge](https://img.shields.io/badge/2011_August-Homeless_since-blue?style=flat)](https://HunterThinks.com/support)
[![YouTube Channel Subscribers](https://img.shields.io/youtube/channel/subscribers/UC3Gx7kz61009NbhpRtPP7tw)](https://www.youtube.com/@HunterHogan)

[![CC-BY-NC-4.0](https://raw.githubusercontent.com/hunterhogan/analyzeAudio/refs/heads/main/.github/CC-BY-NC-4.0.png)](https://creativecommons.org/licenses/by-nc/4.0/)
