# Core

- Python package for measuring named audio aspects from audio/video filenames or decoded audio arrays.
- Public API re-exported from `analyzeAudio/__init__.py`: `analyzeAudioFile`, `analyzeAudioListPathFilenames`, `audioAspects`, `getListAvailableAudioAspects`.
- Importing `analyzeAudio` imports analyzer modules for registration side effects. New analyzers should use `registrationAudioAspect(aspectName)` so `audioAspects` receives the callable and its inspected parameter list.
- `audioAspectsRegistry.py` is the orchestration center:
  - `audioAspects`: mapping from public aspect names to `{analyzer, analyzerParameters}`.
  - `registrationAudioAspect`: decorator; for analyzer returns annotated as `numpy.ndarray`, auto-registers an additional `<aspectName> mean` analyzer.
  - `analyzeAudioFile`: validates path existence, reads audio with `soundfile`, transposes waveform to channel-first, computes torch tensor/spectrogram/magnitude/power once, dispatches analyzers by matching parameter names against local bindings.
  - `analyzeAudioListPathFilenames`: uses `ProcessPoolExecutor`, `hunterMakesPy.parseParameters.defineConcurrencyLimit`, and clears per-file analyzer cache entries as futures finish.
- Analyzer module roles:
  - `analyzersUseFilename.py`: FFmpeg/FFprobe-backed filename analyzers; `ffprobeShotgunAndCache()` parses lavfi JSON through `pythonizeFFprobe()` and caches results.
  - `analyzersUseWaveform.py`: librosa waveform features: tempogram, RMS, tempo, zero-crossing rate.
  - `analyzersUseSpectrogram.py`: librosa spectrogram features: chromagram, spectral contrast/bandwidth/centroid/flatness.
  - `analyzersUseTensor.py`: torchmetrics SRMR analyzer.
  - `pythonator.py`: converts FFprobe JSON, especially lavfi frame tags, to Python/numpy structures.
  - `ffmpeg.py`: Colab-only FFmpeg upgrade helper.
- `tests/` currently contains minimal pytest coverage; `tests/dataSamples/` contains audio/video fixtures used for manual or future tests.
- External runtime invariant: FFmpeg and FFprobe binaries must be available in `PATH` for filename analyzers and README examples.
- Read `mem:tech_stack` for runtime/development tooling, `mem:suggested_commands` for useful commands, `mem:conventions` for project/user coding rules, and `mem:task_completion` before wrapping up coding work.