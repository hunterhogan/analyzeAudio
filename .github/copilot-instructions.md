# Copilot Instructions for analyzeAudio

## Project Overview

This is an audio analysis library that measures various aspects of audio files using a plugin-based architecture. The core concept is a **registry pattern** where analyzer functions are decorated with `@registrationAudioAspect('Aspect Name')` to register themselves in the global `audioAspects` dictionary.

## Architecture

### Core Components
- **`audioAspectsRegistry.py`**: Central registry and main API (`analyzeAudioFile`, `analyzeAudioListPathFilenames`)
- **Analyzer modules**: Four categories based on data source:
  - `analyzersUseFilename.py` - FFmpeg/FFprobe-based analysis
  - `analyzersUseWaveform.py` - Librosa waveform analysis
  - `analyzersUseSpectrogram.py` - Librosa spectrogram analysis
  - `analyzersUseTensor.py` - PyTorch tensor analysis

### Registration Pattern
All analyzers use the `@registrationAudioAspect('Name')` decorator which:
- Registers the function in `audioAspects` global dict
- Automatically creates "mean" variants for numpy.ndarray returns
- Uses parameter introspection to match available data (waveform, spectrogram, etc.)

### Data Flow
1. Audio file → `soundfile` → waveform array
2. Waveform → STFT → spectrogram (magnitude/power)
3. Waveform → PyTorch tensor for GPU-accelerated analysis
4. FFprobe subprocess calls for metadata-based measurements

## Critical Dependencies

### External Binaries (Required)
- **FFmpeg & FFprobe must be in PATH** - Many analyzers fail without these
- Used for: LUFS, spectral stats, signal quality metrics via `ffprobeShotgunAndCache()`

### Python Dependencies
- `librosa` - Spectral/temporal feature extraction
- `torch`/`torchmetrics` - GPU-accelerated SRMR analysis
- `soundfile` - Audio file I/O
- `Z0Z_tools` - Custom utilities for STFT and data export
- `hunterMakesPy` - Concurrency management utilities

## Development Patterns

### Adding New Analyzers
```python
@registrationAudioAspect('My New Metric')
def analyzeMyMetric(waveform: numpy.ndarray, sampleRate: int) -> float:
    # Parameter names must match variables in analyzeAudioFile()
    return some_calculation(waveform)
```

### Parameter Matching System
Analyzer functions receive parameters by name matching from these available variables:
- `pathFilename`, `waveform`, `sampleRate`, `spectrogram`, `spectrogramMagnitude`, `spectrogramPower`, `tensorAudio`, `pytorchOnCPU`

### Caching Strategy
- `@cachetools.cached(cache=cacheAudioAnalyzers)` for expensive computations
- FFprobe results cached in `ffprobeShotgunAndCache()` with LRU cache
- Cache cleared per-file in batch processing to manage memory

## Testing & Quality

### Test Structure
- `tests/conftest.py` - Sample audio files in `dataSamples/`
- `tests/test_audioAspectsRegistry.py` - Core functionality tests
- `tests/test_other.py` - External dependency tests (hunterMakesPy utilities)

### Test Commands
```bash
pytest                    # Run all tests with 4 parallel workers
pytest -n 1             # Single-threaded for debugging
whatMeasurements         # CLI tool to list all available aspects
```

### Code Quality
- **Ruff** linting with custom config (`ruff.toml`) - tabs over spaces, numpy docstring style
- **Pyright** type checking with extensive type annotations
- **pytest-cov** for coverage reporting in `tests/coverage/`

## Project-Specific Conventions

### Naming Patterns
- Aspect names use title case: "Spectral Flatness", "LUFS integrated"
- Multiple similar metrics distinguished carefully: "Zero-crossing rate" vs "Zero-crossings rate"
- Analyzer functions: `analyze[MetricName]` pattern

### Type System
- Heavy use of `numpy.typing.NDArray` with shape annotations
- `TypeAlias` for audio aspect names
- Custom `TypedDict` for analyzer registry structure

### Windows Path Handling
Special handling in `ffprobeShotgunAndCache()` for lavfi paths:
```python
lavfiPathFilename = pFn.drive.replace(":", "\\\\:")+pathlib.PureWindowsPath(pFn.root,pFn.relative_to(pFn.anchor)).as_posix()
```

## Build & Release

### Installation
```bash
pip install ".[testing]"  # Development with test dependencies
```

### Automated Workflows
- **CI**: Python 3.10-3.13 matrix testing on GitHub Actions
- **Release**: Auto-publish to PyPI when version bumped in `pyproject.toml`
- **Citation**: Auto-update CITATION.cff on push

### Command Line Tools
- `whatMeasurements` entry point lists all available audio aspects
- Use for discovering new measurements and validating installations

## Common Gotchas

1. **FFmpeg dependency** - Many functions silently fail without FFmpeg/FFprobe in PATH
2. **Aspect name precision** - Similar aspect names have different implementations
3. **Memory management** - Batch processing clears caches per-file to prevent OOM
4. **GPU availability** - PyTorch functions auto-detect and adapt to CPU-only systems
5. **Windows paths** - Special escaping needed for FFmpeg lavfi filter paths

## Data Export
Results integrate with `Z0Z_tools.dataTabularTOpathFilenameDelimited()` for saving batch analysis results to tab-delimited files.
