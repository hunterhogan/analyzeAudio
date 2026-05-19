# Tech Stack

- Language/package: typed Python package, `requires-python >=3.11`, currently version `0.0.18` in `pyproject.toml`.
- Build backend: setuptools via `pyproject.toml`; packages discovered automatically; `py.typed` is included as package data.
- Runtime dependencies: `Z0Z_tools`, `cachetools`, `hunterMakesPy`, `librosa`, `numpy`, `torch`, `torchmetrics[audio]`, `tqdm`, `typing_extensions`, plus `standard-aifc`/`standard-sunau` for Python >=3.13.
- External binaries: FFmpeg/FFprobe are required for filename-based analyzers and README workflows.
- CLI entry point: `whatMeasurements = analyzeAudio.audioAspectsRegistry:getListAvailableAudioAspects`.
- Test dependencies: `pytest`, `pytest-cov`, `pytest-xdist` from optional dependency group `testing`.
- Test config: `[tool.pytest]` runs `tests` with `--color=auto -n 4`; coverage writes data under `tests/coverage/` and omits `tests/*`.
- Type checking: `pyrightconfig.json` uses strict mode, strict container inference, no type-ignore comments; `ty.toml` sets platform to all and ignores type-ignore comments.
- Lint/format: Ruff selects `ALL`, line length 140, tab indentation, single quotes, pydocstyle numpy; many local ignores intentionally allow current project style. `.isort.cfg` inserts `from __future__ import annotations`, no import sections, tab indentation, line length 140.
- EditorConfig: Python/default files use UTF-8, LF, tabs with width 4, max line 140, trim trailing whitespace, final newline. Markdown/TOML/YAML use 2 spaces and max line 102.
- CI: `.github/workflows/pythonTests.yml` installs editable package with `[testing]` and runs `pytest` across supported Python versions discovered from `requires-python` and GitHub Actions manifests.
- Documentation metadata: Context7 project URL in `context7.json` is `https://context7.com/hunterhogan/analyzeaudio`.