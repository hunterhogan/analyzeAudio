# Suggested Commands

- Use PowerShell from `C:\apps\analyzeAudio`.
- Install/editable test environment: `python -m pip install -e .[testing]`.
- Run tests with project defaults: `pytest`.
- Run a specific test file: `pytest tests\test_other.py`.
- Run one test: `pytest tests\test_other.py -k testConcurrencyLimit`.
- Run package CLI after editable install or venv activation: `whatMeasurements`.
- Run Ruff lint: `ruff check .`.
- Run Ruff formatter if intentionally formatting code: `ruff format .`.
- Run isort if import organization is needed: `isort .`.
- Run Pyright: `pyright`.
- Check package metadata/build basics: `python -m pip install -e .[testing]` before test runs in fresh environments.
- Useful Windows/PowerShell forms:
  - List tracked-relevant files quickly: `rg --files`.
  - Include dotfiles but avoid local venv/cache blast: prefer targeted `rg --files .github analyzeAudio tests` over `rg --files -uu`.
  - Read a whole text file: `Get-Content -Raw path\to\file.py`.
  - Inspect git worktree: `git status --short`.
- Avoid recursively searching `.venv`, `.git`, caches, and large audio samples unless the task explicitly needs them.