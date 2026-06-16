# Task Completion

- Before finishing any coding task, check the dirty worktree with `git status --short` and distinguish own changes from pre-existing user changes.
- For Python source changes, run the most targeted useful verification first. In `analyzeAudio`, the full pytest suite takes roughly 500 seconds, so do not run it by default; ask first unless the user requested it or the change genuinely requires broad validation:
  - Changed pure code path with existing tests: `pytest <target>` or `pytest -k <name>`.
  - Broader behavior change: `pytest`.
  - Type-heavy change: `pyright` as well as tests.
  - Lint/style-focused change: `ruff check .`; use `ruff format .` only when formatting is intentional.
- For changes involving imports, run `isort .` only when import organization is part of the task or the file already expects isort cleanup.
- For docstring-only changes, verify formatting/lint expectations with `ruff check <path>` when practical; do not run broad expensive audio tests unless the touched code behavior changed.
- For tests added/modified, run the relevant test file or node; then run broader `pytest` if the change affects shared fixtures or package behavior.
- If FFmpeg/FFprobe-backed behavior changes, ensure FFmpeg and FFprobe are available in `PATH`; otherwise report that verification was blocked by missing external binaries.
- Do not run destructive cleanup commands. Ignore `.venv`, caches, generated egg-info, and audio sample binaries unless the task explicitly targets them.
- Final response should mention commands run and any commands not run/blockers.