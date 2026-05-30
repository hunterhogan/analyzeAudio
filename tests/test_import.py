"""Drop-in quality checks for any src-layout package."""

from __future__ import annotations

from typing import TYPE_CHECKING
import importlib

if TYPE_CHECKING:
	from types import ModuleType

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
def test_import_package(package: ModuleType) -> None:
	"""Top-level package imports cleanly."""
	assert package is not None


def test_import_all_modules(all_module_names: list[str]) -> None:
	"""Every sub-module can be imported without errors."""
	errors: list[str] = []
	for moduleName in all_module_names:
		try:
			importlib.import_module(moduleName)
		except Exception as e:  # noqa: PERF203
			errors.append(f'{moduleName}: {e}')
	assert not errors, 'Failed to import modules:\n' + '\n'.join(errors)
