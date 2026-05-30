"""Zero-configuration conftest for src-layout projects."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
import importlib
import pathlib
import pkgutil
import pytest
import warnings

if TYPE_CHECKING:
	from types import ModuleType

def _discover_package_name() -> str:
	project_root = Path(__file__).resolve().parent.parent
	src = project_root
	if not src.is_dir():
		msg = f"'src' directory not found at {src}"
		raise FileNotFoundError(msg)

	packages = [d.name for d in src.iterdir() if d.is_dir() and not d.name.startswith('.') and d.name != '__pycache__']
	if not packages:
		msg = f'No Python packages found in {src}'
		raise FileNotFoundError(msg)

	if len(packages) > 1:
		warnings.warn(f"Multiple packages found in src: {packages}. Using '{packages[0]}' as the main package.", stacklevel=2)
	return packages[0]


@pytest.fixture(scope='session')
def package_name() -> str:
	return _discover_package_name()


@pytest.fixture(scope='session')
def package(package_name: str) -> ModuleType:
	"""The imported top-level package."""  # noqa: DOC201
	return importlib.import_module(package_name)


@pytest.fixture(scope='session')
def all_module_names(package_name: str) -> list[str]:
	"""All module names in the package (including subpackages)."""  # noqa: DOC201
	pkg = importlib.import_module(package_name)
	modules = [package_name]
	prefix = package_name + '.'
	for _idk, moduleName, _lame in pkgutil.walk_packages(pkg.__path__, prefix=prefix):
		modules.append(moduleName)
	return modules

pathDataSamples = pathlib.Path(__file__).parent / "dataSamples"

listOfFiles = ["pink-20RMS60sec.wav",
"pink-40RMS60sec.wav",
"pink-60RMS60sec.wav",
"testParkMono96kHz32float12.1sec.wav",
"testPink2ch7.1sec.wav",
"testSine2ch5sec.wav",
"testSine2ch5secCopy1.wav",
"testTrain2ch48kHz6.3sec.wav",
"testVideo11sec.mkv",
"testWooWooMono16kHz32integerClipping9sec.wav",
]
