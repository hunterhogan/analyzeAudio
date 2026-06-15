"""Install FFmpeg."""
from __future__ import annotations

from operator import getitem
from pathlib import Path
import contextlib
import os
import shutil
import subprocess  # noqa: S404
import sys
import tempfile
import urllib.request

if sys.platform == 'linux':
	import fcntl

def FFmpegGitHub() -> None:
	if os.getenv('GITHUB_ACTIONS') == 'true' and sys.platform == 'linux':

		pathFFmpeg: Path = Path(tempfile.gettempdir(), 'ffmpeg')
		pathFFmpeg.mkdir(parents=True, exist_ok=True)
		if str(pathFFmpeg) not in os.environ.get('PATH', '').split(os.pathsep):
			os.environ['PATH'] = f"{pathFFmpeg}{os.pathsep}{os.environ.get('PATH', '')}"

		with (pathFFmpeg / 'install.lock').open('w') as writeStream:
			fcntl.flock(writeStream.fileno(), fcntl.LOCK_EX)
			versionMajor: int = 0
			with contextlib.suppress(FileNotFoundError):
				processFFprobeVersion: subprocess.CompletedProcess[str] = subprocess.run(
					['ffprobe', '-hide_banner', '-show_entries', 'program_version=version', '-of', 'csv=p=0']  # noqa: S607
					, check=False
					, stdout=subprocess.PIPE
					, stderr=subprocess.DEVNULL
					, text=True
				)
				versionMajor = int(getitem(processFFprobeVersion.stdout.strip().split('.', maxsplit=1), 0))

			if versionMajor < 7:
				pathFilename_xz: Path = pathFFmpeg / 'ffmpeg-release.tar.xz'
				url: str = 'https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-n8.1-latest-linux64-gpl-8.1.tar.xz'
				with urllib.request.urlopen(url) as responseFFmpegReleaseArchive, pathFilename_xz.open('wb') as writeStreamBinary:
					shutil.copyfileobj(responseFFmpegReleaseArchive, writeStreamBinary)
				subprocess.run(['/usr/bin/tar', '-xf', str(pathFilename_xz), '-C', str(pathFFmpeg)], check=True)
				pathFFmpegExecutables: Path = next(pathFFmpeg.glob('ffmpeg-*/bin'), next(pathFFmpeg.glob('ffmpeg-*-static')))
				shutil.copy2(pathFFmpegExecutables / 'ffmpeg', pathFFmpeg / 'ffmpeg')
				shutil.copy2(pathFFmpegExecutables / 'ffprobe', pathFFmpeg / 'ffprobe')
				(pathFFmpeg / 'ffmpeg').chmod(0o755)
				(pathFFmpeg / 'ffprobe').chmod(0o755)

def verifyFFmpegColab() -> None:
	"""Upgrade FFmpeg if needed."""
	if 'google.colab' in sys.modules:
		versionFFmpeg: str = ''
		if Path('/usr/bin/dpkg-query').exists():
			systemProcess_dpkg_query: subprocess.CompletedProcess[str] = subprocess.run(
				['/usr/bin/dpkg-query', '--show', '--showformat=${Version}', 'ffmpeg']
				, check=False
				, stdout=subprocess.PIPE
				, stderr=subprocess.DEVNULL
				, text=True
			)
			if systemProcess_dpkg_query.returncode == 0 and ':' in systemProcess_dpkg_query.stdout:
				_name, versionFFmpeg = systemProcess_dpkg_query.stdout.strip().split(':', maxsplit=1)
		if not versionFFmpeg:
			systemProcessFFprobeVersion: subprocess.CompletedProcess[str] = subprocess.run(
				['ffprobe', '-hide_banner', '-show_entries', 'program_version=version', '-of', 'csv=p=0']  # noqa: S607
				, check=True
				, stdout=subprocess.PIPE
				, text=True
			)
			versionFFmpeg = systemProcessFFprobeVersion.stdout.strip()

		versionMajor, _version = versionFFmpeg.split('.', maxsplit=1)
		if int(versionMajor) < 7:
			filenameFFmpegReleaseArchive: str = 'ffmpeg-release.tar.xz'
			subprocess.run(
				[
					'/usr/bin/wget'
					, '-qO'
					, filenameFFmpegReleaseArchive
					, 'https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz'
				]
				, check=True
			)
			subprocess.run(['/usr/bin/tar', '-xf', filenameFFmpegReleaseArchive], check=True)
			pathFFmpegStaticBuild: Path = next(Path.cwd().glob('ffmpeg-*-amd64-static'))
			subprocess.run(
				[
					'/usr/bin/sudo'
					, '/usr/bin/mv'
					, str(pathFFmpegStaticBuild / 'ffmpeg')
					, str(pathFFmpegStaticBuild / 'ffprobe')
					, '/usr/local/bin/'
				]
				, check=True
			)
			subprocess.run(
				['/usr/bin/sudo', '/usr/bin/apt-get', 'remove', '-y', '--autoremove', 'ffmpeg']
				, check=True
			)
