"""Install FFmpeg."""
from __future__ import annotations

from pathlib import Path
import os
import shutil
import subprocess  # noqa: S404
import sys
import tempfile
import urllib.request

def FFmpegGitHub() -> None:
	if os.getenv('GITHUB_ACTIONS') == 'true' and sys.platform == 'linux':

		import fcntl

		pathDirectoryFFmpeg: Path = Path(os.environ.get('RUNNER_TEMP', tempfile.gettempdir())) / 'analyzeAudio-ffmpeg'
		pathDirectoryFFmpeg.mkdir(parents=True, exist_ok=True)
		if str(pathDirectoryFFmpeg) not in os.environ.get('PATH', '').split(os.pathsep):
			os.environ['PATH'] = f"{pathDirectoryFFmpeg}{os.pathsep}{os.environ.get('PATH', '')}"

		with (pathDirectoryFFmpeg / 'install.lock').open('w') as fileLock:
			fcntl.flock(fileLock.fileno(), fcntl.LOCK_EX)
			versionFFmpeg: str = ''
			majorVersionFFmpeg: int = 0
			try:
				systemProcessFFprobeVersion: subprocess.CompletedProcess[str] = subprocess.run(
					['ffprobe', '-hide_banner', '-show_entries', 'program_version=version', '-of', 'csv=p=0']  # noqa: S607
					, check=False
					, stdout=subprocess.PIPE
					, stderr=subprocess.DEVNULL
					, text=True
				)
			except FileNotFoundError:
				pass
			else:
				if systemProcessFFprobeVersion.returncode == 0:
					versionFFmpeg = systemProcessFFprobeVersion.stdout.strip()
					if versionFFmpeg:
						majorVersionFFmpeg = int(versionFFmpeg.split('.', maxsplit=1)[0])

			if majorVersionFFmpeg < 7:
				with tempfile.TemporaryDirectory(prefix='analyzeAudio-ffmpeg-') as pathDirectoryTemporaryString:
					pathDirectoryTemporary: Path = Path(pathDirectoryTemporaryString)
					pathFilenameFFmpegReleaseArchive: Path = pathDirectoryTemporary / 'ffmpeg-release.tar.xz'
					urllib.request.urlretrieve(
						'https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz'
						, pathFilenameFFmpegReleaseArchive
					)
					subprocess.run(['/usr/bin/tar', '-xf', str(pathFilenameFFmpegReleaseArchive), '-C', str(pathDirectoryTemporary)], check=True)
					pathDirectoryFFmpegStaticBuild: Path = next(pathDirectoryTemporary.glob('ffmpeg-*-static'))
					shutil.copy2(pathDirectoryFFmpegStaticBuild / 'ffmpeg', pathDirectoryFFmpeg / 'ffmpeg')
					shutil.copy2(pathDirectoryFFmpegStaticBuild / 'ffprobe', pathDirectoryFFmpeg / 'ffprobe')
					(pathDirectoryFFmpeg / 'ffmpeg').chmod(0o755)
					(pathDirectoryFFmpeg / 'ffprobe').chmod(0o755)


def verifyFFmpegColab() -> None:
	"""Upgrade FFmpeg if needed."""
	if 'google.colab' in sys.modules:
		versionFFmpeg: str = ''
		if Path('/usr/bin/dpkg-query').exists():
			systemProcessLinuxFFmpegVersion: subprocess.CompletedProcess[str] = subprocess.run(
				['/usr/bin/dpkg-query', '--show', '--showformat=${Version}', 'ffmpeg']
				, check=False
				, stdout=subprocess.PIPE
				, stderr=subprocess.DEVNULL
				, text=True
			)
			if systemProcessLinuxFFmpegVersion.returncode == 0:
				versionFFmpeg = systemProcessLinuxFFmpegVersion.stdout.strip()
				if ':' in versionFFmpeg:
					_versionFFmpegEpochIgnored, versionFFmpeg = versionFFmpeg.split(':', maxsplit=1)
		if not versionFFmpeg:
			systemProcessFFprobeVersion: subprocess.CompletedProcess[str] = subprocess.run(
				['ffprobe', '-hide_banner', '-show_entries', 'program_version=version', '-of', 'csv=p=0']  # noqa: S607
				, check=True
				, stdout=subprocess.PIPE
				, text=True
			)
			versionFFmpeg = systemProcessFFprobeVersion.stdout.strip()

		majorVersionFFmpeg, _versionFFmpegRemainderIgnored = versionFFmpeg.split('.', maxsplit=1)
		if int(majorVersionFFmpeg) < 7:
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
