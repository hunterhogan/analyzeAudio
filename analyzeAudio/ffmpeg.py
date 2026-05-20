"""Install FFmpeg."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

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
		if versionFFmpeg == '':
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

