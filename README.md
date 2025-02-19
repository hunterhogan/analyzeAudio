# analyzeAudio

Measure one or more aspects of one or more audio files.

## Note well: FFmpeg & FFprobe binaries must be in PATH

Some options to [download FFmpeg and FFprobe](https://www.ffmpeg.org/download.html) at ffmpeg.org.

## Some ways to use this package

### Use `analyzeAudioFile` to measure one or more aspects of a single audio file

```python
from analyzeAudio import analyzeAudioFile
listAspectNames = ['LUFS integrated',
                   'RMS peak',
                   'SRMR mean',
                   'Spectral Flatness mean']
listMeasurements = analyzeAudioFile(pathFilename, listAspectNames)
```

### Use `getListAvailableAudioAspects` to get a crude list of aspects this package can measure

The aspect names are accurate, but the lack of additional documentation can make things challenging. 'Zero-crossing rate', 'Zero-crossing rate mean', and 'Zero-crossings rate', for example, are different from each other. ("... lack of additional documentation ...")

```python
import analyzeAudio
analyzeAudio.getListAvailableAudioAspects()
```

### Use `analyzeAudioListPathFilenames` to measure one or more aspects of individual file in a list of audio files

### Use `audioAspects` to call an analyzer function by using the name of the aspect you wish to measure

```python
from analyzeAudio import audioAspects
SI_SDR_channelsMean = audioAspects['SI-SDR mean']['analyzer'](pathFilenameAudioFile, pathFilenameDifferentAudioFile)
```

Retrieve the names of the parameters for an analyzer function with the `['analyzerParameters']` key-name.

```python
from analyzeAudio import audioAspects
print(audioAspects['Chromagram']['analyzerParameters'])
```

## Installation

```sh
pip install analyzeAudio
```

## My recovery

[![Static Badge](https://img.shields.io/badge/2011_August-Homeless_since-blue?style=flat)](https://HunterThinks.com/support)
[![YouTube Channel Subscribers](https://img.shields.io/youtube/channel/subscribers/UC3Gx7kz61009NbhpRtPP7tw)](https://www.youtube.com/@HunterHogan)
