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

### Use `whatMeasurements` command line tool to list available measurements

```sh
(.venv) C:\apps\analyzeAudio> whatMeasurements
['Abs_Peak_count', 'Bit_depth', 'Chromagram', 'Chromagram mean', 'Crest factor', 'DC offset', 'Duration-samples', 'Dynamic range', 'Flat_factor', 'LUFS high', 'LUFS integrated', 'LUFS loudness range', 'LUFS low', 'Max_difference', 'Max_level', 'Mean_difference', 'Min_difference', 'Min_level', 'Noise_floor', 'Noise_floor_count', 'Peak dB', 'Peak_count', 'RMS from waveform', 'RMS from waveform mean', 'RMS peak', 'RMS total', 'RMS_difference', 'RMS_trough', 'SI-SDR mean', 'SRMR', 'SRMR mean', 'Signal entropy', 'Spectral Bandwidth', 'Spectral Bandwidth mean', 'Spectral Centroid', 'Spectral Centroid mean', 'Spectral Contrast', 'Spectral Contrast mean', 'Spectral Flatness', 'Spectral Flatness mean', 'Spectral centroid', 'Spectral crest', 'Spectral decrease', 'Spectral entropy', 'Spectral flatness', 'Spectral flux', 'Spectral kurtosis', 'Spectral mean', 'Spectral rolloff', 'Spectral skewness', 'Spectral slope', 'Spectral spread', 'Spectral variance', 'Tempo', 'Tempo mean', 'Tempogram', 'Tempogram mean', 'Zero-crossing rate', 'Zero-crossing rate mean', 'Zero-crossings rate']
```

## Installation

```sh
pip install analyzeAudio
```

## My recovery

[![Static Badge](https://img.shields.io/badge/2011_August-Homeless_since-blue?style=flat)](https://HunterThinks.com/support)
[![YouTube Channel Subscribers](https://img.shields.io/youtube/channel/subscribers/UC3Gx7kz61009NbhpRtPP7tw)](https://www.youtube.com/@HunterHogan)

[![CC-BY-NC-4.0](https://github.com/hunterhogan/analyzeAudio/blob/main/CC-BY-NC-4.0.png)](https://creativecommons.org/licenses/by-nc/4.0/)
