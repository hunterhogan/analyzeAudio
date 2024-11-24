from numpy.typing import NDArray
from typing import Any, Union
import analyzeAudio.audioAspectsRegistry as audioAspectsRegistry
import numpy
import os
import pathlib
import unittest

class test_audioAspectsRegistry(unittest.TestCase):

    pathFilenameSample = pathlib.Path(__file__).parent / 'dataSamples' / 'testSine2ch5sec.wav'
    pathFilenameSampleCopy1 = pathlib.Path(__file__).parent / 'dataSamples' / 'testSine2ch5secCopy1.wav'

    def test_analyzeAudioFile_nonExistentFile(self):
        pathFilenameNonExistent = self.pathFilenameSample.parent / 'nonExistentFile.wav'
        with self.assertRaises(FileNotFoundError):
            audioAspectsRegistry.analyzeAudioFile(pathFilenameNonExistent, [''])

    def test_analyzeAudioFile_invalidAspect(self):
        invalidAspect = 'invalidAspect'
        result = audioAspectsRegistry.analyzeAudioFile(self.pathFilenameSample, [invalidAspect])
        self.assertEqual(result, ['not found'])

    def test_analyzeAudioListPathFilenames_empty(self):
        emptyList: list[Any] = []
        result = audioAspectsRegistry.analyzeAudioListPathFilenames(emptyList, [])
        self.assertEqual(result, [])

    def test_analyzeAudioListPathFilenames_nonExistentFile(self):
        pathFilenameNonExistent = self.pathFilenameSample.parent / 'nonExistentFile.wav'
        with self.assertRaises(FileNotFoundError): # I expect to fail, but I don't know the form of the exception
            audioAspectsRegistry.analyzeAudioListPathFilenames([pathFilenameNonExistent], [''])

    @audioAspectsRegistry.registrationAudioAspect('stereo') 
    def getStereo(waveform) -> bool:
        """
        Checks if the audio is stereo.
        """
        return waveform.shape[0] == 2

    def test_analyzeAudioFile_stereo(self):
        result = audioAspectsRegistry.analyzeAudioFile(self.pathFilenameSample, ['stereo'])
        self.assertEqual(result, [True])

if __name__ == '__main__':
    unittest.main()
