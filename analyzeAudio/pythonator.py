from collections import defaultdict
from typing import Any, cast, NamedTuple
import json
import numpy

# NOTE hey! hey! hey!
# Is blackdetect broken?
# 1. You don't have pytest tests for anything in the entire fricken package
# 2. You tried to improve the blackdetect code, but you didn't test it with anything
# 3. Search for "uncommentToFixBlackdetect"
# NOTE You changed the code because a static type checker was mad at you. Ask yourself,
# "Are you the tool or is the type checker the tool?"

class Blackdetect(NamedTuple):
	black_start: float | None = None
	black_end: float | None = None

def pythonizeFFprobe(FFprobeJSON_utf8: str):
	FFroot: dict[str, Any] = json.loads(FFprobeJSON_utf8)
	Z0Z_dictionaries: dict[str, numpy.ndarray[Any, Any] | dict[str, numpy.ndarray[Any, Any]]] = {}
	if 'packets_and_frames' in FFroot: # Divide into 'packets' and 'frames'
		FFroot = defaultdict(list, FFroot)
		for packetOrFrame in FFroot['packets_and_frames']:
			if 'type' in packetOrFrame:
				FFroot[section := packetOrFrame['type'] + 's'].append(packetOrFrame)
				del FFroot[section][-1]['type']
			else:
				raise ValueError("'packets_and_frames' for the win!")
		del FFroot['packets_and_frames']

	Z0Z_register = [
		'aspectralstats',
		'astats',
		'r128',
		'signalstats',
	]
	leftCrumbs = False
	if 'frames' in FFroot:
		leftCrumbs = False
		# listTuplesBlackdetect = [] # uncommentToFixBlackdetect
		listTuplesBlackdetect: list[Blackdetect] = []
		for indexFrame, FFframe in enumerate(FFroot['frames']):
			if 'tags' in FFframe:
				if 'lavfi.black_start' in FFframe['tags']:
					# listTuplesBlackdetect.append(float(FFframe['tags']['lavfi.black_start'])) # uncommentToFixBlackdetect
					listTuplesBlackdetect.append(Blackdetect(black_start=float(FFframe['tags']['lavfi.black_start'])))
					del FFframe['tags']['lavfi.black_start']
				if 'lavfi.black_end' in FFframe['tags']:
					# listTuplesBlackdetect[-1] = (listTuplesBlackdetect[-1], float(FFframe['tags']['lavfi.black_end'])) # uncommentToFixBlackdetect
					tupleBlackdetectLast = listTuplesBlackdetect.pop() if listTuplesBlackdetect else Blackdetect()
					match tupleBlackdetectLast.black_end:
						case None:
							listTuplesBlackdetect.append(Blackdetect(tupleBlackdetectLast.black_start, float(FFframe['tags']['lavfi.black_end'])))
						case _:
							if tupleBlackdetectLast.black_start is not None:
								listTuplesBlackdetect.append(tupleBlackdetectLast)
							listTuplesBlackdetect.append(Blackdetect(black_end=(float(FFframe['tags']['lavfi.black_end']))))
					del FFframe['tags']['lavfi.black_end']

				# This is not the way to do it
				for keyName, keyValue in FFframe['tags'].items():
					if 'lavfi' in (keyNameDeconstructed := keyName.split('.'))[0]:
						channel = None
						if (registrant := keyNameDeconstructed[1]) in Z0Z_register:
							keyNameDeconstructed = keyNameDeconstructed[2:]
							if keyNameDeconstructed[0].isdigit():
								channel = int(keyNameDeconstructed[0])
								keyNameDeconstructed = keyNameDeconstructed[1:]
							statistic = '.'.join(keyNameDeconstructed)
							if channel is None:
								while True:
									try:
										Z0Z_dictionaries[registrant][statistic][indexFrame] = float(keyValue)
										break  # If successful, exit the loop
									except KeyError:
										if registrant not in Z0Z_dictionaries:
											Z0Z_dictionaries[registrant] = {}
										elif statistic not in Z0Z_dictionaries[registrant]:
											Z0Z_dictionaries[registrant][statistic] = numpy.zeros(len(FFroot['frames']))
										else:
											raise  # Re-raise the exception
							else:
								while True:
									try:
										Z0Z_dictionaries[registrant][statistic][channel - 1, indexFrame] = float(keyValue)
										break  # If successful, exit the loop
									except KeyError:
										if registrant not in Z0Z_dictionaries:
											Z0Z_dictionaries[registrant] = {}
										elif statistic not in Z0Z_dictionaries[registrant]:
												# NOTE (as of this writing) `registrar` can only understand the generic class `numpy.ndarray` and not more specific typing
												valueSherpa = cast(numpy.ndarray, numpy.zeros((channel, len(FFroot['frames']))))
												Z0Z_dictionaries[registrant][statistic] = valueSherpa
										else:
											raise  # Re-raise the exception
									except IndexError:
										if channel > Z0Z_dictionaries[registrant][statistic].shape[0]:
											Z0Z_dictionaries[registrant][statistic] = numpy.resize(Z0Z_dictionaries[registrant][statistic], (channel, len(FFroot['frames'])))
											# Z0Z_dictionaries[registrant][statistic].resize((channel, len(FFroot['frames'])))
										else:
											raise  # Re-raise the exception

				if not FFframe['tags']: # empty = False
					del FFframe['tags']
			if FFframe:
				leftCrumbs = True
		if listTuplesBlackdetect:
			# 2025-03-06: I am _shocked_ that I was able to create a numpy structured array whenever it was that I originally wrote this code.
			arrayBlackdetect = numpy.array(
				[(
					-1.0 if detect.black_start is None else detect.black_start,
					-1.0 if detect.black_end is None else detect.black_end
				) for detect in listTuplesBlackdetect],
				dtype=[('black_start', numpy.float64), ('black_end', numpy.float64)],
				copy=False
			)
			Z0Z_dictionaries['blackdetect'] = arrayBlackdetect
			# Z0Z_dictionaries['blackdetect'] = numpy.array(listTuplesBlackdetect, dtype=[('black_start', numpy.float32), ('black_end', numpy.float32)], copy=False) # uncommentToFixBlackdetect
	if not leftCrumbs:
		del FFroot['frames']
	return FFroot, Z0Z_dictionaries
