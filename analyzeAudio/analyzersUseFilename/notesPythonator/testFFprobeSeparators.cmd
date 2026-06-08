@ECHO OFF
SET pFn=/data/tests/testPink2ch7.1sec.wav
@REM no: _ -
@REM doesn't work: ' ', " ", \t ^ \^ : ˆ
@REM maybe: ; / ':' or \:
ffprobe -hide_banner -loglevel 0 %pFn% -of flat=s=. -show_entries packets>packets.txt
ffprobe -hide_banner -loglevel 0 %pFn% -of flat=s=',' -show_entries frames>frames.txt
ffprobe -hide_banner -loglevel 0 %pFn% -of flat=s=';' -show_entries library_versions>library_versions.txt
ffprobe -hide_banner -loglevel 0 %pFn% -of flat=s='.' -show_entries packets:frames>packets_frames.txt


@REM Parse everything into a data frame: the parser is static and highly predictable.
@REM Ha! 2026 June 8 I wrote ^^^^^^^^^^ before getting destroyed by dataframes in mapFolding,
@REM astToolFactory, and estimates. I doubt I will want a dataframe.
@REM To get data, 1) build a filtergraph that is compatible with FFprobe
@REM 2) query the data frame
