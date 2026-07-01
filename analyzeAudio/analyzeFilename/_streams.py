from __future__ import annotations

from analyzeAudio.analyzeFilename._wideRange import ffprobeAudioMetadata
from analyzeAudio.registry import registrationStreamMetadata
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from os import PathLike
	from typing import Any

@registrationStreamMetadata('avg_frame_rate')
def get_avg_frame_rate(pathFilename: str | PathLike[Any]) -> str:
	return ffprobeAudioMetadata(pathFilename).get('avg_frame_rate', 'not found')

@registrationStreamMetadata('bit_rate')
def get_bit_rate(pathFilename: str | PathLike[Any]) -> str:
	return ffprobeAudioMetadata(pathFilename).get('bit_rate', 'not found')

@registrationStreamMetadata('bits_per_raw_sample')
def get_bits_per_raw_sample(pathFilename: str | PathLike[Any]) -> str:
	return ffprobeAudioMetadata(pathFilename).get('bits_per_raw_sample', 'not found')

@registrationStreamMetadata('bits_per_sample')
def get_bits_per_sample(pathFilename: str | PathLike[Any]) -> str:
	return ffprobeAudioMetadata(pathFilename).get('bits_per_sample', 'not found')

@registrationStreamMetadata('channel_layout')
def get_channel_layout(pathFilename: str | PathLike[Any]) -> str:
	return ffprobeAudioMetadata(pathFilename).get('channel_layout', 'not found')

@registrationStreamMetadata('channels')
def get_channels(pathFilename: str | PathLike[Any]) -> str:
	return ffprobeAudioMetadata(pathFilename).get('channels', 'not found')

@registrationStreamMetadata('codec_long_name')
def get_codec_long_name(pathFilename: str | PathLike[Any]) -> str:
	return ffprobeAudioMetadata(pathFilename).get('codec_long_name', 'not found')

@registrationStreamMetadata('codec_name')
def get_codec_name(pathFilename: str | PathLike[Any]) -> str:
	return ffprobeAudioMetadata(pathFilename).get('codec_name', 'not found')

@registrationStreamMetadata('codec_tag')
def get_codec_tag(pathFilename: str | PathLike[Any]) -> str:
	return ffprobeAudioMetadata(pathFilename).get('codec_tag', 'not found')

@registrationStreamMetadata('codec_tag_string')
def get_codec_tag_string(pathFilename: str | PathLike[Any]) -> str:
	return ffprobeAudioMetadata(pathFilename).get('codec_tag_string', 'not found')

@registrationStreamMetadata('codec_type')
def get_codec_type(pathFilename: str | PathLike[Any]) -> str:
	return ffprobeAudioMetadata(pathFilename).get('codec_type', 'not found')

@registrationStreamMetadata('disposition_attached_pic')
def get_disposition_attached_pic(pathFilename: str | PathLike[Any]) -> str:
	return ffprobeAudioMetadata(pathFilename).get('disposition.attached_pic', 'not found')

@registrationStreamMetadata('disposition_captions')
def get_disposition_captions(pathFilename: str | PathLike[Any]) -> str:
	return ffprobeAudioMetadata(pathFilename).get('disposition.captions', 'not found')

@registrationStreamMetadata('disposition_clean_effects')
def get_disposition_clean_effects(pathFilename: str | PathLike[Any]) -> str:
	return ffprobeAudioMetadata(pathFilename).get('disposition.clean_effects', 'not found')

@registrationStreamMetadata('disposition_comment')
def get_disposition_comment(pathFilename: str | PathLike[Any]) -> str:
	return ffprobeAudioMetadata(pathFilename).get('disposition.comment', 'not found')

@registrationStreamMetadata('disposition_default')
def get_disposition_default(pathFilename: str | PathLike[Any]) -> str:
	return ffprobeAudioMetadata(pathFilename).get('disposition.default', 'not found')

@registrationStreamMetadata('disposition_dependent')
def get_disposition_dependent(pathFilename: str | PathLike[Any]) -> str:
	return ffprobeAudioMetadata(pathFilename).get('disposition.dependent', 'not found')

@registrationStreamMetadata('disposition_descriptions')
def get_disposition_descriptions(pathFilename: str | PathLike[Any]) -> str:
	return ffprobeAudioMetadata(pathFilename).get('disposition.descriptions', 'not found')

@registrationStreamMetadata('disposition_dub')
def get_disposition_dub(pathFilename: str | PathLike[Any]) -> str:
	return ffprobeAudioMetadata(pathFilename).get('disposition.dub', 'not found')

@registrationStreamMetadata('disposition_forced')
def get_disposition_forced(pathFilename: str | PathLike[Any]) -> str:
	return ffprobeAudioMetadata(pathFilename).get('disposition.forced', 'not found')

@registrationStreamMetadata('disposition_hearing_impaired')
def get_disposition_hearing_impaired(pathFilename: str | PathLike[Any]) -> str:
	return ffprobeAudioMetadata(pathFilename).get('disposition.hearing_impaired', 'not found')

@registrationStreamMetadata('disposition_karaoke')
def get_disposition_karaoke(pathFilename: str | PathLike[Any]) -> str:
	return ffprobeAudioMetadata(pathFilename).get('disposition.karaoke', 'not found')

@registrationStreamMetadata('disposition_lyrics')
def get_disposition_lyrics(pathFilename: str | PathLike[Any]) -> str:
	return ffprobeAudioMetadata(pathFilename).get('disposition.lyrics', 'not found')

@registrationStreamMetadata('disposition_metadata')
def get_disposition_metadata(pathFilename: str | PathLike[Any]) -> str:
	return ffprobeAudioMetadata(pathFilename).get('disposition.metadata', 'not found')

@registrationStreamMetadata('disposition_multilayer')
def get_disposition_multilayer(pathFilename: str | PathLike[Any]) -> str:
	return ffprobeAudioMetadata(pathFilename).get('disposition.multilayer', 'not found')

@registrationStreamMetadata('disposition_non_diegetic')
def get_disposition_non_diegetic(pathFilename: str | PathLike[Any]) -> str:
	return ffprobeAudioMetadata(pathFilename).get('disposition.non_diegetic', 'not found')

@registrationStreamMetadata('disposition_original')
def get_disposition_original(pathFilename: str | PathLike[Any]) -> str:
	return ffprobeAudioMetadata(pathFilename).get('disposition.original', 'not found')

@registrationStreamMetadata('disposition_still_image')
def get_disposition_still_image(pathFilename: str | PathLike[Any]) -> str:
	return ffprobeAudioMetadata(pathFilename).get('disposition.still_image', 'not found')

@registrationStreamMetadata('disposition_timed_thumbnails')
def get_disposition_timed_thumbnails(pathFilename: str | PathLike[Any]) -> str:
	return ffprobeAudioMetadata(pathFilename).get('disposition.timed_thumbnails', 'not found')

@registrationStreamMetadata('disposition_visual_impaired')
def get_disposition_visual_impaired(pathFilename: str | PathLike[Any]) -> str:
	return ffprobeAudioMetadata(pathFilename).get('disposition.visual_impaired', 'not found')

@registrationStreamMetadata('duration')
def get_duration(pathFilename: str | PathLike[Any]) -> str:
	return ffprobeAudioMetadata(pathFilename).get('duration', 'not found')

@registrationStreamMetadata('duration_ts')
def get_duration_ts(pathFilename: str | PathLike[Any]) -> str:
	return ffprobeAudioMetadata(pathFilename).get('duration_ts', 'not found')

@registrationStreamMetadata('id')
def get_id(pathFilename: str | PathLike[Any]) -> str:
	return ffprobeAudioMetadata(pathFilename).get('id', 'not found')

@registrationStreamMetadata('initial_padding')
def get_initial_padding(pathFilename: str | PathLike[Any]) -> str:
	return ffprobeAudioMetadata(pathFilename).get('initial_padding', 'not found')

@registrationStreamMetadata('max_bit_rate')
def get_max_bit_rate(pathFilename: str | PathLike[Any]) -> str:
	return ffprobeAudioMetadata(pathFilename).get('max_bit_rate', 'not found')

@registrationStreamMetadata('nb_frames')
def get_nb_frames(pathFilename: str | PathLike[Any]) -> str:
	return ffprobeAudioMetadata(pathFilename).get('nb_frames', 'not found')

@registrationStreamMetadata('nb_read_frames')
def get_nb_read_frames(pathFilename: str | PathLike[Any]) -> str:
	return ffprobeAudioMetadata(pathFilename).get('nb_read_frames', 'not found')

@registrationStreamMetadata('nb_read_packets')
def get_nb_read_packets(pathFilename: str | PathLike[Any]) -> str:
	return ffprobeAudioMetadata(pathFilename).get('nb_read_packets', 'not found')

@registrationStreamMetadata('profile')
def get_profile(pathFilename: str | PathLike[Any]) -> str:
	return ffprobeAudioMetadata(pathFilename).get('profile', 'not found')

@registrationStreamMetadata('r_frame_rate')
def get_r_frame_rate(pathFilename: str | PathLike[Any]) -> str:
	return ffprobeAudioMetadata(pathFilename).get('r_frame_rate', 'not found')

@registrationStreamMetadata('sample_fmt')
def get_sample_fmt(pathFilename: str | PathLike[Any]) -> str:
	return ffprobeAudioMetadata(pathFilename).get('sample_fmt', 'not found')

@registrationStreamMetadata('sample_rate')
def get_sample_rate(pathFilename: str | PathLike[Any]) -> str:
	return ffprobeAudioMetadata(pathFilename).get('sample_rate', 'not found')

@registrationStreamMetadata('start_pts')
def get_start_pts(pathFilename: str | PathLike[Any]) -> str:
	return ffprobeAudioMetadata(pathFilename).get('start_pts', 'not found')

@registrationStreamMetadata('start_time')
def get_start_time(pathFilename: str | PathLike[Any]) -> str:
	return ffprobeAudioMetadata(pathFilename).get('start_time', 'not found')

@registrationStreamMetadata('time_base')
def get_time_base(pathFilename: str | PathLike[Any]) -> str:
	return ffprobeAudioMetadata(pathFilename).get('time_base', 'not found')
