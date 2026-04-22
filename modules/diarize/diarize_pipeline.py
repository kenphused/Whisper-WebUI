# Adapted from https://github.com/m-bain/whisperX/blob/main/whisperx/diarize.py

import numpy as np
import pandas as pd
import os
from pyannote.audio import Pipeline
from typing import Dict, List, Optional, Union
import torch

from modules.whisper.data_classes import *
from modules.utils.paths import DIARIZATION_MODELS_DIR
from modules.diarize.audio_loader import load_audio, SAMPLE_RATE


class DiarizationPipeline:
    def __init__(
        self,
        model_name="pyannote/speaker-diarization-3.1",
        cache_dir: str = DIARIZATION_MODELS_DIR,
        use_auth_token=None,
        device: Optional[Union[str, torch.device]] = "cpu",
    ):
        if isinstance(device, str):
            device = torch.device(device)
        self.model = Pipeline.from_pretrained(
            model_name,
            use_auth_token=use_auth_token,
            cache_dir=cache_dir
        ).to(device)

    def __call__(self, audio: Union[str, np.ndarray], min_speakers: Optional[int] = None, max_speakers: Optional[int] = None) -> pd.DataFrame:
        if isinstance(audio, str):
            audio = load_audio(audio)
        audio_data = {
            'waveform': torch.from_numpy(audio[None, :]),
            'sample_rate': SAMPLE_RATE
        }
        segments = self.model(audio_data, min_speakers=min_speakers, max_speakers=max_speakers)
        diarize_df = pd.DataFrame(segments.itertracks(yield_label=True), columns=['segment', 'label', 'speaker'])
        diarize_df['start'] = diarize_df['segment'].apply(lambda x: x.start)
        diarize_df['end'] = diarize_df['segment'].apply(lambda x: x.end)
        return diarize_df


def assign_word_speakers(diarize_df: pd.DataFrame, transcript_result: Dict, fill_nearest: bool = False) -> Dict[str, List]:
    transcript_segments = transcript_result["segments"]
    if not transcript_segments:
        return {"segments": transcript_segments}
    if isinstance(transcript_segments[0], Segment):
        transcript_segments = [seg.model_dump() for seg in transcript_segments]

    for seg in transcript_segments:
        seg_intersection = np.minimum(diarize_df['end'], seg['end']) - np.maximum(diarize_df['start'], seg['start'])
        intersected_mask = seg_intersection > 0

        speaker = None
        if intersected_mask.any():
            intersected_df = diarize_df.loc[intersected_mask].copy()
            intersected_df['intersection'] = seg_intersection[intersected_mask]
            speaker = intersected_df.groupby("speaker")["intersection"].sum().sort_values(ascending=False).index[0]
        elif fill_nearest:
            speaker = diarize_df.assign(intersection=seg_intersection).sort_values(
                by=["intersection"], ascending=False
            )["speaker"].values[0]

        if speaker is not None:
            seg["speaker"] = speaker

        if 'words' in seg and seg['words'] is not None:
            for word in seg['words']:
                if 'start' in word:
                    word_intersection = np.minimum(diarize_df['end'], word['end']) - np.maximum(
                        diarize_df['start'], word['start']
                    )
                    word_intersected_mask = word_intersection > 0

                    word_speaker = None
                    if word_intersected_mask.any():
                        word_intersected_df = diarize_df.loc[word_intersected_mask].copy()
                        word_intersected_df['intersection'] = word_intersection[word_intersected_mask]
                        word_speaker = word_intersected_df.groupby("speaker")["intersection"].sum().sort_values(
                            ascending=False
                        ).index[0]
                    elif fill_nearest:
                        word_speaker = diarize_df.assign(intersection=word_intersection).sort_values(
                            by=["intersection"], ascending=False
                        )["speaker"].values[0]

                    if word_speaker is not None:
                        word["speaker"] = word_speaker

    return {"segments": transcript_segments}


class DiarizationSegment:
    def __init__(self, start, end, speaker=None):
        self.start = start
        self.end = end
        self.speaker = speaker
