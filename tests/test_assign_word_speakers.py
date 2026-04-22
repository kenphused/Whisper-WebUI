"""
Unit tests for diarize_pipeline.assign_word_speakers.
All tests are CPU-only — no GPU or model download required.
"""
import numpy as np
import pandas as pd
import pytest

from modules.diarize.diarize_pipeline import assign_word_speakers
from modules.whisper.data_classes import Segment


def make_diarize_df(rows):
    """rows: list of (start, end, speaker)"""
    df = pd.DataFrame(rows, columns=["start", "end", "speaker"])
    return df


def make_result(segments):
    """segments: list of dicts with start/end/text and optional words"""
    return {"segments": segments}


# ------------------------------------------------------------------
# Basic speaker assignment
# ------------------------------------------------------------------

def test_single_segment_single_speaker():
    diarize_df = make_diarize_df([(0.0, 5.0, "SPEAKER_00")])
    result = make_result([{"start": 1.0, "end": 3.0, "text": "hello", "words": None}])
    out = assign_word_speakers(diarize_df, result)
    assert out["segments"][0]["speaker"] == "SPEAKER_00"


def test_no_overlap_no_speaker_assigned():
    diarize_df = make_diarize_df([(10.0, 15.0, "SPEAKER_00")])
    result = make_result([{"start": 0.0, "end": 2.0, "text": "hello", "words": None}])
    out = assign_word_speakers(diarize_df, result)
    assert "speaker" not in out["segments"][0]


def test_fill_nearest_assigns_closest_speaker():
    diarize_df = make_diarize_df([(10.0, 15.0, "SPEAKER_00")])
    result = make_result([{"start": 0.0, "end": 2.0, "text": "hello", "words": None}])
    out = assign_word_speakers(diarize_df, result, fill_nearest=True)
    assert out["segments"][0]["speaker"] == "SPEAKER_00"


def test_multiple_speakers_picks_max_overlap():
    diarize_df = make_diarize_df([
        (0.0, 2.0, "SPEAKER_00"),
        (1.5, 5.0, "SPEAKER_01"),
    ])
    # seg overlaps SPEAKER_01 more (1.5 seconds) than SPEAKER_00 (0.5 seconds)
    result = make_result([{"start": 1.5, "end": 4.0, "text": "overlap", "words": None}])
    out = assign_word_speakers(diarize_df, result)
    assert out["segments"][0]["speaker"] == "SPEAKER_01"


def test_empty_segments_returns_empty():
    diarize_df = make_diarize_df([(0.0, 5.0, "SPEAKER_00")])
    result = make_result([])
    out = assign_word_speakers(diarize_df, result)
    assert out["segments"] == []


# ------------------------------------------------------------------
# Word-level speaker assignment
# ------------------------------------------------------------------

def test_word_level_speaker_assignment():
    diarize_df = make_diarize_df([
        (0.0, 1.5, "SPEAKER_00"),
        (1.5, 3.0, "SPEAKER_01"),
    ])
    words = [
        {"start": 0.2, "end": 0.8, "word": "hello"},
        {"start": 1.6, "end": 2.2, "word": "world"},
    ]
    result = make_result([{"start": 0.0, "end": 3.0, "text": "hello world", "words": words}])
    out = assign_word_speakers(diarize_df, result)
    seg_words = out["segments"][0]["words"]
    assert seg_words[0]["speaker"] == "SPEAKER_00"
    assert seg_words[1]["speaker"] == "SPEAKER_01"


def test_words_without_start_field_skipped():
    diarize_df = make_diarize_df([(0.0, 5.0, "SPEAKER_00")])
    words = [{"word": "no_timestamp"}]  # no 'start' key
    result = make_result([{"start": 0.0, "end": 2.0, "text": "test", "words": words}])
    out = assign_word_speakers(diarize_df, result)
    # should not raise and word should have no speaker
    assert "speaker" not in out["segments"][0]["words"][0]


# ------------------------------------------------------------------
# diarize_df is not mutated
# ------------------------------------------------------------------

def test_diarize_df_not_mutated():
    diarize_df = make_diarize_df([(0.0, 5.0, "SPEAKER_00")])
    original_columns = list(diarize_df.columns)
    result = make_result([{"start": 1.0, "end": 3.0, "text": "hi", "words": None}])
    assign_word_speakers(diarize_df, result)
    assert list(diarize_df.columns) == original_columns, "diarize_df columns were mutated"


# ------------------------------------------------------------------
# Segment dataclass input
# ------------------------------------------------------------------

def test_accepts_segment_dataclass_input():
    diarize_df = make_diarize_df([(0.0, 5.0, "SPEAKER_00")])
    segments = [Segment(start=1.0, end=3.0, text="hello")]
    result = {"segments": segments}
    out = assign_word_speakers(diarize_df, result)
    assert out["segments"][0]["speaker"] == "SPEAKER_00"
