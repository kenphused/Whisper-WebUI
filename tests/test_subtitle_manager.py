import json
import os
import tempfile
import pytest

from modules.utils.subtitle_manager import (
    WriteSRT, WriteVTT, WriteTXT, WriteLRC, WriteTSV, WriteJSON,
    format_timestamp, time_str_to_seconds,
    get_start, get_end, get_writer, generate_file, safe_filename,
)
from modules.whisper.data_classes import Segment, Word


# ------------------------------------------------------------------
# Shared fixtures
# ------------------------------------------------------------------

SAMPLE_SEGMENTS = [
    Segment(start=0.0, end=2.5, text="Hello world"),
    Segment(start=3.0, end=5.8, text="How are you?"),
    Segment(start=6.1, end=9.0, text="Fine, thanks."),
]

SAMPLE_RESULT = {"segments": [s.model_dump() for s in SAMPLE_SEGMENTS]}

# Segments with word-level timestamps (needed to exercise iterate_result word path)
WORD_SEGMENTS = [
    Segment(
        start=0.0, end=2.5, text="Hello world",
        words=[
            Word(start=0.0, end=1.0, word="Hello", probability=0.9),
            Word(start=1.1, end=2.5, word=" world", probability=0.95),
        ]
    ),
    Segment(
        start=3.0, end=5.8, text="How are you?",
        words=[
            Word(start=3.0, end=3.5, word="How", probability=0.9),
            Word(start=3.6, end=4.0, word=" are", probability=0.85),
            Word(start=4.1, end=5.8, word=" you?", probability=0.9),
        ]
    ),
]

WORD_RESULT = {"segments": [s.model_dump() for s in WORD_SEGMENTS]}


# ------------------------------------------------------------------
# format_timestamp / time_str_to_seconds
# ------------------------------------------------------------------

@pytest.mark.parametrize("seconds,decimal_marker,always_hours", [
    (0.0, ",", True),
    (3661.5, ",", True),
    (90.123, ".", False),
    (3599.999, ",", True),
])
def test_timestamp_roundtrip(seconds, decimal_marker, always_hours):
    ts = format_timestamp(seconds, always_include_hours=always_hours, decimal_marker=decimal_marker)
    recovered = time_str_to_seconds(ts, decimal_marker=decimal_marker)
    assert abs(recovered - seconds) < 0.001


def test_format_timestamp_hours_omitted_when_zero():
    ts = format_timestamp(61.0, always_include_hours=False, decimal_marker=".")
    assert not ts.startswith("00:")


def test_format_timestamp_hours_included_when_nonzero():
    ts = format_timestamp(3661.0, always_include_hours=False, decimal_marker=",")
    assert ts.startswith("01:")


# ------------------------------------------------------------------
# get_start / get_end
# ------------------------------------------------------------------

def test_get_start_with_words():
    segs = [s.model_dump() for s in WORD_SEGMENTS]
    assert get_start(segs) == 0.0


def test_get_end_with_words():
    segs = [s.model_dump() for s in WORD_SEGMENTS]
    assert get_end(segs) == 5.8


def test_get_start_fallback_no_words():
    segs = [{"start": 1.5, "end": 3.0, "words": []}]
    assert get_start(segs) == 1.5


def test_get_end_fallback_no_words():
    segs = [{"start": 1.5, "end": 3.0, "words": []}]
    assert get_end(segs) == 3.0


# ------------------------------------------------------------------
# ResultWriter.__call__ accepts List[Segment] directly
# ------------------------------------------------------------------

def test_result_writer_accepts_segment_list():
    with tempfile.TemporaryDirectory() as tmpdir:
        writer = WriteSRT(output_dir=tmpdir)
        writer(SAMPLE_SEGMENTS, "seg_list_test")  # pass list, not dict
        parsed = writer.to_segments(os.path.join(tmpdir, "seg_list_test.srt"))
    assert len(parsed) == len(SAMPLE_SEGMENTS)
    assert parsed[0].text.strip() == SAMPLE_SEGMENTS[0].text.strip()


# ------------------------------------------------------------------
# WriteTXT
# ------------------------------------------------------------------

def test_txt_write_and_parse():
    with tempfile.TemporaryDirectory() as tmpdir:
        writer = WriteTXT(output_dir=tmpdir)
        writer(SAMPLE_RESULT, "test")
        out_path = os.path.join(tmpdir, "test.txt")
        assert os.path.exists(out_path)
        parsed = writer.to_segments(out_path)

    # to_segments splits on '\n' and keeps all lines including a trailing empty one
    non_empty = [s for s in parsed if s.text.strip()]
    assert len(non_empty) == len(SAMPLE_SEGMENTS)
    for orig, parsed_seg in zip(SAMPLE_SEGMENTS, non_empty):
        assert orig.text.strip() == parsed_seg.text.strip()
    # TXT segments have no timestamps
    assert non_empty[0].start is None
    assert non_empty[0].end is None


def test_txt_content_is_plain_text():
    with tempfile.TemporaryDirectory() as tmpdir:
        writer = WriteTXT(output_dir=tmpdir)
        writer(SAMPLE_RESULT, "test")
        content = open(os.path.join(tmpdir, "test.txt")).read()

    assert "-->" not in content
    assert "Hello world" in content


# ------------------------------------------------------------------
# WriteSRT
# ------------------------------------------------------------------

def test_srt_write_and_parse():
    with tempfile.TemporaryDirectory() as tmpdir:
        writer = WriteSRT(output_dir=tmpdir)
        writer(SAMPLE_RESULT, "test")
        out_path = os.path.join(tmpdir, "test.srt")
        assert os.path.exists(out_path)
        parsed = writer.to_segments(out_path)

    assert len(parsed) == len(SAMPLE_SEGMENTS)
    for orig, parsed_seg in zip(SAMPLE_SEGMENTS, parsed):
        assert abs(parsed_seg.start - orig.start) < 0.001
        assert abs(parsed_seg.end - orig.end) < 0.001
        assert parsed_seg.text.strip() == orig.text.strip()


def test_srt_blocks_have_sequential_indices():
    with tempfile.TemporaryDirectory() as tmpdir:
        writer = WriteSRT(output_dir=tmpdir)
        writer(SAMPLE_RESULT, "test")
        content = open(os.path.join(tmpdir, "test.srt")).read()

    blocks = [b for b in content.strip().split("\n\n") if b.strip()]
    for i, block in enumerate(blocks, start=1):
        first_line = block.strip().split("\n")[0]
        assert first_line == str(i), f"Block {i} unexpected index: {first_line!r}"


def test_srt_timestamps_monotonic():
    with tempfile.TemporaryDirectory() as tmpdir:
        writer = WriteSRT(output_dir=tmpdir)
        writer(SAMPLE_RESULT, "test")
        parsed = writer.to_segments(os.path.join(tmpdir, "test.srt"))

    for i in range(1, len(parsed)):
        assert parsed[i].start >= parsed[i - 1].start


def test_srt_with_word_timestamps():
    """Exercises the iterate_result word-level path (lines 148-233)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        writer = WriteSRT(output_dir=tmpdir)
        writer(WORD_RESULT, "test")
        parsed = writer.to_segments(os.path.join(tmpdir, "test.srt"))

    assert len(parsed) == len(WORD_SEGMENTS)
    for orig, parsed_seg in zip(WORD_SEGMENTS, parsed):
        assert abs(parsed_seg.start - orig.start) < 0.001
        assert abs(parsed_seg.end - orig.end) < 0.001


def test_srt_with_highlight_words_does_not_crash():
    """highlight_words=True triggers the <u>markup</u> path in iterate_result."""
    with tempfile.TemporaryDirectory() as tmpdir:
        writer = WriteSRT(output_dir=tmpdir)
        writer(WORD_RESULT, "highlighted", highlight_words=True)
        content = open(os.path.join(tmpdir, "highlighted.srt")).read()

    assert "<u>" in content
    assert "-->" in content


def test_srt_segment_with_none_text_is_skipped():
    """Segments with None text must be skipped by iterate_result (line 237)."""
    result = {
        "segments": [
            {"start": 0.0, "end": 1.0, "text": None, "words": None},
            {"start": 2.0, "end": 3.0, "text": "Real text", "words": None},
        ]
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        writer = WriteSRT(output_dir=tmpdir)
        writer(result, "test")
        parsed = writer.to_segments(os.path.join(tmpdir, "test.srt"))

    texts = [s.text.strip() for s in parsed if s.text]
    assert "Real text" in texts
    assert "None" not in texts


# ------------------------------------------------------------------
# WriteVTT
# ------------------------------------------------------------------

def test_vtt_write_and_parse():
    with tempfile.TemporaryDirectory() as tmpdir:
        writer = WriteVTT(output_dir=tmpdir)
        writer(SAMPLE_RESULT, "test")
        out_path = os.path.join(tmpdir, "test.vtt")
        assert os.path.exists(out_path)
        parsed = writer.to_segments(out_path)

    assert len(parsed) == len(SAMPLE_SEGMENTS)
    for orig, parsed_seg in zip(SAMPLE_SEGMENTS, parsed):
        assert abs(parsed_seg.start - orig.start) < 0.001
        assert abs(parsed_seg.end - orig.end) < 0.001
        assert parsed_seg.text.strip() == orig.text.strip()


def test_vtt_starts_with_webvtt_header():
    with tempfile.TemporaryDirectory() as tmpdir:
        writer = WriteVTT(output_dir=tmpdir)
        writer(SAMPLE_RESULT, "test")
        content = open(os.path.join(tmpdir, "test.vtt")).read()

    assert content.startswith("WEBVTT")


def test_vtt_blocks_have_sequential_indices():
    with tempfile.TemporaryDirectory() as tmpdir:
        writer = WriteVTT(output_dir=tmpdir)
        writer(SAMPLE_RESULT, "test")
        content = open(os.path.join(tmpdir, "test.vtt")).read()

    blocks = [b for b in content.split("\n\n") if b.strip() and not b.strip().startswith("WEBVTT")]
    for i, block in enumerate(blocks, start=1):
        first_line = block.strip().split("\n")[0]
        assert first_line == str(i), f"VTT block {i} missing index: {first_line!r}"


def test_vtt_fixture_parses_correctly():
    fixture_path = os.path.join(os.path.dirname(__file__), "test_vtt.vtt")
    writer = WriteVTT(output_dir=".")
    segments = writer.to_segments(fixture_path)
    assert len(segments) == 2
    assert abs(segments[0].start - 0.5) < 0.001
    assert abs(segments[1].end - 4.3) < 0.001


def test_srt_fixture_parses_correctly():
    fixture_path = os.path.join(os.path.dirname(__file__), "test_srt.srt")
    writer = WriteSRT(output_dir=".")
    segments = writer.to_segments(fixture_path)
    assert len(segments) == 2
    assert abs(segments[0].start - 0.0) < 0.001


# ------------------------------------------------------------------
# WriteLRC
# ------------------------------------------------------------------

def test_lrc_write_and_parse():
    with tempfile.TemporaryDirectory() as tmpdir:
        writer = WriteLRC(output_dir=tmpdir)
        writer(SAMPLE_RESULT, "test")
        out_path = os.path.join(tmpdir, "test.lrc")
        assert os.path.exists(out_path)
        parsed = writer.to_segments(out_path)

    assert len(parsed) == len(SAMPLE_SEGMENTS)
    for orig, parsed_seg in zip(SAMPLE_SEGMENTS, parsed):
        assert abs(parsed_seg.start - orig.start) < 0.001
        assert abs(parsed_seg.end - orig.end) < 0.001
        assert parsed_seg.text.strip() == orig.text.strip()


def test_lrc_format_uses_bracket_timestamps():
    with tempfile.TemporaryDirectory() as tmpdir:
        writer = WriteLRC(output_dir=tmpdir)
        writer(SAMPLE_RESULT, "test")
        content = open(os.path.join(tmpdir, "test.lrc")).read()

    assert "[" in content and "]" in content
    assert "-->" not in content


def test_lrc_align_words_path():
    """align_lrc_words=True writes word-embedded timestamps inline (line 334-335)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        writer = WriteLRC(output_dir=tmpdir)
        writer(WORD_RESULT, "aligned", align_lrc_words=True)
        content = open(os.path.join(tmpdir, "aligned.lrc")).read()

    # The aligned format embeds per-word timestamps directly in the text
    assert content.strip(), "aligned LRC output should not be empty"
    assert "[" in content


# ------------------------------------------------------------------
# WriteTSV
# ------------------------------------------------------------------

def test_tsv_write_produces_correct_header():
    with tempfile.TemporaryDirectory() as tmpdir:
        writer = WriteTSV(output_dir=tmpdir)
        writer(SAMPLE_RESULT, "test")
        content = open(os.path.join(tmpdir, "test.tsv")).read()

    first_line = content.split("\n")[0]
    assert first_line == "start\tend\ttext"


def test_tsv_timestamps_are_integer_milliseconds():
    with tempfile.TemporaryDirectory() as tmpdir:
        writer = WriteTSV(output_dir=tmpdir)
        writer(SAMPLE_RESULT, "test")
        lines = open(os.path.join(tmpdir, "test.tsv")).read().strip().split("\n")

    # skip header
    for line in lines[1:]:
        start_ms, end_ms, text = line.split("\t")
        assert start_ms.isdigit(), f"start not integer ms: {start_ms}"
        assert end_ms.isdigit(), f"end not integer ms: {end_ms}"


def test_tsv_row_count_matches_segments():
    with tempfile.TemporaryDirectory() as tmpdir:
        writer = WriteTSV(output_dir=tmpdir)
        writer(SAMPLE_RESULT, "test")
        lines = open(os.path.join(tmpdir, "test.tsv")).read().strip().split("\n")

    assert len(lines) - 1 == len(SAMPLE_SEGMENTS)  # minus header


def test_tsv_values_match_segments():
    with tempfile.TemporaryDirectory() as tmpdir:
        writer = WriteTSV(output_dir=tmpdir)
        writer(SAMPLE_RESULT, "test")
        lines = open(os.path.join(tmpdir, "test.tsv")).read().strip().split("\n")

    for line, orig in zip(lines[1:], SAMPLE_SEGMENTS):
        start_ms, end_ms, text = line.split("\t")
        assert abs(int(start_ms) - round(orig.start * 1000)) <= 1
        assert abs(int(end_ms) - round(orig.end * 1000)) <= 1
        assert text == orig.text.strip()


# ------------------------------------------------------------------
# WriteJSON
# ------------------------------------------------------------------

def test_json_write_produces_valid_json():
    with tempfile.TemporaryDirectory() as tmpdir:
        writer = WriteJSON(output_dir=tmpdir)
        writer(SAMPLE_RESULT, "test")
        with open(os.path.join(tmpdir, "test.json")) as f:
            data = json.load(f)

    assert "segments" in data
    assert len(data["segments"]) == len(SAMPLE_SEGMENTS)


def test_json_content_matches_input():
    with tempfile.TemporaryDirectory() as tmpdir:
        writer = WriteJSON(output_dir=tmpdir)
        writer(SAMPLE_RESULT, "test")
        with open(os.path.join(tmpdir, "test.json")) as f:
            data = json.load(f)

    for orig, seg in zip(SAMPLE_SEGMENTS, data["segments"]):
        assert abs(seg["start"] - orig.start) < 0.001
        assert abs(seg["end"] - orig.end) < 0.001
        assert seg["text"] == orig.text


# ------------------------------------------------------------------
# get_writer
# ------------------------------------------------------------------

@pytest.mark.parametrize("fmt,cls", [
    ("srt", WriteSRT),
    ("vtt", WriteVTT),
    ("txt", WriteTXT),
    ("lrc", WriteLRC),
    ("tsv", WriteTSV),
    ("json", WriteJSON),
])
def test_get_writer_returns_correct_type(fmt, cls):
    with tempfile.TemporaryDirectory() as tmpdir:
        writer = get_writer(fmt, tmpdir)
    assert isinstance(writer, cls)


def test_get_writer_all_returns_callable():
    with tempfile.TemporaryDirectory() as tmpdir:
        writer = get_writer("all", tmpdir)
    assert callable(writer)


def test_get_writer_all_creates_all_formats():
    with tempfile.TemporaryDirectory() as tmpdir:
        write_all = get_writer("all", tmpdir)
        write_all(SAMPLE_RESULT, "multi")
        created = os.listdir(tmpdir)

    extensions = {os.path.splitext(f)[1].lstrip(".") for f in created}
    assert extensions == {"srt", "vtt", "txt", "lrc", "tsv", "json"}


def test_get_writer_format_is_case_insensitive():
    with tempfile.TemporaryDirectory() as tmpdir:
        writer = get_writer("SRT", tmpdir)
    assert isinstance(writer, WriteSRT)


def test_get_writer_strips_leading_dot():
    with tempfile.TemporaryDirectory() as tmpdir:
        writer = get_writer(".srt", tmpdir)
    assert isinstance(writer, WriteSRT)


# ------------------------------------------------------------------
# generate_file
# ------------------------------------------------------------------

def test_generate_file_returns_content_and_path():
    with tempfile.TemporaryDirectory() as tmpdir:
        content, path = generate_file("srt", tmpdir, SAMPLE_RESULT, "out", add_timestamp=False)
        assert isinstance(content, str)
        assert os.path.exists(path)
        assert path.endswith(".srt")


def test_generate_file_webvtt_alias():
    """'WebVTT' format string must resolve to .vtt file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        content, path = generate_file("WebVTT", tmpdir, SAMPLE_RESULT, "out", add_timestamp=False)

    assert path.endswith(".vtt")
    assert content.startswith("WEBVTT")


def test_generate_file_add_timestamp_true():
    with tempfile.TemporaryDirectory() as tmpdir:
        _, path = generate_file("srt", tmpdir, SAMPLE_RESULT, "out", add_timestamp=True)

    basename = os.path.splitext(os.path.basename(path))[0]
    # filename should be "out-<timestamp>" where timestamp is 10 digits
    assert basename.startswith("out-")
    timestamp_part = basename[len("out-"):]
    assert timestamp_part.isdigit() and len(timestamp_part) == 10


def test_generate_file_lrc_highlight_converts_to_align():
    """highlight_words=True on LRC should produce aligned-word format (line 440-441)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        content, path = generate_file(
            "lrc", tmpdir, WORD_RESULT, "out",
            add_timestamp=False, highlight_words=True
        )

    assert path.endswith(".lrc")
    assert content.strip()


def test_generate_file_accepts_segment_list():
    with tempfile.TemporaryDirectory() as tmpdir:
        content, path = generate_file("txt", tmpdir, SAMPLE_SEGMENTS, "out", add_timestamp=False)

    assert "Hello world" in content


# ------------------------------------------------------------------
# safe_filename
# ------------------------------------------------------------------

def test_safe_filename_replaces_invalid_chars():
    result = safe_filename('my<file>:name/test|file?.txt')
    assert "<" not in result
    assert ">" not in result
    assert ":" not in result
    assert "/" not in result
    assert "|" not in result
    assert "?" not in result
    assert "_" in result


def test_safe_filename_preserves_valid_name():
    name = "my_video_2024.mp4"
    assert safe_filename(name) == name


def test_safe_filename_truncates_long_name_with_extension():
    long_name = "a" * 200 + ".mp4"  # 204 chars → must truncate
    result = safe_filename(long_name)
    assert len(result) <= 200
    assert result.endswith(".mp4")


def test_safe_filename_truncates_long_name_no_extension():
    long_name = "a" * 210
    result = safe_filename(long_name)
    assert len(result) <= 200


def test_safe_filename_empty_string():
    assert safe_filename("") == ""


# ------------------------------------------------------------------
# Edge cases
# ------------------------------------------------------------------

def test_single_segment():
    result = {"segments": [Segment(start=0.0, end=1.0, text="Hi").model_dump()]}
    with tempfile.TemporaryDirectory() as tmpdir:
        writer = WriteSRT(output_dir=tmpdir)
        writer(result, "single")
        parsed = writer.to_segments(os.path.join(tmpdir, "single.srt"))
    assert len(parsed) == 1
    assert parsed[0].text.strip() == "Hi"


def test_empty_segments_writes_empty_srt():
    result = {"segments": []}
    with tempfile.TemporaryDirectory() as tmpdir:
        writer = WriteSRT(output_dir=tmpdir)
        writer(result, "empty")
        parsed = writer.to_segments(os.path.join(tmpdir, "empty.srt"))
    assert parsed == []


def test_empty_segments_writes_empty_lrc():
    result = {"segments": []}
    with tempfile.TemporaryDirectory() as tmpdir:
        writer = WriteLRC(output_dir=tmpdir)
        writer(result, "empty")
        parsed = writer.to_segments(os.path.join(tmpdir, "empty.lrc"))
    assert parsed == []
