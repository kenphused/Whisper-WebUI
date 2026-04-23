# Refactor: Wire backend transcription router through WhisperFactory

## Problem

The backend transcription router (`backend/routers/transcription/router.py`) hardcodes
`FasterWhisperInference` directly. This means the backend API always uses faster-whisper
regardless of what is configured, while the Gradio UI path correctly routes through
`WhisperFactory` and respects the configured implementation.

Concrete consequences:
- `openai-whisper` and `insanely-fast-whisper` are available as implementations in
  `WhisperFactory` but unreachable via the backend API.
- There is no `whisper_type` field in `backend/configs/config.yaml`, so the backend
  implementation cannot be changed without touching code.
- The Gradio UI and the backend API behave differently for the same config — a user
  expecting parity between the two entry points will get faster-whisper from the API
  even if they configured something else.

## Root cause

`get_pipeline()` (line 44) constructs `FasterWhisperInference` directly instead of
delegating to `WhisperFactory.create_whisper_inference()`. The cache type annotation
on line 27 also locks the type to `FasterWhisperInference`.

## Steps to fix

### 1. Add `whisper_type` to `backend/configs/config.yaml`

```yaml
whisper:
  whisper_type: faster-whisper   # add this line
  model_size: large-v2
  compute_type: float16
  enable_offload: true
```

Valid values: `faster-whisper`, `whisper`, `insanely_fast_whisper` (matches `WhisperImpl` enum in `modules/whisper/data_classes.py`).

### 2. Update `backend/routers/transcription/router.py`

**Replace the direct import:**
```python
# remove
from modules.whisper.faster_whisper_inference import FasterWhisperInference

# add
from modules.whisper.whisper_factory import WhisperFactory
from modules.whisper.base_transcription_pipeline import BaseTranscriptionPipeline
```

**Widen the cache type (line 27):**
```python
# before
_pipeline_cache: Dict[Tuple[str, str], "FasterWhisperInference"] = {}

# after — key gains whisper_type so different implementations don't collide
_pipeline_cache: Dict[Tuple[str, str, str], BaseTranscriptionPipeline] = {}
```

**Rewrite `get_pipeline()` (lines 44–51):**
```python
def get_pipeline() -> BaseTranscriptionPipeline:
    config = load_server_config()["whisper"]
    whisper_type = config.get("whisper_type", "faster-whisper")
    key = (whisper_type, config["model_size"], config["compute_type"])
    if key not in _pipeline_cache:
        inferencer = WhisperFactory.create_whisper_inference(
            whisper_type=whisper_type,
            output_dir=BACKEND_CACHE_DIR,
        )
        inferencer.update_model(
            model_size=config["model_size"],
            compute_type=config["compute_type"],
        )
        _pipeline_cache[key] = inferencer
    return _pipeline_cache[key]
```

All three implementations (`FasterWhisperInference`, `WhisperInference`,
`InsanelyFastWhisperInference`) share the same `update_model(model_size, compute_type)`
signature defined on `BaseTranscriptionPipeline`, so no further changes are needed.

## Verification

- Set `whisper_type: whisper` in `backend/configs/config.yaml` and confirm the backend
  API uses `WhisperInference` (check model load logs).
- Existing backend tests in `backend/tests/` should pass without modification since
  they exercise the endpoint, not the pipeline construction directly.
- The `TEST_ENV` override in `config_loader.py` only patches `model_size` and
  `compute_type` — add `whisper_type` patching there if tests need to cover
  non-default implementations.
