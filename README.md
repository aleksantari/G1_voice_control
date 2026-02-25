# G1 Voice Control

Language-based control pipeline for the Unitree G1 humanoid robot, designed for voice-controlled endoscope assistance in surgical settings.

Speak a command into a microphone (or type it), and the system produces a validated, structured robot command with action, magnitude, and confidence — ready to send to the robot via DDS.

## Architecture

```
Microphone / Audio File / Text Input
          |
          v
  ┌──────────────┐
  │  Whisper STT  │   Local speech-to-text (skipped for text input)
  │  (base.en)    │
  └──────┬───────┘
         v
  ┌──────────────┐     ┌────────────────┐
  │  LLM Parser  │────>│ Regex Fallback │   If LLM fails or low confidence
  │ (GPT-4o-mini)│     │  (local, <1ms) │
  └──────┬───────┘     └───────┬────────┘
         v                     v
  ┌──────────────────────────────┐
  │   CommandValidator           │   Confidence threshold check
  │   (STOP always passes)      │
  └──────────────┬───────────────┘
                 v
  ┌──────────────────────────────┐
  │   RobotCommand (Pydantic)    │   Validated JSON: action, magnitude,
  │   → DDS topics to robot      │   value_mm, confidence, frame
  └──────────────────────────────┘
```

## Project Structure

```
G1_voice_control/
├── config/
│   ├── config.yaml          # Model settings, thresholds (committed)
│   └── settings.py          # Config loader (YAML + .env secrets)
├── schema/
│   ├── command_schema.py    # Action/Magnitude enums, RobotCommand model
│   └── test_schema.py       # 36 tests
├── parser/
│   ├── llm_parser.py        # GPT-4o-mini command parser
│   ├── prompt_templates.py  # System prompt and user template
│   ├── regex_fallback.py    # Regex-based fallback parser
│   ├── test_parser.py       # 4 tests (mocked)
│   └── test_regex_fallback.py  # 14 tests
├── stt/
│   ├── stt_whisper.py       # Local Whisper transcription
│   ├── audio_recorder.py    # Mic recording via PulseAudio (parec)
│   └── test_stt.py          # 5 tests (mocked)
├── pipeline/
│   ├── pipeline.py          # Main pipeline (wires everything together)
│   ├── fallback.py          # FallbackManager + CommandValidator
│   └── test_pipeline.py     # 5 tests (mocked)
├── demo/
│   ├── pipeline_cli.py      # Interactive CLI (text / audio / mic modes)
│   └── text_demo.py         # Batch LLM test script (16 test cases)
├── pyproject.toml
├── requirements.txt
└── .env.example             # API key template
```

## Supported Commands

### Actions

| Action | Example Phrases |
|--------|----------------|
| MOVE_UP | "move up", "raise", "higher" |
| MOVE_DOWN | "move down", "lower" |
| MOVE_LEFT | "go left", "nudge left" |
| MOVE_RIGHT | "go right", "nudge right" |
| MOVE_FORWARD | "advance", "push in", "deeper" |
| RETRACT | "retract", "pull back", "withdraw" |
| ROTATE_LEFT | "rotate left", "twist left", "counter-clockwise" |
| ROTATE_RIGHT | "rotate right", "twist right", "clockwise" |
| STOP | "stop", "freeze", "hold", "don't move" |

### Magnitudes

| Magnitude | Value | Trigger Words |
|-----------|-------|---------------|
| SMALL | 2.0 mm | "a little", "slightly", "nudge", "tiny" |
| MID | 4.0 mm | (default — no trigger word needed) |
| BIG | 6.0 mm | "a lot", "significantly", "way", "far" |

## Setup

### 1. Clone and create conda environment

```bash
git clone <repo-url>
cd G1_voice_control

conda create -n voice_control python=3.11 -y
conda activate voice_control
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
pip install -e .
```

### 3. System dependencies

```bash
# FFmpeg (required by Whisper for audio decoding)
sudo apt install ffmpeg

# PulseAudio tools (required for microphone recording via PipeWire)
sudo apt install pulseaudio-utils
```

### 4. OpenAI API key

```bash
cp .env.example .env
# Edit .env and add your OpenAI API key:
# OPENAI_API_KEY=sk-...
```

### 5. Verify installation

```bash
python -m pytest schema/ parser/ stt/ pipeline/ -v
```

All 64 tests should pass.

## Usage

The pipeline has three modes, all accessed through the CLI:

### Text mode — type commands directly

```bash
python demo/pipeline_cli.py text
```

```
Text mode — type a command, or 'quit' to exit.

> move up a little
  Action:     MOVE_UP
  Magnitude:  SMALL (2.0mm)
  Confidence: 0.95
  Source:     llm
  Valid:      True — ok
  Latency:    Parse=312.4ms

> quit
```

### Audio file mode — process a pre-recorded file

```bash
python demo/pipeline_cli.py audio path/to/recording.wav
```

Supports WAV, MP3, MP4, M4A, FLAC, and any format FFmpeg can decode.

### Microphone mode — live push-to-talk

```bash
python demo/pipeline_cli.py mic
```

Press ENTER to start recording, speak your command, press ENTER to stop.

## Running Tests

```bash
# All tests (64 total)
python -m pytest schema/ parser/ stt/ pipeline/ -v

# Individual modules
python -m pytest schema/test_schema.py -v          # 36 schema tests
python -m pytest parser/test_parser.py -v           # 4 LLM parser tests
python -m pytest parser/test_regex_fallback.py -v   # 14 regex tests
python -m pytest stt/test_stt.py -v                 # 5 STT tests
python -m pytest pipeline/test_pipeline.py -v       # 5 pipeline tests
```

All tests are mocked (no API calls, no model downloads, no mic access required).

## Configuration

Settings are in `config/config.yaml`:

```yaml
whisper:
  model_size: "base.en"       # Whisper model variant (tiny.en, base.en, small.en, etc.)
  language: "en"
  device: "cpu"               # "cuda" for GPU acceleration

llm:
  provider: "openai"
  model: "gpt-4o-mini"        # LLM for command parsing
  max_tokens: 256
  temperature: 0.0            # Deterministic output

pipeline:
  confidence_threshold: 0.7   # Minimum confidence to accept a command
  enable_fallback: true       # Use regex when LLM fails
  enable_confirmation: false
```

Secrets (API keys) go in `.env` (gitignored), not in config.yaml.

## How It Works

### Two-tier parsing

1. **LLM parser** (primary) — sends the transcribed text to GPT-4o-mini with a surgical command system prompt. Returns structured JSON with action, magnitude, and confidence. Handles novel phrasing and surgical synonyms.

2. **Regex fallback** (backup) — if the LLM is unavailable, times out, or returns low confidence (< 0.5), a local regex parser pattern-matches against known keywords. Returns confidence = 0.6. If no pattern matches, returns a safe STOP with confidence = 0.0.

### Safety design

- **STOP is always prioritized** — the regex parser checks for stop words before any other pattern, and the validator always passes STOP commands regardless of confidence.
- **The pipeline never throws exceptions** — every failure mode produces a safe STOP command.
- **Two confidence thresholds** — 0.5 triggers the fallback (LLM was unsure), 0.7 is the validation threshold (command is trustworthy enough to execute).
