"""Interactive CLI for the language control pipeline.

Usage:
    conda run -n voice_control python demo/pipeline_cli.py text
    conda run -n voice_control python demo/pipeline_cli.py audio path/to/file.wav
    conda run -n voice_control python demo/pipeline_cli.py mic
"""

import sys

from pipeline.pipeline import LanguageControlPipeline


def show(result):
    """Pretty-print a pipeline result dict."""
    cmd = result["command"]
    mag = cmd.magnitude.value if cmd.magnitude else "None"
    val = f"{cmd.value_mm}mm" if cmd.value_mm else "None"

    print(f"  Action:     {cmd.action.value}")
    print(f"  Magnitude:  {mag} ({val})")
    print(f"  Confidence: {cmd.confidence:.2f}")
    print(f"  Source:     {result['source']}")
    print(f"  Valid:      {result['valid']} — {result['message']}")

    stt = result["latency_stt_ms"]
    parse = result["latency_parse_ms"]
    if stt > 0:
        print(f"  Latency:    STT={stt:.1f}ms  Parse={parse:.1f}ms")
    else:
        print(f"  Latency:    Parse={parse:.1f}ms")
    print()


def run_text(pipe):
    """Interactive text input loop."""
    print("Text mode — type a command, or 'quit' to exit.\n")
    while True:
        try:
            text = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text or text.lower() == "quit":
            break
        result = pipe.process_text(text)
        show(result)


def run_audio(pipe, path):
    """Process a single audio file."""
    print(f"Processing: {path}\n")
    result = pipe.process_audio_file(path)
    print(f"  Transcription: \"{result['text']}\"")
    show(result)


def run_mic(pipe):
    """Push-to-talk microphone loop."""
    print("Mic mode — press ENTER to record, ENTER to stop. Type 'quit' to exit.\n")
    while True:
        try:
            check = input("Ready? (ENTER to record, 'quit' to exit): ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if check.lower() == "quit":
            break
        result = pipe.process_microphone()
        print(f"  Transcription: \"{result['text']}\"")
        show(result)


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("text", "audio", "mic"):
        print(__doc__)
        sys.exit(1)

    mode = sys.argv[1]

    if mode == "audio" and len(sys.argv) < 3:
        print("Error: audio mode requires a file path.")
        print("  python demo/pipeline_cli.py audio path/to/file.wav")
        sys.exit(1)

    print("Initializing pipeline...\n")
    pipe = LanguageControlPipeline()
    print("Pipeline ready.\n")

    if mode == "text":
        run_text(pipe)
    elif mode == "audio":
        run_audio(pipe, sys.argv[2])
    elif mode == "mic":
        run_mic(pipe)


if __name__ == "__main__":
    main()
