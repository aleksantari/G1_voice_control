"""Interactive CLI for the language control pipeline.

Usage:
    python demo/pipeline_cli.py text [--zmq]
    python demo/pipeline_cli.py audio path/to/file.wav [--zmq]
    python demo/pipeline_cli.py mic [--zmq]

Add --zmq to publish commands to tcp://*:5556 for endoscope_control.
"""

import sys

from pipeline.pipeline import LanguageControlPipeline


def show(result, publisher=None):
    """Pretty-print a pipeline result dict and optionally publish via ZMQ."""
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

    if publisher and result["valid"]:
        json_str = publisher.publish(cmd)
        print(f"  [ZMQ] Published: {json_str}")
    print()


def run_text(pipe, publisher=None):
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
        show(result, publisher)


def run_audio(pipe, path, publisher=None):
    """Process a single audio file."""
    print(f"Processing: {path}\n")
    result = pipe.process_audio_file(path)
    print(f"  Transcription: \"{result['text']}\"")
    show(result, publisher)


def run_mic(pipe, publisher=None):
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
        show(result, publisher)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = [a for a in sys.argv[1:] if a.startswith("--")]
    use_zmq = "--zmq" in flags

    if not args or args[0] not in ("text", "audio", "mic"):
        print(__doc__)
        sys.exit(1)

    mode = args[0]

    if mode == "audio" and len(args) < 2:
        print("Error: audio mode requires a file path.")
        print("  python demo/pipeline_cli.py audio path/to/file.wav")
        sys.exit(1)

    print("Initializing pipeline...\n")
    pipe = LanguageControlPipeline()

    publisher = None
    if use_zmq:
        from pipeline.zmq_publisher import CommandPublisher
        publisher = CommandPublisher()
        print("[ZMQ] Publishing commands on tcp://*:5556\n")

    print("Pipeline ready.\n")

    try:
        if mode == "text":
            run_text(pipe, publisher)
        elif mode == "audio":
            run_audio(pipe, args[1], publisher)
        elif mode == "mic":
            run_mic(pipe, publisher)
    finally:
        if publisher:
            publisher.close()


if __name__ == "__main__":
    main()
