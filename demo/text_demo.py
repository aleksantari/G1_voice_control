"""Live API test script for the LLM command parser.

Runs real API calls against GPT-4o-mini with test prompts, logs each
response, and flags any that don't match expected results.

Usage:
    conda run -n voice_control python demo/text_demo.py
"""

import time

from parser.llm_parser import LLMCommandParser

# Each test case: input text, expected action, expected magnitude
# For non-commands: expected_action=None means we only check confidence < 0.5
TEST_CASES = [
    # Basic directions
    {"text": "move up", "expected_action": "MOVE_UP", "expected_magnitude": "MID"},
    {"text": "move up a little", "expected_action": "MOVE_UP", "expected_magnitude": "SMALL"},
    {"text": "go up a lot", "expected_action": "MOVE_UP", "expected_magnitude": "BIG"},
    {"text": "move down", "expected_action": "MOVE_DOWN", "expected_magnitude": "MID"},
    {"text": "go left", "expected_action": "MOVE_LEFT", "expected_magnitude": "MID"},
    {"text": "nudge right", "expected_action": "MOVE_RIGHT", "expected_magnitude": "SMALL"},
    # Surgical synonyms
    {"text": "advance significantly", "expected_action": "MOVE_FORWARD", "expected_magnitude": "BIG"},
    {"text": "push in a little", "expected_action": "MOVE_FORWARD", "expected_magnitude": "SMALL"},
    {"text": "retract", "expected_action": "RETRACT", "expected_magnitude": "MID"},
    {"text": "pull back a lot", "expected_action": "RETRACT", "expected_magnitude": "BIG"},
    # Rotation
    {"text": "rotate left", "expected_action": "ROTATE_LEFT", "expected_magnitude": "MID"},
    {"text": "twist right slightly", "expected_action": "ROTATE_RIGHT", "expected_magnitude": "SMALL"},
    # Stop variants
    {"text": "stop", "expected_action": "STOP", "expected_magnitude": None},
    {"text": "freeze", "expected_action": "STOP", "expected_magnitude": None},
    {"text": "hold position", "expected_action": "STOP", "expected_magnitude": None},
    # Non-command (should get low confidence)
    {"text": "the weather is nice today", "expected_action": None, "expected_magnitude": None},
]


def run_tests():
    parser = LLMCommandParser()
    passed = 0
    failed = 0
    results = []

    print(f"Running {len(TEST_CASES)} live API tests...\n")

    for case in TEST_CASES:
        text = case["text"]
        start = time.perf_counter()
        cmd = parser.parse(text)
        elapsed_ms = (time.perf_counter() - start) * 1000

        action_str = cmd.action.value
        mag_str = cmd.magnitude.value if cmd.magnitude else "None"
        val_str = f"{cmd.value_mm}mm" if cmd.value_mm else "None"

        # Check pass/fail
        if case["expected_action"] is None:
            # Non-command: only check confidence < 0.5
            ok = cmd.confidence < 0.5
            if ok:
                label = "[PASS]"
                passed += 1
            else:
                label = "[FAIL]"
                failed += 1
            print(
                f"  {label}  \"{text}\"\n"
                f"         → {action_str} / {mag_str} / {val_str}  "
                f"(conf={cmd.confidence:.2f}, {elapsed_ms:.0f}ms)\n"
                f"         expected: confidence < 0.5"
            )
        else:
            action_ok = action_str == case["expected_action"]
            mag_ok = mag_str == (case["expected_magnitude"] or "None")
            ok = action_ok and mag_ok

            if ok:
                label = "[PASS]"
                passed += 1
                print(
                    f"  {label}  \"{text}\"\n"
                    f"         → {action_str} / {mag_str} / {val_str}  "
                    f"(conf={cmd.confidence:.2f}, {elapsed_ms:.0f}ms)"
                )
            else:
                label = "[FAIL]"
                failed += 1
                print(
                    f"  {label}  \"{text}\"\n"
                    f"         → got {action_str} / {mag_str}, "
                    f"expected {case['expected_action']} / {case['expected_magnitude']}  "
                    f"(conf={cmd.confidence:.2f}, {elapsed_ms:.0f}ms)"
                )

        print()

    # Summary
    total = passed + failed
    print("=" * 50)
    print(f"Results: {passed}/{total} passed, {failed} failed")
    if failed == 0:
        print("All tests passed!")
    print("=" * 50)


if __name__ == "__main__":
    run_tests()
