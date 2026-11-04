import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import json, pytest
from agent import *


def test_calculator_math_and_safety():
    assert calculator("10 * log10(100)") == 20
    with pytest.raises(ValueError):
        calculator("__import__('os').system('echo hi')")


def test_repair_handles_prose_and_trailing_comma():
    assert parse_action('Sure!\n```json\n{"final": 3,}\n```')["final"] == 3
    with pytest.raises(ValueError):
        parse_action('Sure {"final": 3}', repair=False)


def test_convert_both_directions():
    assert convert(2, "GHz", "Hz") == 2e9
    assert convert(2e9, "Hz", "GHz") == 2


class Stuck:
    def complete(self, log):
        return json.dumps({"tool": "lookup", "args": {"name": "nope"}})


def test_step_limit_stops_a_stuck_model():
    r = run_agent("x", Stuck(), max_steps=5)
    assert r.stopped == "step limit" and r.answer is None
