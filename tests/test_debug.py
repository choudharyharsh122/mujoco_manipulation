"""File tracing is opt-in and must preserve return values and exceptions."""
import pytest
from mission_mujoco.debug import log, traced


@traced("Inner computation")
def inner(value, fail=False):
    log("Inspect current value", value=value)
    if fail:
        raise ValueError("example failure")
    return value + 1


@traced("Example run", run_scope=True)
def example_run(out, verbose=False, fail=False):
    return inner(2, fail=fail)


def test_verbose_writes_labeled_file_without_console_output(tmp_path, capsys):
    assert example_run(tmp_path, verbose=True) == 3
    text = (tmp_path / "debug_trace.log").read_text()
    for label in ("In function: inner", "In file: test_debug.py",
                  "Task / what this function is doing: Inner computation",
                  "Passed arguments (summarized):", "value [int]: 2",
                  "Attributes / current values (summarized):",
                  "Event: RETURN", "Returned value (summarized): 3"):
        assert label in text
    assert capsys.readouterr().out == ""
    # The context and file handle do not leak out of the completed run.
    previous = text
    inner(9)
    assert (tmp_path / "debug_trace.log").read_text() == previous


def test_disabled_does_not_create_trace(tmp_path, capsys):
    assert example_run(tmp_path) == 3
    assert not (tmp_path / "debug_trace.log").exists()
    assert capsys.readouterr().out == ""


def test_failure_is_recorded_and_propagated(tmp_path, capsys):
    with pytest.raises(ValueError, match="example failure"):
        example_run(tmp_path, verbose=True, fail=True)
    text = (tmp_path / "debug_trace.log").read_text()
    assert "Event: ERROR" in text
    assert "ValueError: example failure" in text
    assert capsys.readouterr().out == ""
    assert inner(2) == 3
    assert (tmp_path / "debug_trace.log").read_text() == text


@pytest.mark.parametrize("fail", [False, True])
def test_depth_consistent_for_entry_body_and_exit(tmp_path, fail):
    if fail:
        with pytest.raises(ValueError):
            example_run(tmp_path, verbose=True, fail=True)
    else:
        example_run(tmp_path, verbose=True)
    entries = []
    for block in (tmp_path / "debug_trace.log").read_text().split("=" * 72):
        fields = dict(line.split(": ", 1) for line in block.splitlines() if ": " in line)
        if "Event" in fields:
            entries.append((fields["In function"], fields["Event"], int(fields["Call depth"])))
    exit_event = "ERROR" if fail else "RETURN"
    assert entries == [
        ("example_run", "ENTER", 0),
        ("inner", "ENTER", 1),
        ("inner", "Inspect current value", 1),
        ("inner", exit_event, 1),
        ("example_run", exit_event, 0),
    ]
