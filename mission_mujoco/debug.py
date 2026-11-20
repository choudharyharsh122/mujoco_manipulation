"""Optional educational file tracing; scoped to a run and disabled by default."""
from contextlib import ExitStack
from contextvars import ContextVar
from datetime import datetime, timezone
from functools import wraps
from inspect import signature
from pathlib import Path
from reprlib import Repr

_sink = ContextVar("mission_trace_sink", default=None)
_depth = ContextVar("mission_trace_depth", default=0)
_context = ContextVar("mission_trace_context", default=None)
_formatter = Repr()
_formatter.maxstring = 220
_formatter.maxother = 350
_formatter.maxlist = 8
_formatter.maxtuple = 8
_formatter.maxdict = 8


def _write(event, *, arguments=None, attributes=None, result=None, error=None):
    stream = _sink.get()
    if stream is None:
        return
    name, file, task = _context.get() or ("unknown", "unknown", "Trace event")
    lines = [
        "=" * 72,
        f"Time (UTC): {datetime.now(timezone.utc).isoformat()}",
        f"Event: {event}",
        f"Call depth: {_depth.get()}",
        f"In function: {name}",
        f"In file: {file}",
        f"Task / what this function is doing: {task}",
    ]
    for label, values in (("Passed arguments (summarized)", arguments),
                          ("Attributes / current values (summarized)", attributes)):
        if values is not None:
            lines.append(label + ":")
            if not values:
                lines.append("  (none)")
            for key, value in values.items():
                lines.append(f"  {key} [{type(value).__name__}]: {_formatter.repr(value)}")
    if event == "RETURN":
        lines.append("Returned value (summarized): " + _formatter.repr(result))
    if error is not None:
        lines.append(f"Error (propagated to caller): {type(error).__name__}: {error}")
    stream.write("\n".join(lines) + "\n\n")
    stream.flush()


def log(message, **values):
    """Write an in-function event to the active run's file, never stdout."""
    if _sink.get() is not None:
        _write(message, attributes=values)


def traced(description, *, run_scope=False):
    """Record entry, return and errors in <out>/debug_trace.log when verbose=True."""
    def decorate(function):
        sig = signature(function)

        @wraps(function)
        def wrapper(*args, **kwargs):
            with ExitStack() as stack:
                if run_scope:
                    bound = sig.bind(*args, **kwargs)
                    bound.apply_defaults()
                    stream = None
                    if bound.arguments.get("verbose", False):
                        directory = Path(bound.arguments.get("out", "results/run"))
                        directory.mkdir(parents=True, exist_ok=True)
                        stream = stack.enter_context(
                            (directory / "debug_trace.log").open("w", encoding="utf-8")
                        )
                        stream.write("Mission simulation debug trace\nValues are summarized; arrays and collections may be abbreviated.\n\n")
                    token = _sink.set(stream)
                    stack.callback(_sink.reset, token)
                    token = _depth.set(-1)
                    stack.callback(_depth.reset, token)
                if _sink.get() is None:
                    return function(*args, **kwargs)
                inputs = sig.bind(*args, **kwargs)
                inputs.apply_defaults()
                values = {key: value for key, value in inputs.arguments.items() if key != "self"}
                token = _context.set((function.__qualname__, Path(function.__code__.co_filename).name, description))
                stack.callback(_context.reset, token)
                # One depth per traced invocation, including entry, body and exit.
                token = _depth.set(_depth.get() + 1)
                stack.callback(_depth.reset, token)
                _write("ENTER", arguments=values)
                try:
                    result = function(*args, **kwargs)
                except Exception as exc:
                    _write("ERROR", error=exc)
                    raise
                _write("RETURN", result=result)
                return result
        return wrapper
    return decorate
