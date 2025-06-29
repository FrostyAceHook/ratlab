import threading as _threading
from importlib import import_module as _import_module

# Expensive initialisation is done in the background :).

class _Tasks:
    def __init__(self, tasks):
        self._tasks = tasks
        self._objs = {}
        self._loaded = {name: _threading.Event() for name in tasks
                        if name is not None}
        self._started = False

    def start(self):
        if self._started:
            return
        self._started = True
        def target():
            for i, (name, task) in enumerate(self._tasks.items()):
                obj = task()
                if name is not None:
                    self._objs[name] = obj
                    self._loaded[name].set()
        _threading.Thread(target=target, daemon=True).start()

    def __getattr__(self, name):
        objs = object.__getattribute__(self, "_objs")
        loaded = object.__getattribute__(self, "_loaded")
        started = object.__getattribute__(self, "_started")
        start = object.__getattribute__(self, "start")
        if not isinstance(name, str):
            raise TypeError(f"expected string, got '{type(name).__name__}'")
        if name not in loaded:
            raise AttributeError(f"name {repr(name)} is not recognised")
        # If it hasn't been started yet, just start it now instead of throwing or
        # something.
        if not started:
            start()
        event = loaded[name]
        if not event.is_set():
            event.wait()
        obj = objs[name]
        if isinstance(obj, _Failed):
            raise RuntimeError(f"never loaded {repr(name)}, because: {obj.msg}")
        return obj

class _Failed:
    def __init__(self, msg):
        self.msg = msg

def _task_import(path):
    def wrapped():
        try:
            return _import_module(path)
        except ImportError as e:
            e = str(e)
            if e[:1].isupper():
                e = e[:1].lower() + e[1:]
            return _Failed(f"failed to import module {repr(path)} ({e})")
    return wrapped

def _task_matplotlib():
    plt = _import_module("matplotlib.pyplot")
    plt.ion()
    return plt


bg = _Tasks({
    "np": _task_import("numpy"),
    "scipy": _task_import("scipy"),
    "pd": _task_import("pandas"),
    "plt": _task_matplotlib,
})
bg.__doc__ = """
Holds the backgrounds tasks and the objects they return. The object are set as
attributes, and accessing them will block until they are loaded.
"""
