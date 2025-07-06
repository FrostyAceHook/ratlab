import threading as _threading
from importlib import import_module as _import_module

from util import objtname as _objtname

# Expensive initialisation is done in the background :).

class _Failed:
    def __init__(self, msg):
        self._msg = msg
        self._name = None
        self._start_tossing = True
    def _txt(self):
        name = ""
        if self._name is not None:
            name = f" {repr(self._name)}"
        return f"never loaded{name}, because: {self._msg}"
    def _throw(self):
        raise RuntimeError(self._txt())
    def __repr__(self):
        return f"<{self._txt()}>"
    def __getattr__(self, name):
        if not isinstance(name, str):
            raise TypeError(f"expected string, got {_objtname(name)}")
        toss = object.__getattribute__(self, "_start_tossing")
        if toss:
            self._throw()
        raise AttributeError(f"{repr(type(self).__name__)} object has no "
                f"attribute {repr(name)}")


class _Task:
    def __init__(self, name, func, inject=True):
        if not isinstance(name, str):
            raise TypeError(f"expected string for name, got {_objtname(name)}")
        if not callable(func):
            raise TypeError(f"expected callable for func, got {_objtname(func)}")
        self.name = name
        self.func = func
        self.inject = not not inject

    def __call__(self):
        ret = self.func()
        if isinstance(ret, _Failed):
            ret._name = self.name
        return ret

class _Loader:
    def __init__(self, *tasks):
        for task in tasks:
            if not isinstance(task, _Task):
                raise TypeError("expected Task for each task, got "
                        f"{_objtname(task)}")
        if len({task.name for task in tasks}) != len(tasks):
            raise ValueError("got duplicate task names")
        self._tasks = tasks
        self._loaded = {task.name: _threading.Event() for task in tasks
                        if task.inject}
        self._injects = {}
        self._started = False
        self._finished_injects = _threading.Event()

    @property
    def injects(self):
        # Start in-case.
        self.start()
        if not self._finished_injects.is_set():
            self._finished_injects.wait()
        return self._injects

    def start(self):
        if self._started:
            return
        self._started = True
        def target():
            last_inject = -1
            for i, task in enumerate(self._tasks):
                if task.inject:
                    last_inject = i
            if last_inject == -1:
                self._finished_injects.set()
            for i, task in enumerate(self._tasks):
                obj = task()
                if task.inject:
                    self._injects[task.name] = obj
                    self._loaded[task.name].set()
                if i == last_inject:
                    self._finished_injects.set()
        _threading.Thread(target=target, daemon=True).start()

    def __getattr__(self, name):
        loaded = object.__getattribute__(self, "_loaded")
        injects = object.__getattribute__(self, "_injects")
        start = object.__getattribute__(self, "start")
        if not isinstance(name, str):
            raise TypeError(f"expected string, got {_objtname(name)}")
        if name not in loaded:
            raise AttributeError(f"name {repr(name)} is not recognised")
        # If it hasn't been started yet, just start it now instead of throwing
        # or something.
        start()
        event = loaded[name]
        if not event.is_set():
            event.wait()
        obj = injects[name]
        if isinstance(obj, _Failed):
            obj._throw()
        return obj


def _import(path):
    def wrapped():
        try:
            return _import_module(path)
        except ImportError as e:
            e = str(e)
            if e[:1].isupper():
                e = e[:1].lower() + e[1:]
            return _Failed(f"failed to import module {repr(path)} ({e})")
    return wrapped

bg = _Loader(
    _Task("np", _import("numpy")),
    _Task("sympy", _import("sympy"), inject=False),
    _Task("pandas", _import("pandas"), inject=False),
    _Task("scipy", _import("scipy"), inject=False),
)
bg.__doc__ = """
Holds the backgrounds tasks and the objects they return. The object are set as
attributes, and accessing them will block until they are loaded.
"""
