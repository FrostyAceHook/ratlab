import functools as _functools
import inspect as _inspect
import itertools as _itertools
import math as _math
import os as _os
import re as _re
import sys as _sys
from pathlib import Path as _Path
from types import GeneratorType as _GeneratorType



def incremental(func):
    """
    Allows the function to have its arguments specified incrementally. The
    function will be invoked as soon as it can be (i.e. after all mandatory
    arguments are specified).
    - function decorator.
    """
    sig = _inspect.signature(func)
    @_functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            sig.bind(*args, **kwargs)
        except TypeError:
            @_functools.wraps(func)
            def wrapper_next(*args_next, **kwargs_next):
                return wrapper(*(args + args_next), **(kwargs | kwargs_next))
            return wrapper_next
        return func(*args, **kwargs)
    return wrapper



class _Cached:
    def __init__(self, func, sig):
        self.cache = {}
        self.func = func
        self.sig = sig
        _functools.update_wrapper(self, func)

    def keyof(self, *args, **kwargs):
        bound_args = self.sig.bind(*args, **kwargs)
        bound_args.apply_defaults()
        return tuple(bound_args.arguments.items())

    def __contains__(self, args):
        if not isinstance(args, tuple):
            args = (args, )
        return self.iscached(*args)

    def iscached(self, *args, **kwargs):
        key = self.keyof(*args, **kwargs)
        return key in self.cache

    @property
    def values(self):
        return tuple(self.cache.values())

    def __call__(self, *args, **kwargs):
        key = self.keyof(*args, **kwargs)
        if key not in self.cache:
            self.cache[key] = self.func(*args, **kwargs)
        return self.cache[key]

@incremental
def cached(func, forwards_to=None):
    """
    Caches returns from a function, so any identical-inputs will return the same
    object (and potentially be faster who knows).
    - paramed function decorator.
    """
    if forwards_to is None:
        sig = _inspect.signature(func)
    else:
        sig = _inspect.signature(forwards_to)
    return _Cached(func, sig)



class classconst:
    """
    Makes the given method a class property.
    - method decorator.
    """
    def __init__(self, fget):
        self._fget = fget
        _functools.update_wrapper(self, fget)

    def __get__(self, instance, owner):
        return self._fget(owner)


class instconst:
    """
    Makes the given method an instance property.
    - method decorator.
    """
    def __init__(self, fget):
        self._fget = fget
        _functools.update_wrapper(self, fget)

    def __set_name__(self, owner, name):
        self._cache_name = f"_cached_{name}"

    def __get__(self, instance, owner):
        if instance is None:
            return self
        return self(instance)

    def __call__(self, instance):
        return self._fget(instance)


class instcached:
    """
    Makes the given method a cached instance method.
    - method decorator.
    """
    def __init__(self, fget):
        self._fget = fget
        _functools.update_wrapper(self, fget)

    def __set_name__(self, owner, name):
        self._cache_name = f"_cached_{name}"

    def __get__(self, instance, owner):
        if instance is None:
            return self

        if not hasattr(instance, self._cache_name):
            # Get the signature without the first arg.
            sig = _inspect.signature(self._fget)
            params = list(sig.parameters.values())
            params = params[1:]
            sig = sig.replace(parameters=params)
            # Wrap to auto-include instance.
            @_functools.wraps(self._fget)
            def wrapper(*args, **kwargs):
                return self._fget(instance, *args, **kwargs)
            cache = _Cached(wrapper, sig)
            setattr(instance, self._cache_name, cache)
        return getattr(instance, self._cache_name)



def iterable(obj):
    try:
        obj.__iter__()
        return True
    except Exception:
        return False


def tname(t, namespaced=False):
    if not isinstance(t, type):
        raise TypeError(f"expected type, got {tname(type(t))}")
    name = t.__name__
    if not namespaced:
        return f"'{name}'"
    namespace = t.__module__ + "."
    namespace *= namespace not in {"__main__", "builtins"}
    return f"'{namespace}{name}'"


def coloured(cols, txts):
    """
    When given the colour 'cols' and text 'txts' arrays, prints each element of
    the text as its corresponding colour. Colour codes can be found at:
    https://gist.github.com/fnky/458719343aabd01cfb17a3a4f7296797
    """
    if isinstance(cols, int):
        cols = (cols, )
    else:
        cols = list(cols)
    if isinstance(txts, str):
        txts = (txts, )
    else:
        txts = list(txts)
    if len(cols) != len(txts):
        raise TypeError("must give one colour for each piece of text, got "
                f"{len(cols)} colours and {len(txts)} texts")
    for col in cols:
        if not isinstance(col, int):
            raise TypeError("expected integer colour code, got "
                    f"{_tname(type(col))}")
    for txt in txts:
        if not isinstance(txt, str):
            raise TypeError(f"expected string text, got {_tname(type(txt))}")
    segs = [] # funny
    active = False
    for c, t in zip(cols, txts):
        if not t:
            continue
        if c >= 0:
            segs.append(f"\x1B[38;5;{c}m")
            active = True
        elif active:
            segs.append("\x1B[0m")
            active = False
        segs.append(t)
    return "".join(segs) + "\x1B[0m" * active

# Hack to enable console escape codes.
_os.system("")

def nonctrl(string):
    """
    Returns 'string' with all console control codes removed.
    """
    control_code = _re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
    return control_code.sub("", string)


def entry(name, desc=None, *, width=100, pwidth=20, lead=2):
    """
    Returns a string of the form "<name> .... <desc>", with total width as given
    and points stopping at 'pwidth' at the earliest. If 'desc' is none, just
    returns 'name' wrapped at 'width'. 'lead' spaces will be inserted before each
    line, to pad the string.
    """
    name = " ".join(name.split())
    parts = []
    pad_to = 0
    wrapme = ""
    first = True
    if desc is not None:
        desc = " ".join(desc.split())
        left = " " * lead + name + " .."
        left += "." * (pwidth - len(left)) + " "
        parts.append(left)
        pad_to = len(left)
        wrapme = desc
    else:
        first = False
        pad_to = lead
        wrapme = name

    while wrapme:
        line = wrapme[:width - pad_to]
        if len(line) == width - pad_to and " " in line:
            line = line[:line.rindex(" ")]
        wrapme = wrapme[len(line):].lstrip()
        pad = " " * (0 if first else pad_to)
        parts.append(pad + line + "\n" * (not not wrapme))
        first = False
    return "".join(parts)



def immutable(cls):
    """
    Make the given class immutable (outside the `__init__` method), however
    allows the creation of new underscore attributes. When a class inherits from
    this, it will be mutable except for the members which were assigned during
    the immutable classes `__init__`.
    - class decorator.
    """

    if hasattr(cls, "__slots__"):
        raise ValueError("cannot combine __slots__ and @immutable")

    cls_init = cls.__init__

    def __init__(self, *args, **kwargs):
        if not hasattr(self, "_in_init"):
            self._in_init = 1
            self._frozen = set()
        else:
            self._in_init += 1

        old = set(self.__dict__.keys())
        cls_init(self, *args, **kwargs)
        new = set(self.__dict__.keys())

        self._frozen |= new - old
        self._in_init -= 1

    def check(self, name):
        cooked = False
        if getattr(self, "_in_init", 1):
            cooked = False
        else:
            cooked = name in self._frozen
            if type(self) is cls:
                cooked |= not name.startswith("_")
        if cooked:
            raise AttributeError(f"cannot modify attribute {repr(name)} of an "
                    "immutable instance")

    def __setattr__(self, name, value):
        check(self, name)
        super(cls, self).__setattr__(name, value)

    def __delattr__(self, name):
        check(self, name)
        super(cls, self).__delattr__(name)

    cls.__init__ = __init__
    cls.__setattr__ = __setattr__
    cls.__delattr__ = __delattr__
    return cls



def get_locals(func, *args, **kwargs):
    """
    Returns `func_ret, func_locals`, where locals is a dict of all the locals
    variables on return from `func`.
    """
    top_frame = None
    lcls = None
    prev_trace = _sys.gettrace()
    def catch_locals(frame, event, arg):
        nonlocal top_frame, lcls
        # Remember the top-level frame.
        if top_frame is None:
            top_frame = frame
        # If its the return from the top-level invocation of `func`, catch the
        # locals.
        if (event == "return" and frame.f_code == func.__code__
                              and frame.f_back == top_frame):
            # idk if this should be a deepcopy, but u run into pickling errors if
            # u try, and so far (skul lemoji) the shallow copy hasnt had issues.
            lcls = frame.f_locals.copy()
            return prev_trace # pop back to old trace.
        return catch_locals

    # wrapper to know this is the top-level return.
    def call():
        return func(*args, **kwargs)

    try:
        _sys.settrace(catch_locals)
        ret = call()
    finally:
        # don leak the trace.
        _sys.settrace(prev_trace)

    return ret, lcls


@incremental
def templated(creator, parents=(), decorators=(), metaclass=type):
    """
    Used for templated classes. Essentially transforms a function into a class,
    and the fully-defined type can be created via `func[params]`. Basically the
    whole deal is that when this function is invoked, any locals that are around
    at return are manhandled into a new class with those as attributes. Also,
    since the returns from this function must be the same object you can
    "forward" the created class to another call of the template function.
    WARNING: to use `super()`, you must instead do `super(func[params], self)`,
             with the exact params that this class is using.
    - paramed "function" decorator.
    """

    if not isinstance(parents, tuple):
        parents = (parents, )
    if not isinstance(decorators, tuple):
        decorators = (decorators, )

    # i luv validation.
    assert all(isinstance(p, type) for p in parents)
    assert all(callable(d) for d in decorators)

    sig = _inspect.signature(creator)
    param_names = [p.name for p in sig.parameters.values()]
    param_str = lambda x: x.__name__ if isinstance(x, type) else repr(x)

    @cached(forwards_to=creator)
    def create_class(*params):
        # Get all the attributes of the class (which are the local vars that the
        # function defines).
        ret, lcls = get_locals(creator, *params)
        if ret is not None:
            if not isinstance(ret, type):
                raise ValueError("non-none and non-type return from templated "
                        "function-class")
            if ret not in create_class.values:
                raise ValueError("can only return other instantiations of this "
                        "template from the template")
            return ret

        # Get the template parameters independantly.
        params = [(name, lcls[name]) for name in param_names]

        # Make the class name.
        args = (f"{k}={param_str(v)}" for k,v in params)
        cls_name = f"{creator.__name__}[{', '.join(args)}]"

        # Add a reference to the creator.
        if "template" in lcls:
            raise AttributeError("cannot assign to \"template\"")
        lcls["template"] = Creator

        # Setup the class with the thangs, ensuring the specified parents.
        cls = metaclass(cls_name, parents, lcls)

        # Add the template params to the doc.
        cls.__doc__ = creator.__doc__
        if cls.__doc__ is None:
            cls.__doc__ = "Templated class."
        for name, value in params:
            cls.__doc__ += f"\n    .{name:<10}  {param_str(value)}"

        # Apply any decorators, bottom-up.
        for dec in decorators[::-1]:
            cls = dec(cls)

        return cls

    def __instancecheck__(self, instance): # im literally the best
        return isinstance(instance, create_class.values)
    def __subclasscheck__(self, subclass):
        return any(issubclass(subclass, t) for t in create_class.values)

    def __contains__(self, cls):
        return cls in create_class.values

    def __getitem__(self, params):
        if not isinstance(params, tuple):
            params = (params, )
        return create_class(*params)

    def __call__(self, *args, **kwargs):
        raise NotImplementedError("use square brackets to pass template "
                "params")

    attrs = {
        "params": param_names,
        "_creator": creator,
        "__instancecheck__": __instancecheck__,
        "__subclasscheck__": __subclasscheck__,
        "__contains__": __contains__,
        "__getitem__": __getitem__,
        "__call__": __call__,
    }

    CreatorCreator = type("CreatorCreator", (), attrs)
    Creator = CreatorCreator()
    _functools.update_wrapper(Creator, creator)
    return Creator
    # not confusing at all


def simplest_ratio(x):
    """ Returns the simplest ratio `n, d` s.t. `n / d == x`. """
    if not isinstance(x, float):
        raise TypeError(f"expected float for 'x', got {tname(type(x))}")
    if not _math.isfinite(x):
        raise ValueError(f"expected finite float for 'x', got: {x}")

    # gripped and ripped from fractions module.
    def limit_denom(numer, denom, max_denom):
        if denom <= max_denom:
            return numer, denom
        n, d = numer, denom
        p0, q0, p1, q1 = 0, 1, 1, 0
        while True:
            a = n//d
            q2 = q0 + a*q1
            if q2 > max_denom:
                break
            p0, q0, p1, q1 = p1, q1, p0 + a*p1, q2
            n, d = d, n - a*d
        k = (max_denom - q0)//q1
        if 2*d*(q0 + k*q1) <= denom:
            return p1, q1
        else:
            return p0 + k*p1, q0 + k*q1

    if x == 0.0:
        return 0, 1
    if x < 0.0:
        n, d = simplest_ratio(-x)
        return -n, d
    n, d = x.as_integer_ratio()
    for i in range(0, _math.floor(_math.log10(d)) + 1):
        n0, d0 = limit_denom(n, d, 10 ** i)
        if n0 / d0 == x:
            n = n0
            d = d0
            break
    g = _math.gcd(n, d)
    return n // g, d // g
