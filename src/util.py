import functools as _functools
import inspect as _inspect
import math as _math
import os as _os
import re as _re
import sys as _sys
import weakref as _weakref



def incremental(func):
    """
    Allows the function to have its arguments specified incrementally. The
    function will be invoked as soon as it can be (i.e. after all mandatory
    arguments are specified). Oh this is called currying.
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



def maybe_pack(x, aslist=False):
    """
    When a function takes an argument which is a tuple/list of items, it's
    helpful to also support being given a single item which then gets packed into
    a 1-length tuple/list, which this facilitates. Always returns a tuple (or
    a list if 'aslist', copying if already a list).
    """
    if isinstance(x, (tuple, list)):
        x = [tuple, list][not not aslist](x)
    else:
        x = [tuple, list][not not aslist]([x])
    return x

def maybe_unpack(xs, dont_unpack=(), also_dont_unpack=(str, bytes, dict)):
    """
    When a function takes vargs, it's helpful to also support being given a
    single iterable, which this facilitates. Always returns a tuple. Note that
    not all iterables should be unpacked, so `dont_unpack` facilitates not
    unpacking some types.
        def func(*args):
            args = maybe_unpack(args)
            return args
        assert func(1, 2, 3) == func([1, 2, 3])
    WARNING: care must be taken when forwarding to a function which also uses
             `maybe_unpack` (to ensure args aren't double-unpacked). In general,
             follow one of these two strategies:
        def forwards(*args):
            # don't call `maybe_unpack`.
            return func(*args) # unpack.
        def also_forwards(*args):
            args = maybe_unpack(args) # call `maybe_unpack`.
            return func(args) # dont unpack.
    """
    if len(xs) != 1:
        return xs
    x = xs[0]
    # dont unpack all iterables.
    dont = maybe_pack(dont_unpack) + maybe_pack(also_dont_unpack)
    if isinstance(x, dont):
        return xs # dont.
    if not iterable(x):
        return xs
    return tuple(x)



class _Cached:
    def __init__(self, func, sig):
        self._cache = {}
        self.func = func
        self.sig = sig
        _functools.update_wrapper(self, func)
    def _get(self, key):
        return self._cache[key]
    def _has(self, key):
        return key in self._cache
    def _set(self, key, value):
        self._cache[key] = value

    def _keyof(self, *args, **kwargs):
        bound_args = self.sig.bind(*args, **kwargs)
        bound_args.apply_defaults()
        return tuple(bound_args.arguments.items())
    def _fromkey(self, key):
        POSITIONAL_ONLY = _inspect.Parameter.POSITIONAL_ONLY
        POSITIONAL_OR_KEYWORD = _inspect.Parameter.POSITIONAL_OR_KEYWORD
        VAR_POSITIONAL = _inspect.Parameter.VAR_POSITIONAL
        KEYWORD_ONLY = _inspect.Parameter.KEYWORD_ONLY
        VAR_KEYWORD = _inspect.Parameter.VAR_KEYWORD
        args = []
        kwargs = {}
        items = dict(key)
        for name, param in self.sig.parameters.items():
            if param.kind in {POSITIONAL_ONLY, POSITIONAL_OR_KEYWORD}:
                if name in items:
                    args.append(items[name])
            elif param.kind == KEYWORD_ONLY:
                if name in items:
                    kwargs[name] = items[name]
            elif param.kind == VAR_POSITIONAL:
                if name in items:
                    args.extend(items[name])
            elif param.kind == VAR_KEYWORD:
                if name in items:
                    kwargs.update(items[name])
            else:
                assert False
        return tuple(args), kwargs

    def iscached(self, *args, **kwargs):
        key = self._keyof(*args, **kwargs)
        return key in self._cache

    @property
    def allcached(self):
        return [self._fromkey(key) for key in self._cache.keys()]

    def __call__(self, *args, **kwargs):
        key = self._keyof(*args, **kwargs)
        if key not in self._cache:
            value = self.func(*args, **kwargs)
            self._set(key, value)
            return value
        return self._get(key)

class _WeakCached(_Cached):
    # Need an object wrapper to ensure all types can be weakrefed.
    class _Wrapped:
        def __init__(self, value):
            self.value = value
        def __repr__(self):
            return f"_Wrapped({self.value})"

    def __init__(self, func, sig):
        super().__init__(func, sig)
        self._cache = _weakref.WeakValueDictionary()

    def _get(self, key):
        # Unwrap if we wrapped it.
        value = self._cache[key]
        if isinstance(value, self._Wrapped):
            return value.value
        return value
    def _set(self, key, value):
        # Wrap non-weakrefable values.
        if not hasattr(value, "__weakref__"):
            value = self._Wrapped(value)
        self._cache[key] = value

@incremental
def cached(func, forwards_to=None):
    """
    Caches returns from a function, so any identical-inputs will return the same
    object (and potentially be faster who knows). Note this is not a capped-size
    cache, and may explode your computer but will also never forget.
    - paramed function decorator.
    """
    if forwards_to is None:
        sig = _inspect.signature(func)
    else:
        sig = _inspect.signature(forwards_to)
    return _Cached(func, sig)

@incremental
def weakcached(func, forwards_to=None):
    """
    Same as `cached` but doesn't increment reference counts for the cached
    values, so they won't continue to be cached once nothing else references
    them.
    - paramed function decorator.
    """
    if forwards_to is None:
        sig = _inspect.signature(func)
    else:
        sig = _inspect.signature(forwards_to)
    return _WeakCached(func, sig)



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
    """
    Returns true if the given object can have `__iter__` called without an
    exception (hopefully meaning it's iterable).
    """
    try:
        obj.__iter__()
        return True
    except Exception:
        return False


def tname(t, namespaced=False, quoted=True):
    """
    Returns a string of the given type, surrounded in single quotes. If the type
    has the `_tname` property that will be used for the name, otherwise it will
    be constructed from (optionally) the module and `__name__`.
    """
    if not isinstance(t, type):
        raise TypeError(f"expected type, got {objtname(t)}")
    finish = lambda s: f"'{s}'" if quoted else s
    if hasattr(t, "_tname"):
        return finish(t._tname)
    name = t.__name__
    if not namespaced:
        return finish(name)
    namespace = t.__module__ + "."
    namespace *= namespace not in {"__main__", "builtins"}
    return finish(namespace + name)

def objtname(obj, namespaced=False, quoted=True):
    """
    Alias for 'tname(type(obj))'.
    """
    return tname(type(obj), namespaced=namespaced, quoted=quoted)



# Hack to enable console escape codes.
_os.system("")

def coloured(cols, txts):
    """
    When given the colour 'cols' and text 'txts' arrays, prints each element of
    the text as its corresponding colour. If any text is already coloured, it is
    left as that colour. Colour codes can be found at:
    https://gist.github.com/fnky/458719343aabd01cfb17a3a4f7296797
    """
    cols = maybe_pack(cols, aslist=True)
    txts = maybe_pack(txts, aslist=True)
    if len(cols) != len(txts):
        raise TypeError("must give one colour for each piece of text, got "
                f"{len(cols)} colours and {len(txts)} texts")
    for col in cols:
        if not isinstance(col, int):
            raise TypeError(f"expected integer colour code, got {objtname(col)}")
    for txt in txts:
        if not isinstance(txt, str):
            raise TypeError(f"expected string text, got {objtname(txt)}")

    leading_off_code = _re.compile(r"^\x1B\[0m")
    leading_col_code = _re.compile(r"^\x1B\[38;5;([0-9]+)m")
    leading_ctrl_code = _re.compile(r"^\x1B\[[0-9;]*[A-Za-z]")
    col_code = lambda c: f"\x1B[38;5;{c}m" if c >= 0 else "\x1B[0m"

    # Decompose into a list of characters (without control codes) and which
    # colour they should be.
    chars = []
    codes = []
    full = "".join(txts)
    cur_col = -1
    while True:
        while True:
            # Check for an off code.
            match = leading_off_code.match(full)
            if match is not None:
                cur_col = -1
                full = full[len(match.group(0)):]
                continue
            # Check for a colour code.
            match = leading_col_code.match(full)
            if match is not None:
                cur_col = int(match.group(1))
                full = full[len(match.group(0)):]
                continue
            # Ignore any other code.
            match = leading_ctrl_code.match(full)
            if match is not None:
                full = full[len(match.group(0)):]
                continue
            break
        if not full:
            break
        codes.append(cur_col)
        chars.append(full[0])
        full = full[1:]

    # Construct the coloured string, only colouring where necessary.
    wants = [c for col, txt in zip(cols, txts)
               for c in [col] * len(nonctrl(txt))]
    assert len(chars) == len(codes)
    assert len(chars) == len(wants)
    for i, c in enumerate(codes):
        if c != -1:
            wants[i] = c
    # Reset to nothing at the end.
    chars.append("")
    wants.append(-1)

    segs = [] # funny
    prev = -1
    for w, c in zip(wants, chars):
        if w != prev:
            segs.append(col_code(w))
            prev = w
        segs.append(c)
    return "".join(segs)

def nonctrl(string):
    """
    Returns 'string' with all console control codes removed.
    """
    if not isinstance(string, str):
        raise TypeError(f"expected a str 'string', got {objtname(string)}")
    control_code = _re.compile(r"\x1B\[[0-9;]*[A-Za-z]")
    return control_code.sub("", string)

def idxctrl(string, i):
    """
    Returns the correct index into 'string' which indexes the character
    'nonctrl(string)[i]'.
    """
    if not isinstance(string, str):
        raise TypeError(f"expected a str 'string', got {objtname(string)}")
    if not isinstance(i, int):
        raise TypeError(f"expected an integer 'i', got {objtname(i)}")
    if i >= len(string):
        return i
    if i < 0:
        raise IndexError(f"expected a positive index, got: {i}")
    leading_control_code = _re.compile(r"^(?:\x1B\[[0-9;]*[A-Za-z])+")
    missing = 0
    for _ in range(i + 1):
        new = leading_control_code.sub("", string)
        missing += len(string) - len(new)
        string = new[1:]
    return missing + i


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
        left += "." * (pwidth - len(nonctrl(left))) + " "
        parts.append(left)
        pad_to = len(nonctrl(left))
        wrapme = desc
    else:
        first = False
        pad_to = lead
        wrapme = name

    while wrapme:
        line = wrapme[:idxctrl(wrapme, width - pad_to)]
        if len(nonctrl(line)) == width - pad_to and " " in line:
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



def singleton(cls):
    """
    Returns a single instantiation of the given class, and prevents further
    creation of the class.
    """
    instance = cls()
    def throw(cls, *args, **kwargs):
        raise TypeError("cannot create another instance of singleton "
                f"{tname(cls)}")
    cls.__new__ = throw
    return instance




class _Templated:
    def __init__(self, creator, parents, decorators, metaclass, screeners):
        _functools.update_wrapper(self, creator)

        # Create a base class which the others can inherit from. This class will
        # have the parents, and then decorators and metaclass will be used for
        # the instantiated classes.
        self.Base = type(creator.__name__, parents, {})
        self.creator = creator
        self.decorators = decorators
        self.metaclass = metaclass
        self.screeners = screeners

        # Make the default namer.
        def namer(*param_values):
            tostr = lambda x: x.__name__ if isinstance(x, type) else repr(x)
            args = ", ".join(map(tostr, param_values))
            return f"{self.Base.__name__}[{args}]"
        self.namer = namer
        self.tnamer = None # may be set by user.

        # Make the creator. Not a method bc i dont think cached would work lmao.
        @weakcached(forwards_to=creator)
        def makecls(*param_values):
            # Screen params.
            for screener in self.screeners:
                param_values = screener(param_values)
            # Check if we've had these params.
            if makecls.iscached(*param_values):
                return makecls(*param_values)
            # Get all the attributes of the class (which are the local vars that
            # the function defines).
            attributes = self.creator(*param_values)
            if not isinstance(attributes, dict):
                raise TypeError("expected a dict return from template, got "
                        f"{objtname(attributes)}")
            name = self.namer(*param_values)
            # Create class using metaclass and inheriting from the base.
            cls = self.metaclass(name, (self.Base, ), attributes)
            # Apply any decorators, bottom-up.
            for dec in self.decorators[::-1]:
                cls = dec(cls)
            # Cop creators doc.
            if cls.__doc__ is None:
                cls.__doc__ = self.creator.__doc__
            # Cop a tname.
            if self.tnamer is not None and not hasattr(cls, "_tname"):
                cls._tname = self.tnamer(*param_values)
            return cls
        self._makecls = makecls

    def __call__(self, *args, **kwargs):
        raise TypeError("use square brackets to instantiate a templated class")

    def __getitem__(self, param_values):
        if not isinstance(param_values, tuple):
            param_values = (param_values, )
        return self._makecls(*param_values)

    # im literally the best
    def __instancecheck__(self, instance):
        return isinstance(instance, self.Base)
    def __subclasscheck__(self, subclass):
        return issubclass(subclass, self.Base)

    def add_screener(self, screener):
        if not callable(screener):
            raise TypeError("expected callable screener, got "
                    f"{objtname(screener)}")
        # Might have already been added.
        if screener in self.screeners:
            return
        # Check that it wouldn't have changed any returns from already created
        # classes.
        for args, _ in self._makecls.allcached:
            try:
                nocando = (args != screener(args))
            except Exception as e:
                raise RuntimeError("screener added too late, template already "
                        "instantiated in a case where screener failed for "
                        f"params of: {args}") from e
            if nocando:
                raise RuntimeError("screener added too late, template already "
                        "instantiated in a should-have-been-screened case for "
                        f"params of: {args}")
        self.screeners.append(screener)

    def screener(self, screener):
        # Provide nice syntax of `@Template.screener`.
        self.add_screener(screener)
        return screener

@incremental
def templated(creator, parents=(), decorators=(), metaclass=type, screeners=()):
    """
    Used for templated classes. Essentially transforms a function into a class,
    and the fully-defined type can be created via `func[params]`. The whole deal
    is that when this function is invoked, it defines all the normal methods a
    class would have and then returns the `locals()`, which are manhandled into a
    new class with those as attributes. Any dictionary return is accepted for the
    attr lookups. The instantiated classes will have the given 'parents' and have
    the given 'decorators' applied. 'screeners' should all be functions which
    take the template parameters and return new template parameters:
    'screener(params)'; essentially "screening" the parameters to ensure that
    equivalent parameters do not result in distinct classes.
    WARNING: to use `super()`, you must instead do `super(func[params], self)`,
             with the exact params that this class is using.
    EXTRA WARNING: be very careful with mutable values anywhere near a template,
                   generally as much as possible should be immutable. Template
                   parameters (especially) must be immutable.
    - paramed "function" decorator.
    """
    parents = maybe_pack(parents)
    decorators = maybe_pack(decorators)
    screeners = maybe_pack(screeners, aslist=True)
    # i luv validation.
    if not callable(creator):
        raise TypeError(f"expected callable 'creator', got {objtname(creator)}")
    for p in parents:
        if not isinstance(p, type):
            raise TypeError("expected a type for each parent, got "
                    f"{objtname(p)}")
    for d in decorators:
        if not callable(d):
            raise TypeError("expected a callable for each decorator, got "
                    f"{objtname(d)}")
    if not isinstance(metaclass, type):
        raise TypeError("expected a type for the metaclass, got "
                f"{objtname(metaclass)}")
    if not issubclass(metaclass, type):
        raise TypeError("expected a type that inherits from type for the "
                f"metaclass, got {tname(metaclass)}")
    for s in screeners:
        if not callable(s):
            raise TypeError("expected a callable for each screener, got "
                    f"{objtname(s)}")
    return _Templated(creator, parents, decorators, metaclass, screeners)



def simplest_ratio(x):
    """ Returns the simplest ratio `n, d` s.t. `n / d == x`. """
    if not isinstance(x, float):
        raise TypeError(f"expected float for 'x', got {objtname(x)}")
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
