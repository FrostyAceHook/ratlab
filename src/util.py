import functools as _functools
import inspect as _inspect
import sys as _sys


class classproperty:
    """
    Makes the given method a class property.
    - method decorator.
    """
    def __init__(self, fget):
        self.fget = fget
        _functools.update_wrapper(self, fget)

    def __get__(self, instance, owner):
        return self.fget(owner)


def tname(t):
    m = str(t.__module__)
    m = "" if m == "__main__" else m + "."
    return m + t.__name__


def immutable(cls):
    """
    Make the given class immutable (outside the `__init__` method). When a class
    inherits from this, it will be mutable except for the members which were
    assigned during the immutable classes `__init__`.
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
            cooked = type(self) is cls or name in self._frozen
        if cooked:
            raise AttributeError("cannot modify an immutable instance")

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



@immutable
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



def get_locals(func, /, *args, **kwargs):
    """
    Returns `func_ret, func_locals`, where locals is a dict of all the locals
    variables on return from `func`.
    """
    top_frame = None
    lcls = None
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
            return None # no need to keep calling this trace.
        return catch_locals

    # wrapper to know this is the top-level return.
    def call():
        return func(*args, **kwargs)

    try:
        _sys.settrace(catch_locals)
        ret = call()
    finally:
        # don leak the trace.
        _sys.settrace(None)

    return ret, lcls


@incremental
def templated(creator, parents=(), decorators=(), metaclass=type):
    """
    Used for templated classes. Essentially transforms a function into a class,
    and the fully-defined type can be created via `func[params]`. Basically the
    whole deal is that when this function is invoked, any locals that are around
    at return are manhandled into a new class with those as attributes.
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
            raise ValueError("non-none return from templated function")

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
        "__instancecheck__": __instancecheck__,
        "__subclasscheck__": __subclasscheck__,
        "__contains__": __contains__,
        "__getitem__": __getitem__,
        "__call__": __call__,
        "params": param_names,
    }

    CreatorCreator = type("CreatorCreator", (), attrs)
    Creator = CreatorCreator()
    _functools.update_wrapper(Creator, creator)
    return Creator
    # not confusing at all
