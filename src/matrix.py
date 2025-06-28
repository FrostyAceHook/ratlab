import inspect as _inspect
import itertools as _itertools
import math as _math
import types as _types

from util import (
    coloured as _coloured, nonctrl as _nonctrl, entry as _entry, tname as _tname,
    iterable as _iterable, immutable as _immutable, templated as _templated,
    classconst as _classconst, instconst as _instconst,
)


# Derive from to indicate it may implement field methods.
class Field:

    # few methods have defaults, see `ExampleField` for all methods.

    @_classconst
    def from_bool(cls, x):
        return cls.one if x else cls.zero

    @_classconst
    def consts(cls):
        return dict()

    @_classconst
    def exposes(cls):
        return dict()

    @classmethod
    def eq(cls, a, b):
        return cls.issame(a, b)

    @classmethod
    def atan2(cls, y, x):
        if "pi" not in cls.consts:
            raise NotImplementedError("cannot represent pi")
        pi = cls.consts["pi"]
        two = cls.from_int(2)

        if cls.lt(cls.zero, x):
            return cls.atan(cls.div(y, x))
        elif cls.lt(x, cls.zero):
            if cls.lt(cls.zero, y):
                return cls.add(cls.atan(cls.div(y, x)), pi)
            else:
                return cls.sub(cls.atan(cls.div(y, x)), pi)
        elif cls.eq(x, cls.zero):
            if cls.lt(cls.zero, y):
                return cls.div(pi, two)
            elif cls.lt(y, cls.zero):
                return cls.div(cls.sub(cls.zero, pi), two)
            elif cls.eq(y, cls.zero):
                return cls.zero # atan(0/0)
            else:
                raise TypeError(f"value '{repr(y)}' could not be ordered "
                        f"against zero ('{repr(cls.zero)}')")
        else:
            raise TypeError(f"value '{repr(x)}' could not be ordered against "
                    f"zero ('{repr(cls.zero)}')")

    def __repr__(s):
        return s.rep(s, False)



# Example field type, demonstrating maximum functionality. Note no internals are
# included, only the return type is specified.
class ExampleField(Field):

    @classmethod
    def from_bool(cls, x): # bool -> cls
        assert isinstance(x, bool)
        return cls()
    @classmethod
    def from_int(cls, x): # int -> cls
        assert isinstance(x, int)
        return cls()
    @classmethod
    def from_float(cls, x): # float -> cls
        assert isinstance(x, float)
        return cls()
    @classmethod
    def from_complex(cls, x): # complex -> cls
        assert isinstance(x, complex)
        return cls()
    @classmethod
    def from_str(cls, x): # str -> cls (note this isnt atoi)
        assert isinstance(x, str)
        return cls()

    @classmethod
    def to_int(cls, a): # int(a)
        return 0
    @classmethod
    def to_float(cls, a): # float(a)
        return 0.0
    @classmethod
    def to_complex(cls, a): # complex(a)
        return 0.0j

    @_classconst
    def zero(cls): # additive identity.
        return cls(0)
    @_classconst
    def one(cls): # multiplicative identity.
        return cls(1)

    @_classconst
    def consts(cls): # map of str to elements, which will be mapped into the
                     # variable space on field selection.
        return {"e": cls(2.71828), "pi": cls(3.14159)}

    @_classconst
    def exposes(cls): # map of str attribute names to their type. these attrs
                      # will be exposed in the matrices.
        cls().attr = True # "attr" must be an object property.
        return {"attr": bool}

    @classmethod
    def add(cls, a, b): # a+b
        return cls()
    @classmethod
    def sub(cls, a, b): # a-b
        return cls()
    @classmethod
    def absolute(cls, a): # |a|
        return cls()

    @classmethod
    def mul(cls, a, b): # a*b
        return cls()
    @classmethod
    def div(cls, a, b): # a/b
        return cls()

    @classmethod
    def power(cls, a, b): # a^b
        return cls()
    @classmethod
    def root(cls, a, b): # a^(1/b)
        return cls()
    @classmethod
    def log(cls, a, b): # log_a(b)
        return cls()

    @classmethod
    def sin(cls, a): # sin(a)
        return cls()
    @classmethod
    def cos(cls, a): # cos(a)
        return cls()
    @classmethod
    def tan(cls, a): # tan(a)
        return cls()

    @classmethod
    def asin(cls, a): # sin^-1(a)
        return cls()
    @classmethod
    def acos(cls, a): # cos^-1(a)
        return cls()
    @classmethod
    def atan(cls, a): # tan^-1(a)
        return cls()
    @classmethod
    def atan2(cls, y, x): # tan^-1(y / x), but quadrant-aware.
        return cls()

    @classmethod
    def diff(cls, y, x): # (d/dx y)
        return cls()
    @classmethod
    def intt(cls, y, x): # (int y dx)
        return cls()
    @classmethod
    def def_intt(cls, y, x, a, b): # (int(a..b) y dx)
        return cls()

    @classmethod
    def conj(cls, a): # a = x+iy, conj(a) = x-iy
        return cls()

    @classmethod
    def issame(cls, a, b): # a is identical to b, must return bool
        return True
    # Comparisons don't necessarily have bool returns, they may also return any
    # Field type (including this class). However, when this returned object is
    # cast to bool, it must be true iff the original comparison is always true.
    # Note that the return type of comparisons must be consistent across calls
    # with different values (but eq may be different from lt).
    @classmethod
    def eq(cls, a, b): # a == b
        return ...
    @classmethod
    def lt(cls, a, b): # a < b
        return ...

    @classmethod
    def hashed(cls, a): # hash(a)
        return hash(0)

    @classmethod
    def rep(cls, a, short): # repr(a), with short+long form.
        if short:
            return "d ... j"
        return "dujjdujj"




# Generic wrapper for any type which isnt a field, attempting to make it one.
@_templated(parents=Field, decorators=_immutable)
def GenericField(T):
    if T in {tuple, list, dict, _types.FunctionType,
            _types.BuiltinFunctionType, _types.MethodType,
            _types.BuiltinMethodType}:
        raise TypeError("it cannot be good if we tryna make a matrix outta "
                f"{_tname(T)}")

    @classmethod
    def _generic(cls, x):
        return GenericField[type(x)](x)

    def __init__(self, v):
        if not isinstance(v, T):
            raise TypeError(f"expected {_tname(T)}, got {_tname(type(v))}")
        self.obj = v

    @classmethod
    def from_bool(cls, x):
        return cls(T(x))
    @classmethod
    def from_int(cls, x):
        return cls(T(x))
    @classmethod
    def from_float(cls, x):
        return cls(T(x))
    @classmethod
    def from_complex(cls, x):
        return cls(T(x))
    @classmethod
    def from_str(cls, x):
        if issubclass(T, (bool, int, float)): # dont do parsing.
            raise NotImplementedError()
        return cls(T(x))

    @classmethod
    def to_int(cls, a):
        return int(a.obj)
    @classmethod
    def to_float(cls, a):
        return float(a.obj)
    @classmethod
    def to_complex(cls, a):
        return complex(a.obj)

    @_classconst
    def zero(cls):
        return cls(T(0))
    @_classconst
    def one(cls):
        return cls(T(1))

    @classmethod
    def add(cls, a, b):
        return cls._generic(a.obj + b.obj)
    @classmethod
    def sub(cls, a, b):
        return cls._generic(a.obj - b.obj)
    @classmethod
    def absolute(cls, a):
        return cls._generic(abs(a.obj))

    @classmethod
    def mul(cls, a, b):
        return cls._generic(a.obj * b.obj)
    @classmethod
    def div(cls, a, b):
        return cls._generic(a.obj / b.obj)

    @classmethod
    def power(cls, a, b):
        return cls._generic(a.obj ** b.obj)
    @classmethod
    def root(cls, a, b):
        return cls._generic(a.obj ** (T(1) / b.obj))

    @classmethod
    def eq(cls, a, b):
        return a.obj == b.obj
    @classmethod
    def lt(cls, a, b):
        return a.obj < b.obj

    @classmethod
    def hashed(cls, a):
        return hash(a.obj)

    @classmethod
    def rep(cls, a, short):
        if T is bool:
            # most pythonic python ever written?
            return ["nogh", "yeagh", "N", "Y"][a.obj + 2*short]
        return repr(a.obj)

    # templated.
    return locals()



def _is_overridden(space, field, name):
    if name not in space:
        return False
    if field is None:
        return True
    got = space[name]
    expect = getattr(Single[field], name)
    if type(got) is not type(expect):
        return True
    try:
        return not expect.issame(got)
    except Exception:
        return True # assume nothing but the worst.

def _get_field(field, xs=()):
    if field is None:
        if xs:
            field = xs[0].field
        else:
            if lits._field is None:
                raise RuntimeError("must specify a field using 'lits'")
            field = lits._field
    return field

def _maybe_unpack(xs):
    if len(xs) != 1:
        return xs
    x = xs[0]
    # dont unpack all iterables lmao.
    if isinstance(x, (Matrix, str, bytes, dict)):
        return xs
    if not _iterable(x):
        return xs
    return tuple(x)

def _get_space(depth=1):
    # Get the globals of the frame `depth` calls above the caller (so depth+1
    # calls above this function)
    frame = None
    try:
        for i in range(depth + 2):
            if i == 0:
                frame = _inspect.currentframe()
            else:
                frame = frame.f_back
            if frame is None:
                # should always find i=0 and i=1 (current and caller).
                raise RuntimeError(f"call depth too short, got {i - 2}/{depth}")
        return frame.f_globals
    finally:
        del frame # don leak.


def lits(field, inject=True):
    """
    Sets the current/default field to the given field and injects constants such
    as 'e' and 'pi' into the globals.
    """
    try:
        # try create a field out of it.
        if field is not None:
            Single[field]
    except Exception as e:
        raise TypeError("expected a valid field class, got "
                f"{_tname(type(field))}") from e

    prev_field = lits._field
    lits._field = field
    if not inject:
        return
    # Inject constants into the space.
    space = _get_space()
    for name in lits._injects:
        # Dont wipe vars the user has set.
        if _is_overridden(space, prev_field, name):
            continue
        if field is None:
            space.pop(name, None)
            continue
        try:
            space[name] = getattr(Single[field], name)
        except Exception:
            pass
lits._field = None
lits._injects = ("e", "pi")


class Shape:
    """
    Sequence of the length of each dimension, with implicitly infinite trailing
    1s (or 0s if empty). Note that the first dimension is the number of rows and
    the second is the number of columns (further dimensions have no intrinsic
    meaning, other than repeating matrices).
    """

    def __init__(s, *lens):
        lens = _maybe_unpack(lens)
        for l in lens:
            if not isinstance(l, int):
                raise TypeError("dimension lengths must be ints, got "
                        f"{_tname(type(l))}")
        if any(l < 0 for l in lens):
            raise ValueError(f"dimension lengths cannot be negative, got {lens}")
        # Trim any trailing 1s (note this collapses single to ()).
        while lens and lens[-1] == 1:
            lens = lens[:-1]
        # Collapse empty.
        if _math.prod(lens) == 0:
            lens = (0, )
        s._lens = lens

    @_classconst
    def empty(cls):
        return cls(0)
    @_classconst
    def single(cls):
        return cls()

    @_instconst
    def size(s):
        """
        Number of cells.
        """
        return _math.prod(s._lens)
    @_instconst
    def ndim(s):
        """
        Number of dimensions. Note empty has "-1 dimensions" and single has 0.
        """
        return -1 if not s else len(s)
    @_instconst
    def lastaxis(s):
        """
        Index of the last axis with non-1 length.
        """
        return max(0, len(s) - 1)

    @_instconst
    def isempty(s):
        """
        Is empty? (0x0)
        """
        return s.size == 0
    @_instconst
    def issingle(s):
        """
        Is only one cell? (1x1)
        """
        return s.size == 1
    @_instconst
    def isvec(s):
        """
        Has at-most one axis with length >1? (empty and single count as vectors)
        """
        return sum(l > 1 for l in s._lens) <= 1
    @_instconst
    def iscol(s):
        """
        Is column vector? (empty and single count as column vectors)
        """
        return s[0] == s.size
    @_instconst
    def isrow(s):
        """
        Is row vector? (empty and single count as row vectors)
        """
        return s[1] == s.size
    @_instconst
    def issquare(s):
        """
        Is square matrix? (only 2D matrices can be square)
        """
        return s.ndim <= 2 and s[0] == s[1]

    def __iter__(s):
        if not s:
            raise ValueError("cannot iterate axes of empty")
        return s._lens.__iter__()

    def __len__(s):
        if not s:
            raise ValueError("cannot get length of empty")
        return s._lens.__len__()

    def __getitem__(s, axis):
        """
        Returns the length along the given axis.
        """
        # Oob dimensions are implicitly 1 (unless empty, in which case 0).
        if not isinstance(axis, int):
            raise TypeError("expected an integer axis, got "
                    f"{_tname(type(axis))}")
        if axis < 0:
            raise ValueError(f"axis cannot be negative, got: {axis}")
        if axis >= s.ndim:
            return 1 if s.size else 0
        return s._lens.__getitem__(axis)

    def withaxis(s, axis, length):
        """
        New shape with the given length along 'axis'.
        """
        if not isinstance(axis, int):
            raise TypeError("expected an integer axis, got "
                    f"{_tname(type(axis))}")
        if axis < 0:
            raise ValueError(f"axis cannot be negative, got: {axis}")
        if s.isempty:
            return s
        newshape = list(s._lens) + [1] * (axis + 1 - len(s._lens))
        newshape[axis] = length
        return Shape(newshape)

    def insertaxis(s, axis, length):
        """
        New shape with the given length inserted at 'axis'.
        """
        if not isinstance(axis, int):
            raise TypeError("expected an integer axis, got "
                    f"{_tname(type(axis))}")
        if axis < 0:
            raise ValueError(f"axis cannot be negative, got: {axis}")
        if s.isempty:
            return s
        newshape = list(s._lens) + [1] * (axis - len(s._lens))
        newshape.insert(axis, length)
        return Shape(newshape)

    def insert(s, axis, shape):
        """
        New shape with the given shape inserted at 'axis'.
        """
        if not isinstance(axis, int):
            raise TypeError("expected an integer axis, got "
                    f"{_tname(type(axis))}")
        if axis < 0:
            raise ValueError(f"axis cannot be negative, got: {axis}")
        if not isinstance(shape, Shape):
            raise TypeError("expected a Shape shape, got "
                    f"{_tname(type(shape))}")
        if shape.isempty:
            raise TypeError("expected a non-empty shape to insert")
        if s.isempty:
            return s
        newshape = list(s._lens) + [1] * (axis - len(s._lens))
        newshape[axis:axis] = shape._lens
        return Shape(newshape)

    def dropaxis(s, axis):
        """
        New shape with the given axis removed, however this axis must not affect
        the memory layout of the cells - it must be 1 for non-empty shapes and 0
        for empty shapes.
        """
        if not isinstance(axis, int):
            raise TypeError("expected an integer axis, got "
                    f"{_tname(type(axis))}")
        if axis < 0:
            raise ValueError(f"axis cannot be negative, got: {axis}")
        if s.isempty:
            return s
        if s[axis] != 1:
            raise ValueError(f"cannot drop axis with length >1, axis {axis} "
                    f"has length {s[axis]}")
        return Shape(l for i, l in enumerate(s._lens) if i != axis)


    def __bool__(s):
        """
        Alias for 'not s.isempty'.
        """
        return not s.isempty

    def __eq__(s, o):
        """
        Is same shape?
        """
        if isinstance(o, tuple):
            o = Shape(o)
        if not isinstance(o, Shape):
            return NotImplemented
        return s._lens == o._lens

    def __hash__(s):
        return hash(s._lens)

    def __repr__(s):
        # Pad to at-least 2d.
        if s.ndim < 2:
            shape = (s[0], s[1])
        else:
            shape = s
        return "x".join(map(str, shape))

    def stride(s, axis):
        """
        Returns the element-offset between consequentive elements along the given
        axis, when stored in canon flattened format.
        """
        if s.isempty:
            raise ValueError("cannot find stride of empty")
        if not isinstance(axis, int):
            raise TypeError(f"expected an integer axis, got "
                    f"{_tname(type(axis))}")
        if axis < 0:
            raise ValueError(f"axis cannot be negative, got {axis}")
        axis = min(axis, s.ndim)
        return _math.prod(s._lens[:axis])


    @property
    def indices(s):
        """
        Iterable of all indices for this shape. An index is a tuple of per-axis
        offsets.
        """
        if s.isempty:
            raise ValueError("cannot iterate empty")
        slices = [range(l) for l in s]
        revidxs = _itertools.product(*reversed(slices))
        return (tuple(reversed(revidx)) for revidx in revidxs)

    @property
    def offsets(s):
        """
        Iterable of all offsets for this shape. An offset is an integer which
        indexes a canon flattened array of this shape.
        """
        if s.isempty:
            raise ValueError("cannot iterate empty")
        return range(s.size)


    def index(s, offset):
        """
        Index for the given offset.
        """
        if s.isempty:
            raise ValueError("cannot index empty")
        if not isinstance(offset, int):
            raise TypeError(f"expected an integer offset, got "
                    f"{_tname(type(offset))}")
        if offset < 0:
            raise ValueError(f"offset cannot be negative, got {offset}")
        if offset >= s.size:
            raise IndexError(f"offset {offset} out of bounds for size "
                    f"{s.size}")
        return tuple(offset % s.stride(ii) for ii in range(s.ndim))

    def offset(s, *ijk):
        """
        Offset for the given index.
        """
        if s.isempty:
            raise ValueError("cannot index empty")
        ijk = _maybe_unpack(ijk)
        for ii, i in enumerate(ijk):
            if not isinstance(i, int):
                raise TypeError("expected an integer index, got "
                        f"{_tname(type(i))} at index {ii}")
        oldijk = ijk
        ijk = [i + (i < 0) * s[ii] for ii, i in enumerate(ijk)]
        for ii, i in enumerate(ijk):
            if i not in range(s[ii]):
                raise IndexError(f"index {oldijk[ii]} out of bounds for axis "
                        f"{ii} with length {s[ii]}")
        return sum(i * s.stride(ii) for ii, i in enumerate(ijk))



class NO_SEED:
    def __repr__(self):
        return "NO_SEED"
NO_SEED = NO_SEED() # :(


# Matrices.
@_templated(decorators=_immutable)
def Matrix(field, shape):
    """
    Fixed-sized n-dimensional sequence of elements.
    """

    if not isinstance(field, type):
        raise TypeError(f"expected a type for field, got: {repr(field)}, of "
                f"type {_tname(type(field))}")
    if issubclass(field, Matrix):
        raise TypeError("mate a matrix of matrices? calm down")

    # Special field types:
    # - bool (able to be used for indexing and overrides the dflt behaviour of
    #       __bool__)
    # - string (for all matrix repr)
    # - float/complex (numpy internals for speed) (also expose .isnan and .isinf)

    # Wrap non-field classes.
    if not issubclass(field, Field):
        return Matrix[GenericField[field], shape]
    # Wrap non-Shape shapes.
    if not isinstance(shape, Shape):
        return Matrix[field, Shape(shape)]

    @classmethod
    def _need(cls, method, extra=""):
        if extra:
            extra = f" ({extra})"
        if not hasattr(cls.field, method):
            thing = {
                "from_bool": "cannot cast from bool",
                "from_int": "cannot cast from int",
                "from_float": "cannot cast from float",
                "from_complex": "cannot cast from complex",
                "from_str": "cannot cast from string",

                "to_int": "cannot cast to int",
                "to_float": "cannot cast to float",
                "to_complex": "cannot cast to complex",

                "zero": "no additive identity (zero)",
                "one": "no multiplicative identity (one)",

                "consts": "no constants",

                "add": "cannot do addition",
                "sub": "cannot do subtraction",
                "absolute": "cannot do absolution",

                "mul": "cannot do multiplication",
                "div": "cannot do division",

                "power": "cannot do powers",
                "root": "cannot do roots",
                "log": "cannot do natural lograithm",

                "sin": "cannot do sin",
                "cos": "cannot do cos",
                "tan": "cannot do tan",
                "asin": "cannot do asin",
                "acos": "cannot do acos",
                "atan": "cannot do atan",

                "diff": "cannot do derivatives",
                "intt": "cannot do integration",
                "def_intt": "cannot do (definite) integration",

                "conj": "cannot do complex conjugation",

                "eq": "cannot do equality",
                "lt": "cannot do ordering",

                "issame": "cannot check if identical",

                "hashed": "cannot hash",

                "rep": "cannot stringify",
            }
            msg = f"{thing[method]} over field {_tname(cls.field)}{extra}"
            raise NotImplementedError(msg)

    @classmethod
    def _f(cls, method):
        cls._need(method)
        return getattr(cls.field, method)


    def __init__(s, cells, _checkme=True):
        # living on the edge.
        if not _checkme:
            if not isinstance(cells, tuple):
                raise TypeError("sod off mate enable the checks")
            s._cells = cells
            return

        # `cells` is a flattened iterable of each cell, progressing through the
        # matrix in the order of shape (so col-major for 2d).
        if not _iterable(cells):
            raise TypeError(f"cells must be iterable, got {_tname(type(cells))}")

        # Often when a function returns from a new field, its not appropriately
        # wrapped, but we only make this exemption for non-field types.
        if issubclass(s.field, GenericField):
            cells = list(cells)
            for i in range(len(cells)):
                if isinstance(cells[i], s.field.T):
                    cells[i] = s.field(cells[i])

        cells = tuple(cells)
        for i, cell in enumerate(cells):
            if not isinstance(cell, s.field):
                raise TypeError(f"expected cells of type {_tname(s.field)}, got "
                        f"{_tname(type(cell))} (occured at index {i}, had "
                        f"value: {repr(cell)})")
        if len(cells) != shape.size:
            raise TypeError(f"expected {shape.size} cells to matched flattened "
                    f"size of {shape}, got {len(cells)}")

        s._cells = cells


    @_classconst
    def eye(cls):
        """
        Identity matrix.
        """
        if not cls.issquare:
            raise TypeError("only square matricies have an identity matrix, got "
                    f"{cls.shape}")
        cells = [cls._f("zero")] * cls.shape.size if cls.shape else ()
        n = cls.shape[0]
        for i in range(n):
            cells[i*n + i] = cls._f("one")
        return cls(cells)
    @_classconst
    def zeros(cls):
        """
        Zero-filled matrix.
        """
        cells = (cls._f("zero"), ) * cls.shape.size if cls.shape else ()
        return cls(cells)
    @_classconst
    def ones(cls):
        """
        One-filled matrix.
        """
        cells = (cls._f("one"), ) * cls.shape.size if cls.shape else ()
        return cls(cells)

    @_classconst
    def zero(cls):
        """
        Single zero.
        """
        return single(cls._f("zero"), field=cls.field)
    @_classconst
    def one(cls):
        """
        Single one.
        """
        return single(cls._f("one"), field=cls.field)
    @_classconst
    def e(cls):
        """
        Single euler's number (2.71828...).
        """
        if "e" in cls.field.consts:
            e = cls.field.consts["e"]
        else:
            cls._need("from_float", "to represent e")
            e = cls._f("from_float")(_math.e)
        return single(e, field=cls.field)
    @_classconst
    def pi(cls):
        """
        Single pi (3.14159...).
        """
        if "pi" in cls.field.consts:
            pi = cls.field.consts["pi"]
        else:
            cls._need("from_float", "to represent pi")
            pi = cls._f("from_float")(_math.pi)
        return single(pi, field=cls.field)
    @_classconst
    def i(cls):
        """
        Single imaginary unit.
        """
        if "i" in cls.field.consts:
            i = cls.field.consts["i"]
        else:
            cls._need("from_complex", "to represent i")
            i = cls._f("from_complex")(1j)
        return single(i, field=cls.field)

    @_classconst
    def size(cls):
        """
        Number of cells.
        """
        return cls.shape.size
    @_classconst
    def ndim(cls):
        """
        Number of dimensions. Note empty has -1 dimensions and single has 0.
        """
        return cls.shape.ndim
    @_classconst
    def lastaxis(cls):
        """
        Index of the last axis with non-1 length.
        """
        return cls.shape.lastaxis

    @_classconst
    def isempty(cls):
        """
        Is empty? (0x0)
        """
        return cls.shape.isempty
    @_classconst
    def issingle(cls):
        """
        Is only one cell? (1x1)
        """
        return cls.shape.issingle
    @_classconst
    def isvec(cls):
        """
        Has at-most one axis with length >1? (empty and single count as vectors)
        """
        return cls.shape.isvec
    @_classconst
    def iscol(cls):
        """
        Is column vector? (empty and single count as column vectors)
        """
        return cls.shape.iscol
    @_classconst
    def isrow(cls):
        """
        Is row vector? (empty and single count as row vectors)
        """
        return cls.shape.isrow
    @_classconst
    def issquare(cls):
        """
        Is square matrix? (only 2D matrices can be square)
        """
        return cls.shape.issquare


    @classmethod
    def cast(cls, *xs, broadcast=True):
        """
        Attempts to cast each object to a matrix over 'field', all with the same
        size. Note that the shape of this class is ignored. If 'broadcast' is
        false, shapes will be left unchanged.
        """
        xs = _maybe_unpack(xs)
        if not xs:
            return ()

        def conv(x):
            if isinstance(x, Matrix):
                if cls.field != x.field:
                    raise TypeError("cannot cast matrices to different fields, "
                            f"got {_tname(cls.field)} and {_tname(x.field)}")
                return x
            convs = {bool: "from_bool", int: "from_int", float: "from_float",
                    complex: "from_complex", str: "from_str"}
            cell = None
            if isinstance(x, cls.field):
                cell = x
            elif issubclass(cls.field, GenericField):
                # the things i do for GenericField...
                if isinstance(x, cls.field.T):
                    cell = cls.field(x)
            elif type(x) in convs.keys():
                cell = cls._f(convs[type(x)])(x)
            if cell is None:
                raise TypeError(f"{_tname(type(x))} cannot be cast to "
                        f"{_tname(cls.field)}")
            return single(cell, field=cls.field)
        xs = [conv(x) for x in xs]

        # Handle empties.
        if all(x.isempty for x in xs):
            return xs

        # If no broadcasting, don't do broadcasting, so the final result won't be
        # broadcasted.
        if not broadcast:
            return xs

        if any(x.isempty for x in xs):
            raise TypeError("cannot operate with a mix of empty and non-empty "
                    "matrices")

        ndim = max(x.ndim for x in xs)
        shape = Shape(max(x.shape[i] for x in xs) for i in range(ndim))
        def broadcast(x):
            xshape = tuple(x.shape) + (1, ) * (ndim - len(x.shape))
            if not all(a == b or a == 1 for a, b in zip(xshape, shape)):
                raise TypeError(f"cannot broadcast {x.shape} to {shape}")
            return x.rep(y // x for x, y in zip(xshape, shape))
        xs = [broadcast(x) for x in xs]
        return xs


    class _At:
        def __init__(s, matrix):
            s._mat = matrix
        def __getitem__(s, ijk):
            s = s._mat
            if not isinstance(ijk, tuple):
                ijk = (ijk, )
            if not ijk:
                return empty(s.field)
            if s.isempty:
                raise ValueError("cannot index empty")
            def process(i, ii):
                if isinstance(i, slice):
                    return list(range(*i.indices(s.shape[ii])))
                if _iterable(i):
                    return list(i)
                if isinstance(i, int):
                    return [i]
                raise TypeError("expected an integer or slice access, got "
                        f"{_tname(type(i))} for axis {ii}")
            ijk += (slice(None), ) * (s.ndim - len(ijk))
            slices = [process(i, ii) for ii, i in enumerate(ijk)]
            for axis, idxs in enumerate(slices):
                for ii, i in enumerate(idxs):
                    idxs[ii] += s.shape[axis] * (i < 0)
                    if idxs[ii] not in range(s.shape[axis]):
                        raise IndexError(f"index {i} out of bounds for axis "
                                f"{axis} with length {s.shape[axis]}")
            slices = slices[:s.ndim]
            revidxs = _itertools.product(*reversed(slices))
            cells = tuple(s._at(*reversed(revidx)) for revidx in revidxs)
            newshape = Shape(map(len, slices))
            return Matrix[s.field, newshape](cells, _checkme=False)
    @_instconst
    def at(s):
        """
        Submatrix of the given indices (may be a single cell).
        """
        return _At(s)

    def _at(s, *ijk):
        return s._cells[s.shape.offset(ijk)]


    def dropaxis(s, axis):
        """
        Removes the given axis, however this axis must already have length 1 for
        non-empty matrices.
        """
        return Matrix[s.field, s.shape.dropaxis(axis)](s._cells, _checkme=False)


    @_instconst
    def ravel(s):
        """
        Vector of cells in natural iteration order (sequential axes).
        """
        return Matrix[s.field, (s.shape.size, )](s._cells, _checkme=False)


    def along(s, axis):
        """
        Tuple of perpendicular matrices along the given axis.
        """
        if not isinstance(axis, int):
            raise TypeError(f"expected integer axis, got {_tname(type(axis))}")
        if axis < 0:
            raise ValueError(f"cannot have negative axis, got: {axis}")
        # Empty is empty.
        if s.isempty:
            return ()
        if axis >= s.ndim:
            return (s, )
        def idx(j):
            idx = [slice(None)] * s.ndim
            idx[axis] = j
            return tuple(idx)
        return tuple(s.at[idx(j)] for j in range(s.shape[axis]))


    def permute(s, *order):
        """
        Permutes the axes into the given order (like a transpose).
        """
        for axis in order:
            if not isinstance(axis, int):
                raise TypeError(f"expected integer axis, got "
                        f"{_tname(type(axis))}")
        if any(axis < 0 for axis in order):
            raise ValueError(f"cannot have negative axes, got: {order}")
        if len(set(order)) != len(order):
            raise ValueError(f"cannot duplicate axes, got: {order}")
        if max(order) != len(order) - 1:
            raise ValueError(f"missing axis for swap, axis {max(order)} is "
                    "dangling")
        # Empty and single are invariant under permutation.
        if s.isempty or s.issingle:
            return s
        order += tuple(range(s.ndim))[len(order):]
        while len(order) > s.ndim and order[-1] == len(order) - 1:
            order = order[:-1]
        newshape = Shape(s.shape[axis] for axis in order)
        if s.isvec:
            # Vector has only one memory format.
            cells = s._cells
        else:
            idxs = newshape.indices
            invorder = [0] * len(order)
            for ii, i in enumerate(order):
                invorder[i] = ii
            remap = lambda ri: [ri[invorder[axis]] for axis in range(s.ndim)]
            cells = tuple(s._at(*remap(idx)) for idx in idxs)
        return Matrix[s.field, newshape](cells, _checkme=False)


    def rep(s, *counts):
        """
        Repeats this matrix the given number of times along each dimension.
        """
        counts = _maybe_unpack(counts)
        for count in counts:
            if not isinstance(count, int):
                raise TypeError("expected an integer count, got "
                        f"{_tname(type(count))}")
        if any(count < 0 for count in counts):
            raise ValueError(f"cannot have negative counts, got: {counts}")
        mul = counts + (1, ) * (s.ndim - len(counts))
        shp = tuple(s.shape) + (1, ) * (len(counts) - s.ndim)
        newshape = Shape(a * b for a, b in zip(shp, mul))
        if newshape.size == 0:
            return Matrix[s.field, (0, )](())
        if newshape.size == 1:
            return s
        cells = [0] * newshape.size
        for new in newshape.indices:
            old = tuple(a % b for a, b in zip(new, shp))
            cells[newshape.offset(new)] = s._cells[s.shape.offset(old)]
        return Matrix[s.field, newshape](tuple(cells), _checkme=False)

    def rep_along(s, axis, count):
        """
        Repeats this matrix 'count' times along 'axis'.
        """
        if not isinstance(axis, int):
            raise TypeError(f"expected an integer axis, got "
                    f"{_tname(type(axis))}")
        if axis < 0:
            raise ValueError(f"axis cannot be negative, got: {axis}")
        return s.rep((1, )*axis + (count, ))


    def __iter__(s):
        """
        Vector-only cell iterate.
        """
        if not s.isvec:
            raise TypeError(f"only vectors have bare iteration, got {s.shape} "
                    "(use .along or .ravel for matrix iterate)")
        return (single(x, field=s.field) for x in s._cells)

    def __getitem__(s, i):
        """
        Vector-only cell access.
        """
        if not s.isvec:
            raise TypeError(f"only vectors have bare getitem, got {s.shape} "
                    "(use .at for matrix cell access)")
        if isinstance(i, int):
            i += s.size * (i < 0)
            if i not in range(s.size):
                raise IndexError(f"index {i} out of bounds for size {s.size}")
        elif isinstance(i, slice):
            # all slices are valid indexers.
            pass
        else:
            raise TypeError("expected integer or slice to index vector, got "
                    f"{_tname(type(i))}")
        xs = s._cells.__getitem__(i)
        if not isinstance(xs, tuple):
            xs = (xs, )
        if not xs:
            return empty(s.field)
        if s.issingle:
            newshape = Shape(len(xs))
        else:
            newshape = s.shape.withaxis(s.lastaxis, len(xs))
        return Matrix[s.field, newshape](xs, _checkme=False)

    def __len__(s):
        """
        Vector-only cell count.
        """
        if not s.isvec:
            raise TypeError(f"only vectors have bare length, got {s.shape} (use "
                    ".size for matrix cell count)")
        return len(s._cells)

    def tolist(s, rtype):
        """
        Vector-only convert to list of 'rtype', where 'rtype' is either 'int',
        'float', or 'complex'.
        """
        if not s.isvec:
            raise TypeError(f"only vectors can convert to list, got {s.shape} "
                    "(maybe try .ravel.tolist to get a 1d list of matrix cells)")
        if not isinstance(rtype, type):
            raise ValueError("expected type for 'rtype', got "
                    f"{_tname(type(rtype))}")
        if issubclass(rtype, int):
            func = s._f("to_int")
        elif issubclass(rtype, float):
            func = s._f("to_float")
        elif issubclass(rtype, complex):
            func = s._f("to_complex")
        else:
            raise ValueError("expected one of int, float, or complex for "
                    f"'rtype', got {_tname(rtype)}")
        return list(func(x) for x in s._cells)

    @property
    def tolist_i(s):
        """
        Alias for 'm.tolist(rtype=int).
        """
        return s.tolist(rtype=int)
    @property
    def tolist_f(s):
        """
        Alias for 'm.tolist(rtype=float).
        """
        return s.tolist(rtype=float)
    @property
    def tolist_c(s):
        """
        Alias for 'm.tolist(rtype=complex).
        """
        return s.tolist(rtype=complex)


    @_instconst
    def cols(s):
        """
        Tuple of columns, for 2D matrices.
        """
        if s.ndim > 2:
            raise TypeError(f"only 2D matrices can use .cols, got {s.shape} "
                    "(use .along for other matrices)")
        return s.along(0)
    @_instconst
    def rows(s):
        """
        Tuple of rows, for 2D matrices.
        """
        if s.ndim > 2:
            raise TypeError(f"only 2D matrices can use .rows, got {s.shape} "
                    "(use .along for other matrices)")
        return s.along(1)
    @_instconst
    def colmajor(s):
        """
        Vector of cells in column-major order, for 2D matrices.
        """
        if s.ndim > 2:
            raise TypeError(f"only 2D matrices can use .colmajor, got {s.shape} "
                    "(use .ravel (maybe with .permute) for other matrices)")
        return s.ravel
    @_instconst
    def rowmajor(s):
        """
        Vector of cells in row-major order, for 2D matrices.
        """
        if s.ndim > 2:
            raise TypeError(f"only 2D matrices can use .rowmajor, got {s.shape} "
                    "(use .ravel (maybe with .permute) for other matrices)")
        return s.T.ravel


    @_instconst
    def T(s):
        """
        Swaps the first two axes.
        """
        return s.permute(1, 0)

    @_instconst
    def inv(s):
        """
        Inverse matrix, for square 2D matrices.
        """
        if not s.issquare:
            raise TypeError(f"cannot invert a non-square matrix, got {s.shape}")
        if s.det == s.zero:
            raise ValueError("cannot invert a non-invertible matrix, got det=0")
        aug = hstack(s, s.eye)
        aug = aug.rref
        return aug.at[:, s.shape[0]:]

    @_instconst
    def diag(s):
        """
        Vector of diagonal elements, for 2D matrices.
        """
        if s.ndim > 2:
            raise TypeError(f"only 2D matrices have a diagonal, got {s.shape}")
        cells = (s._at(i, i) for i in range(min(s.shape)))
        return Matrix[s.field, (min(s.shape), )](cells)


    @_instconst
    def isdiag(s):
        """
        Is diagonal matrix? (square, and only diagonal is non-zero)
        """
        if s.ndim > 2:
            raise TypeError(f"only 2D matrices can be diagonal, got {s.shape}")
        if not s.issquare:
            return False
        for i in range(s.shape[0]):
            for j in range(s.shape[1]):
                if i == j:
                    continue
                if s.at[i, j] != s.zero:
                    return False
        return True

    @_instconst
    def isuppertri(s):
        """
        Is upper-triangular matrix? (square, and below diagonal is zero)
        """
        if s.ndim > 2:
            raise TypeError(f"only 2D matrices can be triangular, got {s.shape}")
        if not s.issquare:
            return False
        for i in range(s.shape[0]):
            for j in range(i):
                if s.at[i, j] != s.zero:
                    return False
        return True

    @_instconst
    def islowertri(s):
        """
        Is lower-triangular matrix? (square, and above diagonal is zero)
        """
        if s.ndim > 2:
            raise TypeError(f"only 2D matrices can be triangular, got {s.shape}")
        return s.T.isuppertri

    @_instconst
    def isorthogonal(s):
        """
        Is orthogonal matrix? (transpose == inverse)
        """
        if s.ndim > 2:
            raise TypeError(f"only 2D matrices can be orthogonal, got {s.shape}")
        if not s.issquare:
            return False
        if s.det == s.zero:
            return False
        return bool(s.T == s.inv)

    @_instconst
    def issymmetric(s):
        """
        Is symmetric matrix? (square, and below diagonal = above diagonal)
        """
        if s.ndim > 2:
            raise TypeError(f"only 2D matrices can be symmetric, got {s.shape}")
        if not s.issquare:
            return False
        for i in range(s.shape[0]):
            for j in range(i):
                if s.at[i, j] != s.at[j, i]:
                    return False
        return True


    @classmethod
    def _apply(cls, func, *xs, rtype=None, over_field=True):
        # `over_field` is for a non-matrix returning func which operates directly
        # on the field elements, not on them wrapped in a single matrix.

        # Cast me.
        if not xs:
            return empty(cls.field if rtype is None else rtype)
        xs = cls.cast(*xs)

        # Wrap me.
        def f(y):
            nonlocal rtype
            ret = func(*y)
            if rtype is None:
                rtype = type(ret)
                if issubclass(rtype, Matrix):
                    if over_field:
                        raise TypeError("'func' cannot return a matrix")
                    if rtype.isempty:
                        raise TypeError("'func' cannot return an empty matrix")
            elif not isinstance(ret, rtype):
                raise TypeError(f"expected {_tname(rtype)} typed return from "
                        f"'func' for consistency, got {_tname(type(ret))})")
            return ret

        # Do me.
        elts = zip(*(x._cells for x in xs))
        rshape = xs[0].shape
        if over_field:
            # Eval the points now to find rtype and chuck it in a matrix.
            cells = tuple(f(y) for y in elts)
        else:
            # Otherwise, need to handle the possibility of a matrix return. If a
            # matrix is returned, they must all be the same shape and they are
            # appended onto new axes.
            wrapme = lambda y: (single(z, field=cls.field) for z in y)
            cells = tuple(f(wrapme(y)) for y in elts)
            if issubclass(rtype, Matrix):
                # To maintain correct memory ordering we cant just concat the
                # cells, we gotta grab the first from each, then the second, etc.
                cells = tuple(mat._cells[off]
                        for off in rtype.shape.offsets
                        for mat in cells)
                rshape = rshape.insert(rtype.ndim, rtype.shape)
                rtype = rtype.field

        if rtype is None:
            rtype = cls.field
        return Matrix[rtype, rshape](cells)

    @classmethod
    def applyto(cls, func, *xs, rtype=None):
        """
        Constructs a matrix from the results of 'func(a, b, ...)' for all zipped
        elements in '*xs'. If 'rtype' is non-none, hints/enforces the return type
        from 'func'. If 'func' returns a non-single matrix, the shape of the
        return will have these elements appended into new axes.
        """
        if not callable(func):
            raise TypeError("expected callable 'func', got "
                    f"{_tname(type(func))}")
        if rtype is not None and isinstance(rtype, type):
            raise TypeError("expected none or type for 'rtype', got "
                    f"{_tname(type(rtype))}")
        return cls._apply(func, *xs, rtype=rtype, over_field=False)

    def apply(s, func, *os, rtype=None):
        """
        Alias for 'M.applyto(func, s, *os, rtype=rtype)'.
        """
        return type(s).applyto(func, s, *os, rtype=rtype)


    def _fold(s, func, seed=NO_SEED, axis=None, right=False, over_field=True):
        # `over_field` is for a field returning func which operates directly
        # on the field elements, not on them wrapped in a single matrix.

        if right:
            def f(a, b):
                return func(b, a)
            order = reversed
        else:
            f = func
            order = lambda x: x

        if s.isempty and seed is NO_SEED:
            raise TypeError("must specify 'seed' when folding over empty")

        if axis is None:
            rawcells = order(s._cells)
            cells = rawcells if over_field else map(single, rawcells)
            for x in cells:
                if seed is NO_SEED:
                    seed = x
                    continue
                seed = f(seed, x)
            return single(seed) if over_field else seed
        for x in order(s.along(axis)):
            if seed is NO_SEED:
                seed = x
                continue
            seed = x._apply(f, seed, x, over_field=over_field)
        return seed

    def fold(s, func, seed=NO_SEED, axis=None, right=False):
        """
        Constructs a matrix from the results of sequentially evaluating 'func'
        with the running value and the next value. Looks like:
            'func( ... func(func(m[0], m[1]), m[2]), ... m[-1])'
        Or if 'right':
            'func(m[0], ... func(m[-3], func(m[-2], m[-1])) ... )'
        If 'seed' is not none, acts as-if seed was inserted at the start of the
        sequence (or the end if 'right'). 'seed' may be anything which satisfies
        the function, and it may be a matrix of the correct perpendicular size to
        seed a fold along an axis with different values for each run. If 'axis'
        is none, this folding is performed along the ravelled array, otherwise it
        is performed along that axis in parallel.
        """
        if not callable(func):
            raise TypeError("expected callable 'func', got "
                    f"{_tname(type(func))}")
        if axis is not None:
            if not isinstance(axis, int):
                raise TypeError("expected an integer axis, got "
                        f"{_tname(type(axis))}")
            if axis < 0:
                raise ValueError(f"axis cannot be negative, got {axis}")
        return s._fold(func, seed=seed, axis=axis, right=right, over_field=False)


    def __pos__(s):
        """
        Element-wise NOTHING.
        """
        return s
    def __neg__(s):
        """
        Element-wise negation.
        """
        f = lambda x: s._f("sub")(s._f("zero"), x)
        return s._apply(f, s)
    def __abs__(s):
        """
        Element-wise absolution.
        """
        return s._apply(s._f("absolute"), s)

    def __add__(s, o):
        """
        Element-wise addition.
        """
        return s._apply(s._f("add"), s, o)
    def __radd__(s, o):
        return s._apply(lambda a, b: s._f("add")(b, a), s, o)
    def __sub__(s, o):
        """
        Element-wise subtraction.
        """
        return s._apply(s._f("sub"), s, o)
    def __rsub__(s, o):
        return s._apply(lambda a, b: s._f("sub")(b, a), s, o)
    def __mul__(s, o):
        """
        Element-wise multiplication (use '@' for matrix multiplication).
        """
        return s._apply(s._f("mul"), s, o)
    def __rmul__(s, o):
        return s._apply(lambda a, b: s._f("mul")(b, a), s, o)
    def __truediv__(s, o):
        """
        Element-wise division.
        """
        return s._apply(s._f("div"), s, o)
    def __rtruediv__(s, o):
        return s._apply(lambda a, b: s._f("div")(b, a), s, o)
    def __pow__(s, o):
        """
        Element-wise power.
        """
        return s._apply(s._f("power"), s, o)
    def __rpow__(s, o):
        return s._apply(lambda a, b: s._f("power")(b, a), s, o)

    @_instconst
    def sqrt(s):
        """
        Element-wise square root.
        """
        s._need("from_int", "to represent 2")
        two = s._f("from_int")(2)
        return s._apply(lambda x: s._f("root")(x, two), s)
    @_instconst
    def cbrt(s):
        """
        Element-wise cube root.
        """
        s._need("from_int", "to represent 3")
        three = s._f("from_int")(3)
        return s._apply(lambda x: s._f("root")(x, three), s)

    def root(s, n):
        """
        Element-wise nth root.
        """
        return s._apply(s._f("root"), s, n)

    @_instconst
    def exp(s):
        """
        Element-wise natural exponential.
        """
        base = s.e._cells[0]
        return s._apply(lambda x: s._f("power")(base, x), s)
    @_instconst
    def exp2(s):
        """
        Element-wise base-2 exponential.
        """
        s._need("from_int", "to represent 2")
        base = s._f("from_int")(2)
        return s._apply(lambda x: s._f("power")(base, x), s)
    @_instconst
    def exp10(s):
        """
        Element-wise base-10 exponential.
        """
        s._need("from_int", "to represent 10")
        base = s._f("from_int")(10)
        return s._apply(lambda x: s._f("power")(base, x), s)

    @_instconst
    def ln(s):
        """
        Element-wise natural logarithm.
        """
        base = s.e._cells[0]
        return s._apply(lambda x: s._f("log")(base, x), s)
    @_instconst
    def log2(s):
        """
        Element-wise base-2 logarithm.
        """
        s._need("from_int", "to represent 2")
        base = s._f("from_int")(2)
        return s._apply(lambda x: s._f("log")(base, x), s)
    @_instconst
    def log10(s):
        """
        Element-wise base-10 logarithm.
        """
        s._need("from_int", "to represent 10")
        base = s._f("from_int")(10)
        return s._apply(lambda x: s._f("log")(base, x), s)
    def log(s, base):
        """
        Element-wise base-specified logarithm.
        """
        return s._apply(lambda a, b: s._f("log")(b, a), s, base)


    @_instconst
    def sin(s):
        """
        Element-wise trigonometric sine.
        """
        return s._apply(s._f("sin"), s)
    @_instconst
    def cos(s):
        """
        Element-wise trigonometric cosine.
        """
        return s._apply(s._f("cos"), s)
    @_instconst
    def tan(s):
        """
        Element-wise trigonometric tangent.
        """
        return s._apply(s._f("tan"), s)

    @_instconst
    def asin(s):
        """
        Element-wise trigonometric inverse-sine.
        """
        return s._apply(s._f("asin"), s)
    @_instconst
    def acos(s):
        """
        Element-wise trigonometric inverse-cosine.
        """
        return s._apply(s._f("acos"), s)
    @_instconst
    def atan(s):
        """
        Element-wise trigonometric inverse-tangent.
        """
        return s._apply(s._f("atan"), s)

    @_instconst
    def torad(s):
        """
        Converts degrees to radians, alias for 'm * (180/pi)'.
        """
        # maybe preverse largest subproducts.
        return s / (180 / s.pi)
    @_instconst
    def todeg(s):
        """
        Converts radians to degrees, alias for 'm * (pi/180)'.
        """
        return s * (180 / s.pi)

    @_instconst
    def sind(s):
        """
        Element-wise trigonometric sine, with input in degrees.
        """
        return s._apply(s._f("sin"), s.torad)
    @_instconst
    def cosd(s):
        """
        Element-wise trigonometric cosine, with input in degrees.
        """
        return s._apply(s._f("cos"), s.torad)
    @_instconst
    def tand(s):
        """
        Element-wise trigonometric tangent, with input in degrees.
        """
        return s._apply(s._f("tan"), s.torad)

    @_instconst
    def asind(s):
        """
        Element-wise trigonometric inverse-sine, with output in degrees.
        """
        return s._apply(s._f("asin"), s).todeg
    @_instconst
    def acosd(s):
        """
        Element-wise trigonometric inverse-cosine, with output in degrees.
        """
        return s._apply(s._f("acos"), s).todeg
    @_instconst
    def atand(s):
        """
        Element-wise trigonometric inverse-tangent, with output in degrees.
        """
        return s._apply(s._f("atan"), s).todeg


    def diff(s, x):
        """
        Element-wise derivative with respect to 'x'.
        """
        s, x = cls.cast(s, x)
        return s._apply(s._f("diff"), s, x)

    def intt(s, x, *bounds):
        """
        Element-wise integral with respect to 'x'. If bounds are provided,
        evaluates the definite integral.
        """
        bounds = _maybe_unpack(bounds)
        if not bounds:
            s, x = cls.cast(s, x)
            return s._apply(s._f("intt"), s, x)
        if len(bounds) != 2:
            raise TypeError(f"must specify 0 or 2 bounds, got {len(bounds)}")
        lo, hi = bounds
        s, x, lo, hi = cls.cast(s, x, lo, hi)
        return s._apply(s._f("def_intt"), x, lo, hi)


    @_instconst
    def conj(s):
        """
        Element-wise complex conjugate.
        """
        return s._apply(s._f("conj"), s)
    @_instconst
    def real(s):
        """
        Element-wise take-real.
        """
        if s.isempty: # avoid divving by non-empty.
            return s
        return (s + s.conj) / 2
    @_instconst
    def imag(s):
        """
        Element-wise take-imaginary.
        """
        if s.isempty:
            return s
        return (s - s.conj) / (2 * s.i)

    @_instconst
    def abs(s):
        """
        Element-wise absolution.
        """
        return s._apply(s._f("absolute"), s)
    @_instconst
    def sign(s):
        """
        Evaluates to the integer -1, 0, or 1 corresponding to <0, =0, and >0.
        Throws if unable to categorise.
        """
        if not s.issingle:
            raise NotImplementedError("lemme whip up specialised bool first")
        neg = bool(s <= 0)
        pos = bool(s >= 0)
        if neg + pos == 0:
            raise ValueError("could not determine sign")
        return pos - neg # one of -1, 0, or 1.


    def __eq__(s, o):
        """
        Element-wise equality (cast return to bool to determine if all pairs are
        equal).
        """
        return s._apply(s._f("eq"), s, o)
    def __ne__(s, o):
        return s._apply(lambda a, b: not s._f("eq")(a, b), s, o)
    def __lt__(s, o):
        """
        Element-wise ordering (cast return to bool to determine if all pairs are
        ordered).
        """
        return s._apply(s._f("lt"), s, o)
    def __le__(s, o):
        f = lambda a, b: s._f("eq")(a, b) or s._f("lt")(a, b)
        return s._apply(f, s, o)
    def __gt__(s, o):
        return s._apply(lambda a, b: s._f("lt")(b, a), s, o)
    def __ge__(s, o):
        f = lambda a, b: s._f("eq")(a, b) or s._f("lt")(b, a)
        return s._apply(f, s, o)

    def issame(s, o):
        """
        Element-wise identical check. Note this is different to '==' (which
        checks for equivalent values, and may be different than identical
        values).
        """
        return s._apply(s._f("issame"), s, o)


    @_instconst
    def det(s):
        """
        Determinant, for 2D square matrices.
        """
        if s.ndim > 2:
            raise TypeError("only 2D matrices have a determinant, got "
                    f"{s.shape}")
        if not s.issquare:
            raise TypeError(f"only square matrices a determinant, got {s.shape}")

        if s.isempty:
            return 1 # det([]) defined as 1.
        if s.issingle:
            return s # det(x) = x.

        zero = s._f("zero")
        mul = s._f("mul")
        add = s._f("add")
        sub = s._f("sub")

        def submatrix(cells, size, row):
            return [cells[i + j*size]
                    for j in range(1, size)
                    for i in range(size)
                    if i != row]

        def determinant(cells, size):
            if size == 2:
                a, c, b, d = cells
                return sub(mul(a, d), mul(b, c))
            det = zero
            for row in range(size):
                subcells = submatrix(cells, size, row)
                subsize = size - 1
                subdet = determinant(subcells, subsize)
                if row & 1:
                    det = sub(det, mul(cells[row], subdet))
                else:
                    det = add(det, mul(cells[row], subdet))
            return det

        return single(determinant(s._cells, s.shape[0]), field=s.field)

    @_instconst
    def trace(s):
        """
        Sum of diagonal elements, for 2D square matrices.
        """
        if s.ndim > 2:
            raise TypeError(f"only 2D matrices have a trace, got {s.shape}")
        if not s.issquare:
            raise TypeError(f"only square matrices have a trace, got {s.shape}")
        return s.diag.summ

    @_instconst
    def mag(s):
        """
        Euclidean distance, for vectors.
        """
        if not s.isvec:
            raise TypeError(f"only vectors have a magnitude, got {s.shape}")
        return (s & s).sqrt

    def __and__(s, o):
        """
        Vector dot product.
        """
        o, = s.cast(o)
        if not s.isvec or not o.isvec:
            raise TypeError(f"only vectors have a dot product, got {s.shape} "
                    f"and {o.shape}")
        if s.size != o.size:
            raise TypeError("only equal-length vectors have a dot product, "
                    f"got {s.shape} and {o.shape}")
        dot = s._f("zero")
        if not s.shape:
            return single(dot, field=s.field)
        mul = s._f("mul")
        add = s._f("add")
        for a, b in zip(s._cells, o._cells):
            dot = add(dot, mul(a, b))
        return single(dot, field=s.field)
    def __rand__(s, o):
        o, = s.cast(o)
        return o.__and__(s)

    def __or__(s, o):
        """
        3-element vector cross product.
        """
        o, = s.cast(o)
        if not s.isvec or not o.isvec:
            raise TypeError(f"only vectors have a cross product, got {s.shape} "
                    f"and {o.shape}")
        if s.size != 3 or o.size != 3:
            raise TypeError("only 3-element vectors have a cross product, got "
                    f"{s.shape} and {o.shape}")
        ax, ay, az = s._cells
        bx, by, bz = o._cells
        mul = s._f("mul")
        sub = s._f("sub")
        cells = (
            sub(mul(ay, bz), mul(az, by)),
            sub(mul(az, bx), mul(ax, bz)),
            sub(mul(ax, by), mul(ay, bx)),
        )
        # If both vectors are the same shape, keep it. Otherwise use colvec.
        if s.shape == o.shape:
            newshape = s.shape
        else:
            newshape = (3, )
        return Matrix[s.field, newshape](cells)
    def __ror__(s, o):
        o, = s.cast(o)
        return o.__or__(s)

    def __matmul__(s, o):
        """
        Matrix multiplication.
        """
        if s.ndim > 2 or o.ndim > 2:
            raise TypeError(f"only 2D matrices have matrix multiplication, got "
                    f"{s.shape} and {o.shape}")
        o, = s.cast(o)
        if s.shape[1] != o.shape[0]:
            raise TypeError("need equal inner dimension lengths for matrix "
                    f"multiplication, got {s.shape} @ {o.shape}")
        newshape = Shape(s.shape[0], o.shape[1])
        if not newshape:
            return empty(s.field)
        mul = s._f("mul")
        add = s._f("add")
        cells = [s._f("zero")] * newshape.size
        # blazingly fast new matrix multiplication algorithm scientists are
        # dubbing the "naive method" (i think it means really smart).
        for i in range(s.shape[0]):
            for j in range(o.shape[1]):
                r_off = newshape.offset(i, j)
                for k in range(s.shape[1]):
                    s_off = s.shape.offset(i, k)
                    o_off = o.shape.offset(k, j)
                    prod = mul(s._cells[s_off], o._cells[o_off])
                    cells[r_off] = add(cells[r_off], prod)
        return Matrix[s.field, newshape](cells)
    def __rmatmul__(s, o):
        o, = s.cast(o)
        return o.__matmul__(s)

    def __xor__(s, exp):
        """
        Matrix power (repeated self matrix multiplication, possibly inversed).
        """
        if isinstance(exp, Matrix) and exp.issingle:
            exp = int(exp)
        if not isinstance(exp, int):
            raise TypeError("expected an integer exponent, got "
                    f"{_tname(type(exp))}")
        if not s.issquare:
            raise TypeError("only square matrices have exponentiation, got "
                    f"{s.shape})")
        if exp < 0:
            return s.inv ^ (-exp)
        power = s.eye
        running = s
        while True:
            if (exp & 1):
                power @= running
            exp >>= 1
            if not exp:
                break
            running @= running
        return power


    @_instconst
    def rref(s):
        """
        Reduced row echelon form, for 2D matrices.
        """
        if s.ndim > 2:
            raise TypeError(f"only 2D matrices can be rrefed, got {s.shape}")

        eqz = lambda x: s._f("eq")(x, s._f("zero"))
        eqo = lambda x: s._f("eq")(x, s._f("one"))
        add = lambda a, b: s._f("add")(a, b)
        mul = lambda a, b: s._f("mul")(a, b)
        neg = lambda x: s._f("sub")(s._f("zero"), x)
        rec = lambda x: s._f("div")(s._f("one"), x)

        cells = list(s._cells)
        rows, cols = s.shape[0], s.shape[1]

        def row_swap(row1, row2):
            for j in range(cols):
                k1 = row1 + j * rows
                k2 = row2 + j * rows
                cells[k1], cells[k2] = cells[k2], cells[k1]

        def row_mul(row, by):
            for j in range(cols):
                idx = row + j * rows
                cells[idx] = mul(by, cells[idx])

        def row_add(src, dst, by):
            for i in range(cols):
                src_k = src + i * rows
                dst_k = dst + i * rows
                cells[dst_k] = add(cells[dst_k], mul(by, cells[src_k]))

        lead = 0
        for r in range(rows):
            if lead >= cols:
                break

            i = r
            while eqz(cells[i + lead * rows]):
                i += 1
                if i == rows:
                    i = r
                    lead += 1
                    if lead == cols:
                        break
            if lead == cols:
                break
            row_swap(i, r)

            idx = r + lead * rows
            pivot_value = cells[idx]
            if not eqz(pivot_value):
                row_mul(r, rec(pivot_value))
            # Check its 1.
            pivot_value = cells[idx]
            if not eqo(pivot_value):
                raise ValueError(f"couldn't make cell =1, cell is: "
                        f"{repr(pivot_value)}")

            for i in range(rows):
                if i != r:
                    idx = i + lead * rows
                    row_lead_value = cells[idx]
                    if not eqz(row_lead_value):
                        row_add(r, i, neg(row_lead_value))
                    # Check its 0.
                    row_lead_value = cells[idx]
                    if not eqz(row_lead_value):
                        raise ValueError("couldn't make cell =0, cell is: "
                                f"{repr(row_lead_value)}")

            lead += 1

        return Matrix[s.field, s.shape](cells)

    @_instconst
    def pivots(s):
        """
        Tuple of RREF pivot column indices.
        """
        rref = s.rref
        return tuple(i for i, c in enumerate(rref.cols)
                if 1 == sum(not s._f("eq")(x, s._f("zero")) for x in c)
                and 1 == sum(not s._f("eq")(x, s._f("one")) for x in c))

    @_instconst
    def nonpivots(s):
        """
        Tuple of RREF non-pivot column indices.
        """
        return tuple(j for j in range(s.shape[1]) if j not in s.pivots)


    @_instconst
    def colspace(s):
        """
        Tuple of basis vectors for column space.
        """
        return tuple(s.cols[p] for p in s.pivots)

    @_instconst
    def rowspace(s):
        """
        Tuple of basis vectors for row space.
        """
        sys = s.rref
        nonzeros = (i for i, r in enumerate(sys.rows)
                    if any(s._f("eq")(x, s._f("zero")) for x in r))
        return tuple(sys.rows[i].T for i in nonzeros)

    @_instconst
    def nullspace(s):
        """
        Tuple of basis vectors for null space.
        """
        sys = s.rref # implied zero-vec augment.
        def find_first_one(xs):
            for i, x in enumerate(xs):
                if s._f("eq")(x, s._f("one")):
                    return i
            return None
        pivotat = tuple(find_first_one(sys.rows[i]) for i in range(s.shape[0]))
        basis = [[s._f("zero")]*s.shape[1] for _ in sys.nonpivots]
        for n, j in enumerate(sys.nonpivots):
            for i in range(s.shape[0]):
                if pivotat[i] is None or pivotat[i] > j:
                    basis[n][j] = 1
                    break
                basis[n][pivotat[i]] = -sys.at[i, j]
        return tuple(Matrix[s.field, (s.shape[1], )](x) for x in basis)



    @_instconst
    def summ(s):
        """
        Sum of all elements, alias for 'm.summ_along(None)'.
        """
        return s.summ_along(None)
    def summ_along(s, axis):
        """
        Additive sum of the values along the given axis. If 'axis' is none,
        returns the sum over all elements.
        """
        return s._fold(s._f("add"), s._f("zero"), axis=axis)

    @_instconst
    def prod(s):
        """
        Product of all elements, alias for 'm.prod_along(None)'.
        """
        return s.prod_along(None)
    def prod_along(s, axis):
        """
        Multiplicative product of the values along the given axis. If 'axis' is
        none, returns the product over all elements.
        """
        return s._fold(s._f("mul"), s._f("one"), axis=axis)

    @_instconst
    def minn(s):
        """
        Minimum of all elements, alias for 'm.minn_along(None)'.
        """
        return s.minn_along(None)
    def minn_along(s, axis):
        """
        Minimum of the values along the given axis. If 'axis' is none, returns
        the minimum over all elements. In the case of ties, the earlier occurence
        is kept.
        """
        if s.isempty:
            raise TypeError("cannot find minimum of empty")
        lt = s._f("lt")
        return s._fold(lambda a, b: b if lt(b, a) else a, axis=axis)

    @_instconst
    def maxx(s):
        """
        Maximum of all elements, alias for 'm.maxx_along(None)'.
        """
        return s.maxx_along(None)
    def maxx_along(s, axis):
        """
        Maximum of the values along the given axis. If 'axis' is none, returns
        the maximum over all elements. In the case of ties, the earlier occurence
        is kept.
        """
        if s.isempty:
            raise TypeError("cannot find maximum of empty")
        lt = s._f("lt")
        return s._fold(lambda a, b: b if lt(a, b) else a, axis=axis)

    @_instconst
    def mean(s):
        """
        Arithmetic mean of all elements, alias for 'm.mean_along(None)'.
        """
        return s.mean_along(None)
    def mean_along(s, axis):
        """
        Arithmetic mean of the values along the given axis. If 'axis' is none,
        returns the arithmetic mean over all elements.
        """
        if s.isempty:
            raise TypeError("cannot find arithmetic mean of empty")
        n = s.size if axis is None else s.shape[axis]
        return s.summ_along(axis) / n
    @_instconst
    def ave(s):
        """
        Alias for 'm.mean'.
        """
        return s.mean
    def ave_along(s, axis):
        """
        Alias for 'm.mean_along(axis)'.
        """
        return s.mean_along(axis)

    @_instconst
    def geomean(s):
        """
        Geometric mean of all elements, alias for 'm.geomean_along(None)'.
        """
        return s.geomean_along(None)
    def geomean_along(s, axis):
        """
        Geometric mean of the values along the given axis. If 'axis' is none,
        returns the geometric mean over all elements.
        """
        if s.isempty:
            raise TypeError("cannot find geometric mean of empty")
        n = s.size if axis is None else s.shape[axis]
        return s.prod_along(axis).root(n)

    @_instconst
    def harmean(s):
        """
        Harmonic mean of all elements, alias for 'm.harmean_along(None)'.
        """
        return s.harmean_along(None)
    def harmean_along(s, axis):
        """
        Harmonic mean of the values along the given axis. If 'axis' is none,
        returns the harmonic mean over all elements.
        """
        if s.isempty:
            raise TypeError("cannot find harmonic mean of empty")
        n = s.size if axis is None else s.shape[axis]
        return n / (s.one / s).summ_along(axis)

    @_instconst
    def quadmean(s):
        """
        Quadratic mean (root-mean-square) of all elements, alias for
        'm.quadmean_along(None)'.
        """
        return s.quadmean_along(None)
    def quadmean_along(s, axis):
        """
        Quadratic mean (root-mean-square) of the values along the given axis. If
        'axis' is none, returns the quadratic mean over all elements.
        """
        if s.isempty:
            raise TypeError("cannot find quadratic mean of empty")
        n = s.size if axis is None else s.shape[axis]
        return ((s * s).summ_along(axis) / n).sqrt


    @_instconst
    def obj(s):
        """
        Cast a single to the object it contains.
        """
        if not s.issingle:
            raise TypeError("expected single for scalar cast to object, got "
                    f"{s.shape}")
        return s._cells[0]

    def __bool__(s):
        """
        Cast to bool, returning true iff all elements are non-zero.
        """
        return not any(s._f("eq")(x, s._f("zero")) for x in s._cells)
    def __int__(s):
        """
        Cast a single to int.
        """
        if not s.issingle:
            raise TypeError("expected single for scalar cast to int, got "
                    f"{s.shape}")
        x = s._f("to_int")(s._cells[0])
        if not isinstance(x, int):
            raise TypeError("expected 'to_int' to return an int, got "
                    f"{_tname(type(x))}")
        return x
    def __float__(s):
        """
        Cast a single to float.
        """
        if not s.issingle:
            raise TypeError("expected single for scalar cast to float, got "
                    f"{s.shape}")
        x = s._f("to_float")(s._cells[0])
        if not isinstance(x, float):
            raise TypeError("expected 'to_float' to return a float, got "
                    f"{_tname(type(x))}")
        return x
    def __complex__(s):
        """
        Cast a single to complex.
        """
        if not s.issingle:
            raise TypeError("expected single for scalar cast to complex, got "
                    f"{s.shape}")
        x = s._f("to_complex")(s._cells[0])
        if not isinstance(x, complex):
            raise TypeError("expected 'to_complex' to return a complex, got "
                    f"{_tname(type(x))}")
        return x

    def __hash__(s):
        return hash((s.shape, ) + tuple(s._f("hashed")(x) for x in s._cells))

    def __repr__(s, short=None):
        if short is None:
            short = doesdflt2short()
        short = not not short

        if s.isempty:
            return "my boy "*(not short) + "M.T."

        rep_ = s._f("rep")
        rep = lambda x: rep_(x, short)

        if s.issingle:
            return rep(s._cells[0])

        if short and not s.isvec:
            # Shorten elements of zero to a single dot.
            def repme(x):
                if repme.can_eq_zero and repme.eq(x, repme.zero):
                    return "."
                return rep(x)
            repme.can_eq_zero = True
            try:
                repme.eq = s._f("eq")
                repme.zero = s._f("zero")
            except NotImplementedError:
                repme.can_eq_zero = False
        else:
            repme = rep

        # cheeky matrix of the reps to make access easy.
        reps = Matrix[GenericField[str], s.shape](repme(x) for x in s._cells)
        width = max(len(_nonctrl(r.obj)) for r in reps._cells)
        return reps._repr_str(short, width, s.lastaxis, allow_flat=True)

    def _repr_str(s, short, width, axis, allow_flat=False):
        # this method is a helper, and is only defined for matrices over the
        # GenericField[str].

        if axis > 1:
            along = s.along(axis)
            layers = [x._repr_str(short, width, axis - 1) for x in along]
            def encapsulate(r):
                r = r.split("\n")
                s = r[0] + "".join(f"\n  {line}" for line in r[1:])
                return "[ " + s + " ]"
            layers = (encapsulate(r) for r in layers)
            return "\n".join(layers)

        # 2d print.

        # Print col vecs as rows with marked transpose, if allowed.
        suffix = ""
        height = s.shape[0]
        if allow_flat and s.iscol:
            suffix = _coloured(40, "ᵀ")
            height = s.shape[1]
        cols = []
        for i, r in enumerate(s._cells):
            if not i % height:
                cols.append([])
            cell = " " * (width - len(_nonctrl(r.obj))) + r.obj
            cols[-1].append(cell)
        rows = list(zip(*cols))
        padded = (not short) or (width > 3)
        join = lambda x: "  ".join(x) if padded else " ".join(x)
        wrap = lambda x: f"[ {x} ]" if padded else f"[{x}]"
        str_rows = (wrap(join(row)) for row in rows)
        return "\n".join(str_rows) + suffix
    if field is not GenericField[str]:
        del _repr_str


    # not an actual class, gotta fulfill the templated promises.
    ret = locals()

    # Add the methods which expose the attributes the field says to.
    for attr, rtype in field.exposes.items():
        def getter(s, attr=attr, rtype=rtype): # capture current attr and rtype.
            cells = (getattr(x, attr) for x in s._cells)
            return Matrix[rtype, s.shape](cells)
        getter = _instconst(getter)
        getter.__name__ = attr
        getter.__doc__ = getattr(field, attr).__doc__
        ret[attr] = getter

    # Done.
    return ret


Matrix.namer = lambda F, s: f"{s} Matrix[{_tname(F, quoted=False)}]"



class Single:
    def __getitem__(self, field):
        return Matrix[field, Shape.single]
Single = Single()
Single.__doc__ = """
Refers to the single (1x1) matrix type over the given field. Note this is not a
proper templated class, only a thin wrapper.
"""

def single(x, *, field=None):
    """
    Single (1x1) matrix of the given object.
    """
    if field is None:
        field = type(x)
    return Single[field]((x, ))

def issingle(x):
    """
    True iff 'x' is a matrix with only one cell.
    """
    return isinstance(x, Matrix) and x.issingle


class Empty:
    def __getitem__(self, field):
        return Matrix[field, Shape.empty]
Empty = Empty()
Empty.__doc__ = """
Refers to the empty (0x0) matrix type over the given field. Note this is not a
proper templated class, only a thin wrapper.
"""

def empty(field):
    """
    Empty (0x0) matrix over the given field.
    """
    return Empty[field](())

def isempty(x):
    """
    True iff 'x' is a matrix with no cells.
    """
    return isinstance(x, Matrix) and x.isempty



def castall(xs, broadcast=True, *, field=None):
    """
    When given an sequence of matrices, returns them cast to the same field and
    (optionally) broadcast to the same shape.
    """
    if not _iterable(xs):
        raise TypeError(f"expected an iterable for xs, got {_tname(type(xs))}")
    xs = list(xs)
    if not xs:
        return ()
    for x in xs:
        if isinstance(x, Matrix):
            Mat = type(x)
            break
    else:
        Mat = Single[_get_field(field)]
    return Mat.cast(*xs, broadcast=broadcast)



def det(x, *, field=None):
    """
    Alias for 'x.det'.
    """
    x, = castall([x], field=field)
    return x.det
def trace(x, *, field=None):
    """
    Alias for 'x.trace'.
    """
    x, = castall([x], field=field)
    return x.trace
def mag(x, *, field=None):
    """
    Alias for 'x.mag'.
    """
    x, = castall([x], field=field)
    return x.mag
def dot(x, y, *, field=None):
    """
    Alias for 'x & y'.
    """
    x, y = castall([x, y], field=field, broadcast=False)
    return x & y
def cross(x, y, *, field=None):
    """
    Alias for 'x | y'.
    """
    x, y = castall([x, y], field=field, broadcast=False)
    return x | y
def sqrt(x, *, field=None):
    """
    Alias for 'x.sqrt'.
    """
    x, = castall([x], field=field)
    return x.sqrt
def cbrt(x, *, field=None):
    """
    Alias for 'x.cbrt'.
    """
    x, = castall([x], field=field)
    return x.cbrt
def root(base, x, *, field=None):
    """
    Alias for 'x.root(base)'.
    """
    x, = castall([x], field=field)
    return x.root(base)
def exp(x, *, field=None):
    """
    Alias for 'x.exp'.
    """
    x, = castall([x], field=field)
    return x.exp
def exp2(x, *, field=None):
    """
    Alias for 'x.exp2'.
    """
    x, = castall([x], field=field)
    return x.exp2
def exp10(x, *, field=None):
    """
    Alias for 'x.exp10'.
    """
    x, = castall([x], field=field)
    return x.exp10
def ln(x, *, field=None):
    """
    Alias for 'x.ln'.
    """
    x, = castall([x], field=field)
    return x.ln
def log2(x, *, field=None):
    """
    Alias for 'x.log2'.
    """
    x, = castall([x], field=field)
    return x.log2
def log10(x, *, field=None):
    """
    Alias for 'x.log10'.
    """
    x, = castall([x], field=field)
    return x.log10
def log(base, x, *, field=None):
    """
    Alias for 'x.log(base)'.
    """
    x, = castall([x], field=field)
    return x.log(base)
def sin(x, *, field=None):
    """
    Alias for 'x.sin'.
    """
    x, = castall([x], field=field)
    return x.sin
def cos(x, *, field=None):
    """
    Alias for 'x.cos'.
    """
    x, = castall([x], field=field)
    return x.cos
def tan(x, *, field=None):
    """
    Alias for 'x.tan'.
    """
    x, = castall([x], field=field)
    return x.tan
def asin(x, *, field=None):
    """
    Alias for 'x.asin'.
    """
    x, = castall([x], field=field)
    return x.asin
def acos(x, *, field=None):
    """
    Alias for 'x.acos'.
    """
    x, = castall([x], field=field)
    return x.acos
def atan(x, *, field=None):
    """
    Alias for 'x.atan'.
    """
    x, = castall([x], field=field)
    return x.atan
def atan2(y, x, *, field=None):
    """
    Quadrant-aware 'atan(y / x)'.
    """
    y, x = castall([y, x], field=field)
    return y._apply(y._f("atan2"), y, x)
def torad(degrees, *, field=None):
    """
    Alias for 'degrees.torad'.
    """
    x, = castall([degrees], field=field)
    # maybe preverse largest subproducts.
    return x.torad
def todeg(radians, *, field=None):
    """
    Alias for 'radians.todeg'.
    """
    x, = castall([radians], field=field)
    return x.todeg
def sind(x, *, field=None):
    """
    Alias for 'x.sind'.
    """
    x, = castall([x], field=field)
    return x.sind
def cosd(x, *, field=None):
    """
    Alias for 'x.cosd'.
    """
    x, = castall([x], field=field)
    return x.cosd
def tand(x, *, field=None):
    """
    Alias for 'x.tand'.
    """
    x, = castall([x], field=field)
    return x.tand
def asind(x, *, field=None):
    """
    Alias for 'x.asind'.
    """
    x, = castall([x], field=field)
    return x.asind
def acosd(x, *, field=None):
    """
    Alias for 'x.acosd'.
    """
    x, = castall([x], field=field)
    return x.acosd
def atand(x, *, field=None):
    """
    Alias for 'x.atand'.
    """
    x, = castall([x], field=field)
    return x.atand
def atand2(y, x, *, field=None):
    """
    Quadrant-aware 'atand(y / x)'.
    """
    y, x = castall([y, x], field=field)
    return y._apply(y._f("atan2"), y, x).todeg
def diff(y, x, *, field=None):
    """
    Alias for 'y.diff(x)'.
    """
    y, x = castall([y, x], field=field)
    return y.diff(x)
def intt(y, x, *bounds, field=None):
    """
    Alias for 'y.intt(x, *bounds)'.
    """
    bounds = _maybe_unpack(bounds)
    if not bounds:
        y, x = castall([y, x], field=field)
        return y.intt(x)
    if len(bounds) != 2:
        raise TypeError(f"must specify 0 or 2 bounds, got {len(bounds)}")
    lo, hi = bounds
    y, x, lo, hi = castall([y, x, lo, hi], field=field)
    return y.intt(x, (lo, hi))
def conj(x, *, field=None):
    """
    Alias for 'x.conj'.
    """
    x, = castall([x], field=field)
    return x.conj
def real(x, *, field=None):
    """
    Alias for 'x.real'.
    """
    x, = castall([x], field=field)
    return x.real
def imag(x, *, field=None):
    """
    Alias for 'x.imag'.
    """
    x, = castall([x], field=field)
    return x.imag
def sign(x, *, field=None):
    """
    Alias for 'x.sign'.
    """
    x, = castall([x], field=field)
    return x.sign
def issame(x, y, *, field=None):
    """
    Alias for 'x.issame(y)'.
    """
    x, y = castall([x, y], field=field)
    return x.issame(y)
def rref(x, *, field=None):
    """
    Alias for 'x.rref'.
    """
    x, = castall([x], field=field)
    return x.rref
def pivots(x, *, field=None):
    """
    Alias for 'x.pivots'.
    """
    x, = castall([x], field=field)
    return x.pivots
def nonpivots(x, *, field=None):
    """
    Alias for 'x.nonpivots'.
    """
    x, = castall([x], field=field)
    return x.nonpivots
def colspace(x, *, field=None):
    """
    Alias for 'x.colspace'.
    """
    x, = castall([x], field=field)
    return x.colspace
def rowspace(x, *, field=None):
    """
    Alias for 'x.rowspace'.
    """
    x, = castall([x], field=field)
    return x.rowspace
def nullspace(x, *, field=None):
    """
    Alias for 'x.nullspace'.
    """
    x, = castall([x], field=field)
    return x.nullspace
def summ(x, axis=None, *, field=None):
    """
    Alias for 'x.summ_along(axis)'.
    """
    x, = castall([x], field=field)
    return x.summ_along(axis)
def prod(x, axis=None, *, field=None):
    """
    Alias for 'x.prod_along(axis)'.
    """
    x, = castall([x], field=field)
    return x.prod_along(axis)
def minn(x, axis=None, *, field=None):
    """
    Alias for 'x.minn_along(axis)'.
    """
    x, = castall([x], field=field)
    return x.minn_along(axis)
def maxx(x, axis=None, *, field=None):
    """
    Alias for 'x.maxx_along(axis)'.
    """
    x, = castall([x], field=field)
    return x.maxx_along(axis)
def mean(x, axis=None, *, field=None):
    """
    Alias for 'x.mean_along(axis)'.
    """
    x, = castall([x], field=field)
    return x.mean_along(axis)
def ave(x, axis=None, *, field=None):
    """
    Alias for 'x.mean_along(axis)'.
    """
    x, = castall([x], field=field)
    return x.mean_along(axis)
def geomean(x, axis=None, *, field=None):
    """
    Alias for 'x.geomean_along(axis)'.
    """
    x, = castall([x], field=field)
    return x.geomean_along(axis)
def harmean(x, axis=None, *, field=None):
    """
    Alias for 'x.harmean_along(axis)'.
    """
    x, = castall([x], field=field)
    return x.harmean_along(axis)
def quadmean(x, axis=None, *, field=None):
    """
    Alias for 'x.quadmean_along(axis)'.
    """
    x, = castall([x], field=field)
    return x.quadmean_along(axis)
def logmean(x, y, *, field=None):
    """
    Logarithmic mean of 'x' and 'y': (x - y) / ln(x / y)
    """
    x, y = castall([x, y], field=field)
    f = lambda a, b: a if (a == b) else (a - b) / (a / b).ln
    return x.apply(f, y)


def short(x, *, field=None):
    """
    Prints a short string representation of 'x'.
    """
    x, = castall([x], field=field)
    print(x.__repr__(short=True))

def long(x, *, field=None):
    """
    Prints a long string representation of 'x'.
    """
    x, = castall([x], field=field)
    print(x.__repr__(short=False))

def doesdflt2short():
    """
    True if the current default matrix print is short, false if it's long.
    """
    return doesdflt2short.v
doesdflt2short.v = True

class _Dflt2:
    def __init__(self, short, doc):
        self.short = False
        self.store = None
        self.__doc__ = doc

    def __call__(self):
        doesdflt2short.v = True

    def __enter__(self):
        self.store = doesdflt2short.v
        doesdflt2short.v = self.short
    def __exit__(self, type, value, traceback):
        if self.store is not None:
            doesdflt2short.v = self.store
            self.store = None
        return False
dflt2short = _Dflt2(True, "Changes the default matrix print to short. Can also "
        "be used as a context manager to temporarily set.")
dflt2long = _Dflt2(False, "Changes the default matrix print to long. Can also "
        "be used as a context manager to temporarily set.")


def stack(axis, *xs, field=None):
    """
    Stacks the given matrices along the given axis.
    """
    xs = _maybe_unpack(xs)
    xs = castall(xs, field=field, broadcast=False)
    field = _get_field(field, xs)
    if not isinstance(axis, int):
        raise TypeError(f"expected an integer axis, got {_tname(type(axis))}")
    if axis < 0:
        raise ValueError(f"axis cannot be negative, got {axis}")


    # Stacking empty do NOTHIGN and stacking nothign do EMPTY.
    if not xs:
        return empty(field)
    if all(x.isempty for x in xs):
        return empty(field)

    # Huge stack of one item.
    if len(xs) == 1:
        return xs[0]

    # Check perpendicular sizes.
    perpshape = xs[0].shape.withaxis(axis, 1)
    newshape = perpshape.withaxis(axis, sum(x.shape[axis] for x in xs))
    for x in xs:
        if x.shape.withaxis(axis, 1) != perpshape:
            raise TypeError(f"expected a perpendicular shape of {perpshape} for "
                    f"stacking, got {x.shape}")
    # Get lookups.
    offset = [0] * len(xs)
    for i, x in enumerate(xs[:-1]):
        offset[i + 1] = offset[i] + x.shape[axis]
    offset.append(newshape[axis] + 1) # unreachable offset.
    lookup = [0] * newshape[axis]
    for i in range(1, newshape[axis]):
        lookup[i] = lookup[i - 1] + (i >= offset[lookup[i - 1] + 1])
    # Get the cells.
    cells = [0] * newshape.size
    for new in newshape.indices:
        i = 0 if axis >= len(new) else new[axis]
        j = lookup[i]
        k = i - offset[j]
        subcells = xs[j]._cells
        subshape = xs[j].shape
        subidx = tuple(k if ii == axis else i for ii, i in enumerate(new))
        cells[newshape.offset(new)] = subcells[subshape.offset(subidx)]
    return Matrix[field, newshape](cells)

def vstack(*xs, field=None):
    """
    Vertically concatenates the given matrices.
    """
    return stack(0, *xs, field=field)

def hstack(*xs, field=None):
    """
    Horizontally concatenates the given matrices.
    """
    return stack(1, *xs, field=field)

def ravel(*xs, field=None):
    """
    Concatenated vector of the raveled cells of all given matrices.
    """
    xs = _maybe_unpack(xs)
    xs = castall(xs, field=field, broadcast=False)
    field = _get_field(field, xs)
    size = sum(x.size for x in xs)
    cells = [None] * size
    off = 0
    for x in xs:
        cells[off:off + x.size] = x._cells
        off += x.size
    return Matrix[field, (size, )](cells)

def rep(x, *counts, field=None):
    """
    Repeats the given matrix the given number of times along each dimension.
    """
    x, = castall([x], field=field)
    return x.rep(*counts)

def rep_along(x, axis, count, *, field=None):
    """
    Repeats the given matrix 'count' times along 'axis'.
    """
    x, = castall([x], field=field)
    return x.rep_along(axis, count)

def tovec(*xs, field=None):
    """
    Concatenated column vector of the given values and iterables.
    """
    # dont maybe unpack xs lmao.
    cells = []
    for x in xs:
        if not _iterable(x):
            x, = castall([x], field=field)
        if isinstance(x, Matrix):
            if not x.isvec:
                raise TypeError("expected vectors to concatenate into column, "
                        f"got {x.shape}")
            cells.extend(x._cells)
            if field is None:
                field = x.field
            continue
        for y in x:
            if not isinstance(y, Matrix):
                y, = castall([y], field=field)
            if not y.issingle:
                raise TypeError("expected iterable to contain singles to "
                        f"concatenate into column, got {y.shape}")
            if field is None:
                field = y.field
            cells.append(y._cells[0])
    field = _get_field(field)
    return Matrix[field, (len(cells), )](cells)


def eye(n, *, field=None):
    """
    2D identity matrix, of the given size.
    """
    field = _get_field(field)
    if not isinstance(n, int):
        raise TypeError(f"expected an integer size, got {_tname(type(n))}")
    if n < 0:
        raise ValueError(f"cannot have negative size, got: {n}")
    return Matrix[field, (n, n)].eye

def zeros(*counts, field=None):
    """
    Zero-filled matrix of the given size, defaulting to square if only one axis
    length is given.
    """
    field = _get_field(field)
    counts = _maybe_unpack(counts)
    for count in counts:
        if not isinstance(count, int):
            raise TypeError("expected an integer count, got "
                    f"{_tname(type(count))}")
    if any(count < 0 for count in counts):
        raise ValueError(f"cannot have negative counts, got: {counts}")
    if len(counts) == 1:
        counts *= 2
    return Matrix[field, counts].zeros

def ones(*counts, field=None):
    """
    One-filled matrix of the given size, defaulting to square if only one axis
    length is given.
    """
    field = _get_field(field)
    counts = _maybe_unpack(counts)
    for count in counts:
        if not isinstance(count, int):
            raise TypeError("expected an integer count, got "
                    f"{_tname(type(count))}")
    if any(count < 0 for count in counts):
        raise ValueError(f"cannot have negative counts, got: {counts}")
    if len(counts) == 1:
        counts *= 2
    return Matrix[field, counts].ones

def diag(*xs, field=None):
    """
    Matrix with a diagonal of the given elements and zeros elsewhere.
    """
    diagvec = tovec(*xs, field=field)
    n = diagvec.size
    Mat = Matrix[diagvec.field, (n, n)]
    if n == 1:
        cells = diagvec._cells
    else:
        cells = [Mat._f("zero")] * (n*n) if n else []
        for i in range(n):
            cells[i*n + i] = diagvec._cells[i]
    return Mat(cells)

def linspace(x0, x1, n, *, field=None):
    """
    Vector of 'n' linearly spaced values starting at 'x0' and ending at 'x1'.
    """
    if not isinstance(n, int):
        raise TypeError(f"expected an integer n, got {_tname(type(n))}")
    if n < 0:
        raise ValueError(f"expected n >= 0, got: {n}")
    x0, x1 = castall([x0, x1], field=field)
    field = _get_field(field, [x0, x1])
    if n == 0:
        return empty[field]
    if x0.isempty:
        raise TypeError("'x0' cannot be empty when n>0")
    if x1.isempty:
        raise TypeError("'x1' cannot be empty when n>0")
    if n == 1:
        return x0
    # Lerp each value.
    step = (x1 - x0) / (n - 1)
    x = (x0 + step * i for i in range(n))
    # Stack these along a new axis.
    return stack(x0.ndim, x)

def logspace(x0, x1, n, *, field=None):
    """
    Vector of 'n' logarithmically spaced values starting at 'x0' and ending at
    'x1'.
    """
    if not isinstance(n, int):
        raise TypeError(f"expected an integer n, got {_tname(type(n))}")
    if n < 0:
        raise ValueError(f"expected n >= 0, got: {n}")
    x0, x1 = castall([x0, x1], field=field)
    field = _get_field(field, [x0, x1])
    if n == 0:
        return empty[field]
    if x0.isempty:
        raise TypeError("'x0' cannot be empty when n>0")
    if x1.isempty:
        raise TypeError("'x1' cannot be empty when n>0")
    if n == 1:
        return x0
    # Just log, do linear, then exp.
    x0 = x0.ln
    x1 = x1.ln
    step = (x1 - x0) / (n - 1)
    x = ((x0 + step * i).exp for i in range(n))
    return stack(x0.ndim, x)


def lerp(x, X, Y, extend=False, *, field=None):
    """
    1-dimensional linear interpolation of 'X' and 'Y' at the points specified in
    'x'. 'X' must be a vector, but all axes other than the first along 'Y' are
    considered the y-values and the interpolated y-value for all elements of 'x'
    will be returned. If 'extend' is true, allows 'x' to be outside the range of
    'X'.
    """
    x, X, Y = castall([x, X, Y], field=field, broadcast=False)
    if not X.isvec:
        raise TypeError(f"expected a vector 'X', got {X.shape}")
    if X.size != Y.shape[0]:
        raise TypeError("expected one 'Y' data point for each 'X', got "
                f"{X.size} x-values and {Y.shape[0]} y-values")
    Ys = [y.dropaxis(0) for y in Y.along(0)]
    if x.isempty:
        return x
    if X.size == 0:
        raise TypeError("expected non-empty 'X' and 'Y' for a non-empty 'x'")
    xlo = X[0]
    xhi = X[-1]
    ascending = (xlo < xhi)
    if not ascending:
        xlo, xhi = xhi, xlo
    if X.size == 1:
        def flat_interpolate_so_its_not_really_an_interpolate_at_all(z):
            if not extend and z != X:
                raise ValueError("expected 'x' within the range of 'X', got: "
                        f"{z}, which is outside: {xlo} .. {xhi}")
            return Ys[0]
        return x.apply(flat_interpolate_so_its_not_really_an_interpolate_at_all)
    if xlo == xhi:
        raise ValueError(f"expected unique 'X'-values, got duplicates: {xlo} "
                f"and {xhi}")
    def interpolate(z):
        a, b = 0, len(X) - 1
        xa, xb = X[a], X[b]
        while b > a + 1:
            c = (a + b) // 2
            xc = X[c]
            if xc == xa:
                raise ValueError("expected unique 'X'-values, got duplicates: "
                        f"{xa} and {xc}")
            if xc == xb:
                raise ValueError("expected unique 'X'-values, got duplicates: "
                        f"{xc} and {xb}")
            if xc == z:
                return Ys[c]
            if (z > xc) == ascending:
                if (xa > xc) == ascending:
                    raise ValueError("expected sorted 'X'-values, they weren't")
                a, xa = c, xc
            else:
                if (xb < xc) == ascending:
                    raise ValueError("expected sorted 'X'-values, they weren't")
                b, xb = c, xc

        if not extend and (z < xa or xb < z):
            raise ValueError(f"expected 'x' within the range of 'X', got: {z}, "
                    f"which is outside: {xlo} .. {xhi}")

        ya, yb = Ys[a], Ys[b]
        return ya + (yb - ya) * ( (z - xa) / (xb - xa) )

    return x.apply(interpolate)



def rootnr(f, df, x0, tol=1e-8, max_iters=120):
    """
    Solves the root of the given function: f(root) = 0; given its derivation
    df(x) = (d/dx f(x)) and an initial guess for the root ~= x0.
    """
    if isinstance(x0, Matrix):
        if not x0.issingle:
            raise TypeError("cannot root find a function with a non-single "
                    f"matrix input, got {x0.shape} shaped initial root guess")
    x = x0
    for _ in range(max_iters):
        fx = f(x)
        if isinstance(fx, Matrix):
            if not fx.issingle:
                raise TypeError("cannot root find a non-single matrix "
                        f"function, got {fx.shape}")
        dfx = df(x)
        if isinstance(dfx, Matrix):
            if not dfx.issingle:
                raise TypeError("cannot root find a non-single matrix "
                        f"function derivative, got {dfx.shape}")
        if dfx == 0:
            raise ZeroDivisionError("(d/dx f(x)) =0")
        x_new = x - fx / dfx
        if abs(x_new - x) < tol:
            return x_new
        x = x_new
    raise RuntimeError(f"no convergence within {max_iter} iterations")

def rootbi(f, a, b, tol=1e-8, max_iters=120):
    """
    Solves the root of the given function: f(root) = 0; given a lower and upper
    bound of the root, a <= root <= b, where sign(f(a)) != sign(f(b)).
    """
    if isinstance(a, Matrix):
        if not a.issingle:
            raise TypeError("cannot root find a function with a non-single "
                    f"matrix input, got {a.shape} shaped initial 'a'")
    if isinstance(b, Matrix):
        if not b.issingle:
            raise TypeError("cannot root find a function with a non-single "
                    f"matrix input, got {b.shape} shaped initial 'b'")
    fa = f(a)
    fb = f(b)
    if isinstance(fa, Matrix):
        if not fa.issingle:
            raise TypeError("cannot root find a non-single matrix function, got "
                    f"{fa.shape}")
    if isinstance(fb, Matrix):
        if not fb.issingle:
            raise TypeError("cannot root find a non-single matrix function, got "
                    f"{fb.shape}")
    if abs(fa) < tol:
        return a
    if abs(fb) < tol:
        return b
    def sign(x, fx):
        neg = bool(fx <= 0)
        pos = bool(fx >= 0)
        if neg + pos == 0:
            raise ValueError("could not determine sign of f(x), got: "
                    f"f({repr(x)}) = {repr(fx)}")
        return pos - neg
    if sign(a, fa) == sign(b, fb):
        raise ValueError("f(x) must have opposite signs at a and b, got: "
                f"f({repr(a)}) = {repr(fa)}, and: f({repr(b)}) = {repr(fb)}")
    for _ in range(max_iters):
        c = (a + b) / 2
        fc = f(c)
        if isinstance(fc, Matrix):
            if not fc.issingle:
                raise TypeError("cannot root find a non-single matrix "
                        f"function, got {fc.shape}")
        if abs(fc) < tol or (b - a) / 2 < tol:
            return c
        if sign(c, fc) == sign(a, fa):
            a, fa = c, fc
        else:
            b, fb = c, fc
    raise RuntimeError(f"no convergence within {max_iter} iterations")



def oderk4(f, T, x0, *, field=None):
    """
    Solves the given differential: f(t, x) = (d/dt x); over the given time values
    and using the given initial state x0. The time must be a vector, but the
    state may be matrix of any size and the solution states will be stacked along
    a new axis.
    """
    if not isinstance(T, Matrix):
        if not _iterable(T):
            raise TypeError(f"expected iterable for 'T', got {_tname(type(T))}")
        T = tovec(T, field=field)
    if not T.isvec:
        raise TypeError(f"expected vector for 'T', got {T.shape}")
    x0, = castall([x0], field=field)
    field = _get_field(field, [x0])
    if T.isempty:
        return empty(field)
    X = [None] * len(T)
    X[0] = x0
    for i in range(1, len(T)):
        x = X[i - 1]
        t = T[i]
        h = T[i] - T[i - 1]
        k1 = h * f(t, x)
        k2 = h * f(t + h/2, x + k1/2)
        k3 = h * f(t + h/2, x + k2/2)
        k4 = h * f(t + h, x + k3)
        X[i] = x + (k1 + 2*k2 + 2*k3 + k4) / 6
    return stack(x0.ndim, X)





def mvars(long=None):
    """
    Prints all matrix variables in the current space.
    """
    from syntax import KW_PREV

    if long is None:
        long = not doesdflt2short()

    # Trim down to the matrix variables.
    space = _get_space()
    mspace = {k: v for k, v in space.items() if isinstance(v, Matrix)}
    # Dont include the "last result" variable.
    mspace.pop(KW_PREV, None)
    for name in lits._injects:
        if name not in mspace:
            continue
        if not _is_overridden(space, lits._field, name):
            mspace.pop(name, None)

    if not mspace:
        print(_coloured(245, "no matrix variables."))
    for name, value in mspace.items():
        pre = _coloured([208, 161], [name, " = "])
        pad = " " * (len(name) + len(" = "))
        mat = value.__repr__(short=not long)
        mat = mat.replace("\n", "\n" + pad)
        print(pre + mat)


def mhelp(*, field=None):
    """
    Prints the signature and doc of the functions in this file (the matrix
    functions).
    """
    field = _get_field(field)
    width = 100
    def classify(c):
        if c.isalnum() or c == "_":
            return "identifier"
        if c in ".+-*/%=&|@<>^~:":
            return "operator"
        if c in "()[]{}, ":
            return "boring"
        raise Exception(f"dunno {repr(c)}")
    def colourof(txt, prev_txt, next_txt, key, leading):
        issci = lambda t: t is not None and t[:-1].isdigit() and t.endswith("e")
        if txt.isdigit() or issci(txt):
            return 135
        if (txt == "-" or txt == "+") and issci(prev_txt):
            return 135
        if txt in {"None", "False", "True", "NO_SEED"}:
            return 135
        if key == "identifier":
            if leading and txt[0].isupper():
                return 38
            if next_txt is not None and next_txt[0] in "([":
                return 112
            if prev_txt == "." and next_txt is None:
                return 153
            return 208
        if key == "operator":
            return 161
        if key == "boring":
            return -1
        raise Exception(f"dunno {repr(key)}")
    def print_entry(name, desc):
        txts = []
        cols = []
        leading = True
        grouped = _itertools.groupby(name, key=classify)
        keytxts = [(k, "".join(g)) for k, g in grouped]
        for i in range(len(keytxts)):
            key, txt = keytxts[i]
            prev_txt = None if i == 0 else keytxts[i - 1][1]
            next_txt = None if i == len(keytxts) - 1 else keytxts[i + 1][1]
            col = colourof(txt, prev_txt, next_txt, key, leading)
            txts.append(txt)
            cols.append(col)
            leading = False
        name = _coloured(cols, txts)
        print(_entry(name, desc, width=width, pwidth=22, lead=2))

    Mat = Empty[field]
    attrs = {name: attr for name, attr in vars(Mat).items()
            if attr.__doc__ is not None
            and name != "__module__"
            and name != "__doc__"
            and name != "_tname"
            and name != "template"}

    # also chuck the other functions in this file.
    funcs = {name: obj for name, obj in globals().items()
            if obj.__doc__ is not None
            and callable(obj)
            and obj.__module__ == __name__
            and name != "Matrix"}

    # Explicitly print a couple.
    print_entry("Matrix[field, shape]", Matrix.__doc__)
    print_entry("M.field", "Cell type.")
    print_entry("M.shape", "'Shape' object with the length of each dimension.")
    attrs.pop("field", None)
    attrs.pop("shape", None)


    func_ops = {
        "__bool__": "bool",
        "__int__": "int",
        "__float__": "float",
        "__complex__": "complex",
        "__abs__": "abs",
        "__iter__": "iter",
        "__len__": "len",
        "__hash__": "hash",
        "__repr__": "repr",
    }
    un_ops = {
        "__pos__": "+",
        "__neg__": "-",
        "__invert__": "~",
    }
    bin_ops = {
        "__add__": "+",
        "__sub__": "-",
        "__mul__": "*",
        "__truediv__": "/",
        "__floordiv__": "//",
        "__mod__": "%",
        "__pow__": "**",
        "__and__": "&",
        "__or__": "|",
        "__eq__": "==",
        "__neq__": "!=",
        "__lt__": "<",
        "__le__": "<=",
        "__gt__": ">",
        "__ge__": ">=",
        "__matmul__": "@",
    }
    for name, attr in attrs.items():
        m = "m"
        if isinstance(attr, (classmethod, _classconst)):
            m = "M"
        isclassmethod = isinstance(attr, classmethod)
        isinstmethod = False
        if not isclassmethod:
            isinstmethod = callable(attr)
            isinstmethod &= not isinstance(attr, (_classconst, _instconst))

        if name in func_ops:
            expr = f"{func_ops[name]}({m})"
        elif name in un_ops:
            expr = f"{un_ops[name]}{m}"
        elif name in bin_ops:
            expr = f"{m} {bin_ops[name]} {m}"
        elif name == "__xor__":
            expr = f"{m} ^ exp"
        elif name == "at":
            expr = f"{m}.at[rows, cols, ...]"
        elif name == "__getattr__":
            expr = f"{m}.attr"
        elif isclassmethod or isinstmethod:
            if isclassmethod:
                sig = _inspect.signature(attr.__func__)
            else:
                sig = _inspect.signature(attr)
            sig = str(sig)
            sig = sig[1:-1]
            sig = sig[sig.index(",") + 1:].strip() if "," in sig else ""
            if name == "__getitem__":
                expr = f"{m}[{sig}]"
            elif name == "__call__":
                expr = f"{m}({sig})"
            else:
                expr = f"{m}.{name}({sig})"
        else:
            expr = f"{m}.{name}"
        print_entry(expr, attr.__doc__)

    for name, func in funcs.items():
        # Print `Single` just before `single()`.
        if func is single:
            print_entry("Single[field]", Single.__doc__)
        # Print `Empty` just before `empty()`.
        if func is empty:
            print_entry("Empty[field]", Empty.__doc__)

        doc = func.__doc__
        if name == "mhelp":
            doc = "Prints this message."

        sig = _inspect.signature(func)
        sig = str(sig)
        sig = sig[1:-1]
        if sig.endswith(", *, field=None"):
            sig = sig[:-len(", *, field=None")]
        elif sig.endswith("*, field=None"):
            sig = sig[:-len("*, field=None")]
        elif sig.endswith(", field=None"):
            sig = sig[:-len(", field=None")]
        expr = f"{name}({sig})"
        print_entry(expr, doc)
