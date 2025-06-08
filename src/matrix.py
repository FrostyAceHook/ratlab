import inspect as _inspect
import itertools as _itertools
import math as _math
import types as _types

from util import (
    classconst as _classconst, instconst as _instconst, tname as _tname,
    iterable as _iterable, immutable as _immutable, templated as _templated,
)


# Derive from to indicate it may implement field methods.
class Field:

    # few methods have defaults, see `ExampleField` for all methods.

    @_classconst
    def from_bool(cls, x):
        return cls.one if x else cls.zero

    @_classconst
    def consts(cls):
        return {}

    @classmethod
    def atan2(cls, y, x):
        if "pi" in cls.consts:
            pi = cls.consts["pi"]
        else:
            pi = cls.from_float(_math.pi)
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
        rep = s.repr_short if hasattr(s, "repr_short") else s.repr_long
        return rep(s)



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
    def repr_short(cls, a): # repr(a)
        return "d ... j"
    @classmethod
    def repr_long(cls, a): # long(a)
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
    def repr_short(cls, a):
        if T is bool:
            return "Y" if a.obj else "N"
        return repr(a.obj)
    @classmethod
    def repr_long(cls, a):
        if T is bool:
            return "yeagh" if a.obj else "nogh"
        return repr(a.obj)



# Current field.
def lits(field, *, space=None):
    """
    Sets the current/default field to the given field and injects constants such
    as 'e' and 'pi' into the space.
    """
    if field is not None:
        if not isinstance(field, type):
            raise TypeError(f"expected a field class, got {_tname(type(field))}")
        if not issubclass(field, Field):
            raise TypeError("expected a field class which inherits from "
                    f"{_tname(Field)}, got {_tname(field)} (which doesn't)")

    if field is not None and space is not None:
        # Inject constants, but only if they aren't currently overridden.
        OldMat = Single[lits.field] if lits.field is not None else None
        NewMat = Single[field]
        def isoverridden(name):
            if name not in space:
                return False
            if OldMat is None:
                return True
            got = space[name]
            expect = getattr(OldMat, name)
            if type(got) is not type(expect):
                return True
            return got != expect
        def inject(name):
            if isoverridden(name):
                return
            try:
                space[name] = getattr(NewMat, name)
            except NotImplementedError:
                pass
        inject("e")
        inject("pi")
    lits.field = field
lits.field = None



def _get_field(field):
    if field is None:
        if lits.field is None:
            raise RuntimeError("specify a field using 'lits'")
        field = lits.field
    return field



def _maybe_unpack(xs):
    if len(xs) == 0 or len(xs) > 1:
        return xs
    x = xs[0]
    if isinstance(x, Matrix):
        return xs
    if not _iterable(x):
        return xs
    return x if isinstance(x, list) else tuple(x)



class Shape(tuple):
    """
    Tuple of the length of each dimension.
    """

    def __new__(cls, shape):
        try:
            shape = tuple(shape)
        except Exception:
            raise TypeError("shape must be an iterable of the length of each "
                    "dimension")
        for s in shape:
            if not isinstance(s, int):
                raise TypeError("dimension lengths must be ints, got "
                        f"{_tname(type(s))}")
        if any(s < 0 for s in shape):
            raise ValueError("dimension lengths cannot be negative, got "
                    f"{shape}")
        # Trim any trailing 1s (note this collapses single to ()).
        while shape and shape[-1] == 1:
            shape = shape[:-1]
        # Collapse empty.
        if _math.prod(shape) == 0:
            shape = (0, )
        return super().__new__(cls, shape)

    @_classconst
    def empty(cls):
        return cls((0, ))
    @_classconst
    def single(cls):
        return cls(())

    @_instconst
    def size(self):
        return _math.prod(self)
    @_instconst
    def ndim(self):
        return -1 if not self else len(self)

    def __getitem__(self, axis):
        # Oob dimensions are implicitly 1 (unless empty, in which case 0).
        if not isinstance(axis, int):
            raise TypeError(f"expected an integer axis, got "
                    f"{_tname(type(axis))}")
        if axis < 0:
            raise ValueError(f"axis cannot be negative, got {axis}")
        if axis >= self.ndim:
            return 1 if self.size else 0
        return super().__getitem__(axis)

    def withaxis(self, axis, length):
        if self.isempty:
            return self
        shape = list(self) + [1] * (axis + 1 - len(self))
        shape[axis] = length
        return Shape(shape)

    def __bool__(self):
        return self.size > 0

    def __eq__(self, other):
        if not isinstance(other, tuple):
            return NotImplemented
        other = Shape(other)
        if self.ndim != other.ndim:
            return False
        return all(a == b for a, b in zip(self, other))

    def __hash__(self):
        return super().__hash__()

    @_instconst
    def isempty(self):
        return self.size == 0
    @_instconst
    def issingle(self):
        return self.size == 1
    @_instconst
    def isvec(self):
        return sum(s == 1 for s in self) >= self.ndim - 1
    @_instconst
    def iscol(self):
        return self[0] == self.size
    @_instconst
    def isrow(self):
        return self[1] == self.size
    @_instconst
    def issquare(self):
        return len(self) <= 2 and self[0] == self[1]

    def __repr__(self):
        # Pad to at-least 2d.
        shape = tuple(self)
        if self.ndim < 2:
            shape = (self[0], self[1])
        return "x".join(map(str, shape))

    def stride(self, axis):
        if not isinstance(axis, int):
            raise TypeError(f"expected an integer axis, got "
                    f"{_tname(type(axis))}")
        if axis < 0:
            raise ValueError(f"axis cannot be negative, got {axis}")
        if self.isempty:
            raise ValueError("cannot find stride of empty")
        axis = min(axis, self.ndim)
        return _math.prod(tuple(self)[:axis])


    def offset(self, *ijk):
        if self.isempty:
            raise ValueError("cannot index empty")
        ijk = _maybe_unpack(ijk)
        for ii, i in enumerate(ijk):
            if not isinstance(i, int):
                raise TypeError("expected an integer index, got "
                        f"{_tname(type(i))} at index {ii}")
        ijk = [i + (i < 0) * self[ii] for ii, i in enumerate(ijk)]
        for ii, i in enumerate(ijk):
            if i not in range(self[ii]):
                raise IndexError(f"index {i} out-of-bounds for axis {ii} with "
                        f"length {self[ii]}")
        return sum(i * self.stride(ii) for ii, i in enumerate(ijk))

    def offsets(self):
        if self.isempty:
            raise ValueError("cannot iterate empty")
        return range(self.size)

    def index(self, offset):
        if self.isempty:
            raise ValueError("cannot index empty")
        if not isinstance(offset, int):
            raise TypeError(f"expected an integer offset, got "
                    f"{_tname(type(offset))}")
        if offset < 0:
            raise ValueError(f"offset cannot be negative, got {offset}")
        if offset >= self.size:
            raise IndexError(f"offset {offset} out-of-bounds for size "
                    f"{self.size}")
        return tuple(offset % self.stride(ii) for ii in range(self.ndim))

    def indices(self):
        if self.isempty:
            raise ValueError("cannot iterate empty")
        slices = [range(s) for s in self]
        revidxs = _itertools.product(*reversed(slices))
        return (tuple(reversed(revidx)) for revidx in revidxs)




# Matrices.
@_templated(parents=Field, decorators=_immutable)
def Matrix(field, shape):
    """
    Fixed-sized n-dimensional sequence of elements. 'field' is the class of the
    elements, and 'shape' is a tuple of the length along each dimension. 'shape'
    is an empty tuple for a single element, has a zero anywhere in it for an
    empty matrix, and is implicitly infinite with appended 1s. For a typical 2x2
    matrix, the shape is (row count, col count).
    """

    if not isinstance(field, type):
        raise TypeError(f"expected a type for field, got {field} of type "
                f"{_tname(type(field))}")
    if issubclass(field, Matrix):
        raise TypeError("mate a matrix of matrices? calm down")

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
        if not hasattr(field, method):
            thing = {
                "from_int": "cannot cast from int",
                "from_float": "cannot cast from float",
                "from_complex": "cannot cast from complex",

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

                "eq": "cannot do equality",
                "lt": "cannot do ordering",

                "hashed": "cannot hash",

                "repr_short": "cannot short string represent",
                "repr_long": "cannot long string represent",
            }
            msg = f"{thing[method]} over field {_tname(field)}{extra}"
            raise NotImplementedError(msg)

    @classmethod
    def _f(cls, method):
        cls._need(method)
        return getattr(field, method)


    def __init__(self, cells):
        # `cells` is a flattened iterable of each cell, progressing through the
        # matrix in the order of shape (so col-major for 2d).
        if not _iterable(cells):
            raise TypeError(f"cells must be iterable, got {_tname(type(cells))}")

        # Often when a function returns from a new field, its not appropriately
        # wrapped, but we only make this exemption for non-field types.
        if issubclass(field, GenericField):
            cells = list(cells)
            for i in range(len(cells)):
                if isinstance(cells[i], field.T):
                    cells[i] = field(cells[i])

        cells = tuple(cells)
        for i in range(len(cells)):
            if not isinstance(cells[i], field):
                raise TypeError(f"expected cells of type {_tname(field)}, got "
                        f"{_tname(type(cells[i]))} (occured at index {i}, had "
                        f"value {cells[i]})")
        if len(cells) != shape.size:
            raise TypeError("cells must match flattened shape (expected "
                    f"{shape}={shape.size}, got {len(cells)})")

        self._cells = cells


    @_classconst
    def eye(cls):
        """
        Identity matrix.
        """
        if not cls.issquare:
            raise TypeError("only square matricies have an identity (got size "
                    f"{_size_str})")
        cells = [cls._f("zero")] * shape.size if shape else ()
        n = shape[0]
        for i in range(n):
            cells[i*n + i] = cls._f("one")
        return cls(cells)
    @_classconst
    def zeros(cls):
        """
        Zero-filled matrix.
        """
        cells = (cls._f("zero"), ) * shape.size if shape else ()
        return cls(cells)
    @_classconst
    def ones(cls):
        """
        One-filled matrix.
        """
        cells = (cls._f("one"), ) * shape.size if shape else ()
        return cls(cells)

    @_classconst
    def zero(cls):
        """
        Single zero.
        """
        return single(cls._f("zero"), field=field)
    @_classconst
    def one(cls):
        """
        Single one.
        """
        return single(cls._f("one"), field=field)
    @_classconst
    def e(cls):
        """
        Single euler's number.
        """
        if "e" in field.consts:
            e = field.consts["e"]
        else:
            cls._need("from_float", "to represent e")
            e = cls._f("from_float")(_math.e)
        return single(e, field=field)
    @_classconst
    def pi(cls):
        """
        Single pi.
        """
        if "pi" in field.consts:
            pi = field.consts["pi"]
        else:
            cls._need("from_float", "to represent pi")
            pi = cls._f("from_float")(_math.pi)
        return single(pi, field=field)

    @_classconst
    def size(s):
        """
        Number of cells.
        """
        return shape.size
    @_classconst
    def ndim(s):
        """
        Number of dimensions. Note empty has -1 dimensions and single has 0.
        """
        return shape.ndim

    @_classconst
    def isempty(cls):
        """
        Is empty matrix? (0x0)
        """
        return shape.isempty
    @_classconst
    def issingle(cls):
        """
        Is only one cell? (1x1)
        """
        return shape.issingle
    @_classconst
    def isvec(cls):
        """
        Has at-most one axis with length >1? (empty and single count as vectors)
        """
        return shape.isvec
    @_classconst
    def iscol(cls):
        """
        Is column vector? (empty and single count as column vectors)
        """
        return shape.iscol
    @_classconst
    def isrow(cls):
        """
        Is row vector? (empty and single count as row vectors)
        """
        return shape.isrow
    @_classconst
    def issquare(cls):
        """
        Is square matrix?
        """
        return shape.issquare


    @classmethod
    def cast(cls, *xs, broadcast=True):
        """
        Attempts to cast each object to a matrix over 'field', all with the same
        size. Note that the shape of this class is ignored. If 'broadcast' is
        false, no shape altering will be performed.
        """
        xs = _maybe_unpack(xs)
        if not xs:
            raise ValueError("cannot cast no objects")

        def conv(x):
            if isinstance(x, Matrix):
                return x
            convs = {bool: "from_bool", int: "from_int", float: "from_float",
                    complex: "from_complex", str: "from_str"}
            if type(x) not in convs.keys():
                raise TypeError(f"{_tname(type(x))} cannot operate with "
                        f"{_tname(field)}")
            x = cls._f(convs[type(x)])(x)
            return single(x, field=field)
        xs = [conv(x) for x in xs]

        if not all(field == x.field for x in xs):
            if field != o.field:
                raise TypeError("cannot operate on matrices of different fields "
                        f"({_tname(field)} vs {_tname(o.field)})")

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
            xshape = x.shape + (1, ) * (ndim - len(x.shape))
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
                raise ValueError("cannot index empty matrix")
            def process(i, ii):
                if isinstance(i, slice):
                    return tuple(range(*i.indices(shape[ii])))
                if _iterable(i):
                    return i
                if isinstance(i, int):
                    return (i, )
                raise TypeError("expected an integer or slice access, got "
                        f"{_tname(type(i))} for axis {ii}")
            ijk += (slice(None), ) * (s.ndim - len(ijk))
            slices = [process(i, ii) for ii, i in enumerate(ijk)]
            for ii, slc in enumerate(slices[s.ndim:]):
                for i in slc:
                    if i != 0:
                        raise IndexError(f"index {i} out-of-bounds for axis "
                                f"{s.ndim + ii} with length 1")
            slices = slices[:s.ndim]
            revidxs = _itertools.product(*reversed(slices))
            cells = tuple(s._at(*reversed(revidx)) for revidx in revidxs)
            return Matrix[field, tuple(map(len, slices))](cells)
    @_instconst
    def at(s):
        """
        Submatrix of the given indices (may be a single cell).
        """
        return _At(s)

    def _at(s, *ijk):
        return s._cells[shape.offset(ijk)]


    @_instconst
    def ravel(s):
        """
        Vector of cells in natural iteration order (sequential axes).
        """
        if shape[0] == shape.size:
            return s # skip type checks and such.
        return Matrix[field, (shape.size, )](s._cells)


    def along(s, *axes):
        """
        Tuple of perpendicular matrices along the given axis. When given multiple
        axes, recursively traverses those axes for each matrix along the previous
        traversals and returns them flattened.
        """
        for axis in axes:
            if not isinstance(axis, int):
                raise TypeError(f"expected integer axis, got "
                        f"{_tname(type(axis))}")
        for axis in axes:
            if axis < 0:
                raise ValueError(f"cannot have negative axis, got {axis}")
        if len(set(axes)) != len(axes):
            raise ValueError(f"cannot duplicate axes, got {axes}")
        # Empty is empty.
        if s.isempty:
            return ()
        # Recurse until only along 1 axis.
        if len(axes) > 1:
            return tuple(y for x in s.along(axes[0]) for y in x.along(axes[1:]))
        if not axes:
            return (s, )
        axis = axes[0]
        if axis >= s.ndim:
            return (s, )
        def idx(j):
            idx = [slice(None)] * s.ndim
            idx[axis] = j
            return tuple(idx)
        return tuple(s.at[idx(j)] for j in range(shape[axis]))


    def permute(s, *order):
        """
        Permutes the axes into the given order (like a transpose).
        """
        for axis in order:
            if not isinstance(axis, int):
                raise TypeError(f"expected integer axis, got "
                        f"{_tname(type(axis))}")
        for axis in order:
            if axis < 0:
                raise ValueError(f"cannot have negative axis, got {axis}")
        if len(set(order)) != len(order):
            raise ValueError(f"cannot duplicate axes, got {order}")
        if max(order) != len(order) - 1:
            raise ValueError(f"missing axis for swap, axis {max(order)} is "
                    "dangling")
        # Empty and single are invariant under permutation.
        if s.isempty or s.issingle:
            return s
        order += tuple(range(s.ndim))[len(order):]
        while len(order) > s.ndim and order[-1] == len(order) - 1:
            order = order[:-1]
        newshape = Shape(shape[axis] for axis in order)
        if s.isvec:
            # Vector has only one memory format.
            cells = s._cells
        else:
            idxs = newshape.indices()
            invorder = [0] * len(order)
            for ii, i in enumerate(order):
                invorder[i] = ii
            remap = lambda ri: [ri[invorder[axis]] for axis in range(s.ndim)]
            cells = (s._at(*remap(idx)) for idx in idxs)
        return Matrix[field, newshape](cells)


    def rep(s, *counts):
        """
        Repeats this matrix the given number of times along each dimension.
        """
        counts = _maybe_unpack(counts)
        for count in counts:
            if not isinstance(count, int):
                raise TypeError("expected an integer count, got "
                        f"{_tname(type(rows))}")
        for count in counts:
            if count < 0:
                raise ValueError(f"count cannot be negative, got {count}")
        mul = counts + (1, ) * (s.ndim - len(counts))
        shp = tuple(shape) + (1, ) * (len(counts) - s.ndim)
        newshape = Shape(a * b for a, b in zip(shp, mul))
        if newshape.size == 0:
            return Matrix[field, (0, )](())
        if newshape.size == 1:
            return s
        cells = [0] * newshape.size
        for new in newshape.indices():
            old = tuple(a % b for a, b in zip(new, shp))
            cells[newshape.offset(new)] = s._cells[shape.offset(old)]
        return Matrix[field, newshape](cells)

    def repalong(s, axis, count):
        """
        Repeats this matrix 'count' times along 'axis'.
        """
        if not isinstance(axis, int):
            raise TypeError(f"expected an integer axis, got {_tname(type(axis))}")
        if axis < 0:
            raise ValueError(f"axis cannot be negative, got {axis}")
        return s.rep((1, )*axis + (count, ))


    def __iter__(s):
        """
        Vector-only cell iterate.
        """
        if not s.isvec:
            raise TypeError(f"only vectors have bare iteration, got {shape} "
                    "(use .along and friends for matrix iterate)")
        return (single(x, field=field) for x in s._cells)

    def __getitem__(s, i):
        """
        Vector-only cell access.
        """
        if not s.isvec:
            raise TypeError(f"only vectors have bare getitem, got {shape} (use "
                    ".at for matrix cell access)")
        xs = s._cells.__getitem__(i)
        if not isinstance(xs, tuple):
            xs = (xs, )
        shp = (len(xs), ) if shape[1] == 1 else (1, len(xs))
        return Matrix[field, shp](xs)

    def __len__(s):
        """
        Vector-only cell count.
        """
        if not s.isvec:
            raise TypeError(f"only vectors have bare length, got {shape} (use "
                    ".size for matrix cell count)")
        return len(s._cells)

    @_instconst
    def cols(s):
        """
        Tuple of column vectors
        """
        if s.ndim > 2:
            raise TypeError(f"only 2D matrices have .cols, got {shape} (use "
                    ".along for other matrices)")
        return s.along(0)
    @_instconst
    def rows(s):
        """
        Tuple of row vectors.
        """
        if s.ndim > 2:
            raise TypeError(f"only 2D matrices have .rows, got {shape} (use "
                    ".along for other matrices)")
        return s.along(1)
    @_instconst
    def colmajor(s):
        """
        Vector of cells in column-major order.
        """
        if s.ndim > 2:
            raise TypeError(f"only 2D matrices have .colmajor, got {shape} (use "
                    ".ravel (maybe with .permute) for other matrices)")
        return s.ravel
    @_instconst
    def rowmajor(s):
        """
        Vector of cells in row-major order.
        """
        if s.ndim > 2:
            raise TypeError(f"only 2D matrices have .rowmajor, got {shape} (use "
                    ".ravel (maybe with .permute) for other matrices)")
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
            raise TypeError(f"cannot invert a non-square matrix, got {shape}")
        if s.det == s.zero:
            raise ValueError("cannot invert a non-invertible matrix, got det=0")
        aug = hstack(s, s.eye)
        aug = aug.rref
        return aug.at[:, shape[0]:]

    @_instconst
    def diag(s):
        """
        Vector of diagonal elements, for 2D matrices.
        """
        if s.ndim > 2:
            raise TypeError(f"only 2D matrices have .diag, got {shape}")
        cells = (s._at(i, i) for i in range(min(shape)))
        return Matrix[field, (min(shape), )](cells)


    @_instconst
    def isdiag(s):
        """
        Is diagonal matrix? (square, and only diagonal is non-zero)
        """
        if s.ndim > 2:
            raise TypeError(f"only 2D matrices have .isdiag, got {shape}")
        if not s.issquare:
            return False
        for i in range(shape[0]):
            for j in range(shape[1]):
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
            raise TypeError(f"only 2D matrices have .isuppertri, got {shape}")
        if not s.issquare:
            return False
        for i in range(shape[0]):
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
            raise TypeError(f"only 2D matrices have .islowertri, got {shape}")
        return s.T.isuppertri

    @_instconst
    def isorthogonal(s):
        """
        Is orthogonal matrix? (transpose == inverse)
        """
        if s.ndim > 2:
            raise TypeError(f"only 2D matrices have .isorthogonal, got {shape}")
        return bool(s.T == s.inv)

    @_instconst
    def issymmetric(s):
        """
        Is symmetric matrix? (square, and below diagonal = above diagonal)
        """
        if s.ndim > 2:
            raise TypeError(f"only 2D matrices have .issymmetric, got {shape}")
        if not s.issquare:
            return False
        for i in range(shape[0]):
            for j in range(i):
                if s.at[i, j] != s.at[j, i]:
                    return False
        return True


    @_instconst
    def det(s):
        """
        Determinant, for 2D square matrices.
        """
        if s.ndim > 2:
            raise TypeError(f"only 2D matrices have determinants, got {shape}")
        if not s.issquare:
            raise TypeError("only square matrices have determinants, got "
                    f"{shape}")

        if s.isempty:
            return 1 # det([]) defined as 1.

        def submatrix(cells, size, row, col):
            return [cells[i*size + j]
                    for i in range(size)
                    for j in range(size)
                    if (i != row and j != col)]

        def determinant(cells, size):
            if size == 1:
                return cells[0]
            if size == 2: # for speed only.
                return cells[0]*cells[3] - cells[1]*cells[2]
            det = s.zero
            for j in range(size):
                subcells = submatrix(cells, size, 0, j)
                subsize = size - 1
                subdet = determinant(subcells, subsize)
                if j & 1:
                    det -= cells[j] * subdet
                else:
                    det += cells[j] * subdet
            return det

        return determinant(s._cells, shape[0])

    @_instconst
    def trace(s):
        """
        Sum of diagonal elements, for 2D square matrices.
        """
        if s.ndim > 2:
            raise TypeError(f"only 2D matrices have traces, got {shape}")
        if not s.issquare:
            raise TypeError(f"only square matrices have traces, got {shape}")
        trace = s.zero
        for i in range(shape[0]):
            trace += s.at[i, i]
        return trace

    @_instconst
    def mag(s):
        """
        Euclidean distance, for vectors.
        """
        if not s.isvec:
            raise TypeError(f"only vectors have a magnitude, got {shape}")
        return (s & s).sqrt


    @_instconst
    def rref(s):
        """
        Reduced row echelon form, for 2D matrices.
        """
        if s.ndim > 2:
            raise TypeError(f"only 2D matrices can be rrefed, got {shape}")

        eqz = lambda x: s._f("eq")(x, s._f("zero"))
        eqo = lambda x: s._f("eq")(x, s._f("one"))
        add = lambda a, b: s._f("add")(a, b)
        mul = lambda a, b: s._f("mul")(a, b)
        neg = lambda x: s._f("sub")(s._f("zero"), x)
        rec = lambda x: s._f("div")(s._f("one"), x)

        def row_swap(shape, cells, row1, row2):
            cols = shape[1]
            for j in range(cols):
                k1 = row1 * cols + j
                k2 = row2 * cols + j
                cells[k1], cells[k2] = cells[k2], cells[k1]

        def row_mul(shape, cells, row, by):
            cols = shape[1]
            for j in range(cols):
                idx = row * cols + j
                cells[idx] = mul(by, cells[idx])

        def row_add(shape, cells, src, by, dst):
            cols = shape[1]
            for i in range(cols):
                src_k = src * cols + i
                dst_k = dst * cols + i
                cells[dst_k] = add(cells[dst_k], mul(by, cells[src_k]))

        cells = list(s._cells)

        rows, cols = shape[0], shape[1]
        lead = 0
        for r in range(rows):
            if lead >= cols:
                break

            i = r
            while eqz(cells[cols*i + lead]):
                i += 1
                if i == rows:
                    i = r
                    lead += 1
                    if lead == cols:
                        break
            if lead == cols:
                break
            row_swap(shape, cells, i, r)

            pivot_value = cells[cols*r + lead]
            if not eqz(pivot_value):
                row_mul(shape, cells, r, rec(pivot_value))
            # Check its 1.
            pivot_value = cells[cols*r + lead]
            if not eqo(pivot_value):
                raise ValueError(f"could not make cell =one, cell is "
                        f"{pivot_value}")

            for i in range(rows):
                if i != r:
                    idx = cols*i + lead
                    row_lead_value = cells[cols*i + lead]
                    if not eqz(row_lead_value):
                        row_add(shape, cells, r, neg(row_lead_value), i)
                    # Check its 0.
                    row_lead_value = cells[cols*i + lead]
                    if not eqz(row_lead_value):
                        raise ValueError("could not make cell =zero, cell is "
                                f"{row_lead_value}")

            lead += 1

        return Matrix[field, shape](cells)

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
        return tuple(j for j in range(shape[1]) if j not in s.pivots)


    @_instconst
    def colspace(s):
        """
        Basis for column space.
        """
        return tuple(s.cols[p] for p in s.pivots)

    @_instconst
    def rowspace(s):
        """
        Basis for row space.
        """
        rref = s.rref
        nonzeros = (i for i, r in enumerate(rref.rows)
                    if any(s._f("eq")(x, s._f("zero")) for x in r))
        return tuple(rref.rows[i].T for i in nonzeros)

    @_instconst
    def nullspace(s):
        """
        Basis for null space.
        """
        sys = s.rref # implied zero-vec augment.
        def find_first_one(xs):
            for i, x in enumerate(xs):
                if s._f("eq")(x, s._f("one")):
                    return i
            return None
        pivotat = tuple(find_first_one(sys.rows[i]) for i in range(shape[0]))
        basis = [[s._f("zero")]*shape[1] for _ in sys.nonpivots]
        for n, j in enumerate(sys.nonpivots):
            for i in range(shape[0]):
                if pivotat[i] is None or pivotat[i] > j:
                    basis[n][j] = 1
                    break
                basis[n][pivotat[i]] = -sys.at[i, j]
        return tuple(Matrix[field, (shape[1], )](x) for x in basis)


    def _reprfr(s, width, ndim, allow_flat=False):
        assert field is GenericField[str]

        if ndim > 2:
            layers = (x._reprfr(width, ndim - 1) for x in s.along(ndim - 1))
            def encapsulate(r):
                r = r.split("\n")
                s = r[0] + "".join(f"\n  {line}" for line in r[1:])
                return "[ " + s + " ]"
            layers = (encapsulate(r) for r in layers)
            return "\n".join(layers)

        # 2d print.

        # Print col vecs as rows with "'", if allowed.
        suffix = ""
        height = shape[0]
        if allow_flat and s.iscol:
            suffix = "'"
            height = shape[1]
        cols = []
        for i, r in enumerate(s._cells):
            if not i % height:
                cols.append([])
            cols[-1].append(f"{r.obj:>{width}}")
        rows = list(zip(*cols))
        padded = (width > 3)
        join = lambda x: "  ".join(x) if padded else " ".join(x)
        wrap = lambda x: f"[ {x} ]" if padded else f"[{x}]"
        str_rows = (wrap(join(row)) for row in rows)
        return "\n".join(str_rows) + suffix

    def _repr(s, islong, width=None):
        if s.isempty:
            return "my boy "*islong + "M.T."

        rep = s._f("repr_long" if islong else "repr_short")

        if s.issingle:
            return rep(s._cells[0])

        if not islong and not s.isvec:
            # Shorten elements of zero to a single dot.
            rep_ = rep
            def rep(x):
                if rep.can_eq_zero and s._f("eq")(x, s._f("zero")):
                    return "."
                return rep_(x)
            rep.can_eq_zero = True
            try:
                s._need("eq")
                s._need("zero")
            except NotImplementedError:
                rep.can_eq_zero = False

        # cheeky matrix of the reps to make access easy.
        reps = Matrix[GenericField[str], shape](rep(x) for x in s._cells)
        width = max(len(r.obj) for r in reps._cells)
        return reps._reprfr(width, s.ndim, allow_flat=True)

    @_instconst
    def repr_short(s):
        """
        Short string representation.
        """
        return s._repr(False)

    @_instconst
    def repr_long(s):
        """
        Long string representation.
        """
        return s._repr(True)


    @classmethod
    def _eltwise(cls, func, *xs, rtype=None, pierce=True):
        # Cast me.
        if not xs:
            raise ValueError("cannot operate on no matrices")
        xs = cls.cast(xs)

        # Apply me.
        def f(y):
            nonlocal rtype
            ret = func(*y)
            if isinstance(ret, Matrix):
                if not ret.issingle:
                    raise TypeError("expected 'func' to return a single matrix, "
                            f"got {ret.shape}")
                ret = ret._cells[0]
            if rtype is None:
                rtype = type(ret)
            elif not isinstance(ret, rtype):
                raise TypeError(f"expected {_tname(rtype)} typed return from "
                        f"'func' for consistency, got {_tname(type(ret))})")
            return ret
        zipped = zip(*(x._cells for x in xs))
        if pierce:
            # need these in tuple to eval now and find rtype.
            cells = tuple(f(y) for y in zipped)
        else:
            cells = tuple(f(single(z, field=field) for z in y) for y in zipped)
        return Matrix[rtype, xs[0].shape](cells)

    @classmethod
    def eltwise(cls, func, *xs, rtype=None):
        """
        Constructs a matrix from the results of 'func(a, b, ...)' for all zipped
        elements in '*xs'. If 'rtype' is non-none, hints/enforces the return type
        from 'func'.
        """
        return cls._eltwise(func, *xs, rtype=rtype, pierce=False)

    def apply(s, func, *os, rtype=None):
        """
        Alias for 'M.eltwise(func, s, *os, rtype=rtype)'.
        """
        return s.eltwise(func, s, *os, rtype=rtype)


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
        return s._eltwise(f, s)
    def __abs__(s):
        """
        Element-wise absolution.
        """
        return s._eltwise(s._f("absolute"), s)

    def __add__(s, o):
        """
        Element-wise addition.
        """
        return s._eltwise(s._f("add"), s, o)
    def __radd__(s, o):
        return s._eltwise(lambda a, b: s._f("add")(b, a), s, o)
    def __sub__(s, o):
        """
        Element-wise subtraction.
        """
        return s._eltwise(s._f("sub"), s, o)
    def __rsub__(s, o):
        return s._eltwise(lambda a, b: s._f("sub")(b, a), s, o)
    def __mul__(s, o):
        """
        Element-wise multiplication (use '@' for matrix multiplication).
        """
        return s._eltwise(s._f("mul"), s, o)
    def __rmul__(s, o):
        return s._eltwise(lambda a, b: s._f("mul")(b, a), s, o)
    def __truediv__(s, o):
        """
        Element-wise division.
        """
        return s._eltwise(s._f("div"), s, o)
    def __rtruediv__(s, o):
        return s._eltwise(lambda a, b: s._f("div")(b, a), s, o)
    def __pow__(s, o):
        """
        Element-wise power.
        """
        return s._eltwise(s._f("power"), s, o)
    def __rpow__(s, o):
        return s._eltwise(lambda a, b: s._f("power")(b, a), s, o)

    @_instconst
    def sqrt(s):
        """
        Element-wise square root.
        """
        s._need("from_int", "to represent 2")
        two = s._f("from_int")(2)
        return s._eltwise(lambda x: s._f("root")(x, two), s)
    @_instconst
    def cbrt(s):
        """
        Element-wise cube root.
        """
        s._need("from_int", "to represent 3")
        three = s._f("from_int")(3)
        return s._eltwise(lambda x: s._f("root")(x, three), s)

    def root(s, n):
        """
        Element-wise nth root.
        """
        return s._eltwise(s._f("root"), s, n)

    @_instconst
    def exp(s):
        """
        Element-wise natural exponential.
        """
        base = s.e._cells[0]
        return s._eltwise(lambda x: s._f("power")(base, x), s)
    @_instconst
    def exp2(s):
        """
        Element-wise base-2 exponential.
        """
        s._need("from_int", "to represent 2")
        base = s._f("from_int")(2)
        return s._eltwise(lambda x: s._f("power")(base, x), s)
    @_instconst
    def exp10(s):
        """
        Element-wise base-10 exponential.
        """
        s._need("from_int", "to represent 10")
        base = s._f("from_int")(10)
        return s._eltwise(lambda x: s._f("power")(base, x), s)

    @_instconst
    def ln(s):
        """
        Element-wise natural logarithm.
        """
        base = s.e._cells[0]
        return s._eltwise(lambda x: s._f("log")(base, x), s)
    @_instconst
    def log2(s):
        """
        Element-wise base-2 logarithm.
        """
        s._need("from_int", "to represent 2")
        base = s._f("from_int")(2)
        return s._eltwise(lambda x: s._f("log")(base, x), s)
    @_instconst
    def log10(s):
        """
        Element-wise base-10 logarithm.
        """
        s._need("from_int", "to represent 10")
        base = s._f("from_int")(10)
        return s._eltwise(lambda x: s._f("log")(base, x), s)
    def log(s, base):
        """
        Element-wise base-specified logarithm.
        """
        return s._eltwise(lambda a, b: s._f("log")(b, a), s, base)

    @_instconst
    def sin(s):
        """
        Element-wise trigonometric sine.
        """
        return s._eltwise(s._f("sin"), s)
    @_instconst
    def cos(s):
        """
        Element-wise trigonometric cosine.
        """
        return s._eltwise(s._f("cos"), s)
    @_instconst
    def tan(s):
        """
        Element-wise trigonometric tangent.
        """
        return s._eltwise(s._f("tan"), s)

    @_instconst
    def asin(s):
        """
        Element-wise trigonometric inverse-sine.
        """
        return s._eltwise(s._f("asin"), s)
    @_instconst
    def acos(s):
        """
        Element-wise trigonometric inverse-cosine.
        """
        return s._eltwise(s._f("acos"), s)
    @_instconst
    def atan(s):
        """
        Element-wise trigonometric inverse-tangent.
        """
        return s._eltwise(s._f("atan"), s)


    def __eq__(s, o):
        """
        Element-wise equality (cast return to bool to determine if all pairs are
        equal).
        """
        return s._eltwise(s._f("eq"), s, o)
    def __ne__(s, o):
        return s._eltwise(lambda a, b: not s._f("eq")(a, b), s, o)
    def __lt__(s, o):
        """
        Element-wise ordering (cast return to bool to determine if all pairs are
        ordered strictly ascending).
        """
        return s._eltwise(s._f("lt"), s, o)
    def __le__(s, o):
        f = lambda a, b: s._f("eq")(a, b) or s._f("lt")(a, b)
        return s._eltwise(f, s, o)
    def __gt__(s, o):
        return s._eltwise(lambda a, b: s._f("lt")(b, a), s, o)
    def __ge__(s, o):
        f = lambda a, b: s._f("eq")(a, b) or s._f("lt")(b, a)
        return s._eltwise(f, s, o)


    def __and__(s, o):
        """
        Vector dot product.
        """
        o, = s.cast(o)
        if not s.isvec or not o.isvec:
            raise TypeError(f"can only dot product vectors (got sizes "
                    f"{s._size_str} and {o._size_str})")
        if len(s) != len(o):
            raise TypeError(f"can only dot product same-size vectors (got sizes "
                    f"{s._size_str} and {o._size_str})")
        mul = s._f("mul")
        add = s._f("add")
        dot = s._f("zero")
        for a, b in zip(s._cells, o._cells):
            dot = add(dot, mul(a, b))
        return single(dot, field=field)
    def __rand__(s, o):
        o, = s.cast(o)
        return o.__and__(s)

    def __or__(s, o):
        """
        3D vector cross product.
        """
        o, = s.cast(o)
        if not s.isvec or not o.isvec:
            raise TypeError(f"can only cross product vectors (got sizes "
                    f"{s._size_str} and {o._size_str})")
        if len(s) != 3 or len(o) != 3:
            raise TypeError(f"can only cross product 3-element vectors (got "
                    f"sizes {s._size_str} and {o._size_str})")
        shp = (3, 1) # column vector takes precedence.
        if lhs.shape == (1, 3) and rhs.shape == (1, 3):
            # but keep row vector if both are.
            shp = hstack
        ax, ay, az = lhs._cells
        bx, by, bz = rhs._cells
        mul = s._f("mul")
        sub = s._f("sub")
        cells = (
            sub(mul(ay, bz), mul(az, by)),
            sub(mul(az, bx), mul(ax, bz)),
            sub(mul(ax, by), mul(ay, bx)),
        )
        return Matrix[field, shp](cells)
    def __ror__(s, o):
        o, = s.cast(o)
        return o.__or__(s)

    def __matmul__(s, o):
        """
        Matrix multiplication.
        """
        o, = s.cast(o)
        if s.shape[1] != o.shape[0]:
            raise TypeError("incorrect size for matrix multiplication "
                    f"({s._size_str} times {o._size_str})")
        shp = s.shape[0], o.shape[1]
        mul = s._f("mul")
        add = s._f("add")
        cells = [s._f("zero")] * _prod(shp)
        # blazingly fast new matrix multiplication algorithm scientists are
        # dubbing the "naive method" (i think it means really smart).
        for i in range(s.shape[0]):
            for j in range(o.shape[1]):
                for k in range(s.shape[1]):
                    s_idx = s.shape[1]*i + k
                    o_idx = o.shape[1]*k + j
                    r_idx = shp[1]*i + j
                    prod = mul(s._cells[s_idx], o._cells[o_idx])
                    cells[r_idx] = add(cells[r_idx], prod)
        return Matrix[field, shp](cells)
    def __rmatmul__(s, o):
        o, = s.cast(o)
        return o.__matmul__(s)

    def __xor__(s, exp):
        """
        Matrix power (repeated self matrix multiplication, with possible
        inverse).
        """
        if isinstance(exp, field):
            exp = s._f("to_int")(exp)
        if not isinstance(exp, int):
            raise TypeError("can only raise a matrix to an integer power (got "
                    f"{_tname(type(exp))})")
        if not s.issquare:
            raise TypeError("can only raise square matrices to a power (got "
                    f"{s._size_str})")
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


    def __bool__(s):
        """
        True iff any elements are non-zero (or non-one if field has no zero).
        """
        if hasattr(field, "zero"):
            return not all(s._f("eq")(x, s._f("zero")) for x in s._cells)
        if hasattr(field, "one"):
            return not all(s._f("eq")(x, s._f("one")) for x in s._cells)
        raise NotImplementedError("no zero or one element in field "
                f"{_tname(field)}, must specify an element to compare to")

    def __int__(s):
        """
        Cast a single to int.
        """
        if not s.issingle:
            raise TypeError(f"cannot cast {s._size_str} matrix to int (requires "
                    "1x1)")
        return s._f("to_int")(s._cells[0])
    def __float__(s):
        """
        Cast a single to float.
        """
        if not s.issingle:
            raise TypeError(f"cannot cast {s._size_str} matrix to float "
                    "(requires 1x1)")
        return s._f("to_float")(s._cells[0])
    def __complex__(s):
        """
        Cast a single to complex.
        """
        if not s.issingle:
            raise TypeError(f"cannot cast {s._size_str} matrix to complex "
                    "(requires 1x1)")
        return s._f("to_complex")(s._cells[0])

    def __hash__(s):
        return hash(tuple(s._f("hashed")(x) for x in s._cells))

    def __repr__(s):
        """
        Short string representation (use 'long' or '.repr_long' for a long
        string).
        """
        return s.repr_short



    # Its so dangerous bruh.
    def __getattr__(s, attr, rtype=None):
        """
        If a non-matrix attribute is accessed, it will be retrived from each
        element instead.
        """
        cells = object.__getattribute__(s, "_cells") # cooked.
        bad = False
        if attr.startswith("_"):
            bad = True
        elif cells:
            try:
                # need tuple to eval now.
                cells = tuple(getattr(x, attr) for x in cells)
            except AttributeError:
                bad = True
        else:
            try:
                bad = not hasattr(s._f("zero"), attr)
            except NotImplementedError:
                try:
                    bad = not hasattr(s._f("one"), attr)
                except NotImplementedError:
                    bad = False # giv up.
        if bad:
            raise AttributeError(f"{_tname(type(s))} object has no attribute "
                    f"{repr(attr)} (and neither do the {_tname(field)} "
                    "elements)")
        if not cells:
            if rtype is None:
                rtype = field
            return empty(rtype)
        if rtype is None:
            rtype = type(cells[0])
        for cell in cells:
            if not isinstance(cell, rtype):
                raise TypeError(f"expected {_tname(rtype)} typed return from "
                        f"'.__getattr__({repr(attr)})' for consistency, got "
                        f"{_tname(type(cell))})")
        return Matrix[rtype, shape](cells)



class Single:
    def __getitem__(self, field):
        return Matrix[field, Shape.single]
Single = Single()
Single.__doc__ = """
Refers to the single matrix (1x1) over the given field. Note this is not a proper
templated class, only a thin wrapper.
"""

def single(x, *, field=None):
    """
    Returns the given object as a single matrix (1x1).
    """
    if field is None:
        field = type(x)
    return Single[field]((x, ))

def issingle(a):
    """
    Returns true iff 'a' is a matrix with only one cell.
    """
    return isinstance(a, Matrix) and a.issingle


class Empty:
    def __getitem__(self, field):
        return Matrix[field, Shape.empty]
Empty = Empty()
Empty.__doc__ = """
Refers to the empty matrix (0x0) over the given field. Note this is not a proper
templated class, only a thin wrapper.
"""

def empty(field):
    """
    Returns an empty matrix (0x0) instance over the given field.
    """
    return Empty[field](())

def isempty(a):
    """
    Returns true iff 'a' is a matrix with no cells.
    """
    return isinstance(a, Matrix) and a.isempty



def sqrt(x, *, field=None):
    """
    Alias for 'x.sqrt'.
    """
    field = _get_field(field)
    if not isinstance(x, Matrix):
        x, = Single[field].cast(x)
    return x.sqrt
def cbrt(x, *, field=None):
    """
    Alias for 'x.cbrt'.
    """
    field = _get_field(field)
    if not isinstance(x, Matrix):
        x, = Single[field].cast(x)
    return x.cbrt
def root(base, x, *, field=None):
    """
    Alias for 'x.root(base)'.
    """
    field = _get_field(field)
    if not isinstance(x, Matrix):
        x, = Single[field].cast(x)
    return x.root(base)

def exp(x, *, field=None):
    """
    Alias for 'x.exp'.
    """
    field = _get_field(field)
    if not isinstance(x, Matrix):
        x, = Single[field].cast(x)
    return x.exp
def exp2(x, *, field=None):
    """
    Alias for 'x.exp2'.
    """
    field = _get_field(field)
    if not isinstance(x, Matrix):
        x, = Single[field].cast(x)
    return x.exp2
def exp10(x, *, field=None):
    """
    Alias for 'x.exp10'.
    """
    field = _get_field(field)
    if not isinstance(x, Matrix):
        x, = Single[field].cast(x)
    return x.exp10

def ln(x, *, field=None):
    """
    Alias for 'x.ln'.
    """
    field = _get_field(field)
    if not isinstance(x, Matrix):
        x, = Single[field].cast(x)
    return x.ln
def log2(x, *, field=None):
    """
    Alias for 'x.log2'.
    """
    field = _get_field(field)
    if not isinstance(x, Matrix):
        x, = Single[field].cast(x)
    return x.log2
def log10(x, *, field=None):
    """
    Alias for 'x.log10'.
    """
    field = _get_field(field)
    if not isinstance(x, Matrix):
        x, = Single[field].cast(x)
    return x.log10
def log(base, x, *, field=None):
    """
    Alias for 'x.log(base)'.
    """
    field = _get_field(field)
    if not isinstance(x, Matrix):
        x, = Single[field].cast(x)
    return x.log(base)
def sin(x, *, field=None):
    """
    Alias for 'x.sin'.
    """
    field = _get_field(field)
    if not isinstance(x, Matrix):
        x, = Single[field].cast(x)
    return x.sin
def cos(x, *, field=None):
    """
    Alias for 'x.cos'.
    """
    field = _get_field(field)
    if not isinstance(x, Matrix):
        x, = Single[field].cast(x)
    return x.cos
def tan(x, *, field=None):
    """
    Alias for 'x.tan'.
    """
    field = _get_field(field)
    if not isinstance(x, Matrix):
        x, = Single[field].cast(x)
    return x.tan
def asin(x, *, field=None):
    """
    Alias for 'x.asin'.
    """
    field = _get_field(field)
    if not isinstance(x, Matrix):
        x, = Single[field].cast(x)
    return x.asin
def acos(x, *, field=None):
    """
    Alias for 'x.acos'.
    """
    field = _get_field(field)
    if not isinstance(x, Matrix):
        x, = Single[field].cast(x)
    return x.acos
def atan(x, *, field=None):
    """
    Alias for 'x.atan'.
    """
    field = _get_field(field)
    if not isinstance(x, Matrix):
        x, = Single[field].cast(x)
    return x.atan

def atan2(y, x, *, field=None):
    """
    Quadrant-aware 'atan(y / x)'.
    """
    field = _get_field(field)
    x, y = Single[field].cast(x, y)
    return y._eltwise(y._f("atan2"), y, x)



def long(a):
    """
    Prints a long string representation of 'a'.
    """
    if not isinstance(a, Matrix):
        raise TypeError(f"expected matrix, got {_tname(type(a))}")
    print(a.repr_long)



def stack(axis, *xs, field=None):
    """
    Stacks the given arrays along the given axis.
    """
    field = _get_field(field)
    xs = _maybe_unpack(xs)
    if not xs:
        raise ValueError("cannot stack no matrices")
    xs = Single[field].cast(xs, broadcast=False)

    if not isinstance(axis, int):
        raise TypeError(f"expected an integer axis, got {_tname(type(axis))}")
    if axis < 0:
        raise ValueError(f"axis cannot be negative, got {axis}")

    # Stacking empty do NOTHIGN.
    if all(x.isempty for x in xs):
        return empty(field)
    if len(xs) == 1:
        return xs[0]

    # Check perpendicular sizes.
    perpshape = xs[0].shape.withaxis(axis, 1)
    newshape = perpshape.withaxis(axis, sum(x.shape[axis] for x in xs))
    for x in xs:
        if x.shape.withaxis(axis, 1) != perpshape:
            raise TypeError(f"expected {perpshape} perpendicular shape for "
                    f"stacking, got {x.shape})")
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
    for new in newshape.indices():
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
    return stack(0, xs, field=field)

def hstack(*xs, field=None):
    """
    Horizontally concatenates the given matrices.
    """
    return stack(1, xs, field=field)

def ravel(*xs, field=None):
    """
    Returns a vector of the cells of all given matrices.
    """
    field = _get_field(field)
    xs = _maybe_unpack(xs)
    xs = Single[field].cast(xs, broadcast=False)
    size = sum(x.size for x in xs)
    cells = ()
    for x in xs:
        cells += x._cells
    return Matrix[field, (size, )](cells)

def rep(x, *counts, field=None):
    """
    Repeats the given matrix the given number of times along each dimension.
    """
    field = _get_field(field)
    x, = Single[field].cast(x)
    return x.rep(counts)

def repalong(x, axis, count, *, field=None):
    """
    Repeats the given matrix 'count' times along 'axis'.
    """
    field = _get_field(field)
    x, = Single[field].cast(x)
    return x.repalong(axis, count)


# def mat(*xs, field=None):
#     """
#     Returns a matrix from the given iterables.
#     """
#     stack = []
#     while stack:


def diag(*xs, field=None):
    """
    Creates a matrix with a diagonal of the given elements and zeros elsewhere.
    """
    field = _get_field(field)
    xs = _maybe_unpack(xs)
    diagvec = ravel(*xs, field=field)
    n = len(diagvec)
    Mat = Matrix[field, (n, n)]
    if n == 1:
        cells = diagvec._cells
    else:
        cells = [Mat._f("zero")] * (n*n) if n else []
        for i in range(n):
            cells[i*n + i] = diagvec._cells[i]
    return Mat(cells)


def eye(n, *, field=None):
    """
    Identity matrix, of the given size.
    """
    field = _get_field(field)
    return Matrix[field, (n, n)].eye

def zeros(rows, cols=None, *, field=None):
    """
    Zero-filled matrix, defaulting to square if only one size given.
    """
    field = _get_field(field)
    cols = rows if cols is None else cols
    return Matrix[field, (rows, cols)].zeros

def ones(rows, cols=None, *, field=None):
    """
    One-filled matrix, defaulting to square if only one size given.
    """
    field = _get_field(field)
    cols = rows if cols is None else cols
    return Matrix[field, (rows, cols)].ones



def summ(*xs, field=None):
    """
    Returns the additive sum of the given values.
    """
    field = _get_field(field)
    xs = _maybe_unpack(xs)
    if not xs:
        return Single[field].zero
    r = Single[field].zero
    for x in xs:
        if not isinstance(x, Matrix):
            x, = Single[field].cast(x)
        for y in x:
            r += y
    return r

def prod(*xs, field=None):
    """
    Returns the multiplicative product of the given values.
    """
    field = _get_field(field)
    xs = _maybe_unpack(xs)
    if not xs:
        return Single[field].one
    r = Single[field].one
    for x in xs:
        if not isinstance(x, Matrix):
            x, = Single[field].cast(x)
        for y in x:
            r *= y
    return r

def minn(*xs, field=None):
    """
    Returns the minimum of the given values (first occurrence in the case of
    ties).
    """
    field = _get_field(field)
    xs = _maybe_unpack(xs)
    if not xs:
        raise ValueError("cannot find minimum of no elements")
    r = None
    for x in xs:
        if not isinstance(x, Matrix):
            x, = Single[field].cast(x)
        for y in x:
            if r is None:
                r = y
            elif y < r:
                r = y
    return r

def maxx(*xs, field=None):
    """
    Returns the maximum of the given values (first occurrence in the case of
    ties).
    """
    field = _get_field(field)
    xs = _maybe_unpack(xs)
    if not xs:
        raise ValueError("cannot find maximum of no elements")
    r = None
    for x in xs:
        if not isinstance(x, Matrix):
            x, = Single[field].cast(x)
        for y in x:
            if r is None:
                r = y
            elif y > r:
                r = y
    return r


def mean(*xs, field=None):
    """
    Returns the arithmetic mean of the given values.
    """
    field = _get_field(field)
    xs = _maybe_unpack(xs)
    if not xs:
        raise ValueError("cannot find the arithmetic mean no elements")
    r = Single[field].zero
    n = Single[field].zero
    for x in xs:
        if not isinstance(x, Matrix):
            x, = Single[field].cast(x)
        for y in x:
            r += y
            n += Single[field].one
    return r / n
def ave(*xs, field=None):
    """
    Alias for 'mean(*xs)'.
    """
    return mean(*xs, field=field)

def geomean(*xs, field=None):
    """
    Returns the geometric mean of the given values.
    """
    field = _get_field(field)
    xs = _maybe_unpack(xs)
    if not xs:
        raise ValueError("cannot find the geometric mean no elements")
    r = Single[field].one
    n = Single[field].zero
    for x in xs:
        if not isinstance(x, Matrix):
            x, = Single[field].cast(x)
        for y in x:
            r *= y
            n += Single[field].one
    return r.root(n)

def harmean(*xs, field=None):
    """
    Returns the harmonic mean of the given values.
    """
    field = _get_field(field)
    xs = _maybe_unpack(xs)
    if not xs:
        raise ValueError("cannot find the harmonic mean no elements")
    r = Single[field].zero
    n = Single[field].zero
    for x in xs:
        if not isinstance(x, Matrix):
            x, = Single[field].cast(x)
        for y in x:
            r += y.one / y
            n += Single[field].one
    return n / r

def quadmean(*xs, field=None):
    """
    Returns the quadratic mean (root-mean-square) of the given values.
    """
    field = _get_field(field)
    xs = _maybe_unpack(xs)
    if not xs:
        raise ValueError("cannot find the root-mean-square no elements")
    r = Single[field].zero
    n = Single[field].zero
    for x in xs:
        if not isinstance(x, Matrix):
            x, = Single[field].cast(x)
        for y in x:
            r += y * y
            n += Single[field].one
    return (r / n).sqrt

def logmean(x, y, *, field=None):
    """
    Returns the logarithmic mean of 'x' and 'y': (x - y) / ln(x / y)
    """
    field = _get_field(field)
    x, y = Single[field].cast(x, y)
    f = lambda a, b: a if (a == b) else (a - b) / (a / b).ln
    return x.apply(f, y)




class series:
    def __init__(self, xs):
        if not isinstance(xs, Matrix):
            raise TypeError("expected a matrix for the argument of series, got "
                    f"{_tname(type(xs))}")
        self.xs = xs

    def __repr__(self):
        s = repr(self.xs)
        f = self.xs.field
        s += f"\n  sum .... {summ(self.xs, field=f)}"
        s += f"\n  prod ... {prod(self.xs, field=f)}"
        s += f"\n  ave .... {ave(self.xs, field=f)}"
        mi = minn(self.xs, field=f)
        ma = maxx(self.xs, field=f)
        s += f"\n  min .... {mi}"
        s += f"\n  max .... {ma}"
        dfu = float(100 * ma / mi)
        dfv = float(100 * (1 - mi/ma))
        s += f"\n  diff ... {ma - mi} (^ {dfu:.3g}%) (v {dfv:.3g}%)"
        return s




def mhelp():
    def printme(name, long):
        name = "  " + name
        print(name, end="")
        s = long.strip().replace("\n", " ")
        w = 0
        while s:
            if w:
                print(w * " ", end="")
            else:
                w = len(name)
            line = s[:100 - w]
            if len(line) == 100 - w and " " in line:
                line = line[:line.rindex(" ")]
            print(line)
            s = s[len(line):].lstrip()

    def print_attr(name, desc):
        s = f"{name} .."
        s += "." * (18 - len(s)) + " "
        printme(s, desc)

    printme("Matrix - ", Matrix.__doc__)
    print_attr("M.field", "Cell type.")
    print_attr("M.shape", "Tuple of the size of each dimension.")

    Mat = Single[Field]
    attrs = [(name, attr) for name, attr in vars(Mat).items()
            if attr.__doc__ is not None
            and name != "__module__"
            and name != "__doc__"
            and name != "template"
            and name not in Matrix.params]

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
    for name, attr in attrs:
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
        print_attr(expr, attr.__doc__)

    # also chuck the other functions in this file.
    funcs = []
    this_func = _inspect.currentframe().f_code.co_name
    for obj in globals().values():
        if not _inspect.isfunction(obj):
            continue
        if obj.__module__ != __name__: # must be in this file.
            continue
        if obj.__name__ == this_func: # mustnt be this func.
            continue
        if obj.__doc__ is None:
            continue
        funcs.append(obj)

    for func in funcs:
        # Print `Single` just before `single()`.
        if func is single:
            print_attr("Single[field]", Single.__doc__)

        sig = _inspect.signature(func)
        sig = str(sig)
        sig = sig[1:-1]
        if sig.endswith(", *, field=None"):
            sig = sig[:-len(", *, field=None")]
        elif sig.endswith(", field=None"):
            sig = sig[:-len(", field=None")]
        elif sig.endswith(", *, space=None"):
            sig = sig[:-len(", *, space=None")]
        name = f"{func.__name__}({sig})"
        print_attr(name, func.__doc__)
