import cmath as _cmath
import functools as _functools
import inspect as _inspect
import itertools as _itertools
import math as _math
import types as _types

from util import (
    coloured as _coloured, nonctrl as _nonctrl, entry as _entry, tname as _tname,
    objtname as _objtname, iterable as _iterable, immutable as _immutable,
    templated as _templated, classconst as _classconst, instconst as _instconst,
    maybe_unpack as _maybe_unpack, add_to as _add_to,
)
from bg import bg as _bg



class Field:
    """
    Base class for implementing operation over elements. Cannot do operations on
    entire matrices, only performs for single elements of those matrices. See
    'ExampleField' descriptions of all methods, as this class implements only the
    mandatory methods.
    """

    @classmethod
    def dtype(cls):
        return _bg.np.dtype(object)

    @classmethod
    def fromobj(cls, obj):
        if isinstance(obj, bool):
            return cls.one() if obj else cls.zero()
        raise TypeError(f"cannot create {_tname(cls)} from {_objtname(obj)} "
                f"(value: {obj})")

    @classmethod
    def consts(cls):
        return dict()




class ExampleField(Field):
    """
    Example field type, demonstrating maximum functionality. Note no internals
    are included, only the argument and return types of each method are
    specified.
    """

    @classmethod
    def dtype(cls): # backing numpy array dtype.
        return object

    @classmethod
    def fromobj(cls, obj):
        return ...

    @classmethod
    def to_int(cls, a):
        return int()
    @classmethod
    def to_float(cls, a):
        return float()
    @classmethod
    def to_complex(cls, a):
        return complex()

    @classmethod
    def consts(cls): # map of str to elements, all optional.
        # The names below are all specially recognised, otherwise it may be
        # anything.
        return {
            "__0__": 0, # additive identity.
            "__1__": 0, # multiplicative identity.
            "__i__": 1j, # imaginary unit.
            "__e__": 2.71828, # euler's number.
            "__pi__": 3.14159, # pi.
        }

    @classmethod
    def add(cls, a, b): # a+b
        return ... # something this field can still operate on.
    @classmethod
    def sub(cls, a, b): # a-b
        return ...

    @classmethod
    def abs(cls, a): # |a|
        return ...
    @classmethod
    def real(cls, a): # Re(a), but dont implement if non-complex
        return ...
    @classmethod
    def imag(cls, a): # Im(a), but dont implement if non-complex
        return ...

    @classmethod
    def mul(cls, a, b): # a*b
        return ...
    @classmethod
    def div(cls, a, b): # a/b
        return ...

    @classmethod
    def power(cls, a, b): # a^b
        return ...
    @classmethod
    def root(cls, a, b): # a^(1/b)
        return ...
    @classmethod
    def log(cls, a, b): # log_a(b)
        return ...

    @classmethod
    def sin(cls, a): # sin(a)
        return ...
    @classmethod
    def cos(cls, a): # cos(a)
        return ...
    @classmethod
    def tan(cls, a): # tan(a)
        return ...

    @classmethod
    def asin(cls, a): # sin^-1(a)
        return ...
    @classmethod
    def acos(cls, a): # cos^-1(a)
        return ...
    @classmethod
    def atan(cls, a): # tan^-1(a)
        return ...
    @classmethod
    def atan2(cls, y, x): # tan^-1(y / x), but quadrant-aware.
        return ...

    @classmethod
    def diff(cls, y, x): # (d/dx y)
        return ...
    @classmethod
    def intt(cls, y, x): # (int y dx)
        return ...
    @classmethod
    def def_intt(cls, y, x, a, b): # (int(a..b) y dx)
        return ...

    @classmethod
    def issame(cls, a, b): # a is identical to b, must return bool
        return True

    @classmethod
    def eq(cls, a, b): # a == b
        return bool() # common for comparison to return bool, but again, can be
                      # any field.
    @classmethod
    def neq(cls, a, b): # a != b
        return bool()
        # another common impl:
        return not cls.eq(a, b)
    @classmethod
    def lt(cls, a, b): # a < b
        return bool()
    @classmethod
    def le(cls, a, b): # a <= b
        return bool()
        return cls.lt(a, b) or cls.eq(a, b)

    @classmethod
    def hashed(cls, a): # hash(a)
        return hash()

    @classmethod
    def rep(cls, a, short): # repr(a), with short+long form.
        assert isinstance(short, bool)
        if short:
            return "d ... j"
        return "dujjdujj"





def _is_overridden(space, field, name):
    if name not in space:
        return False
    if field is None:
        return True
    got = space[name]
    try:
        expect = getattr(Single[field], name)
    except Exception:
        return True # assume nothing but the worst.
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

def _maybe_unpack_mats(xs):
    return _maybe_unpack(xs, dont_unpack=Matrix)

def _maybe_unpack_ints(xs):
    xs = _maybe_unpack_mats(xs)
    if len(xs) == 1 and isinstance(xs[0], Matrix) and xs[0].isvec:
        return tuple(xs[0].numpyvec(int))
    return xs


def lits(field, inject=True, *, space=None):
    """
    Sets the current/default field to the given field and injects constants such
    as 'e' and 'pi' into the globals.
    """
    try:
        # try create a field out of it.
        if field is not None:
            Single[field]
    except Exception as e:
        raise TypeError(f"expected a valid field class, got {_objtname(field)}")\
                from e

    prev_field = lits._field
    lits._field = field
    if not inject:
        return
    # Inject constants into the space.
    if space is None:
        space = _get_space()
    for name in lits._injects:
        # Dont wipe vars the user has set.
        if _is_overridden(space, prev_field, name):
            continue
        didset = False
        if field is not None:
            try:
                space[name] = getattr(Single[field], name)
                didset = True
            except Exception:
                pass
        if not didset:
            space.pop(name, None)
lits._field = None
lits._injects = ("e", "pi")


"""
Cheeky rant about how matrix interals will interoperate with numpy, since we have
both different shaping logic and different memory layout expectations.

Firstly, for empty the canonical shape and backing is:
    shape = (0,)
    npshape = (0,)
    backing = np.array([])

Now for the general case, consider a matrix shape of:
    shape = (l0, l1, l2, ...)
    # meaning 'l2' matrices each with 'l0' rows and 'l1' columns.
Numpy would order that same shape as:
    npshape = (..., l2, l0, l1)
Aka, a 2d transpose and a reverse. So, when converting from matrix shape to array
shape, use:
    npshape = shape[2::-1] + shape[:2]
This means the numpy shape is always 2d, but this actually solves another problem
where numpy distinguishes between more cases than us:
    npshape=(1, 2, 3)  |  shape=(2, 3)
       npshape=(2, 3)  |  shape=(2, 3)
Since we consider all shapes implicitly infinite, the above are equivalent. To
ensure consistent matrix operation when using the numpy arrays, we keep it 2D.
Note also that conversion from npshape to shape is not needed (currently :)), so
handling this in reverse is not considered.
Nevermind :(
Going from a numpy shape, firstly treat 1d as column vector, secondly perform the
inverse to-numpy permutation, thirdly discard trailing 1s.

Now for the memory layout issue. Matrix expects memory to be laid out where each
additional axis is consider blocks of the previous data, so for a 3d example:
    [ [1 3]
      [2 4] ]
    [ [5 7]
      [6 8] ]
The in-memory flattened layout is sorted ascending. If this memory was
interpreted as a numpy array of the converted npshape, it would look like:
    "C"-style ordering:
    [ [1 2]
      [3 4] ]
    [ [5 6]
      [7 8] ]
    "F"-style ordering:
    [ [1 5]
      [3 7] ]
    [ [2 6]
      [4 8] ]
Yikes, neither are correct. Therefore, we can only pick one out of:
    - consistent memory layout
    - consistent data interpretation
    # where consistent means the matrix and the backing array agree.
Note this interpretation is irrelevant for vector shapes, so we will only
consider non-vector matrices when deciding. The two essentially mean that:
    For consistent memory layout:
      - Any non-element-wise operations (like matrix multiply, .at, .along, etc.)
        require re-ordering the backing array into data-correct, then applying
        the numpy operation, then re-ordering to layout-correct again.
      - .tonumpy requires a re-order.
      - requires extra shape handling to convert from shape -> npshape ->
        np_consistent_memory_layout_shape
    For consistent data interpretation:
      - Any operations defined by memory layout (like .ravel, .reshape, .etc)
        require re-ordering the backing array into layout-correct, then applying
        the numpy operation, then re-ordering to data-correct again.
I'm going to pick consistent data interpretation, to prioritise operations like
matrix multiply over operations like reshape (plus .ravel is most used for
vectors and there's no speed penalty there).
"""


class Permuter:
    """
    Sequence of axis indices representing a new axis ordering, with implicitly
    infinite trailing entries which are their own index (meaning no-op).
    """

    def __init__(p, *order):
        """
        Creates a permuter for the given axis ordering.
        """
        if len(order) == 1 and isinstance(order[0], Shape):
            order = order[0]._order
        else:
            order = _maybe_unpack_ints(order)
            for l in order:
                if not isinstance(l, int):
                    raise TypeError("dimension lengths must be ints, got "
                            f"{_objtname(l)}")
            if any(l < 0 for l in order):
                raise ValueError(f"dimension lengths cannot be negative, got: "
                        f"{order}")
            # Check ordering is logical.
            if len(set(order)) != len(order):
                raise ValueError(f"cannot duplicate axes, got: {order}")
            if order and max(order) != len(order) - 1:
                assert max(order) > len(order) - 1
                raise ValueError(f"missing axis for swap, axis {max(order)} is "
                        "dangling")
        # Trim any trailing no-ops.
        while order and order[-1] == len(order) - 1:
            order = order[:-1]
        p._order = order
    @classmethod
    def tonumpy(cls, ndim):
        """
        Permutation from matrix shape of the number of dimensions to the
        equivalent numpy shape.
        """
        if not isinstance(ndim, int):
            raise TypeError("expected an integer number of dimensions, got "
                    f"{_objtname(ndim)}")
        if ndim < 0:
            raise ValueError("cannot have a negative number of dimensions, "
                    f"got: {ndim}")
        return Permuter(tuple(range(ndim - 1, 1, -1)) + (0, 1))
    @classmethod
    def fromnumpy(cls, ndim):
        """
        Inverse permutation of 'P.tonumpy(ndim)'.
        """
        if not isinstance(ndim, int):
            raise TypeError("expected an integer number of dimensions, got "
                    f"{_objtname(ndim)}")
        if ndim < 0:
            raise ValueError("cannot have a negative number of dimensions, "
                    f"got: {ndim}")
        return Permuter.tonumpy(ndim).inv


    def order(p, ndim):
        """
        Tuple of permutation for the given number of dimensions.
        """
        if not isinstance(ndim, int):
            raise TypeError("expected an integer number of dimensions, got "
                    f"{_objtname(ndim)}")
        if ndim < 0:
            raise ValueError("cannot have a negative number of dimensions, "
                    f"got: {ndim}")
        order = p._order
        # Add ids for remaining.
        order += tuple(range(len(order), ndim))
        return order

    @_instconst
    def ndim(p):
        """
        Highest permuted axis index.
        """
        return len(p._order)

    @_instconst
    def inv(p):
        """
        Inversion of this permutation, reversing its effects.
        """
        inv = [None] * len(p._order)
        for i, axis in enumerate(p._order):
            inv[axis] = i
        return Permuter(inv)

    def __iter__(p):
        raise TypeError("cannot iterate permuter (for the (likely) intended "
                "behaviour iterate 'p.order(p.ndim)')")

    def __len__(p):
        raise TypeError("all permuters are implicitly infinite (for the number "
                "of dimensions use 's.ndim')")

    def __getitem__(p, i):
        """
        Returns the new axis for this index.
        """
        if not isinstance(i, int):
            raise TypeError(f"expected integer index, got {_objtname(i)}")
        if i >= len(p._order):
            return i
        return p._order[i]

    def __call__(p, *seq):
        """
        Returns the permutation of the given sequence.
        """
        if len(seq) == 1 and isinstance(seq[0], (Permuter, Shape)):
            seq = seq[0]
            rtype = type(seq)
            ndim = seq.ndim
        else:
            seq = _maybe_unpack(seq)
            rtype = tuple
            ndim = len(seq)
        return rtype(seq[axis] for axis in p.order(ndim))

    def __repr__(p):
        return f"<permuter: {p._order}>"



class Shape:
    """
    Sequence of the length of each dimension, with implicitly infinite trailing
    1s (or 0s if empty). Note that the first dimension is the number of rows and
    the second is the number of columns (further dimensions have no intrinsic
    meaning, other than repeating matrices).
    """

    def __init__(s, *lens):
        """
        Creates a shape with the given axis lengths.
        """
        if len(lens) == 1 and isinstance(lens[0], Shape):
            lens = lens[0]._lens
        else:
            lens = _maybe_unpack_ints(lens)
            for l in lens:
                if not isinstance(l, int):
                    raise TypeError("dimension lengths must be ints, got "
                            f"{_objtname(l)}")
            if any(l < 0 for l in lens):
                raise ValueError(f"dimension lengths cannot be negative, got: "
                        f"{lens}")
            # Collapse empty.
            if _math.prod(lens) == 0:
                lens = (0, )
            # Trim any trailing 1s (note this collapses single to ()).
            while lens and lens[-1] == 1:
                lens = lens[:-1]
        s._lens = lens
    @_instconst
    def tonumpy(s):
        """
        Equivalent numpy shape tuple.
        """
        # empty canonical shape is (0, )
        if s.isempty:
            return (0, )
        npshape = Permuter.tonumpy(s.ndim)(s)
        # otherwise pad to at-least 2 dimensions.
        return tuple(npshape[i] for i in range(max(2, s.ndim)))
    @classmethod
    def fromnumpy(cls, npshape):
        """
        Equivalent shape to a numpy shape tuple.
        """
        if not isinstance(npshape, tuple):
            raise TypeError("expected tuple for 'npshape', got "
                    f"{_objtname(npshape)}")
        if len(npshape) == 0:
            raise ValueError("cannot have empty numpy shape")
        for l in npshape:
            if not isinstance(l, int):
                raise TypeError("numpy shape elements must be ints, got "
                        f"{_objtname(l)}")
        if any(l < 0 for l in npshape):
            raise ValueError(f"numpy shape elements cannot be negative, got: "
                    f"{npshape}")
        if npshape == (0, ):
            return Shape.empty
        if len(npshape) == 1:
            npshape += (1, ) # colvec.
        shape = Permuter.fromnumpy(len(npshape))(npshape)
        return Shape(shape)

    @classmethod
    def sqrshape(cls, *lens):
        """
        Returns a shape of the given lengths. If only one length is given,
        returns a square shape of that side-length.
        """
        lens = _maybe_unpack_ints(lens)
        for l in lens:
            if not isinstance(l, int):
                raise TypeError("dimension lengths must be ints, got "
                        f"{_objtname(l)}")
        if any(l < 0 for l in lens):
            raise ValueError(f"dimension lengths cannot be negative, got: "
                    f"{lens}")
        if len(lens) == 1:
            lens *= 2
        return cls(lens)


    @_classconst
    def empty(cls):
        """
        0x0 shape.
        """
        return cls(0)
    @_classconst
    def single(cls):
        """
        1x1 shape.
        """
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
        return -1 if s.isempty else len(s._lens)
    @_instconst
    def lastaxis(s):
        """
        Index of the last axis with non-1 length. Returns 0 for empty and single.
        """
        return max(0, len(s._lens) - 1)

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
        raise TypeError("cannot iterate shape (for the (likely) intended "
                "behaviour use 's[axis] for axis in range(s.ndim)')")

    def __len__(s):
        raise TypeError("all shapes are implicitly infinite (for the number of "
                "dimensions use 's.ndim')")

    def __getitem__(s, axis):
        """
        Returns the length of the given axis.
        """
        # Oob dimensions are implicitly 1 (unless empty, in which case 0).
        if not isinstance(axis, int):
            raise TypeError(f"expected an integer axis, got {_objtname(axis)}")
        if axis < 0:
            raise ValueError(f"axis cannot be negative, got: {axis}")
        if axis >= s.ndim:
            return 1 if s.size else 0
        return s._lens.__getitem__(axis)

    def insert(s, axis, shape):
        """
        Inserts the given 'shape' at position 'axis', shifting the previous axis
        lengths up.
        """
        if not isinstance(axis, int):
            raise TypeError(f"expected an integer axis, got {_objtname(axis)}")
        if axis < 0:
            raise ValueError(f"axis cannot be negative, got: {axis}")
        if not isinstance(shape, Shape):
            raise TypeError(f"expected a Shape shape, got {_objtname(shape)}")
        if shape.isempty:
            raise TypeError("cannot insert empty shape")
        if s.isempty:
            return s
        newshape = list(s._lens) + [1] * (axis - len(s._lens))
        newshape[axis:axis] = shape._lens
        return Shape(newshape)

    def withaxis(s, axis, length):
        """
        Replaces the length of 'axis' with the given 'length'.
        """
        if not isinstance(axis, int):
            raise TypeError(f"expected an integer axis, got {_objtname(axis)}")
        if axis < 0:
            raise ValueError(f"axis cannot be negative, got: {axis}")
        if s.isempty:
            return s
        newshape = list(s._lens) + [1] * (axis + 1 - len(s._lens))
        newshape[axis] = length
        return Shape(newshape)

    def dropaxis(s, axis):
        """
        Removes the given axis, shifting the later axes down. Note this axis must
        already have length 1 or 0 (aka it must not have contributed to the
        size).
        """
        if not isinstance(axis, int):
            raise TypeError(f"expected an integer axis, got {_objtname(axis)}")
        if axis < 0:
            raise ValueError(f"axis cannot be negative, got: {axis}")
        if s.isempty:
            return s
        if s[axis] != 1:
            raise ValueError(f"cannot drop axis with length >1, axis {axis} "
                    f"has length {s[axis]}")
        return Shape(l for i, l in enumerate(s._lens) if i != axis)

    def check_broadcastable(s, o):
        if not isinstance(o, Shape):
            raise TypeError(f"expected a shape, got {_objtname(o)}")
        if s.isempty and o.isempty:
            return
        if s.isempty:
            raise ValueError("cannot broadcast from empty")
        if o.isempty:
            raise ValueError("cannot broadcast to empty")
        for axis in range(max(o.ndim, s.ndim)):
            if s[axis] == 1:
                continue
            if s[axis] != o[axis]:
                raise ValueError(f"cannot broadcast axis {axis} with length "
                        f"{s[axis]} to {o[axis]}, when attempting to broadcast "
                        f"{s} to {o}")

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
        shape = (s[0], s[1]) if s.ndim < 2 else s._lens
        return "x".join(map(str, shape))




class NO_SEED:
    def __repr__(self):
        return "NO_SEED"
NO_SEED = NO_SEED() # :(


# Matrix.
@_templated(decorators=_immutable)
def Matrix(field, shape):
    """
    Fixed-sized n-dimensional sequence of elements.
    """

    if not isinstance(field, type):
        raise TypeError(f"expected a type for field, got {_objtname(field)}")
    if issubclass(field, Matrix):
        raise TypeError("mate a matrix of matrices? calm down")
    if not isinstance(shape, Shape):
        raise TypeError(f"expected a shape for shape, got {_objtname(shape)}")

    # Note we don't enforce `field` to actually be a Field (i.e. inherit from
    # Field). Without inheriting from Field, we don't recognise many operations,
    # and primarily only allow iteration/access (which may still be desirable).
    # Note the methods will are given defaults even over non-fields:
    # - dtype: object
    # - consts: empty dict
    # - hashed: forwards to `hash`
    # - rep: forwards to `repr`

    @classmethod
    def _doesnthave(cls, thing):
        isfor = {
            "fromobj": "cannot be cast from",

            "to_int": "cannot be cast to int",
            "to_float": "cannot be cast to float",
            "to_complex": "cannot be cast to complex",

            "0": "has no additive identity (zero)",
            "1": "has no multiplicative identity (one)",
            "i": "has no imaginary unit (i)",
            "e": "cannot represent euler's number (e)",
            "pi": "cannot represent pi",

            "add": "cannot do addition",
            "sub": "cannot do subtraction",

            "abs": "cannot do absolution",
            "real": "cannot do complex take-real",
            "imag": "cannot do complex take-imaginary",

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

            "issame": "cannot check if identical",

            "eq": "cannot do equality",
            "neq": "cannot do inequality",
            "lt": "cannot do less-than",
            "le": "cannot do less-than-or-equal-to",

            "hashed": "cannot hash",

            "rep": "cannot stringify",
        }
        if issubclass(cls.field, Field):
            tname = f"field {_tname(cls.field)}"
        else:
            tname = f"non-field class {_tname(cls.field)}"
        raise NotImplementedError(f"{isfor[thing]} over {tname}")
    @classmethod
    def _need(cls, method):
        if issubclass(cls.field, Field):
            has = hasattr(cls.field, method)
        else:
            # We provide a non-field impl of rep.
            has = (method in {"hashed", "rep"})
        # consts and dtype are always implicit if not supplied.
        if method in {"dtype", "consts"}:
            has = True
        if not has:
            cls._doesnthave(method)
    @classmethod
    def _f(cls, method): # Retrives "method" from the field.
        cls._need(method)
        # some things are always defautled even tho field should impl them.
        dflts = {
            "dtype": (lambda: _bg.np.dtype(object)),
            "consts": (lambda: dict()),
        }
        if issubclass(cls.field, Field):
            if method in dflts:
                return getattr(cls.field, method, dflts[method])
            return getattr(cls.field, method)
        else:
            if method in dflts:
                return dflts[method]
            if method == "hashed":
                return lambda a: hash(a)
            if method == "rep":
                return lambda a, short: repr(a)
            assert False



    def __init__(s, cells=None):
        # All cells are stored in a numpy array of the shape `s.shape.tonumpy`,
        # with dtype `s._dtype`.
        if not isinstance(cells, _bg.np.ndarray):
            raise TypeError(f"expected numpy array, got {_objtname(cells)}")
        npshape = s.shape.tonumpy
        if cells.shape != npshape:
            raise ValueError(f"expected {npshape} shape, got {cells.shape}")
        dtype = s._f("dtype")()
        if cells.dtype is not dtype:
            raise ValueError(f"expected {dtype} dtype, got {cells.dtype}")
        # Make it read-only.
        cells.flags["WRITEABLE"] = False
        s._cells = cells


    @classmethod
    def cast(cls, *xs, broadcast=True):
        """
        Attempts to cast each object to a matrix over 'field', all with the same
        size. Note that the shape of this class is ignored. If 'broadcast' is
        false, shapes will be left unchanged.
        """
        xs = _maybe_unpack_mats(xs)
        if not xs:
            return ()

        # Cast to the correct field.
        xs = tuple(x if isinstance(x, Matrix) else single(x) for x in xs)
        xs = tuple(x.tofield(cls.field) for x in xs)

        # If no broadcasting, don't do broadcasting, so the final result won't be
        # broadcast. Also if only one nothing to broadcast against.
        if not broadcast or len(xs) == 1:
            return xs

        # Handle empties.
        if all(x.isempty for x in xs):
            return xs
        if any(x.isempty for x in xs):
            raise TypeError("cannot operate with a mix of empty and non-empty "
                    "matrices")

        # Broadcast all to the same (largest) shape.
        ndim = max(x.ndim for x in xs)
        newshape = Shape(max(x.shape[axis] for x in xs) for axis in range(ndim))
        return tuple(x.broadcast(newshape) for x in xs)


    @_classconst
    def _zero(cls):
        try:
            return cls._f("consts")()["__0__"]
        except Exception:
            cls._doesnthave("0")
    @_classconst
    def _one(cls):
        try:
            return cls._f("consts")()["__1__"]
        except Exception:
            cls._doesnthave("1")
    @_classconst
    def _two(cls):
        try:
            return cls._f("fromobj")(2)
        except Exception:
            raise NotImplementedError("cannot represent 2") from e
    @_classconst
    def _three(cls):
        try:
            return cls._f("fromobj")(3)
        except Exception:
            raise NotImplementedError("cannot represent 3") from e
    @_classconst
    def _ten(cls):
        try:
            return cls._f("fromobj")(10)
        except Exception:
            raise NotImplementedError("cannot represent 10") from e
    @_classconst
    def _180(cls):
        try:
            return cls._f("fromobj")(180)
        except Exception:
            raise NotImplementedError("cannot represent 180") from e
    @_classconst
    def _e(cls):
        try:
            return cls._f("consts")()["__e__"]
        except Exception:
            cls._doesnthave("e")
    @_classconst
    def _pi(cls):
        try:
            return cls._f("consts")()["__pi__"]
        except Exception:
            cls._doesnthave("pi")
    @_classconst
    def _i(cls):
        try:
            return cls._f("fromobj")(1j)
        except Exception as e:
            cls._doesnthave("i")


    @_classconst
    def eye(cls):
        """
        Identity matrix.
        """
        if not cls.issquare:
            raise TypeError("only square matricies have an identity matrix, got "
                    f"{cls.shape}")
        if cls.isempty:
            return cls([])
        if cls.issingle:
            return cls.one
        zero = cls._zero
        one = cls._one
        npshape = cls.shape.tonumpy
        dtype = cls._f("dtype")()
        cells = _bg.np.full(npshape, zero, dtype=dtype)
        _bg.np.fill_diagonal(cells, one)
        return cls(cells)
    @_classconst
    def zeros(cls):
        """
        Zero-filled matrix.
        """
        if cls.isempty:
            return cls([])
        zero = cls._zero
        npshape = cls.shape.tonumpy
        dtype = cls._f("dtype")()
        cells = _bg.np.full(npshape, zero, dtype=dtype)
        return cls(cells)
    @_classconst
    def ones(cls):
        """
        One-filled matrix.
        """
        if cls.isempty:
            return cls([])
        one = cls._one
        npshape = cls.shape.tonumpy
        dtype = cls._f("dtype")()
        cells = _bg.np.full(npshape, one, dtype=dtype)
        return cls(cells)

    @_classconst
    def zero(cls):
        """
        Single zero.
        """
        return single(cls._zero, field=cls.field)
    @_classconst
    def one(cls):
        """
        Single one.
        """
        return single(cls._one, field=cls.field)
    @_classconst
    def e(cls):
        """
        Single euler's number (2.71828...).
        """
        return single(cls._e, field=cls.field)
    @_classconst
    def pi(cls):
        """
        Single pi (3.14159...).
        """
        return single(cls._pi, field=cls.field)
    @_classconst
    def i(cls):
        """
        Single imaginary unit.
        """
        return single(cls._i, field=cls.field)

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
        Index of the last axis with non-1 length. Returns 0 for empty and single.
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


    def numpyarr(s, dtype=None):
        """
        Read-only numpy nd-array of the matrix, of the given 'dtype' and of the
        shape 'm.shape.tonumpy'. If 'dtype' is none, uses the natural dtype of
        the field (typically 'object' unless the field is numeric, note in
        particular that integers are still 'object' to ensure python's unbounded
        integers are used).
        WARNING: numpy uses different shape logic to matrices, so the returned
                 array is always at-least 2D (unless empty). To get a vector
                 (provided 'm' is a vector), use 'm.numpyvec(dtype)'.
        """
        if dtype is None:
            return s._cells
        # TODO:
        raise TypeError("haven don it yet")
    def numpyvec(s, dtype=None):
        """
        Read-only numpy 1d-array of the vector, of the given 'dtype'. If 'dtype'
        is none, uses the natural dtype of the field (typically 'object' unless
        the field is numeric, note in particular that integers are still 'object'
        to ensure python's unbounded integers are used).
        """
        if not s.isvec:
            raise TypeError("expected vector to convert to numpy vector, got "
                    f"{s.shape}")
        if dtype is None:
            return s._cells.reshape(-1)
        # TODO:
        raise TypeError("haven don it yet")


    def tofield(s, newfield):
        """
        Casts all cells to the given field.
        """
        if not isinstance(newfield, type):
            raise TypeError("expected a type for 'newfield', got "
                    f"{_objtname(newfield)}")
        if newfield is s.field:
            return s
        if s.isempty:
            return empty(newfield)
        Mat = Matrix[newfield, s.shape]
        dtype = Mat._f("dtype")()
        fromobj = Mat._f("fromobj")
        fromobj = _bg.np.vectorize(fromobj, otypes=[dtype])
        # cells = fromobj(s._cells).astype(dtype, casting="no", copy=True)
        return Mat(fromobj(s._cells))


    def reshape(s, *newshape):
        """
        Views the ravelled cells under the given 'shape'. The size of the new
        shape must be equal to the size of the current shape (aka cannot change
        the number of elements). Note this is distinct to 'permute', which
        changes axis ordering and therefore alters the ravelled cells.
        """
        newshape = Shape(*newshape)
        if newshape.size != s.shape.size:
            raise ValueError("cannot change size (no. elements) when reshaping, "
                    f"expected {s.shape.size} size, got {newshape.size}")
        # Reshape the correctly (matrix-memory-layout) ravelled cells.
        cells = s.ravel._cells
        cells = cells.reshape(newshape.tonumpy)
        return Matrix[s.field, newshape](cells)

    @_instconst
    def ravel(s):
        """
        Vector of cells in natural iteration order (sequential axes, which is
        row-major), aka the flattened cells.
        """
        # See the rant earlier in this file about numpy interoperation, short
        # answer is the backing array memory layout isnt what we expect so we got
        # work to do.
        if s.ndim < 2:
            # If vector, it doesn't matter.
            cells = s._cells.reshape(-1)
        elif s.ndim == 2:
            # If 2d, can use f-style ordering to get our memory layout.
            cells = s._cells.ravel(order="F")
        else:
            # For higher dimensions, easiest way to get the memory layout we
            # expect is to do 2d matrix transpose then read off c-style. bob the
            # builder type shi.
            npaxes = tuple(range(len(s._cells.ndim) - 2)) + (-1, -2)
            cells = s._cells.transpose(npaxes).ravel(order="C")
        return Matrix.fromnumpy(s.field, cells)

    def broadcast(s, *newshape):
        """
        Broadcasts the matrix to a new shape. Broadcasting allows axes which
        previously had a length of 1 to be "broadcast" to the (larger) requested
        length by repeating along that axis. Note that for all axes in the new
        shape, it must satisfy:
            if current_length == 0: # (only occurs on empties)
                assert new_length == 0
            elif current_length == 1:
                assert new_length > 0
            else:
                assert new_length == current_length
        """
        newshape = Shape(*newshape)
        s.shape.check_broadcastable(newshape)
        cells = _bg.np.broadcast_to(s._cells, newshape.tonumpy)
        return Matrix[s.field, newshape](cells)

    def dropaxis(s, axis):
        """
        Removes the given axis from the shape, shifting the later axes down. Note
        this axis must already have length 1 or 0 (aka it must not have
        contributed to the size, and so dropping it does not remove any cells).
        """
        return s.reshape(s.shape.dropaxis(axis))

    def permute(s, *neworder):
        """
        Permutes the axes into the given order (like a transpose).
        """
        permuter = Permuter(*neworder)
        # Empty and single are invariant under permutation.
        if s.isempty or s.issingle:
            return s
        newshape = permuter(s.shape)
        # Vector handled separately cause its easy (no data reordering).
        if s.isvec:
            cells = s._cells.reshape(newshape.tonumpy)
        else:
            cells = s._cells
            # Gotta make the missus happy. Numpy will never implicitly add
            # dimensions, to lets do that first.
            ndim = max(newshape.ndim, s.ndim)
            npshape = (1, ) * (newshape.ndim - s.ndim) + s.shape.tonumpy
            cells = cells.reshape(npshape)
            # Now gotta make the permutation tuple in the correct format.
            tonp = Permuter.tonumpy(ndim)
            fromnp = Permuter.fromnumpy(ndim)
            nppermuter = tonp(permuter(fromnp))
            nporder = nppermuter.order(ndim)
            cells = cells.transpose(nporder)
            # yay. also check it remains consistent (might have superfluous axes
            # or something)
            cells = cells.reshape(newshape.tonumpy)
        return Matrix[s.field, newshape](cells)


    @_instconst
    def at(s):
        """
        Submatrix of the given indices (may be a single cell). There are two
        modes of access: indexing and masking.
        - Indexing
            Performed when sets of integers are given for each axis, the set
            product of which will be used as indices to form the returned matrix.
            The order of axis access is the same as shape, and is implicitly
            infinite with appended ':'s.
            m.at[0, 1] # row 0, column 1, of every 2d matrix.
            m.at[i, j, k] == m.at[i, j, k, :, :, :, :]
            GETTING:
                Each axis may be a bare integer, a slice, or a sequence of
                integers. All indices must be in bounds for their axis (except
                slices), and repetition of indices within an axis is not allowed.
            SETTING:
                The rhs matrix must be shape-compatible with the accessed shape.
                If the rhs matrix is empty, this is equivalent to ignoring all
                given indices (note this also requires the missing indices to
                produce a hyperrectangular shape (aka a valid matrix shape)).
        - Masking
            Performed when a single shape-compatible boolean matrix is given.
            This creates a vector of the canon matrix traversal where the entry
            is only included if the mask entry is true.
            m.at[[True, False][False, True]] # access diag of 2x2 matrix.
            GETTING:
                Mask must be a shape-compatible boolean matrix. Returns the
                vector of entries corresponding to mask trues.
            SETTING:
                The rhs matrix must be a vector of the same length as the masked
                access. All entries in the accessed matrix which correspond to
                true in the mask will be replaced by the corresponding element
                from the rhs. If the rhs matrix is empty, this is equivalent to
                a reshaped access of the inverted mask (note this also requires
                the inverted mask to produce a hyperrectangular shape (aka a
                valid matrix shape)).
        """
        return _MatrixAt(s)


    def along(s, axis):
        """
        Tuple of perpendicular matrices along the given axis.
        """
        if not isinstance(axis, int):
            raise TypeError(f"expected integer axis, got {_objtname(axis)}")
        if axis < 0:
            raise ValueError(f"cannot have negative axis, got: {axis}")
        # Empty is empty.
        if s.isempty:
            return ()
        if axis >= s.ndim:
            return (s, )
        def idx(i):
            idx = [slice(None)] * s.ndim
            idx[axis] = slice(i, i + 1)
            idx = tuple(idx)
            return Permuter.tonumpy(s.ndim)(idx)
        return tuple(s.at[idx(i)] for i in range(s.shape[axis]))


    def rep(s, *counts):
        """
        Repeats this matrix 'counts[axis]' times for each axis.
        """
        counts = _maybe_unpack_ints(counts)
        for count in counts:
            if not isinstance(count, int):
                raise TypeError("expected an integer count, got "
                        f"{_objtname(count)}")
        if any(count < 0 for count in counts):
            raise ValueError(f"cannot have negative counts, got: {counts}")
        # Use shape as helper.
        counts = Shape(counts)
        if counts.isempty:
            return empty(s.field)
        # Reorder counts to the numpy ordering.
        ndim = max(s.ndim, counts.ndim)
        npcounts = Permuter.tonumpy(ndim)(counts)
        npcounts = tuple(npcounts[axis] for axis in range(ndim))
        cells = _bg.np.tile(s._cells, npcounts)
        return Matrix.fromnumpy(s.field, cells)

    def rep_along(s, axis, count):
        """
        Repeats this matrix 'count' times along 'axis'.
        """
        if not isinstance(axis, int):
            raise TypeError(f"expected an integer axis, got {_objtname(axis)}")
        if axis < 0:
            raise ValueError(f"axis cannot be negative, got: {axis}")
        if not isinstance(count, int):
            raise TypeError(f"expected an integer count, got {_objtname(count)}")
        if count < 0:
            raise TypeError(f"count cannot be negative, got: {count}")
        return s.rep((1, )*axis + (count, ))


    def __iter__(s):
        """
        Vector-only cell iterate.
        """
        if not s.isvec:
            raise TypeError(f"only vectors have bare iteration, got {s.shape} "
                    "(use .along or .ravel for matrix iterate)")
        return (single(x, field=s.field) for x in s._cells.ravel())

    def __getitem__(s, i):
        """
        Vector-only cell access, supporting slices.
        """
        if not s.isvec:
            raise TypeError(f"only vectors have bare getitem, got {s.shape} "
                    "(use .at for matrix cell access)")
        if isinstance(i, int):
            if i < -s.size or i >= s.size:
                raise IndexError(f"index {i} out of bounds for size {s.size}")
            if i < 0:
                i += s.size
            i = slice(i, i + 1)
        if not isinstance(i, slice):
            raise TypeError("expected integer or slice to index vector, got "
                    f"{_objtname(i)}")
        newshape = s.shape.withaxis(s.lastaxis, len(cells))
        cells = s._cells.ravel().__getitem__(i)
        cells = cells.reshape(newshape.tonumpy)
        return Matrix[s.field, newshape](cells)

    def __len__(s):
        """
        Vector-only cell count, alias for 's.size'.
        """
        if not s.isvec:
            raise TypeError(f"only vectors have bare length, got {s.shape} (use "
                    ".size for matrix cell count)")
        return s.size


    @_instconst
    def cols(s):
        """
        Tuple of columns, for 2D matrices.
        """
        if s.ndim > 2:
            raise TypeError(f"only 2D matrices can use .cols, got {s.shape} "
                    "(use .along for other matrices)")
        return s.along(1)
    @_instconst
    def rows(s):
        """
        Tuple of rows, for 2D matrices.
        """
        if s.ndim > 2:
            raise TypeError(f"only 2D matrices can use .rows, got {s.shape} "
                    "(use .along for other matrices)")
        return s.along(0)
    @_instconst
    def colmajor(s):
        """
        Vector of cells in column-major order, for 2D matrices.
        """
        if s.ndim > 2:
            raise TypeError(f"only 2D matrices can use .colmajor, got {s.shape} "
                    "(use .ravel (maybe with .permute) for other matrices)")
        # Can speed up ravelling in 2D by using numpy's two orderings.
        return Matrix.fromnumpy(s.field, s._cells.ravel(order="F"))
    @_instconst
    def rowmajor(s):
        """
        Vector of cells in row-major order, for 2D matrices.
        """
        if s.ndim > 2:
            raise TypeError(f"only 2D matrices can use .rowmajor, got {s.shape} "
                    "(use .ravel (maybe with .permute) for other matrices)")
        return Matrix.fromnumpy(s.field, s._cells.ravel(order="C"))


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
        if s.isempty:
            return s
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
        if s.isempty or s.issingle:
            return s
        return Matrix.fromnumpy(s.field, s._cells.diagonal())


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
        Is upper-triangular matrix? (square, and below main diagonal is zero)
        """
        if s.ndim > 2:
            raise TypeError(f"only 2D matrices can be triangular, got {s.shape}")
        if not s.issquare:
            return False
        for i in range(s.shape[0]):
            for j in range(i + 1, s.shape[1]):
                if s.at[i, j] != s.zero:
                    return False
        return True

    @_instconst
    def islowertri(s):
        """
        Is lower-triangular matrix? (square, and above main diagonal is zero)
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
        # TODO: make all
        return bool(s.T == s.inv)

    @_instconst
    def issymmetric(s):
        """
        Is symmetric matrix? (square, and below main diagonal = above main
        diagonal)
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
    def _apply(cls, func, rfield, *xs, user=False):
        # Expects func to be a nullary which returns the actual function which
        # operates on field elements and returns rfield elements. if `user`,
        # expects a return of `field, numpy array`.

        # Check me.
        if not xs:
            if rfield is None:
                rfield = cls.field
            return empty(rfield)
        # Cast me.
        xs = cls.cast(*xs)
        # Shape me.
        xshape = xs[0].shape
        # Empty me (dujj).
        if xshape.isempty:
            if rfield is None:
                rfield = cls.field
            return tuple(empty(rfield) for _ in xs)
        # Expose me (dujj).
        xs = [x._cells for x in xs]
        # Unwrap me (dujj).
        func = func()
        # Flatten me.
        xs = [x.reshape(-1) for x in xs]
        # Newshape me.
        newshape = xshape
        # First me.
        if user:
            ret_field, first = func(*(x[0] for x in xs))
            if rfield is None:
                rfield = ret_field
            if ret_field != rfield:
                raise TypeError("inconsistent return field, expected "
                        f"{rfield}, got {ret_field}")
            rshape = Shape.fromnumpy(first.shape)
            if rshape.isempty:
                raise TypeError("cannot return empty from applied function")
            newshape = newshape.insert(newshape.ndim, rshape)
        else:
            first = func(*(x[0] for x in xs))
            if rfield is None:
                rfield = type(first)
        # Npshape me.
        npshape = (xshape.size, )
        if user:
            npshape += rshape.tonumpy
        # Dtype me.
        dtype = Matrix[rfield, Shape.empty]._f("dtype")()
        # Preallocate me.
        cells = _bg.np.empty(npshape, dtype)
        # Do me (dujj).
        cells[0] = first
        for i in range(1, npshape[0]):
            if not user:
                ret = func(*(x[i] for x in xs))
            else:
                ret_field, ret = func(*(x[i] for x in xs))
                assert isinstance(ret, _bg.np.ndarray)
                if ret_field != rfield:
                    raise TypeError("inconsistent return field, expected "
                            f"{rfield}, got {ret_field}")
                if ret.dtype != dtype:
                    raise TypeError("inconsistent return dtype, expected "
                            f"{dtype}, got {ret.dtype}")
                if ret.shape != rshape.tonumpy:
                    raise TypeError("inconsistent return shape, expected "
                            f"{rshape}, got {Shape.fromnumpy(ret.shape)}")
            cells[i] = ret
        # Dropaxes me.
        cells = cells.reshape(newshape.tonumpy)
        return Matrix[rfield, newshape](cells)

    @classmethod
    def applyto(cls, func, *xs, rfield=None):
        """
        Constructs a matrix from the results of 'func(a, b, ...)' for all zipped
        elements in '*xs'. If 'func' returns a non-single matrix, the shape of
        the return will have these elements appended into new axes. 'rfield' can
        be used to dictate the field of the return, otherwise it will be
        inferred.
        """
        if not callable(func):
            raise TypeError(f"expected callable 'func', got {_objtname(func)}")
        if rfield is not None and not isinstance(rfield, type):
            raise TypeError("expected type for 'rfield', got "
                    f"{_objtname(rfield)}")
        def wrapped(*ys):
            # Wrap in singles.
            ys = [single(y, field=cls.field) for y in ys]
            ret = func(*ys)
            # Validate me (dujj).
            if not isinstance(ret, Matrix):
                ret = single(ret)
            # Unpack to field and cells.
            return ret.field, ret._cells
        return cls._apply(lambda: wrapped, rfield=rfield, *xs, user=True)

    def apply(s, func, *os, rfield=None):
        """
        Alias for 'M.applyto(func, s, *os, rfield=rfield)'.
        """
        return s.applyto(func, s, *os, rfield=rfield)


    def _fold(s, func, axis=None, seed=NO_SEED, right=False):
        # Expects func to be a nullary which returns the actual function which
        # operates on field elements and returns field elements. If `axis` is
        # none, seed must be a field element, otherwise it must be a
        # broadcastable perpendicular matrix. func always operates on and returns
        # field elements.

        if not s.isempty:
            func = func()
        # otherwise shouldnt be called.

        if right:
            f = lambda a, b: func(b, a)
            order = reversed
        else:
            f = func
            order = lambda x: x

        if axis is None:
            # Iterate through the flat cells.
            flat = s._cells.reshape(-1)
            for x in order(flat):
                if seed is NO_SEED:
                    seed = x
                    continue
                seed = f(seed, x)
            return single(seed, field=s.field)

        perpshape = s.shape.withaxis(axis, 1)
        if seed is not NO_SEED:
            try:
                seed = seed.broadcast(perpshape)
            except ValueError as e:
                raise ValueError("cannot broadcast seed for folding along axis "
                        f"{axis}") from e
        # Iterate along axis.
        for x in order(s.along(axis)):
            if seed is NO_SEED:
                seed = x
                continue
            seed = x._apply(lambda: f, s.field, seed, x)
        if seed is NO_SEED:
            assert s.isempty
            return empty(s.field)
        return seed

    def fold(s, func, axis=None, seed=NO_SEED, right=False):
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
        is performed along that axis in parallel. If 'm' is empty, returns 'seed'
        (which must be specified).
        """
        if not callable(func):
            raise TypeError(f"expected callable 'func', got {_objtname(func)}")
        if axis is not None:
            if not isinstance(axis, int):
                raise TypeError("expected an integer axis, got "
                        f"{_objtname(axis)}")
            if axis < 0:
                raise ValueError(f"axis cannot be negative, got {axis}")
        if seed is not NO_SEED:
            seed, = s.cast(seed, broadcast=False)
            if axis is None:
                if not seed.issingle:
                    raise TypeError("expected a single for 'seed' when folding "
                            f"over all elements, got {seed.shape}")
                seed = seed.obj
        def wrapped(a, b):
            a = single(a, field=s.field)
            b = single(b, field=s.field)
            c = func(a, b)
            if not isinstance(c, Matrix):
                c = single(c, field=s.field)
            if not c.issingle:
                raise TypeError("expected folding function to return a "
                        f"single, got {c.shape}")
            return c.obj
        return s._fold(lambda: wrapped, axis=axis, seed=seed, right=right)


    def __pos__(s):
        """
        Element-wise NOTHING.
        """
        return s
    def __neg__(s):
        """
        Element-wise negation.
        """
        if s.isempty:
            return s
        sub = s._f("sub")
        zero = s._zero
        neg = lambda x: sub(zero, x)
        return s._apply(lambda: neg, s.field, s)
    def __abs__(s):
        """
        Element-wise absolution.
        """
        return s._apply(lambda: s._f("abs"), s.field, s)

    @_instconst
    def abs(s):
        """
        Alias for 'abs(s)'.
        """
        return s.__abs__()
    @_instconst
    def conj(s):
        """
        Element-wise complex conjugate.
        """
        if s.isempty:
            return s
        return s.real - s.i * s.imag
    @_instconst
    def real(s):
        """
        Element-wise take-real.
        """
        return s._apply(lambda: s._f("real"), s.field, s)
    @_instconst
    def imag(s):
        """
        Element-wise take-imaginary.
        """
        return s._apply(lambda: s._f("imag"), s.field, s)

    @_instconst
    def sign(s):
        """
        Element-wise (-1, 0, or 1) corresponding to (<0, =0, or >0).
        """
        if not s.issingle:
            raise NotImplementedError("lemme whip up specialised bool first")
        neg = bool(s <= 0)
        pos = bool(s >= 0)
        if neg + pos == 0:
            raise ValueError(f"could not determine sign of: {s}")
        return pos - neg # one of -1, 0, or 1.

    def __add__(s, o):
        """
        Element-wise addition.
        """
        return s._apply(lambda: s._f("add"), s.field, s, o)
    def __radd__(s, o):
        s, o = s.cast(s, o)
        return o.__add__(s)
    def __sub__(s, o):
        """
        Element-wise subtraction.
        """
        return s._apply(lambda: s._f("sub"), s.field, s, o)
    def __rsub__(s, o):
        s, o = s.cast(s, o)
        return o.__sub__(s)
    def __mul__(s, o):
        """
        Element-wise multiplication (use '@' for matrix multiplication).
        """
        return s._apply(lambda: s._f("mul"), s.field, s, o)
    def __rmul__(s, o):
        s, o = s.cast(s, o)
        return o.__mul__(s)
    def __truediv__(s, o):
        """
        Element-wise division.
        """
        return s._apply(lambda: s._f("div"), s.field, s, o)
    def __rtruediv__(s, o):
        s, o = s.cast(s, o)
        return o.__truediv__(s)
    def __pow__(s, o):
        """
        Element-wise power.
        """
        return s._apply(lambda: s._f("power"), s.field, s, o)
    def __rpow__(s, o):
        s, o = s.cast(s, o)
        return o.__pow__(s)

    @_instconst
    def sqrt(s):
        """
        Element-wise square root.
        """
        if s.isempty:
            return s
        root = s._f("root")
        base = s._two
        rootbase = lambda x: root(x, base)
        return s._apply(lambda: rootbase, s.field, s)
    @_instconst
    def cbrt(s):
        """
        Element-wise cube root.
        """
        if s.isempty:
            return s
        root = s._f("root")
        base = s._three
        rootbase = lambda x: root(x, base)
        return s._apply(lambda: rootbase, s.field, s)

    def root(s, n):
        """
        Element-wise nth root.
        """
        return s._apply(lambda: s._f("root"), s.field, s, n)

    @_instconst
    def exp(s):
        """
        Element-wise natural exponential.
        """
        if s.isempty:
            return s
        power = s._f("power")
        base = s._e
        powbase = lambda x: power(base, x)
        return s._apply(lambda: powbase, s.field, s)
    @_instconst
    def exp2(s):
        """
        Element-wise base-2 exponential.
        """
        if s.isempty:
            return s
        power = s._f("power")
        base = s._two
        powbase = lambda x: power(base, x)
        return s._apply(lambda: powbase, s.field, s)
    @_instconst
    def exp10(s):
        """
        Element-wise base-10 exponential.
        """
        if s.isempty:
            return s
        power = s._f("power")
        base = s._ten
        powbase = lambda x: power(base, x)
        return s._apply(lambda: powbase, s.field, s)

    @_instconst
    def ln(s):
        """
        Element-wise natural logarithm.
        """
        if s.isempty:
            return s
        log = s._f("log")
        base = s._e
        logbase = lambda x: log(base, x)
        return s._apply(lambda: logbase, s.field, s)
    @_instconst
    def log2(s):
        """
        Element-wise base-2 logarithm.
        """
        if s.isempty:
            return s
        log = s._f("log")
        base = s._two
        logbase = lambda x: log(base, x)
        return s._apply(lambda: logbase, s.field, s)
    @_instconst
    def log10(s):
        """
        Element-wise base-10 logarithm.
        """
        if s.isempty:
            return s
        log = s._f("log")
        base = s._ten
        logbase = lambda x: log(base, x)
        return s._apply(lambda: logbase, s.field, s)
    def log(s, base):
        """
        Element-wise base-specified logarithm.
        """
        return s._apply(lambda: s._f("log"), s.field, base, s)


    @_instconst
    def sin(s):
        """
        Element-wise trigonometric sine.
        """
        return s._apply(lambda: s._f("sin"), s.field, s)
    @_instconst
    def cos(s):
        """
        Element-wise trigonometric cosine.
        """
        return s._apply(lambda: s._f("cos"), s.field, s)
    @_instconst
    def tan(s):
        """
        Element-wise trigonometric tangent.
        """
        return s._apply(lambda: s._f("tan"), s.field, s)

    @_instconst
    def asin(s):
        """
        Element-wise trigonometric inverse-sine.
        """
        return s._apply(lambda: s._f("asin"), s.field, s)
    @_instconst
    def acos(s):
        """
        Element-wise trigonometric inverse-cosine.
        """
        return s._apply(lambda: s._f("acos"), s.field, s)
    @_instconst
    def atan(s):
        """
        Element-wise trigonometric inverse-tangent.
        """
        return s._apply(lambda: s._f("atan"), s.field, s)

    @_instconst
    def torad(s):
        """
        Converts degrees to radians, alias for 'm / (180/pi)'.
        """
        if s.isempty:
            return s
        pi = s._pi
        one80 = s._180
        div = s._f("div")
        # maybe preverse largest subproducts.
        one80_pi = div(one80, pi)
        conv = lambda x: div(x, one80_pi)
        return s._apply(lambda: conv, s.field, s)
    @_instconst
    def todeg(s):
        """
        Converts radians to degrees, alias for 'm * (180/pi)'.
        """
        if s.isempty:
            return s
        pi = s._pi
        one80 = s._180
        mul = s._f("mul")
        div = s._f("div")
        # maybe preverse largest subproducts.
        one80_pi = div(one80, pi)
        conv = lambda x: mul(x, one80_pi)
        return s._apply(lambda: conv, s.field, s)

    @_instconst
    def sind(s):
        """
        Element-wise trigonometric sine, with input in degrees.
        """
        return s.torad.sin
    @_instconst
    def cosd(s):
        """
        Element-wise trigonometric cosine, with input in degrees.
        """
        return s.torad.cos
    @_instconst
    def tand(s):
        """
        Element-wise trigonometric tangent, with input in degrees.
        """
        return s.torad.tan

    @_instconst
    def asind(s):
        """
        Element-wise trigonometric inverse-sine, with output in degrees.
        """
        return s.asin.todeg
    @_instconst
    def acosd(s):
        """
        Element-wise trigonometric inverse-cosine, with output in degrees.
        """
        return s.acos.todeg
    @_instconst
    def atand(s):
        """
        Element-wise trigonometric inverse-tangent, with output in degrees.
        """
        return s.atan.todeg


    def diff(s, x):
        """
        Element-wise derivative with respect to 'x'.
        """
        return s._apply(lambda: s._f("diff"), s.field, s, x)

    def intt(s, x, *bounds):
        """
        Element-wise integral with respect to 'x'. If bounds are provided,
        evaluates the definite integral.
        """
        bounds = _maybe_unpack(bounds) # unpack mats also.
        if not bounds:
            return s._apply(lambda: s._f("intt"), s.field, s, x)
        if len(bounds) != 2:
            raise TypeError(f"must specify 0 or 2 bounds, got: {bounds}")
        lo, hi = bounds
        return s._apply(lambda: s._f("def_intt"), s.field, s, x, lo, hi)


    def issame(s, o):
        """
        Element-wise identical check. Note this is different to '==' (which
        checks for equivalent values, and may be different than identical
        values).
        """
        return s._apply(lambda: s._f("issame"), bool, s, o)

    def __eq__(s, o):
        """
        Element-wise equal-to.
        """
        return s._apply(lambda: s._f("eq"), bool, s, o)
    def __ne__(s, o):
        return s._apply(lambda: s._f("neq"), bool, s, o)
    def __lt__(s, o):
        """
        Element-wise less-than.
        """
        return s._apply(lambda: s._f("lt"), bool, s, o)
    def __le__(s, o):
        return s._apply(lambda: s._f("le"), bool, s, o)
    def __gt__(s, o):
        return s._apply(lambda: s._f("lt"), bool, o, s)
    def __ge__(s, o):
        return s._apply(lambda: s._f("le"), bool, o, s)


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

        mul = s._f("mul")
        add = s._f("add")
        sub = s._f("sub")

        # hardcode 2x2.
        if s.shape[0] == 2:
            a, b, c, d = cells.reshape(-1)
            return sub(mul(a, d), mul(b, c))

        def submatrix(cells, size, row):
            return cells[_bg.np.arange(size) != row, 1:]

        def determinant(cells, size):
            # base case 3x3.
            if size == 3:
                a, b, c, d, e, f, g, h, i = cells.reshape(-1)

                ei = mul(e, i)
                fh = mul(f, h)
                di = mul(d, i)
                fg = mul(f, g)
                dh = mul(d, h)
                eg = mul(e, g)
                a_ei_fh = mul(a, sub(ei, fh))
                b_di_fg = mul(b, sub(di, fg))
                c_dh_eg = mul(c, sub(dh, eg))
                return add(sub(a_ei_fh, b_di_fg), c_dh_eg)
                # got an alt not a shift
                aei = mul(mul(a, e), i)
                bfg = mul(mul(b, f), g)
                cdh = mul(mul(c, d), h)
                ceg = mul(mul(c, e), g)
                bdi = mul(mul(b, d), i)
                afh = mul(mul(a, f), h)
                fst = add(add(aei, bfg), cdh)
                snd = add(add(ceg, bdi), afh)
                return sub(fst, snd)

            for row in range(size):
                subcells = submatrix(cells, size, row)
                subsize = size - 1
                subdet = determinant(subcells, subsize)
                subdet = mul(subdet, cells[row, 0])
                if row == 0:
                    det = subdet
                else:
                    if row & 1:
                        det = sub(det, subdet)
                    else:
                        det = add(det, subdet)
            return det

        return single(determinant(cells, s.shape[0]), field=s.field)

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

    def dot(s, o):
        """
        Vector dot product.
        """
        s, o = s.cast(s, o, broadcast=False)
        if not s.isvec or not o.isvec:
            raise TypeError(f"only vectors have a dot product, got {s.shape} "
                    f"and {o.shape}")
        if s.size != o.size:
            raise TypeError("only equal-length vectors have a dot product, "
                    f"got {s.shape} and {o.shape}")
        # Reshape to common vector (defaulting to column).
        newshape = s.shape if (s.shape == o.shape) else Shape(s.size)
        s = s.reshape(newshape)
        o = o.reshape(newshape)
        # Elementwise product then sum.
        return (s * o).summ

    def cross(s, o):
        """
        3-element vector cross product.
        """
        s, o = s.cast(s, o, broadcast=False)
        if not s.isvec or not o.isvec:
            raise TypeError(f"only vectors have a cross product, got {s.shape} "
                    f"and {o.shape}")
        if s.size != 3 or o.size != 3:
            raise TypeError("only 3-element vectors have a cross product, got "
                    f"{s.shape} and {o.shape}")
        # Use crossproduct formula.
        ax, ay, az = s._cells.reshape(-1)
        bx, by, bz = o._cells.reshape(-1)
        mul = s._f("mul")
        sub = s._f("sub")
        cells = _bg.np.array([
                sub(mul(ay, bz), mul(az, by)),
                sub(mul(az, bx), mul(ax, bz)),
                sub(mul(ax, by), mul(ay, bx)),
            ], dtype=s._f("dtype")())
        # If both vectors are the same shape, keep it. Otherwise use colvec.
        newshape = s.shape if (s.shape == o.shape) else Shape(3)
        cells = cells.reshape(newshape.tonumpy)
        return Matrix[s.field, newshape](cells)

    def __matmul__(s, o):
        """
        Matrix multiplication.
        """
        s, o = s.cast(s, o, broadcast=False)
        if s.ndim > 2 or o.ndim > 2:
            raise TypeError(f"only 2D matrices have matrix multiplication, got "
                    f"{s.shape} and {o.shape}")
        if s.shape[1] != o.shape[0]:
            raise TypeError("need equal inner dimension lengths for matrix "
                    f"multiplication, got {s.shape} @ {o.shape}")
        newshape = Shape(s.shape[0], o.shape[1])
        if newshape.isempty:
            return empty(s.field)
        mul = s._f("mul")
        add = s._f("add")
        cells = s.zeros._cells.copy()
        # blazingly fast new matrix multiplication algorithm scientists are
        # dubbing the "naive method" (i think it means really smart).
        for i in range(s.shape[0]):
            for j in range(o.shape[1]):
                for k in range(s.shape[1]):
                    prod = mul(s._cells[i, k], o._cells[k, j])
                    cells[i, j] = add(cells[i, j], prod)
        return Matrix[s.field, newshape](cells)
    def __rmatmul__(s, o):
        s, o = s.cast(s, o, broadcast=False)
        return o.__matmul__(s)

    def __xor__(s, exp):
        """
        Matrix power (repeated self matrix multiplication, possibly inversed).
        """
        if isinstance(exp, Matrix) and exp.issingle:
            try:
                exp = int(exp)
            except Exception as e:
                raise TypeError("expected an integer exponent") from e
        if not isinstance(exp, int):
            raise TypeError("expected an integer exponent, got "
                    f"{_objtname(exp)}")
        if not s.issquare:
            raise TypeError("only square matrices have exponentiation, got "
                    f"{s.shape}")
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

        if s.isempty:
            return s

        zero = s._zero
        one = s._one
        add = s._f("add")
        sub = s._f("sub")
        mul = s._f("mul")
        div = s._f("div")
        eq = s._f("eq")
        neq = s._f("neq")
        eqz = lambda x: eq(zero, x)
        eqo = lambda x: eq(one, x)
        neqz = lambda x: neq(zero, x)
        neqo = lambda x: neq(one, x)
        neg = lambda a: sub(zero, a)
        rec = lambda a: div(one, a)

        cells = s._cells.copy()
        rows, cols = s.shape[0], s.shape[1]

        def row_swap(i1, i2):
            cells[i1, :], cells[i2, :] = cells[i2, :], cells[i1, :]

        def row_mul(i, by):
            for j in range(cols):
                cells[i, j] = mul(cells[i, j], by)

        def row_add(dst, src, by):
            for j in range(cols):
                cells[dst, j] = add(cells[dst, j], mul(cells[src, j], by))

        lead = 0
        for r in range(rows):
            if lead >= cols:
                break

            i = r
            while eqz(cells[i, lead]):
                i += 1
                if i == rows:
                    i = r
                    lead += 1
                    if lead == cols:
                        break
            if lead == cols:
                break
            row_swap(i, r)

            pivot_value = cells[r, lead]
            if neqz(pivot_value):
                row_mul(r, rec(pivot_value))
            # Check its 1.
            pivot_value = cells[r, lead]
            if neqo(pivot_value):
                raise ValueError(f"couldn't make cell =1, cell is: "
                        f"{repr(pivot_value)}")

            for i in range(rows):
                if i == r:
                    continue
                row_lead_value = cells[i, lead]
                if neqz(row_lead_value):
                    row_add(i, r, neg(row_lead_value))
                # Check its 0.
                row_lead_value = cells[i, lead]
                if neqz(row_lead_value):
                    raise ValueError("couldn't make cell =0, cell is: "
                            f"{repr(row_lead_value)}")

            lead += 1

        # Cheeky collapse to exactly zero or one, to remove imprecision.
        @_bg.np.vectorize(otypes=[s._f("dtype")()])
        def collapse(x):
            if eqz(x):
                return zero
            if eqo(x):
                return one
            return x
        cells = collapse(cells)

        return Matrix[s.field, s.shape](cells)

    @_instconst
    def pivots(s):
        """
        Tuple of RREF pivot column indices.
        """
        if s.isempty:
            return ()
        sys = s.rref
        zero = s._zero
        one = s._one
        eq = s._f("eq")
        neq = s._f("neq")
        return tuple(i for i, c in enumerate(sys.cols)
                     if 1 == sum(neq(zero, x) for x in c)
                     and 1 == sum(eq(one, x) for x in c))

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
        if s.isempty:
            return ()
        sys = s.rref
        zero = s._zero
        neq = s._f("neq")
        nonzeros = (i for i, r in enumerate(sys.rows)
                    if any(neq(zero, x) for x in r))
        return tuple(sys.rows[i].T for i in nonzeros)

    @_instconst
    def nullspace(s):
        """
        Tuple of basis vectors for null space.
        """
        sys = s.rref # implied zero-vec augment.
        zero = s._zero
        one = s._one
        eq = s._f("eq")
        sub = s._f("sub")
        neg = lambda a: sub(zero, a)
        def find_first_one(xs):
            for i, x in enumerate(xs):
                if eq(one, x):
                    return i
            return None
        pivotat = tuple(find_first_one(row) for row in sys.rows)
        nonpivots = sys.nonpivots
        Col = Matrix[s.field, Shape(s.shape[1])]
        zeros = Col.zeros._cells
        basis = [zeros.copy() for _ in nonpivots]
        for n, j in enumerate(nonpivots):
            for i in range(s.shape[0]):
                if pivotat[i] is None or pivotat[i] > j:
                    basis[n][j] = one
                    break
                basis[n][pivotat[i]] = neg(sys._cells[i, j])
        return tuple(Col(x) for x in basis)



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
        if s.isempty:
            return s.zero if axis is None else s
        return s._fold(lambda: s._f("add"), axis)

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
        if s.isempty:
            return s.one if axis is None else s
        return s._fold(lambda: s._f("mul"), axis)

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
        take = lambda a, b: b if lt(b, a) else a
        return s._fold(lambda: take, axis)

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
        take = lambda a, b: b if lt(a, b) else a
        return s._fold(lambda: take, axis)

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
        return s._cells[0, 0]

    def __bool__(s):
        """
        Cast a single to bool, returning true iff the element is non-zero.
        """
        if not s.issingle:
            raise TypeError("expected single for scalar cast to bool, got "
                    f"{s.shape}")
        return s._f("neq")(s._zero, s.obj)
    def __int__(s):
        """
        Cast a single to int.
        """
        if not s.issingle:
            raise TypeError("expected single for scalar cast to int, got "
                    f"{s.shape}")
        return s._f("to_int")(s.obj)
    def __float__(s):
        """
        Cast a single to float.
        """
        if not s.issingle:
            raise TypeError("expected single for scalar cast to float, got "
                    f"{s.shape}")
        return s._f("to_float")(s.obj)
    def __complex__(s):
        """
        Cast a single to complex.
        """
        if not s.issingle:
            raise TypeError("expected single for scalar cast to complex, got "
                    f"{s.shape}")
        return s._f("to_complex")(s.obj)

    def __hash__(s):
        return hash((s.shape, ) + tuple(s._f("hashed")(x) for x in s._cells))

    def __repr__(s, short=None, multiline=True):
        if short is None:
            short = doesdflt2short()
        short = not not short

        if s.isempty:
            return "my boy "*(not short) + "M.T."

        rep_ = s._f("rep")
        rep = lambda x: rep_(x, short)

        if s.issingle:
            return rep(s.obj)

        if short and not s.isvec:
            # Shorten elements of zero to a single dot.
            def repme(x):
                if repme.can_dot and repme.issame(x, repme.zero):
                    return "."
                return rep(x)
            repme.can_dot = False
            try:
                repme.issame = s._f("issame")
                repme.zero = s._zero
                repme.can_dot = True
            except Exception:
                pass
        else:
            repme = rep

        # cheeky matrix of the reps to make access easy.
        reps = s._apply(lambda: repme, str, s)
        lens = reps._apply(lambda: len, int, reps)
        width = int(lens.maxx)
        return reps._repr_str(short, width, s.lastaxis, allow_flat=True,
                multiline=multiline)

    def _repr_str(s, short, width, axis, multiline, allow_flat=False):
        # this method is a helper, and is only defined for matrices over str.
        newline = "\n" if multiline else ""
        if axis > 1:
            def tostr(x):
                x = x._repr_str(short, width, axis - 1, multiline)
                x = x.split("\n")
                s = x[0] + "".join(f"\n  {line}" for line in x[1:])
                return "[ " + s + " ]"
            return newline.join(tostr(x) for x in s.along(axis))
        # 2d print.
        # Print col vecs as rows with marked transpose, if allowed.
        suffix = ""
        cells = s._cells
        if allow_flat and s.iscol:
            suffix = _coloured(40, "ᵀ")
            cells = cells.T
        def rowstr(row):
            return [" " * (width - len(_nonctrl(r))) + r for r in row]
        rows = [rowstr(row) for row in cells]
        padded = (not short) or (width > 3)
        join = lambda x: "  ".join(x) if padded else " ".join(x)
        wrap = lambda x: f"[ {x} ]" if padded else f"[{x}]"
        str_rows = (wrap(join(row)) for row in rows)
        return newline.join(str_rows) + suffix
    if field is not str:
        del _repr_str


    # not an actual class, gotta fulfill the templated promises.
    return locals()


class _MatrixAt:
    def __init__(self, matrix):
        self._matrix = matrix
    def _indexfor(self, axis, i):
        length = self._matrix.shape[axis]
        if isinstance(i, slice):
            return i
        if isinstance(i, int):
            if i < -length or i >= length:
                raise IndexError(f"index {i} out of bounds for axis {axis} with "
                        f"length {length}")
            if i < 0:
                i += length
            return slice(i, i + 1)
        if isinstance(i, Matrix):
            i = i.numpyvec(int)
        elif _iterable(i):
            i = list(i)
        if isinstance(i, list):
            i = _bg.np.array(i, dtype=object)
            if i.shape != (len(i), ):
                raise TypeError(f"expected a 1d sequence to index axis {axis}, "
                        f"got shape: {i.shape}")
            for j in i:
                if not isinstance(j, int):
                    raise TypeError("expected a sequence of integers to index "
                            f"axis {axis}, got {_objtname(j)}")
        if not isinstance(i, _bg.np.ndarray):
            raise TypeError("expected a bare integer, slice, mask, or a "
                    f"sequence of integers to index, got {_objtname(i)} for "
                    f"axis {axis}")
        if (i < -length).any() or (i >= length).any():
            oob = (i < -length) | (i >= length)
            raise IndexError(f"axis {axis} with length {length} recieved out of "
                    f"bounds indices: {list(i[oob])}")
        i[i < 0] += length
        return i
    def _indices(self, ijk):
        s = self._matrix
        if s.isempty:
            raise TypeError("cannot index empty matrix")
        if not isinstance(ijk, tuple):
            ijk = (ijk, )
        if not ijk:
            raise TypeError("must specify indices")
        return tuple(self._indexfor(axis, i) for axis, i in enumerate(ijk))

    def __getitem__(self, ijk):
        # TODO:
        # raise NotImplementedError("haven don eit")
        ijk = self._indices(ijk)
        m = self._matrix
        return Matrix.fromnumpy(m.field, m._cells[ijk])
    def __setitem__(self, ijk, rhs):
        # TODO:
        raise NotImplementedError("haven don eit")


@Matrix.screener
def _matrix_screen(params):
    if len(params) == 0:
        raise TypeError("missing 1st argument 'field'")
    if len(params) == 1:
        raise TypeError("missing 2nd argument 'shape'")
    if len(params) > 2:
        raise TypeError("got extra arguments, only accepts 'field' and 'shape'")
    return params
@Matrix.screener
def _matrix_screen_shape(params):
    field, shape = params
    if not isinstance(shape, Shape):
        return field, Shape(*shape)
    return params

# Make a creator from a numpy array.
@_add_to(Matrix, name="fromnumpy")
def _matrix_fromnumpy(field, arr):
    """
    Creates a matrix over the given field from the given numpy array.
    """
    shape = Shape.fromnumpy(arr.shape)
    arr = arr.reshape(shape.tonumpy)
    return Matrix[field, shape](arr)




class Integer(Field):
    @classmethod
    def fromobj(cls, x):
        if isinstance(x, int):
            return x.__int__()
        if isinstance(x, float):
            if not x.is_integer():
                raise ValueError(f"cannot cast non-integer to int, got: {x}")
            return x.__int__()
        if isinstance(x, complex):
            if not (x.imag == 0.0):
                raise TypeError(f"cannot cast complex to int, got: {x}")
            if not x.real.is_integer():
                raise ValueError(f"cannot cast non-integer to int, got: {x}")
            return x.real.__int__()

        if hasattr(x, "__int__"):
            return x.__int__()
        return super().fromobj(x)

    @classmethod
    def to_int(cls, a):
        return a.__int__()
    @classmethod
    def to_float(cls, a):
        return a.__float__()
    @classmethod
    def to_complex(cls, a):
        return complex(a.__float__())

    @classmethod
    def consts(cls):
        return super().consts() | {
            "__0__": 0,
            "__1__": 1,
        }

    @classmethod
    def add(cls, a, b):
        return a + b
    @classmethod
    def sub(cls, a, b):
        return a - b

    @classmethod
    def abs(cls, a):
        return abs(a)

    @classmethod
    def mul(cls, a, b):
        return a * b
    @classmethod
    def div(cls, a, b):
        rem = (a % b)
        if rem:
            raise TypeError(f"expected integer division result over Integer, "
                    f"this ain't: ({a}) / ({b})")
        return a // b

    @classmethod
    def power(cls, a, b):
        return a ** b
    @classmethod
    def root(cls, a, b):
        raise NotImplementedError("iroot")
    @classmethod
    def log(cls, a, b):
        raise NotImplementedError("ilog")

    @classmethod
    def issame(cls, a, b):
        return a == b
    @classmethod
    def eq(cls, a, b):
        return a == b
    @classmethod
    def neq(cls, a, b):
        return a != b
    @classmethod
    def lt(cls, a, b):
        return a < b
    @classmethod
    def le(cls, a, b):
        return a <= b

    @classmethod
    def hashed(cls, a):
        return hash(a)

    @classmethod
    def rep(cls, a, short):
        return repr(a)
        # TODO:
        # ill impl later (with prog)


# Specialise matrices over ints to supply a field.
@Matrix.screener
def _matrix_screen_int(params):
    field, shape = params
    if issubclass(field, int) and not issubclass(field, Field):
        return Integer, shape
    return params



class Float(Field):
    # @classmethod
    # def dtype(cls):
    #     return float

    @classmethod
    def fromobj(cls, x):
        if isinstance(x, int):
            return x.__float__()
        if isinstance(x, float):
            return x.__float__()
        if isinstance(x, complex):
            if not (x.imag == 0.0):
                raise TypeError(f"cannot cast complex to float, got: {x}")
            return x.real.__float__()

        if hasattr(x, "__float__"):
            return x.__float__()
        return super().fromobj(x)

    @classmethod
    def to_int(cls, a):
        if not a.is_integer():
            raise ValueError(f"cannot cast non-integer to int, got: {a}")
        return a.__int__()
    @classmethod
    def to_float(cls, a):
        return a.__float__()
    @classmethod
    def to_complex(cls, a):
        return complex(a.__float__())

    @classmethod
    def consts(cls):
        return super().consts() | {
            "__0__": 0.0,
            "__1__": 1.0,
            "__e__": _math.e,
            "__pi__": _math.pi,
        }

    # @classmethod
    # def exposes(cls):
    #     return {"isnan": bool, "isinf": bool, "isfinite": bool, "signbit": bool}
    # @classmethod
    # def expose(cls, a, name):
    #     if name == "signbit":
    #         return math.copysign(1.0, a) < 0.0
    #     return getattr(math, name)(a)


    @classmethod
    def add(cls, a, b):
        return a + b
    @classmethod
    def sub(cls, a, b):
        return a - b

    @classmethod
    def abs(cls, a):
        return abs(a)

    @classmethod
    def mul(cls, a, b):
        return a * b
    @classmethod
    def div(cls, a, b):
        return a / b

    @classmethod
    def power(cls, a, b):
        return a ** b
    @classmethod
    def root(cls, a, b):
        return a ** (1.0 / b)
    @classmethod
    def log(cls, a, b):
        return _math.log(b) / _math.log(a)

    @classmethod
    def sin(cls, a):
        return _math.sin(a)
    @classmethod
    def cos(cls, a):
        return _math.cos(a)
    @classmethod
    def tan(cls, a):
        return _math.tan(a)

    @classmethod
    def asin(cls, a):
        return _math.asin(a)
    @classmethod
    def acos(cls, a):
        return _math.acos(a)
    @classmethod
    def atan(cls, a):
        return _math.atan(a)
    @classmethod
    def atan2(cls, y, x):
        return _math.atan2(y, x)

    @classmethod
    def issame(cls, a, b):
        # nan is the same as nan.
        if _math.isnan(a) or _math.isnan(b):
            return _math.isnan(a) and _math.isnan(b)
        # zero is not the same as negative zero.
        if a == 0.0 and b == 0.0:
            return _math.copysign(1.0, a) == _math.copysign(1.0, b)
        return a == b

    @classmethod
    def eq(cls, a, b, ulps=15):
        # Equality automatically allows for a few ulps of imprecision, except in
        # some cases like:
        # - nan is never equal to anything.
        # - inf (and negative inf) only equals exactly inf (or negative inf).
        # - zero only equals exactly zero (negative and positive zero are eq).
        # - one (and negative one) only equals exactly one (or negative one).
        if _math.isnan(a) or _math.isnan(b):
            return False
        if (_math.isinf(a) or _math.isinf(b)
                or a == 0.0 or b == 0.0
                or abs(a) == 1.0 or abs(b) == 1.0):
            return a == b
        # we do tricks around here (c my beloved).
        toint = lambda z: struct.unpack("=q", struct.pack("=d", z))[0]
        ux = toint(abs(a))
        uy = toint(abs(b))
        if (a < 0.0) != (b < 0.0):
            return ux + uy <= ulps
        return abs(ux - uy) <= ulps
    @classmethod
    def lt(cls, a, b):
        if _math.isnan(a) or _math.isnan(b):
            return False
        return a < b

    @classmethod
    def neq(cls, a, b):
        if _math.isnan(a) or _math.isnan(b):
            return False
        return not cls.eq(a, b)
    @classmethod
    def le(cls, a, b):
        if _math.isnan(a) or _math.isnan(b):
            return False
        return cls.eq(a, b) or cls.lt(a, b)

    @classmethod
    def hashed(cls, a):
        return hash(a)

    @classmethod
    def rep(cls, a, short):
        return repr(a)
        # TODO:
        # ill impl later (with prog)


# Specialise matrices over floats to use numpy internals for huge speed ups.
@Matrix.screener
def _matrix_screen_float(params):
    field, shape = params
    if issubclass(field, float) and not issubclass(field, Field):
        return Float, shape
    return params
@Matrix.specialiser(lambda field, shape: field is Float)
def _FloatMatrix(Mat):
    # TODO:
    # fast numpy impl.
    return Mat



class Complex(Field):
    # @classmethod
    # def dtype(cls):
    #     return float

    @classmethod
    def fromobj(cls, x):
        if isinstance(x, int):
            return complex(x.__float__())
        if isinstance(x, float):
            return complex(x.__float__())
        if isinstance(x, complex):
            return x.__complex__()

        if hasattr(x, "__complex__"):
            return x.__complex__()
        if hasattr(x, "__float__"):
            return complex(x.__float__())
        return super().fromobj(x)

    @classmethod
    def to_int(cls, a):
        if not a.is_integer():
            raise ValueError(f"cannot cast non-integer to int, got: {a}")
        return a.__int__()
    @classmethod
    def to_float(cls, a):
        return a.__float__()
    @classmethod
    def to_complex(cls, a):
        return complex(a.__float__())

    @classmethod
    def consts(cls):
        return super().consts() | {
            "__0__": complex(0.0),
            "__1__": complex(1.0),
            "__i__": 1j,
            "__e__": complex(_math.e),
            "__pi__": complex(_math.pi),
        }

    @classmethod
    def add(cls, a, b):
        return a + b
    @classmethod
    def sub(cls, a, b):
        return a - b

    @classmethod
    def abs(cls, a):
        return complex(abs(a))
    @classmethod
    def real(cls, a):
        return complex(a.real)
    @classmethod
    def imag(cls, a):
        return complex(a.imag)

    @classmethod
    def mul(cls, a, b):
        return a * b
    @classmethod
    def div(cls, a, b):
        return a / b

    @classmethod
    def power(cls, a, b):
        return a ** b
    @classmethod
    def root(cls, a, b):
        return a ** (1.0 / b)
    @classmethod
    def log(cls, a, b):
        return _cmath.log(b) / _cmath.log(a)

    @classmethod
    def sin(cls, a):
        return _cmath.sin(a)
    @classmethod
    def cos(cls, a):
        return _cmath.cos(a)
    @classmethod
    def tan(cls, a):
        return _cmath.tan(a)

    @classmethod
    def asin(cls, a):
        return _cmath.asin(a)
    @classmethod
    def acos(cls, a):
        return _cmath.acos(a)
    @classmethod
    def atan(cls, a):
        return _cmath.atan(a)
    @classmethod
    def atan2(cls, y, x):
        if a.imag == 0.0 and b.imag == 0.0:
            return complex(_math.atan2(y.real, x.real))
        raise TypeError("cannot perform quadrant-aware atan on complex, got: "
                f"{a}, and {b}")

    @classmethod
    def issame(cls, a, b):
        return Float.issame(a.real, b.real) and Float.issame(a.imag, b.imag)

    @classmethod
    def eq(cls, a, b, ulps=15):
        if _cmath.isnan(a) or _cmath.isnan(b):
            return False
        return Float.eq(a.real, b.real) and Float.eq(a.imag, b.imag)
    @classmethod
    def neq(cls, a, b):
        if _cmath.isnan(a) or _cmath.isnan(b):
            return False
        return not cls.eq(a, b)
    @classmethod
    def lt(cls, a, b):
        if _cmath.isnan(a) or _cmath.isnan(b):
            return False
        if a.imag == 0.0 and b.imag == 0.0:
            return a.real < b.real
        raise TypeError(f"cannot order complex, got: {a}, and {b}")
    @classmethod
    def le(cls, a, b):
        if _cmath.isnan(a) or _cmath.isnan(b):
            return False
        if a.imag == 0.0 and b.imag == 0.0:
            return Float.eq(a.real, b.real) or a.real < b.real
        raise TypeError(f"cannot order complex, got: {a}, and {b}")

    @classmethod
    def hashed(cls, a):
        return hash(a)

    @classmethod
    def rep(cls, a, short):
        return repr(a)
        # TODO:
        # ill impl later (with prog)

# Specialise matrices over floats to use numpy internals for huge speed ups.
@Matrix.screener
def _matrix_screen_complex(params):
    field, shape = params
    if issubclass(field, complex) and not issubclass(field, Field):
        return Complex, shape
    return params
@Matrix.specialiser(lambda field, shape: field is Complex)
def _ComplexMatrix(Mat):
    # TODO:
    # fast numpy impl.
    return Mat




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
    Mat = Single[type(x)]
    cells = _bg.np.array([[x]], dtype=Mat._f("dtype")())
    mat = Mat(cells)
    if field is not None:
        mat = mat.tofield(field)
    return mat

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
    Mat = Empty[field]
    cells = _bg.np.array([], dtype=Mat._f("dtype")())
    return Mat(cells)

def isempty(x):
    """
    True iff 'x' is a matrix with no cells.
    """
    return isinstance(x, Matrix) and x.isempty



def castall(xs, broadcast=True, *, field=None):
    """
    When given an sequence of values/matrices, returns them as matrices cast to
    the same field and (optionally) broadcast to the same shape.
    """
    if not _iterable(xs):
        raise TypeError(f"expected an iterable for xs, got {_objtname(xs)}")
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
    Alias for 'x.dot(y)'.
    """
    x, y = castall([x, y], field=field, broadcast=False)
    return x.dot(y)
def cross(x, y, *, field=None):
    """
    Alias for 'x.cross(y)'.
    """
    x, y = castall([x, y], field=field, broadcast=False)
    return x.cross(y)
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
    return y._apply(lambda: y._f("atan2"), y, x)
def torad(degrees, *, field=None):
    """
    Alias for 'degrees.torad'.
    """
    x, = castall([degrees], field=field)
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
    return y._apply(lambda: y._f("atan2"), y, x).todeg
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
    y, x = castall([y, x], field=field)
    return y.intt(x, *bounds)
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
def conj(x, *, field=None):
    """
    Alias for 'x.conj'.
    """
    x, = castall([x], field=field)
    return x.conj
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
    xs = _maybe_unpack_mats(xs)
    xs = castall(xs, field=field, broadcast=False)
    field = _get_field(field, xs)
    if not isinstance(axis, int):
        raise TypeError(f"expected an integer axis, got {_objtname(axis)}")
    if axis < 0:
        raise ValueError(f"axis cannot be negative, got: {axis}")

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
    for x in xs:
        if x.shape.withaxis(axis, 1) != perpshape:
            raise TypeError(f"expected a perpendicular shape of {perpshape} for "
                    f"stacking, got {x.shape}")
    # Get all cells with the right number of dimensions.
    ndim = max(xs[0].ndim, axis + 1)
    ys = [x._cells for x in xs]
    ys = [y[(None,) * (ndim - y.ndim)] for y in ys]
    npaxis = Permuter.fromnumpy(ndim).order(ndim)[axis]
    cells = _bg.np.concatenate(ys, axis=npaxis)
    return Matrix.fromnumpy(field, cells)

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
    xs = _maybe_unpack_mats(xs)
    xs = castall(xs, field=field, broadcast=False)
    field = _get_field(field, xs)
    cells = _bg.np.concatenate([x.ravel.numpyvec() for x in xs])
    return Matrix.fromnumpy(field, cells)

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
    concat = []
    for x in xs:
        if not _iterable(x):
            x, = castall([x], field=field)
        if isinstance(x, Matrix):
            if not x.isvec:
                raise TypeError("expected vectors to concatenate into column, "
                        f"got {x.shape}")
            concat.append(x.numpyvec())
            if field is None:
                field = x.field
            continue
        for y in x:
            if not isinstance(y, Matrix):
                y, = castall([y], field=field)
            if not y.isvec:
                raise TypeError("expected iterable to contain vectors to "
                        f"concatenate into column, got {y.shape}")
            if field is None:
                field = y.field
            concat.append(y.numpyvec())
    field = _get_field(field)
    cells = _bg.np.concatenate(concat)
    return Matrix.fromnumpy(field, cells)


def eye(n, *, field=None):
    """
    2D identity matrix, of the given size.
    """
    field = _get_field(field)
    if not isinstance(n, int):
        raise TypeError(f"expected an integer size, got {_objtname(n)}")
    if n < 0:
        raise ValueError(f"cannot have negative size, got: {n}")
    return Matrix[field, (n, n)].eye

def zeros(*lens, field=None):
    """
    Zero-filled matrix of the given size, defaulting to square if only one axis
    length is given.
    """
    field = _get_field(field)
    shape = Shape.sqrshape(*lens)
    return Matrix[field, shape].zeros

def ones(*lens, field=None):
    """
    One-filled matrix of the given size, defaulting to square if only one axis
    length is given.
    """
    field = _get_field(field)
    shape = Shape.sqrshape(*lens)
    return Matrix[field, shape].ones

def diag(x, *, field=None):
    """
    If given a vector, returns a diagonal matrix with that diagonal. If given a
    matrix, returns the diagonal of that matrix.
    """
    x, = castall([x], field=field)
    field = _get_field(field, [x])
    if x.isempty or x.issingle:
        return x
    if not x.isvec:
        return x.diag
    Mat = Matrix[field, Shape(x.size, x.size)]
    cells = Mat.zeros._cells.copy()
    _bg.np.fill_diagonal(cells, x.numpyvec())
    return Mat(cells)

def linspace(x0, x1, n, *, field=None):
    """
    Vector of 'n' linearly spaced values starting at 'x0' and ending at 'x1'.
    """
    if not isinstance(n, int):
        raise TypeError(f"expected an integer n, got {_objtname(n)}")
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
        raise TypeError(f"expected an integer n, got {_objtname(n)}")
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
            raise TypeError(f"expected iterable for 'T', got {_objtname(T)}")
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
        "__pow__": "**",
        "__matmul__": "@",
        "__mod__": "%",
        "__and__": "&",
        "__or__": "|",
        "__xor__": "^",
        "__lshift__": ">>",
        "__rshift__": "<<",
        "__eq__": "==",
        "__neq__": "!=",
        "__lt__": "<",
        "__le__": "<=",
        "__gt__": ">",
        "__ge__": ">=",
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
