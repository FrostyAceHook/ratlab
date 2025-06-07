import inspect as _inspect
import math as _math
import types as _types
from math import prod as _prod

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
        self.v = v

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
        return int(a.v)
    @classmethod
    def to_float(cls, a):
        return float(a.v)
    @classmethod
    def to_complex(cls, a):
        return complex(a.v)

    @_classconst
    def zero(cls):
        return cls(T(0))
    @_classconst
    def one(cls):
        return cls(T(1))

    @classmethod
    def add(cls, a, b):
        return cls._generic(a.v + b.v)
    @classmethod
    def sub(cls, a, b):
        return cls._generic(a.v - b.v)
    @classmethod
    def absolute(cls, a):
        return cls._generic(abs(a.v))

    @classmethod
    def mul(cls, a, b):
        return cls._generic(a.v * b.v)
    @classmethod
    def div(cls, a, b):
        return cls._generic(a.v / b.v)

    @classmethod
    def power(cls, a, b):
        return cls._generic(a.v ** b.v)

    @classmethod
    def eq(cls, a, b):
        return bool(a.v == b.v)
    @classmethod
    def lt(cls, a, b):
        return bool(a.v < b.v)

    @classmethod
    def hashed(cls, a):
        return hash(a.v)

    @classmethod
    def repr_short(cls, a):
        if T is bool:
            return "Y" if a.v else "N"
        return repr(a.v)
    @classmethod
    def repr_long(cls, a):
        if T is bool:
            return "yeagh" if a.v else "nogh"
        return repr(a.v)



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



# Matrices.
@_templated(parents=Field, decorators=_immutable)
def Matrix(field, shape):
    """
    Fixed-sized two-dimensional sequence of elements. The general rule is all
    field operations are element-wise, and then all matrix-specific functions and
    operations are implemented independantly.
    """

    if not isinstance(field, type):
        raise TypeError("field must be a type, not something of type "
                f"{_tname(type(field))}")
    if issubclass(field, Matrix):
        raise TypeError("mate a matrix of matrices? calm down")
    # Wrap non-field classes.
    if not issubclass(field, Field):
        return Matrix[GenericField[field], shape]

    if not isinstance(shape, tuple) or len(shape) != 2:
        raise TypeError("shape must be a tuple of (numrows, numcols)")
    if not isinstance(shape[0], int) or not isinstance(shape[1], int):
        raise TypeError("shape must use ints")
    if shape[0] < 0 or shape[1] < 0:
        raise TypeError("cannot have negative size")
    # collapse empty matrices to be over Field with (0,0).
    if _prod(shape) == 0 and shape != (0, 0):
        return Matrix[field, (0, 0)]

    class _TomHollandManIDontWannaTalkToY(str):
        __doc__ = None
    _size_str = _TomHollandManIDontWannaTalkToY(f"{shape[0]}x{shape[1]}")

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


    def __init__(self, cells, print_colvec_flat=True, print_zero_as_dot=True):
        # cells = flattened, with contiguous rows.
        if not _iterable(cells):
            raise TypeError("cells must be a sequence")
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
                raise TypeError(f"got a cell of type {_tname(type(cells[i]))}, "
                        f"expected {_tname(field)} (occured at index {i}, had "
                        f"value {cells[i]})")
        if len(cells) != _prod(shape):
            raise TypeError("cells must match flattened shape (expected "
                    f"{_size_str}={_prod(shape)}, got {len(cells)})")

        self._cells = cells
        self._print_colvec_flat = print_colvec_flat
        self._print_zero_as_dot = print_zero_as_dot


    @_classconst
    def eye(cls):
        """
        Identity matrix.
        """
        if not cls.issquare:
            raise TypeError("only square matricies have an identity (got size "
                    f"{_size_str})")
        n = shape[0]
        cells = [cls._f("zero")] * _prod(shape) if n else ()
        for i in range(n):
            cells[i*n + i] = cls._f("one")
        return cls(cells)
    @_classconst
    def zeros(cls):
        """
        Zero-filled matrix.
        """
        cells = (cls._f("zero"), ) * _prod(shape) if shape[0] else ()
        return cls(cells)
    @_classconst
    def ones(cls):
        """
        One-filled matrix.
        """
        cells = (cls._f("one"), ) * _prod(shape) if shape[0] else ()
        return cls(cells)

    @_classconst
    def zero(cls):
        """
        Single zero.
        """
        return single(cls._f("zero"))
    @_classconst
    def one(cls):
        """
        Single one.
        """
        return single(cls._f("one"))
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
        return single(e)
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
        return single(pi)


    @_classconst
    def isempty(cls):
        """
        Is empty matrix? (0x0)
        """
        return shape[0] == 0

    @_classconst
    def issingle(cls):
        """
        Is only one cell? (1x1)
        """
        return shape == (1, 1)

    @_classconst
    def isvec(cls):
        """
        Is row vector or column vector? (empty and single count as vectors)
        """
        return cls.isempty or shape[0] == 1 or shape[1] == 1

    @_classconst
    def iscol(cls):
        """
        Is column vector? (empty and single count as column vectors)
        """
        return cls.isempty or shape[1] == 1

    @_classconst
    def isrow(cls):
        """
        Is row vector? (empty and single count as row vectors)
        """
        return cls.isempty or shape[0] == 1

    @_classconst
    def issquare(cls):
        """
        Is square matrix?
        """
        return shape[0] == shape[1]


    def __iter__(s):
        """
        Vector-only cell iterate.
        """
        if not s.isvec:
            raise TypeError("only vectors have bare iteration (got size "
                    f"{_size_str}), use .rowmajor or .colmajor for matrix "
                    "iterate")
        return (single(x) for x in s._cells)

    def __getitem__(s, i):
        """
        Vector-only cell access.
        """
        if not s.isvec:
            raise TypeError("only vectors have bare getitem (got size "
                    f"{_size_str}), use .at for matrix cell access")
        xs = s._cells.__getitem__(i)
        if not isinstance(xs, tuple):
            xs = (xs, )
        shp = (len(xs), 1) if shape[1] == 1 else (1, len(xs))
        return Matrix[field, shp](xs)

    def __len__(s):
        """
        Number of cells.
        """
        return len(s._cells)

    @_instconst
    def rowmajor(s):
        """
        Column vector of cells in row-major order.
        """
        return Matrix[field, (_prod(shape), 1)](s._cells)

    @_instconst
    def colmajor(s):
        """
        Column vector of cells in column-major order.
        """
        return Matrix[field, (_prod(shape), 1)](s.T._cells)

    def at(s, i, j):
        """
        Cell at row i, column j.
        """
        if not isinstance(i, int):
            raise TypeError("expected an integer row index (got "
                    f"{_tname(type(i))}")
        if not isinstance(j, int):
            raise TypeError("expected an integer column index (got "
                    f"{_tname(type(j))}")
        ii = i + (i < 0) * shape[0]
        jj = j + (j < 0) * shape[1]
        if ii not in range(shape[0]):
            raise ValueError(f"row index {i} out-of-bounds for {shape[0]} rows")
        if jj not in range(shape[1]):
            raise ValueError(f"column index {j} out-of-bounds for {shape[1]} "
                    "columns")
        return single(s._cells[shape[1]*i + j])
    def _at(s, i, j):
        return s.at(i, j)._cells[0]

    class _SubmatrixGetter:
        def __init__(s, matrix):
            s._matrix = matrix
        def __getitem__(s, ij):
            if not isinstance(ij, tuple):
                raise TypeError("expected both row and column slices")
            if len(ij) != 2:
                raise TypeError("expected only row and column slices, got "
                        f"{len(ij)} slices")
            def process(k, length):
                if isinstance(k, slice):
                    return range(*k.indices(length))
                return k if _iterable(k) else (k, )
            i, j = ij
            rs = process(i, shape[0])
            cs = process(j, shape[1])
            cells = (s._matrix._at(r, c) for r in rs for c in cs)
            shp = len(rs), len(cs)
            return Matrix[field, shp](cells)
    @_instconst
    def sub(s):
        """
        Submatrix of specified rows and columns.
        """
        return _SubmatrixGetter(s)

    @_instconst
    def rows(s):
        """
        Tuple of row vectors.
        """
        rw, cs = shape
        vec = Matrix[field, (1, cs)]
        return tuple(vec(s._at(i, j) for j in range(cs)) for i in range(rw))

    @_instconst
    def cols(s):
        """
        Tuple of column vectors
        """
        rs, cs = shape
        vec = Matrix[field, (rs, 1)]
        return tuple(vec(s._at(i, j) for i in range(rs)) for j in range(cs))


    @_instconst
    def T(s):
        """
        Matrix transpose.
        """
        nr, nc = shape
        if s.isvec:
            # Make sure vector transpose is super fast bc its literally the same
            # in-memory.
            cells = s._cells
        else:
            cells = (s._at(i, j) for j in range(nc) for i in range(nr))
        return Matrix[field, (nc, nr)](cells)

    @_instconst
    def inv(s):
        """
        Matrix inverse.
        """
        if not s.issquare:
            raise TypeError("cannot invert a non-square matrix (got size "
                    f"{s._size_str})")
        if s.det == s.zero:
            raise ValueError("cannot invert a non-invertible matrix (det == 0)")
        aug = hstack(s, s.eye)
        aug = aug.rref
        inverse_cols = aug.cols[shape[0]:]
        return hstack(*inverse_cols)

    @_instconst
    def diag(s):
        """
        Matrix diagonal (as column vector).
        """
        cells = (s._at(i, i) for i in range(min(shape)))
        return Matrix[field, (min(shape), 1)](cells)


    @_instconst
    def isdiag(s):
        """
        Is diagonal matrix? (square, and only diagonal is non-zero)
        """
        if not s.issquare:
            return False
        for i in range(shape[0]):
            for j in range(shape[1]):
                if i == j:
                    continue
                if s.at(i, j) != s.zero:
                    return False
        return True

    @_instconst
    def isuppertri(s):
        """
        Is upper-triangular matrix? (square, and below diagonal is zero)
        """
        if not s.issquare:
            return False
        for i in range(shape[0]):
            for j in range(i):
                if s.at(i, j) != s.zero:
                    return False
        return True

    @_instconst
    def islowertri(s):
        """
        Is lower-triangular matrix? (square, and above diagonal is zero)
        """
        return s.T.isuppertri

    @_instconst
    def isorthogonal(s):
        """
        Is orthogonal matrix? (transpose == inverse)
        """
        return bool(s.T == s.inv)

    @_instconst
    def issymmetric(s):
        """
        Is symmetric matrix? (square, and below diagonal = above diagonal)
        """
        if not s.issquare:
            return False
        for i in range(shape[0]):
            for j in range(i):
                if s.at(i, j) != s.at(j, i):
                    return False
        return True


    @_instconst
    def det(s):
        """
        Matrix determinant.
        """
        if not s.issquare:
            raise TypeError("cannot find determinant of non-square matrix (got "
                    f"size {s._size_str})")

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
            det = s._f("zero")
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
        Matrix trace.
        """
        if not s.issquare:
            raise TypeError("cannot find trace of non-square matrix (got size "
                    f"{s._size_str})")
        trace = s.zero
        for i in range(shape[0]):
            trace += s.at(i, i)
        return trace

    @_instconst
    def mag(s):
        """
        Vector magnitude.
        """
        if not s.isvec:
            raise TypeError("cannot find magnitude of a non-vector (got size "
                    f"{s._size_str})")
        return (s & s).sqrt


    @_instconst
    def rref(s):
        """
        Reduced-row echelon form.
        """

        eqz = lambda x: s._f("eq")(x, s._f("zero"))
        add = lambda a, b: s._f("add")(a, b)
        mul = lambda a, b: s._f("mul")(a, b)
        neg = lambda x: s._f("sub")(s._f("zero"), x)
        rec = lambda x: s._f("div")(s._f("one"), x)

        def row_swap(shape, cells, row1, row2):
            num_cols = shape[1]
            for j in range(num_cols):
                k1 = row1 * num_cols + j
                k2 = row2 * num_cols + j
                cells[k1], cells[k2] = cells[k2], cells[k1]

        def row_mul(shape, cells, row, by):
            num_cols = shape[1]
            for j in range(num_cols):
                idx = row*num_cols + j
                cells[idx] = mul(by, cells[idx])

        def row_add(shape, cells, src, by, dst):
            num_cols = shape[1]
            for i in range(num_cols):
                src_k = src*num_cols + i
                dst_k = dst*num_cols + i
                cells[dst_k] = add(cells[dst_k], mul(by, cells[src_k]))

        cells = list(s._cells)

        num_rows, num_cols = shape
        lead = 0
        for r in range(num_rows):
            if lead >= num_cols:
                break

            i = r
            while eqz(cells[num_cols*i + lead]):
                i += 1
                if i == num_rows:
                    i = r
                    lead += 1
                    if lead == num_cols:
                        break
            if lead == num_cols:
                break
            row_swap(shape, cells, i, r)

            pivot_value = cells[num_cols*r + lead]
            if not eqz(pivot_value):
                row_mul(shape, cells, r, rec(pivot_value))

            for i in range(num_rows):
                if i != r:
                    row_lead_value = cells[num_cols*i + lead]
                    if not eqz(row_lead_value):
                        row_add(shape, cells, r, neg(row_lead_value), i)

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
        Tuple of column space basis (as column vectors).
        """
        return tuple(s.cols[p] for p in s.pivots)

    @_instconst
    def rowspace(s):
        """
        Tuple of row space basis (as column vectors).
        """
        rref = s.rref
        nonzeros = (i for i, r in enumerate(rref.rows)
                    if any(s._f("eq")(x, s._f("zero")) for x in r))
        return tuple(rref.rows[i].T for i in nonzeros)

    @_instconst
    def nullspace(s):
        """
        Basis for null space (as column vectors).
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
                basis[n][pivotat[i]] = -sys.at(i, j)
        return tuple(Matrix[field, (shape[1], 1)](x) for x in basis)

    def _repr(s, islong):
        if s.isempty:
            return "my boy "*islong + "M.T."

        rep = s._f("repr_long" if islong else "repr_short")

        if s.issingle:
            # return "[ " + rep(s._cells[0]) + " ]"
            return rep(s._cells[0])

        if s._print_zero_as_dot and not islong:
            rep_ = rep
            def rep(x):
                if rep.can_eq_zero:
                    iszero = s._f("eq")(x, s._f("zero"))
                    return "." if iszero else rep_(x)
                return rep_(x)
            rep.can_eq_zero = True
            try:
                s._need("eq")
                s._need("zero")
            except NotImplementedError:
                rep.can_eq_zero = False

        multiline = not (s._print_colvec_flat and s.isvec)
        max_len = max(len(rep(x)) for x in s._cells)
        padded = (max_len > 3)
        rows = []
        for k, x in enumerate(s._cells):
            if (k % shape[1]) == 0:
                rows.append([])
            rows[-1].append(f"{rep(x):>{max_len}}")
        join = lambda x: "  ".join(x) if padded else " ".join(x)
        wrap = lambda x: f"[ {x} ]" if padded else f"[{x}]"
        str_rows = (wrap(join(row)) for row in rows)
        return ("\n" if multiline else "").join(str_rows)

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
    def cast(cls, o, keep_single=False):
        """
        Attempts to cast 'o' to a matrix over the same field. If 'o' is a matrix,
        its size will not be altered. Otherwise, 'o' will be a single if
        'keep_single' and the same size as this matrix if not.
        """

        # Any matrix over the same field is okie.
        if isinstance(o, Matrix):
            if field != o.field:
                raise TypeError("cannot operate on matrices of different fields "
                        f"({_tname(field)} vs {_tname(o.field)})")
            return o

        # Try to cast it from basic number types.
        convs = {bool: "from_bool", int: "from_int", float: "from_float",
                 complex: "from_complex", str: "from_str"}
        if type(o) not in convs.keys():
            raise TypeError(f"{_tname(type(o))} cannot operate with "
                    f"{_tname(field)}")
        o = cls._f(convs[type(o)])(o)
        if keep_single:
            # Just single.
            return single(o)
        # Otherwise create a matrix with it as every element.
        return cls((o, ) * _prod(shape))


    @classmethod
    def _eltwise(cls, func, *xs, rettype=None, pierce=True):
        def f(*y):
            nonlocal rettype
            ret = func(*y)
            if isinstance(ret, Matrix):
                if not ret.issingle:
                    raise TypeError("'func' returned a non-single matrix (got "
                            f"{ret._size_str})")
                ret = ret._cells[0]
            if rettype is None:
                rettype = type(ret)
            elif not isinstance(ret, rettype):
                raise TypeError("inconsistently typed return from 'func' "
                        f"(expected {_tname(rettype)}, got "
                        f"{_tname(type(ret))})")
            return ret

        if not xs:
            # MT^2
            if not _prod(shape):
                if rettype is None:
                    rettype = field
                return Matrix[rettype, shape](())
            # Matrix of just the things.
            cell = f()
            return Matrix[rettype, shape]((cell, ) * _prod(shape))

        # Cast all to matricies over the correct field.
        xs = [cls.cast(x, keep_single=True) for x in xs]

        # Find the largest size.
        newshape = (0, 0)
        newsize = 0
        for x in xs:
            size = _prod(x.shape)
            if size > newsize:
                newshape = x.shape
                newsize = _prod(newshape)

        # Assume eltwise operations on MT dont change field.
        if newsize == 0:
            return Matrix[field, newshape](())

        # Enforce size conformity.
        def conform(x):
            # Broadcast singles up.
            if x.issingle and newsize != 1:
                x = Matrix[field, newshape](x._cells * newsize)
            if x.shape != newshape:
                raise TypeError("cannot element-wise operate on matrices of "
                        f"different sizes (expected "
                        f"{newshape[0]}x{newshape[1]}, got {x._size_str})")
            return x
        xs = [conform(x) for x in xs]

        # Apply.
        zipped = zip(*(x._cells for x in xs))
        if pierce:
            cells = tuple(f(*y) for y in zipped)
        else:
            cells = tuple(f(*map(single, y)) for y in zipped)
        return Matrix[rettype, newshape](cells)


    @classmethod
    def eltwise(cls, func, *xs, rettype=None):
        """
        Constructs a matrix from the results of 'func(a, b, ...)' for all zipped
        elements in '*xs'. If 'rettype' is non-none, hints/enforces the return
        type from 'func'.
        """
        return cls._eltwise(func, *xs, rettype=rettype, pierce=False)

    def apply(s, func, *os, rettype=None):
        """
        Alias for 'M.eltwise(func, s, *os, rettype=rettype)'.
        """
        return type(s).eltwise(func, s, *os, rettype=rettype)


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
        o = s.cast(o, keep_single=True)
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
        return single(dot)
    def __rand__(s, o):
        o = s.cast(o, keep_single=True)
        return o.__and__(s)

    def __or__(s, o):
        """
        3D vector cross product.
        """
        o = s.cast(o, keep_single=True)
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
        o = s.cast(o, keep_single=True)
        return o.__or__(s)

    def __matmul__(s, o):
        """
        Matrix multiplication.
        """
        o = s.cast(o, keep_single=True)
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
        o = s.cast(o, keep_single=True)
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
    def __getattr__(s, attr):
        """
        If a non-matrix attribute is accessed, it will be retrived from each
        element instead.
        """
        cells = object.__getattribute__(s, "_cells") # cooked.
        bad = False
        if attr.startswith("_"):
            bad = True
        elif cells:
            bad = not hasattr(cells[0], attr)
        else:
            try:
                bad = not hasattr(s._f("zero"), attr)
            except NotImplementedError:
                try:
                    bad = not hasattr(s._f("one"), attr)
                except NotImplementedError:
                    bad = False # giv up.
        if bad:
            raise AttributeError(f"{repr(type(s).__name__)} object has no "
                    f"attribute {repr(attr)}")
        if not cells:
            return Matrix[Field, (0, 0)](())
        cells = tuple(getattr(x, attr) for x in cells)
        newfield = type(cells[0]) if cells else Field
        return Matrix[newfield, shape](cells)



class Single:
    def __getitem__(self, field):
        return Matrix[field, (1, 1)]
Single = Single()
Single.__doc__ = """
Refers to the type 'Matrix[field, (1, 1)]'. Note this is not a proper templated
class, only a thin wrapper.
"""

def single(x):
    """
    Returns the given object as a single matrix (1x1).
    """
    return Single[type(x)]((x, ))

def issingle(a):
    """
    Returns true iff 'a' is a matrix with only one cell.
    """
    return isinstance(a, Matrix) and a.issingle



def sqrt(x, *, field=None):
    """
    Alias for 'x.sqrt'.
    """
    if field is None:
        if lits.field is None:
            raise RuntimeError("specify a field using 'lits'")
        field = lits.field
    if not isinstance(x, Matrix):
        x = Single[field].cast(x)
    return x.sqrt
def cbrt(x, *, field=None):
    """
    Alias for 'x.cbrt'.
    """
    if field is None:
        if lits.field is None:
            raise RuntimeError("specify a field using 'lits'")
        field = lits.field
    if not isinstance(x, Matrix):
        x = Single[field].cast(x)
    return x.cbrt
def root(base, x, *, field=None):
    """
    Alias for 'x.root(base)'.
    """
    if field is None:
        if lits.field is None:
            raise RuntimeError("specify a field using 'lits'")
        field = lits.field
    if not isinstance(x, Matrix):
        x = Single[field].cast(x)
    return x.root(base)

def exp(x, *, field=None):
    """
    Alias for 'x.exp'.
    """
    if field is None:
        if lits.field is None:
            raise RuntimeError("specify a field using 'lits'")
        field = lits.field
    if not isinstance(x, Matrix):
        x = Single[field].cast(x)
    return x.exp
def exp2(x, *, field=None):
    """
    Alias for 'x.exp2'.
    """
    if field is None:
        if lits.field is None:
            raise RuntimeError("specify a field using 'lits'")
        field = lits.field
    if not isinstance(x, Matrix):
        x = Single[field].cast(x)
    return x.exp2
def exp10(x, *, field=None):
    """
    Alias for 'x.exp10'.
    """
    if field is None:
        if lits.field is None:
            raise RuntimeError("specify a field using 'lits'")
        field = lits.field
    if not isinstance(x, Matrix):
        x = Single[field].cast(x)
    return x.exp10

def ln(x, *, field=None):
    """
    Alias for 'x.ln'.
    """
    if field is None:
        if lits.field is None:
            raise RuntimeError("specify a field using 'lits'")
        field = lits.field
    if not isinstance(x, Matrix):
        x = Single[field].cast(x)
    return x.ln
def log2(x, *, field=None):
    """
    Alias for 'x.log2'.
    """
    if field is None:
        if lits.field is None:
            raise RuntimeError("specify a field using 'lits'")
        field = lits.field
    if not isinstance(x, Matrix):
        x = Single[field].cast(x)
    return x.log2
def log10(x, *, field=None):
    """
    Alias for 'x.log10'.
    """
    if field is None:
        if lits.field is None:
            raise RuntimeError("specify a field using 'lits'")
        field = lits.field
    if not isinstance(x, Matrix):
        x = Single[field].cast(x)
    return x.log10
def log(base, x, *, field=None):
    """
    Alias for 'x.log(base)'.
    """
    if field is None:
        if lits.field is None:
            raise RuntimeError("specify a field using 'lits'")
        field = lits.field
    if not isinstance(x, Matrix):
        x = Single[field].cast(x)
    return x.log(base)
def sin(x, *, field=None):
    """
    Alias for 'x.sin'.
    """
    if field is None:
        if lits.field is None:
            raise RuntimeError("specify a field using 'lits'")
        field = lits.field
    if not isinstance(x, Matrix):
        x = Single[field].cast(x)
    return x.sin
def cos(x, *, field=None):
    """
    Alias for 'x.cos'.
    """
    if field is None:
        if lits.field is None:
            raise RuntimeError("specify a field using 'lits'")
        field = lits.field
    if not isinstance(x, Matrix):
        x = Single[field].cast(x)
    return x.cos
def tan(x, *, field=None):
    """
    Alias for 'x.tan'.
    """
    if field is None:
        if lits.field is None:
            raise RuntimeError("specify a field using 'lits'")
        field = lits.field
    if not isinstance(x, Matrix):
        x = Single[field].cast(x)
    return x.tan
def asin(x, *, field=None):
    """
    Alias for 'x.asin'.
    """
    if field is None:
        if lits.field is None:
            raise RuntimeError("specify a field using 'lits'")
        field = lits.field
    if not isinstance(x, Matrix):
        x = Single[field].cast(x)
    return x.asin
def acos(x, *, field=None):
    """
    Alias for 'x.acos'.
    """
    if field is None:
        if lits.field is None:
            raise RuntimeError("specify a field using 'lits'")
        field = lits.field
    if not isinstance(x, Matrix):
        x = Single[field].cast(x)
    return x.acos
def atan(x, *, field=None):
    """
    Alias for 'x.atan'.
    """
    if field is None:
        if lits.field is None:
            raise RuntimeError("specify a field using 'lits'")
        field = lits.field
    if not isinstance(x, Matrix):
        x = Single[field].cast(x)
    return x.atan

def atan2(y, x, *, field=None):
    """
    Quadrant-aware 'atan(y / x)'.
    """
    if field is None:
        if lits.field is None:
            raise RuntimeError("specify a field using 'lits'")
        field = lits.field
    Mat = Single[field]
    if isinstance(y, Matrix):
        Mat = type(y)
    elif isinstance(x, Matrix):
        Mat = type(x)
    if not isinstance(y, Matrix):
        y = Mat.cast(y)
    if not isinstance(x, Matrix):
        x = Mat.cast(x)
    return y._eltwise(y._f("atan2"), y, x)



def long(a):
    """
    Prints a long string representation of 'a'.
    """
    if not isinstance(a, Matrix):
        raise TypeError(f"expected matrix, got {_tname(type(a))}")
    print(a.repr_long)



def _keep_unpacking(xs):
    if len(xs) == 0 or len(xs) > 1:
        return False
    if isinstance(xs[0], Matrix):
        return False
    return _iterable(xs[0])



def concat(*rows, field=None):
    """
    Concatenates the given matrices as: concat((0, 1), (2, 3)) = [1,2][2,3]
    """
    if field is None:
        if lits.field is None:
            raise RuntimeError("specify a field using 'lits'")
        field = lits.field
    Mat = Single[field]
    cells = []
    shape = [0, 0]
    for row in rows:
        height = None
        xs = []
        if isinstance(row, Matrix):
            raise TypeError("must give a list of matrices for each row, not "
                    "just a bare matrix")
        for x in row:
            if not isinstance(x, Matrix):
                # raise TypeError("must give a matrix to be concatenated, got "
                #         f"{_tname(type(x))}")
                # on god i have changed this line back and forth about 50 times.
                x = Mat.cast(x)
            if x.field != field:
                raise TypeError(f"inconsistent field (expected "
                        f"{_tname(field)}, got {_tname(x.field)}")
            if height is None:
                height = x.shape[0]
            if height != x.shape[0]:
                raise TypeError("inconsistent vertical concat size (expected "
                        f"{height}, got {x.shape[0]})")
            xs.append(x)

        if height is None or height == 0:
            continue

        rowcells = []
        for row in range(height):
            for x in xs:
                rowcells.extend(x.rows[row]._cells)
        assert len(rowcells) % height == 0
        width = len(rowcells) // height
        if not shape[1]:
            shape[1] = width
        if width != shape[1]:
            raise TypeError("inconsistent horizontal concat size (expected "
                    f"{shape[1]}, got {width})")
        cells.extend(rowcells)
        shape[0] += height

    return Matrix[field, tuple(shape)](cells)

def vstack(*xs, field=None):
    """
    Vertically concatenates the given matrices.
    """
    if field is None:
        if lits.field is None:
            raise RuntimeError("specify a field using 'lits'")
        field = lits.field
    if _keep_unpacking(xs):
        return vstack(*xs[0], field=field)
    return concat(*((x, ) for x in xs), field=field)

def hstack(*xs, field=None):
    """
    Horizontally concatenates the given matrices.
    """
    if field is None:
        if lits.field is None:
            raise RuntimeError("specify a field using 'lits'")
        field = lits.field
    if _keep_unpacking(xs):
        return hstack(*xs[0], field=field)
    return concat(xs, field=field)

def ascol(*xs, field=None):
    """
    Returns a column vector of the combined input vectors.
    """
    if field is None:
        if lits.field is None:
            raise RuntimeError("specify a field using 'lits'")
        field = lits.field
    if _keep_unpacking(xs):
        return ascol(*xs[0], field=field)
    def tocol(x):
        if isinstance(x, Matrix) and x.isrow:
            return x.T
        if isinstance(x, Matrix) and not x.iscol:
            raise TypeError(f"cannot give non-vectors, got size {x._size_str}")
        return x
    return concat(*((tocol(x), ) for x in xs), field=field)

def asrow(*xs, field=None):
    """
    Returns a row vector of the combined input vectors.
    """
    if field is None:
        if lits.field is None:
            raise RuntimeError("specify a field using 'lits'")
        field = lits.field
    if _keep_unpacking(xs):
        return asrow(*xs[0], field=field)
    def torow(x):
        if isinstance(x, Matrix) and x.iscol:
            return x.T
        if isinstance(x, Matrix) and not x.isrow:
            raise TypeError(f"cannot give non-vectors, got size {x._size_str}")
        return x
    return concat((torow(x) for x in xs), field=field)

def rep(x, rows, cols=1, *, field=None):
    """
    Repeats the given matrix the given number of times for each direction.
    """
    if field is None:
        if lits.field is None:
            raise RuntimeError("specify a field using 'lits'")
        field = lits.field
    if not isinstance(rows, int):
        raise TypeError("expected an integer number of rows, got "
                f"{_tname(type(rows))}")
    if not isinstance(cols, int):
        raise TypeError("expected an integer number of columns, got "
                f"{_tname(type(cols))}")
    return concat(*((x, ) * cols for _ in range(rows)), field=field)


def diag(*xs, field=None):
    """
    Creates a matrix with a diagonal of the given elements and zeros elsewhere.
    """
    if field is None:
        if lits.field is None:
            raise RuntimeError("specify a field using 'lits'")
        field = lits.field
    if _keep_unpacking(xs):
        return diag(*xs[0], field=field)
    diagvec = ascol(*xs, field=field)
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
    if field is None:
        if lits.field is None:
            raise RuntimeError("specify a field using 'lits'")
        field = lits.field
    return Matrix[field, (n, n)].eye

def zeros(rows, cols=None, *, field=None):
    """
    Zero-filled matrix, defaulting to square if only one size given.
    """
    if field is None:
        if lits.field is None:
            raise RuntimeError("specify a field using 'lits'")
        field = lits.field
    cols = rows if cols is None else cols
    return Matrix[field, (rows, cols)].zeros

def ones(rows, cols=None, *, field=None):
    """
    One-filled matrix, defaulting to square if only one size given.
    """
    if field is None:
        if lits.field is None:
            raise RuntimeError("specify a field using 'lits'")
        field = lits.field
    cols = rows if cols is None else cols
    return Matrix[field, (rows, cols)].ones



def summ(*xs, field=None):
    """
    Returns the additive sum of the given values.
    """
    if field is None:
        if lits.field is None:
            raise RuntimeError("specify a field using 'lits'")
        field = lits.field
    if _keep_unpacking(xs):
        return summ(*xs[0], field=field)
    if not xs:
        return Single[field].zero
    r = Single[field].zero
    for x in xs:
        if not isinstance(x, Matrix):
            x = Single[field].cast(x)
        for y in x:
            r += y
    return r

def prod(*xs, field=None):
    """
    Returns the multiplicative product of the given values.
    """
    if field is None:
        if lits.field is None:
            raise RuntimeError("specify a field using 'lits'")
        field = lits.field
    if _keep_unpacking(xs):
        return prod(*xs[0], field=field)
    if not xs:
        return Single[field].one
    r = Single[field].one
    for x in xs:
        if not isinstance(x, Matrix):
            x = Single[field].cast(x)
        for y in x:
            r *= y
    return r

def minn(*xs, field=None):
    """
    Returns the minimum of the given values (first occurrence in the case of
    ties).
    """
    if field is None:
        if lits.field is None:
            raise RuntimeError("specify a field using 'lits'")
        field = lits.field
    if _keep_unpacking(xs):
        return minn(*xs[0], field=field)
    if not xs:
        raise ValueError("cannot find minimum of no elements")
    r = None
    for x in xs:
        if not isinstance(x, Matrix):
            x = Single[field].cast(x)
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
    if field is None:
        if lits.field is None:
            raise RuntimeError("specify a field using 'lits'")
        field = lits.field
    if _keep_unpacking(xs):
        return maxx(*xs[0], field=field)
    if not xs:
        raise ValueError("cannot find maximum of no elements")
    r = None
    for x in xs:
        if not isinstance(x, Matrix):
            x = Single[field].cast(x)
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
    if field is None:
        if lits.field is None:
            raise RuntimeError("specify a field using 'lits'")
        field = lits.field
    if _keep_unpacking(xs):
        return mean(*xs[0], field=field)
    if not xs:
        raise ValueError("cannot find the arithmetic mean no elements")
    r = Single[field].zero
    n = Single[field].zero
    for x in xs:
        if not isinstance(x, Matrix):
            x = Single[field].cast(x)
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
    if field is None:
        if lits.field is None:
            raise RuntimeError("specify a field using 'lits'")
        field = lits.field
    if _keep_unpacking(xs):
        return geomean(*xs[0], field=field)
    if not xs:
        raise ValueError("cannot find the geometric mean no elements")
    r = Single[field].one
    n = Single[field].zero
    for x in xs:
        if not isinstance(x, Matrix):
            x = Single[field].cast(x)
        for y in x:
            r *= y
            n += Single[field].one
    return r.root(n)

def harmean(*xs, field=None):
    """
    Returns the harmonic mean of the given values.
    """
    if field is None:
        if lits.field is None:
            raise RuntimeError("specify a field using 'lits'")
        field = lits.field
    if _keep_unpacking(xs):
        return harmean(*xs[0], field=field)
    if not xs:
        raise ValueError("cannot find the harmonic mean no elements")
    r = Single[field].zero
    n = Single[field].zero
    for x in xs:
        if not isinstance(x, Matrix):
            x = Single[field].cast(x)
        for y in x:
            r += y.one / y
            n += Single[field].one
    return n / r

def quadmean(*xs, field=None):
    """
    Returns the quadratic mean (root-mean-square) of the given values.
    """
    if field is None:
        if lits.field is None:
            raise RuntimeError("specify a field using 'lits'")
        field = lits.field
    if _keep_unpacking(xs):
        return quadmean(*xs[0], field=field)
    if not xs:
        raise ValueError("cannot find the root-mean-square no elements")
    r = Single[field].zero
    n = Single[field].zero
    for x in xs:
        if not isinstance(x, Matrix):
            x = Single[field].cast(x)
        for y in x:
            r += y * y
            n += Single[field].one
    return (r / n).sqrt

def logmean(x, y, *, field=None):
    """
    Returns the logarithmic mean of 'x' and 'y': (x - y) / ln(x / y)
    """
    if field is None:
        if lits.field is None:
            raise RuntimeError("specify a field using 'lits'")
        field = lits.field
    Mat = Single[field]
    if isinstance(x, Matrix):
        Mat = type(x)
    elif isinstance(y, Matrix):
        Mat = type(y)
    if not isinstance(y, Matrix):
        y = Mat.cast(y)
    if not isinstance(x, Matrix):
        x = Mat.cast(x)
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
            line = s[:90 - w]
            if len(line) == 90 - w and " " in line:
                line = line[:line.rindex(" ")]
            print(line)
            s = s[len(line):].lstrip()

    def print_attr(name, desc):
        s = f"{name} .."
        s += "." * (18 - len(s)) + " "
        printme(s, desc)

    printme("Matrix - ", Matrix.__doc__)
    print_attr("M.field", "Cell type.")
    print_attr("M.shape", "(row count, column count)")

    Mat = Single[Field]
    attrs = [(name, attr) for name, attr in Mat.__dict__.items()
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
        elif name == "sub":
            expr = f"{m}.sub[rows, cols]"
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
