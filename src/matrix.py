import inspect as _inspect
import types as _types
from math import prod as _prod, pi as _pi

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

    @_classconst
    def yes(cls):
        return object()

    @classmethod
    def atan2(cls, y, x):
        if "pi" in cls.consts:
            pi = cls.consts["pi"]
        else:
            pi = cls.from_float(_pi)
        two = cls.from_int(2)

        if cls.lt(cls.zero, x) is cls.yes:
            return cls.atan(cls.div(y, x))
        elif cls.lt(x, cls.zero) is cls.yes:
            if cls.lt(cls.zero, y):
                return cls.add(cls.atan(cls.div(y, x)), pi)
            else:
                return cls.sub(cls.atan(cls.div(y, x)), pi)
        elif cls.eq(x, cls.zero) is cls.yes:
            if cls.lt(cls.zero, y) is cls.yes:
                return cls.div(pi, two)
            elif cls.lt(y, cls.zero) is cls.yes:
                return cls.div(cls.sub(cls.zero, pi), two)
            elif cls.eq(y, cls.zero) is cls.yes:
                return cls.zero # atan(0/0)
            else:
                raise TypeError(f"value '{repr(y)}' could not be ordered "
                        f"against zero ('{repr(cls.zero)}')")
        else:
            raise TypeError(f"value '{repr(x)}' could not be ordered against "
                    f"zero ('{repr(cls.zero)}')")

    @classmethod
    def repr_short(cls, a):
        return cls.repr_long(a)

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
    # Field type (including this class). However, when a comparison is always
    # true, `cls.yes` must be returned (again, this object does not need to be a
    # bool). Note that the return type of comparisons must be consistent across
    # calls with different values (eq and lt must always return the type of
    # `cls.yes`).
    @_classconst
    def yes(cls): # object returned when the result of a comparison is yes.
        return object() # may be `True`.
    @classmethod
    def eq(cls, a, b): # a == b
        return cls.yes
    @classmethod
    def lt(cls, a, b): # a < b
        return cls.yes

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
    # It cannot be good if we tryna make a matrix outa these.
    assert T not in {tuple, list, dict, _types.FunctionType,
            _types.BuiltinFunctionType, _types.MethodType,
            _types.BuiltinMethodType}

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

    @_classconst
    def yes(cls):
        return True
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
        return repr(a.v)
    @classmethod
    def repr_long(cls, a):
        return repr(a.v)




# Matrices.
@_templated(parents=Field, decorators=_immutable)
def Matrix(field, shape):
    """
    Fixed-sized two-dimensional sequence of elements. The general rule is all
    field operations are element-wise, and then all matrix-specific function and
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
    if _prod(shape) == 0 and (shape != (0, 0) or field != Field):
        return Matrix[Field, (0, 0)]

    size_string = f"{shape[0]}x{shape[1]}"

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
                "abs": "cannot do absolution",

                "mul": "cannot do multiplication",
                "div": "cannot do division",

                "power": "cannot do powers",
                "root": "cannot do roots",
                "exp": "cannot do natural exponential",
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
                    f"{size_string}={_prod(shape)}, got {len(cells)})")

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
                    f"{size_string})")
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
        cells = (cls._f("zero"), ) * _prod(shape) if shape[0] else ()
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
                    f"{size_string}), use .rowmajor or .colmajor for matrix "
                    "iterate")
        return iter(s.rowmajor)

    def __getitem__(s, i):
        """
        Vector-only cell access.
        """
        if not s.isvec:
            raise TypeError("only vectors have bare getitem (got size "
                    f"{size_string}), use .at for matrix cell access")
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
        Tuple of row-major cells.
        """
        return tuple(single(x) for x in s._cells)

    @_instconst
    def colmajor(s):
        """
        Tuple of column-major cells.
        """
        return s.T.rowmajor

    def at(s, i, j):
        """
        Cell at row i, column j.
        """
        if not isinstance(i, int):
            raise TypeError("expected an integer row index (got "
                    f"{tname(type(i))}")
        if not isinstance(j, int):
            raise TypeError("expected an integer column index (got "
                    f"{tname(type(j))}")
        ii = i + (i < 0) * shape[0]
        jj = j + (j < 0) * shape[1]
        if ii not in range(shape[0]):
            raise ValueError(f"row index {i} out-of-bounds for {shape[0]} rows")
        if jj not in range(shape[0]):
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
        cells = (s._at(i, j) for j in range(shape[1]) for i in range(shape[0]))
        return Matrix[field, (shape[1], shape[0])](cells)

    @_instconst
    def inv(s):
        """
        Matrix inverse.
        """
        if not s.issquare:
            raise TypeError("cannot invert a non-square matrix (got size "
                    f"{s.size_string})")
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
        cells = (s.at(i, i) for i in range(min(shape)))
        return Matrix[field, (len(cells), 1)](cells)


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
                    f"size {s.size_string})")

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
                    f"{s.size_string})")
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
                    f"{s.size_string})")
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
                if 1 == sum(not eq(x, s._f("zero")) for x in c)
                and 1 == sum(not eq(x, s._f("one")) for x in c))

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
                    if any(eq(x, s.zero) for x in r))
        return tuple(rref.rows[i].T for i in nonzeros)

    @_instconst
    def nullspace(s):
        """
        Basis for null space (as column vectors).
        """
        sys = s.rref # implied zero-vec augment.
        def find_first_one(xs):
            for i, x in enumerate(xs):
                if eq(x, s.one):
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
        rep = s._f("repr_long" if islong else "repr_short")

        if s.isempty:
            return "my boy "*islong + "M.T."

        if s.issingle:
            # return "[ " + rep(s._cells[0]) + " ]"
            return rep(s._cells[0])

        if s._print_zero_as_dot:
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
                s.can_eq_zero = False

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


    # casts `o` to a matrix over the same field, but does no size check.
    @classmethod
    def _cast(cls, o):
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
        # Create a matrix with it as every element.
        return cls((o, ) * _prod(shape))


    @classmethod
    def _eltwise_unary(cls, s, func):
        assert isinstance(s, cls)
        cells = tuple(func(x) for x in s._cells)
        newfield = type(cells[0]) if cells else Field
        return Matrix[newfield, shape](cells)

    @classmethod
    def _eltwise_binary(cls, s, o, func): # tricksy
        assert isinstance(s, cls)

        # unpack over lists specifically.
        if isinstance(o, list):
            return [cls._eltwise_binary(s, x, func) for x in o]

        o = cls._cast(o)
        # Project singles to the full size.
        if s.issingle or o.issingle:
            if s.issingle:
                s = Matrix[field, o.shape](s._cells * _prod(o.shape))
            else:
                o = Matrix[field, s.shape](o._cells * _prod(s.shape))
        if s.shape != o.shape:
            raise TypeError("cannot element-wise operate on matrices of "
                    f"different sizes ({s.size_string} vs {o.size_string})")

        cells = tuple(func(a, b) for a, b in zip(s._cells, o._cells))
        newfield = type(cells[0]) if cells else Field
        return Matrix[newfield, s.shape](cells)


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
        return s._eltwise_unary(s, f)
    def __abs__(s):
        """
        Element-wise absolution.
        """
        return s._eltwise_unary(s, s._f("abs"))

    def __add__(s, o):
        """
        Element-wise addition.
        """
        return s._eltwise_binary(s, o, s._f("add"))
    def __radd__(s, o):
        return s._eltwise_binary(s, o, lambda a, b: s._f("add")(b, a))
    def __sub__(s, o):
        """
        Element-wise subtraction.
        """
        return s._eltwise_binary(s, o, s._f("sub"))
    def __rsub__(s, o):
        return s._eltwise_binary(s, o, lambda a, b: s._f("sub")(b, a))
    def __mul__(s, o):
        """
        Element-wise multiplication (use '@' for matrix multiplication).
        """
        return s._eltwise_binary(s, o, s._f("mul"))
    def __rmul__(s, o):
        return s._eltwise_binary(s, o, lambda a, b: s._f("mul")(b, a))
    def __truediv__(s, o):
        """
        Element-wise division.
        """
        return s._eltwise_binary(s, o, s._f("div"))
    def __rtruediv__(s, o):
        return s._eltwise_binary(s, o, lambda a, b: s._f("div")(b, a))
    def __pow__(s, o):
        """
        Element-wise power.
        """
        return s._eltwise_binary(s, o, s._f("power"))
    def __rpow__(s, o):
        return s._eltwise_binary(s, o, lambda a, b: s._f("power")(b, a))

    @_instconst
    def sqrt(s):
        """
        Element-wise square root.
        """
        cls._need("from_int", "to represent 2")
        two = s._f("from_int")(2)
        return s._eltwise_unary(s, lambda x: s._f("root")(x, two))
    @_instconst
    def cbrt(s):
        """
        Element-wise cube root.
        """
        cls._need("from_int", "to represent 3")
        three = s._f("from_int")(3)
        return s._eltwise_unary(s, lambda x: s._f("root")(x, three))

    def root(s, n):
        """
        Element-wise nth root.
        """
        return s._eltwise_binary(s, n, s._f("root"))

    @_instconst
    def exp(s):
        """
        Element-wise natural exponential.
        """
        return s._eltwise_unary(s, s._f("exp"))
    @_instconst
    def ln(s):
        """
        Element-wise natural logarithm.
        """
        return s._eltwise_unary(s, lambda x: s._f("log")(s._f("exp")(s._f("one")), x))

    def log(s, base):
        """
        Element-wise base-specified logarithm.
        """
        return s._eltwise_binary(s, base, lambda a, b: s._f("log")(b, a))

    @_instconst
    def sin(s):
        """
        Element-wise trigonometric sine.
        """
        return s._eltwise_unary(s, s._f("sin"))
    @_instconst
    def cos(cls, a):
        """
        Element-wise trigonometric cosine.
        """
        return s._eltwise_unary(s, s._f("cos"))
    @_instconst
    def tan(cls, a):
        """
        Element-wise trigonometric tangent.
        """
        return s._eltwise_unary(s, s._f("tan"))

    @_instconst
    def asin(s):
        """
        Element-wise trigonometric inverse-sine.
        """
        return s._eltwise_unary(s, s._f("asin"))
    @_instconst
    def acos(cls, a):
        """
        Element-wise trigonometric inverse-cosine.
        """
        return s._eltwise_unary(s, s._f("acos"))
    @_instconst
    def atan(cls, a):
        """
        Element-wise trigonometric inverse-tangent.
        """
        return s._eltwise_unary(s, s._f("atan"))


    def __eq__(s, o):
        """
        Element-wise equality (use 'eq' to check if all pairs are equal).
        """
        return s._eltwise_binary(s, o, s._f("eq"))
    def __ne__(s, o):
        return s._eltwise_binary(s, o, lambda a, b: not s._f("eq")(a, b))
    def __lt__(s, o):
        """
        Element-wise ordering (use 'lt' to check if all pairs are strictly
        ascending).
        """
        return s._eltwise_binary(s, o, s._f("lt"))
    def __le__(s, o):
        f = lambda a, b: s._f("eq")(a, b) or s._f("lt")(a, b)
        return s._eltwise_binary(s, o, f)
    def __gt__(s, o):
        return s._eltwise_binary(s, o, lambda a, b: s._f("lt")(b, a))
    def __ge__(s, o):
        f = lambda a, b: s._f("eq")(a, b) or s._f("lt")(b, a)
        return s._eltwise_binary(s, o, f)


    def __and__(s, o):
        """
        Vector dot product.
        """
        o = s._cast(o)
        if not s.isvec or not o.isvec:
            raise TypeError(f"can only dot product vectors (got sizes "
                    f"{s.size_string} and {o.size_string})")
        if len(s) != len(o):
            raise TypeError(f"can only dot product same-size vectors (got sizes "
                    f"{s.size_string} and {o.size_string})")
        mul = s._f("mul")
        add = s._f("add")
        dot = s._f("zero")
        for a, b in zip(s._cells, o._cells):
            dot = add(dot, mul(a, b))
        return s._single(dot)
    def __rand__(s, o):
        o = s._cast(o)
        return o.__and__(s)

    def __or__(s, o):
        """
        3D vector cross product.
        """
        o = s._cast(o)
        if not s.isvec or not o.isvec:
            raise TypeError(f"can only cross product vectors (got sizes "
                    f"{s.size_string} and {o.size_string})")
        if len(s) != 3 or len(o) != 3:
            raise TypeError(f"can only cross product 3-element vectors (got "
                    f"sizes {s.size_string} and {o.size_string})")
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
        o = s._cast(o)
        return o.__or__(s)

    def __matmul__(s, o):
        """
        Matrix multiplication.
        """
        o = s._cast(o)
        if s.shape[1] != o.shape[0]:
            raise TypeError("incorrect size for matrix multiplication "
                    f"({s.size_string} times {o.size_string})")
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
                    cells[r_idx] = add(cells[idx], prod)
        return Matrix[field, shp](cells)
    def __rmatmul__(s, o):
        o = s._cast(o)
        return o.__matmul__(s)

    def __call__(s, *values):
        """
        Shorthand for '@ vstack(*values)'
        """
        return s @ vstack(*values)

    def __xor__(s, exp):
        """
        Matrix power.
        """
        if isinstance(exp, field):
            exp = s._f("to_int")(exp)
        if not isinstance(exp, int):
            raise TypeError("can only raise a matrix to an integer power (got "
                    f"{_tname(type(exp))})")
        if not s.issquare:
            raise TypeError("can only raise square matrices to a power (got "
                    f"{s.size_string})")
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
            eqz = lambda x: (s._f("eq")(x, s._f("zero")) is s._f("yes"))
            return not all(eqz(x) for x in s._cells)
        if hasattr(field, "one"):
            eqo = lambda x: (s._f("eq")(x, s._f("one")) is s._f("yes"))
            return not all(eqo(x) for x in s._cells)
        raise NotImplementedError("no zero or one element in field "
                f"{_tname(field)}, must specify an element to compare to")

    def __int__(s):
        """
        Cast a single to int.
        """
        if not s.issingle:
            raise TypeError(f"cannot cast {s.size_string} matrix to int (requires "
                    "1x1)")
        return s._f("to_int")(s._cells[0])
    def __float__(s):
        """
        Cast a single to float.
        """
        if not s.issingle:
            raise TypeError(f"cannot cast {s.size_string} matrix to float "
                    "(requires 1x1)")
        return s._f("to_float")(s._cells[0])
    def __complex__(s):
        """
        Cast a single to complex.
        """
        if not s.issingle:
            raise TypeError(f"cannot cast {s.size_string} matrix to complex "
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



    def __getattr__(s, attr):
        """
        If a non-matrix attribute is accessed, it will be retrived from each
        element instead.
        """
        if attr.startswith("_"):
            raise AttributeError()
        cells = object.__getattribute__(s, "_cells") # cooked.
        cells = tuple(getattr(x, attr) for x in cells)
        field = type(cells[0]) if cells else Field
        return Matrix[field, shape](cells)


def eq(a, b):
    """
    True if all pairs of elements in a and b are equal, false otherwise.
    """
    if not isinstance(a, Matrix):
        raise TypeError(f"expected matrix for lhs, got {tname(type(a))}")
    if not isinstance(b, Matrix):
        raise TypeError(f"expected matrix for rhs, got {tname(type(b))}")
    if a.field != b.field:
        raise TypeError(f"matrix fields must be the same (got {tname(a.field)} "
                f"and {tname(b.field)}")
    return all(x._cells[0] is a._f("yes") for x in (a == b).rowmajor)
def neq(a, b):
    if not isinstance(a, Matrix):
        raise TypeError(f"expected matrix for lhs, got {tname(type(a))}")
    if not isinstance(b, Matrix):
        raise TypeError(f"expected matrix for rhs, got {tname(type(b))}")
    if a.field != b.field:
        raise TypeError(f"matrix fields must be the same (got {tname(a.field)} "
                f"and {tname(b.field)}")
    return all(x._cells[0] is a._f("yes") for x in (a != b).rowmajor)
def lt(a, b):
    """
    True if all pairs of elements in a and b are strictly ascending, false
    otherwise.
    """
    if not isinstance(a, Matrix):
        raise TypeError(f"expected matrix for lhs, got {tname(type(a))}")
    if not isinstance(b, Matrix):
        raise TypeError(f"expected matrix for rhs, got {tname(type(b))}")
    if a.field != b.field:
        raise TypeError(f"matrix fields must be the same (got {tname(a.field)} "
                f"and {tname(b.field)}")
    return all(x._cells[0] is a._f("yes") for x in (a < b).rowmajor)
def le(a, b):
    if not isinstance(a, Matrix):
        raise TypeError(f"expected matrix for lhs, got {tname(type(a))}")
    if not isinstance(b, Matrix):
        raise TypeError(f"expected matrix for rhs, got {tname(type(b))}")
    if a.field != b.field:
        raise TypeError(f"matrix fields must be the same (got {tname(a.field)} "
                f"and {tname(b.field)}")
    return all(x._cells[0] is a._f("yes") for x in (a <= b).rowmajor)
def gt(a, b):
    if not isinstance(a, Matrix):
        raise TypeError(f"expected matrix for lhs, got {tname(type(a))}")
    if not isinstance(b, Matrix):
        raise TypeError(f"expected matrix for rhs, got {tname(type(b))}")
    if a.field != b.field:
        raise TypeError(f"matrix fields must be the same (got {tname(a.field)} "
                f"and {tname(b.field)}")
    return all(x._cells[0] is a._f("yes") for x in (a > b).rowmajor)
def ge(a, b):
    if not isinstance(a, Matrix):
        raise TypeError(f"expected matrix for lhs, got {tname(type(a))}")
    if not isinstance(b, Matrix):
        raise TypeError(f"expected matrix for rhs, got {tname(type(b))}")
    if a.field != b.field:
        raise TypeError(f"matrix fields must be the same (got {tname(a.field)} "
                f"and {tname(b.field)}")
    return all(x._cells[0] is a._f("yes") for x in (a >= b).rowmajor)



def single(x):
    return Matrix[type(x), (1, 1)]((x, ))

def issingle(x):
    """
    Returns true iff 'x' is a matrix with only one cell.
    """
    return isinstance(x, Matrix) and x.issingle


def long(a):
    """
    Prints a long string representation of 'a'.
    """
    if not isinstance(a, Matrix):
        raise TypeError(f"expected matrix, got {tname(type(a))}")
    print(a.repr_long)


def concat(*rows):
    """
    Concatenates the given matrices as:
    concat(([0], [1]), ([2], [3])) -> [1,2][2,3]
    """
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
                x = single(x)
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

    field = type(cells[0]) if _prod(shape) else Field
    return Matrix[field, tuple(shape)](cells)

def vstack(*xs):
    """
    Vertically concatenates the given matrices.
    """
    if len(xs) == 1 and _iterable(xs[0]) and not isinstance(xs[0], Matrix):
        return vstack(*xs[0])
    return concat(*((x, ) for x in xs))

def hstack(*xs):
    """
    Horizontally concatenates the given matrices.
    """
    if len(xs) == 1 and _iterable(xs[0]) and not isinstance(xs[0], Matrix):
        return hstack(*xs[0])
    return concat(xs)

def rep(x, rows, cols=1):
    """
    Repeats the given matrix the given number of times for each
    direction.
    """
    if not isinstance(rows, int):
        raise TypeError("expected an integer number of rows, got "
                f"{tname(type(rows))}")
    if not isinstance(cols, int):
        raise TypeError("expected an integer number of columns, got "
                f"{tname(type(cols))}")
    return concat(*((x, ) * cols for _ in range(rows)))

def diag(*xs):
    """
    Creates a matrix with a diagonal of the given elements and zeros elsewhere.
    """
    if len(xs) == 1 and _iterable(xs[0]) and not issingle(x):
        return diag(*xs[0])
    field = xs[0].field if xs else Field
    n = len(xs)
    Mat = Matrix[field, (n, n)]
    cells = [Mat._f("zero")] * (n*n) if n else []
    for i in range(n):
        cells[i*n + i] = xs[i]
    return Mat(cells)

def eye(n, *, field):
    """
    Identity matrix, of the given size.
    """
    return Matrix[field, (n, n)].eye

def zeros(rows, cols=None, *, field):
    """
    Zero-filled matrix, defaulting to square if only one size given.
    """
    cols = rows if cols is None else cols
    return Matrix[field, (rows, cols)].zeros

def ones(rows, cols=None, *, field):
    """
    One-filled matrix, defaulting to square if only one size given.
    """
    cols = rows if cols is None else cols
    return Matrix[field, (rows, cols)].ones


def sqrt(x):
    """
    Alias for 'x.sqrt', and overloaded for basic number types.
    """
    if isinstance(x, Matrix):
        return x.sqrt
    x = float(x)
    return math.sqrt(x)
def cbrt(x):
    """
    Alias for 'x.cbrt', and overloaded for basic number types.
    """
    if isinstance(x, Matrix):
        return x.cbrt
    x = float(x)
    return math.cbrt(x)
def root(x, n):
    """
    Alias for 'x.root(n)', and overloaded for basic number types.
    """
    if isinstance(x, Matrix):
        return x.root(n)
    x = float(x)
    n = int(n)
    return x ** (1.0 / n)

def exp(x):
    """
    Alias for 'x.exp', and overloaded for basic number types.
    """
    if isinstance(x, Field):
        return x.exp
    x = float(x)
    return math.exp(x)
def log(x):
    """
    Alias for 'x.log', and overloaded for basic number types.
    """
    if isinstance(x, Field):
        return x.log
    x = float(x)
    if x == 0.0:
        return -float("inf")
    return math.log(x)

def sin(x):
    """
    Alias for 'x.sin', and overloaded for basic number types.
    """
    if isinstance(x, Matrix):
        return x.sin
    x = float(x)
    lookup = {pi/2: 1.0, -pi/2: -1.0, pi: 0.0, -pi: 0.0, 2*pi: 0.0}
    return lookup.get(x, math.sin(x))
def cos(x):
    """
    Alias for 'x.cos', and overloaded for basic number types.
    """
    if isinstance(x, Matrix):
        return x.cos
    x = float(x)
    lookup = {pi/2: 0.0, -pi/2: 0.0, pi: -1.0, -pi: -1.0, 2*pi: 1.0}
    return lookup.get(x, math.cos(x))
def tan(x):
    """
    Alias for 'x.tan', and overloaded for basic number types.
    """
    if isinstance(x, Matrix):
        return x.tan
    x = float(x)
    return math.tan(x)

def asin(x):
    """
    Alias for 'x.asin', and overloaded for basic number types.
    """
    if isinstance(x, Matrix):
        return x.asin
    x = float(x)
    lookup = {pi/2: 1.0, -pi/2: -1.0, pi: 0.0, -pi: 0.0, 2*pi: 0.0}
    return lookup.get(x, math.asin(x))
def acos(x):
    """
    Alias for 'x.acos', and overloaded for basic number types.
    """
    if isinstance(x, Matrix):
        return x.acos
    x = float(x)
    lookup = {pi/2: 0.0, -pi/2: 0.0, pi: -1.0, -pi: -1.0, 2*pi: 1.0}
    return lookup.get(x, math.acos(x))
def atan(x):
    """
    Alias for 'x.atan', and overloaded for basic number types.
    """
    if isinstance(x, Matrix):
        return x.atan
    x = float(x)
    return math.atan(x)

def atan2(y, x):
    """
    Quadrant-aware 'atan(y / x)'.
    """
    if isinstance(x, Matrix) or isinstance(y, Matrix):
        if not isinstance(x, Matrix):
            x = y._cast(x)
        if not isinstance(y, Matrix):
            y = x._cast(y)
        return x._eltwise_binary(x, y, x._f("atan2"))
    y = float(y)
    x = float(x)
    return math.atan2(y, x)





def keep_unpacking(xs):
    return len(xs) == 1 and _iterable(xs[0]) and not issingle(xs[0])


def summ(*xs, field):
    if keep_unpacking(xs):
        return summ(*xs[0], field=field)
    if not xs:
        return single(field.zero)
    r = single(field.zero)
    for x in xs:
        r += x
    return r

def prod(*xs, field):
    if keep_unpacking(xs):
        return prod(*xs[0], field=field)
    if not xs:
        return single(field.one)
    r = single(field.one)
    for x in xs:
        r *= x
    return r

def ave(*xs, field):
    if keep_unpacking(xs):
        return ave(*xs[0], field=field)
    if not xs:
        raise ValueError("cannot average no elements")
    r = single(field.zero)
    i = single(field.zero)
    for x in xs:
        r += x
        i += single(field.one)
    return r / i

def minn(*xs, field):
    if keep_unpacking(xs):
        return minn(*xs[0], field=field)
    if not xs:
        raise ValueError("cannot find minimum of no elements")
    r = xs[0]
    for x in xs[1:]:
        if x < r:
            r = x
    return r

def maxx(*xs, field):
    if keep_unpacking(xs):
        return maxx(*xs[0], field=field)
    if not xs:
        raise ValueError("cannot find maximum of no elements")
    r = xs[0]
    for x in xs[1:]:
        if x > r:
            r = x
    return r



class series:
    def __init__(self, xs):
        if not isinstance(xs, Matrix):
            raise TypeError("expected a matrix for the argument of series, got "
                    f"{tname(type(xs))}")
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
        print(name, end="")
        s = long.strip().replace("\n", " ")
        w = 0
        while s:
            if w:
                print(w * " ", end="")
            else:
                w = len(name)
            line = s[:80 - w]
            if len(line) == 80 - w and " " in line:
                line = line[:line.rindex(" ")]
            print(line)
            s = s[len(line):].lstrip()

    def print_attr(name, desc):
        s = f"{name} .."
        s += "." * (16 - len(s)) + " "
        printme(s, desc)

    printme("Matrix - ", Matrix.__doc__)
    print_attr("M.field", "Cell type.")
    print_attr("M.shape", "(row count, column count)")

    cls = Matrix[Field, (1, 1)]
    attrs = [(name, attr) for name, attr in vars(cls).items()
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
        if name in func_ops:
            name = f"{func_ops[name]}(m)"
        elif name in un_ops:
            name = f"{un_ops[name]}m"
        elif name in bin_ops:
            name = f"m {bin_ops[name]} m"
        elif name == "__xor__":
            name = "m ^ exp"
        elif name == "sub":
            name = "m.sub[rows, cols]"
        elif name == "__getattr__":
            name = "m.attr"
        elif callable(attr):
            sig = _inspect.signature(attr)
            sig = str(sig)
            sig = sig[1:-1]
            sig = sig[sig.index(",") + 1:].strip() if "," in sig else ""
            if name == "__getitem__":
                name = f"m[{sig}]"
            elif name == "__call__":
                name = f"m({sig})"
            else:
                name = f"m.{name}({sig})"
        else:
            name = f"m.{name}"
        print_attr(name, attr.__doc__)

    # also chuck the functions operating on or returning matrices.
    extras = [concat, vstack, hstack, rep, diag, eye, zeros, ones]
    for func in extras:
        sig = _inspect.signature(func)
        sig = str(sig)
        sig = sig[1:-1]
        if sig.endswith(", *, field"):
            sig = sig[:-len(", *, field")]
        elif sig.endswith(", field"):
            sig = sig[:-len(", field")]
        name = f"{func.__name__}({sig})"
        print_attr(name, func.__doc__)

def print_all_docstrings():
    current_function = _inspect.currentframe().f_code.co_name
    for name, obj in globals().items():
        if (
            _inspect.isfunction(obj)
            and obj.__module__ == __name__  # Only functions in this file
            and name != current_function    # Skip this function itself
        ):
            print(f"Function: {name}")
            print(_inspect.getdoc(obj) or "[No docstring]")
            print("-" * 40)
