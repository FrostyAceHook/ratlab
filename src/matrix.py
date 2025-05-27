import inspect
from math import prod

from field import Field, fieldof
from util import iterable, classproperty, immutable, templated


# Matrices.
@templated(parents=Field, decorators=immutable)
def Matrix(field, shape):
    """
    Fixed-sized two-dimensional sequence of elements. The general rule is all
    field operations are element-wise, and then all matrix-specific function and
    operations are implemented independantly.
    """
    if not isinstance(field, type) or not issubclass(field, Field):
        raise TypeError("field must be a field class")
    if issubclass(field, Matrix):
        raise TypeError("mate a matrix of matrices? calm down")

    if not isinstance(shape, tuple) or len(shape) != 2:
        raise TypeError("shape must be a tuple of (numrows, numcols)")
    if not isinstance(shape[0], int) or not isinstance(shape[1], int):
        return Matrix[field, (int(shape[0]), int(shape[1]))]
    if shape[0] < 0 or shape[1] < 0:
        raise TypeError("cannot have negative size")
    # collapse empty matrices to be over Field with (0,0).
    if prod(shape) == 0 and (shape != (0, 0) or field != Field):
        return Matrix[Field, (0, 0)]

    def __init__(self, cells, print_colvec_flat=True, print_zero_as_dot=True):
        # cells = flattened, with contiguous rows.
        if not iterable(cells):
            raise TypeError("cells must be a flattened sequence")
        cells = tuple(cells)
        if not all(isinstance(x, field) for x in cells):
            raise TypeError("cells must be of given field")
        if len(cells) != prod(shape):
            raise TypeError("cells must match flattened shape")

        self._cells = cells
        self._print_colvec_flat = print_colvec_flat
        self._print_zero_as_dot = print_zero_as_dot


    @property
    def rows(self):
        """ Tuple of row vectors. """
        num_rows, num_cols = shape
        vec_type = Matrix[field, (1, num_cols)]
        return tuple(vec_type(self.at(i, j) for j in range(num_cols))
                                            for i in range(num_rows))

    @property
    def cols(self):
        """ Tuple of column vectors. """
        num_rows, num_cols = shape
        vec_type = Matrix[field, (num_rows, 1)]
        return tuple(vec_type(self.at(i, j) for i in range(num_rows))
                                            for j in range(num_cols))

    @classproperty
    def isempty(cls):
        """ Is empty matrix? (0x0) """
        return shape[0] == 0

    @classproperty
    def isvec(cls):
        """ Is row vector or column vector? (empty counts as vector) """
        return cls.isempty or shape[0] == 1 or shape[1] == 1

    @classproperty
    def issquare(cls):
        """ Is square matrix? """
        return shape[0] == shape[1]

    @property
    def isdiag(self):
        """ Is diagonal matrix? (square, and only diagonal is non-zero) """
        if not self.issquare:
            return False
        for i in range(shape[0]):
            for j in range(shape[1]):
                if i == j:
                    continue
                if self.at(i, j) != field.zero:
                    return False
        return True

    @property
    def isuppertri(self):
        """ Is upper-triangular matrix? (square, and below diagonal is zero) """
        if not self.issquare:
            return False
        for i in range(shape[0]):
            for j in range(i):
                if self.at(i, j) != field.zero:
                    return False
        return True

    @property
    def islowertri(self):
        """ Is lower-triangular matrix? (square, and above diagonal is zero) """
        return self.T.isuppertri

    @property
    def isorthogonal(self):
        """ Is orthogonal matrix? (transpose == inverse) """
        return self.T == self.inv

    @property
    def issymmetric(self):
        """ Is symmetric matrix? (square, and below diagonal = above diagonal)
        """
        if not self.issquare:
            return False
        for i in range(shape[0]):
            for j in range(i):
                if self.at(i, j) != self.at(j, i):
                    return False
        return True


    @property
    def T(self):
        """ Matrix transpose. """
        shp = shape[1], shape[0]
        cells = [self.at(i, j) for j in range(shape[1]) for i in range(shape[0])]
        return Matrix[field, shp](cells)

    @property
    def inv(self):
        """ Matrix inverse. """
        assert self.issquare, "cannot invert a non-square matrix"
        assert self.det != field.zero, "cannot invert a non-invertible matrix"
        size = shape[0]
        aug = hstack(self, self.eye)
        aug = aug.rref
        inverse_cols = aug.cols[size:]
        return hstack(*inverse_cols)

    @property
    def diag(self):
        """ Matrix diagonal (as column vector). """
        cells = [self.at(i, i) for i in range(min(shape))]
        return Matrix[field, (len(cells), 1)](cells)

    @classproperty
    def eye(cls):
        """ Identity matrix. """
        assert cls.issquare, "only square matricies have an identity"
        n = shape[0]
        cells = [field.zero] * prod(shape) if n else []
        for i in range(n):
            cells[i*n + i] = field.one
        return cls(cells)


    def __iter__(self):
        """ Vector-only cell iterate. """
        assert self.isvec, "cannot iterate a matrix, use .rowmajor or .colmajor"
        return iter(self._cells)

    def __getitem__(self, i):
        """ Vector-only cell access. """
        assert self.isvec, "use .at for 2d matrix cell access"
        i = int(i)
        return self._cells[i]

    def __len__(self):
        """ Number of cells. """
        return len(self._cells)

    @property
    def rowmajor(self):
        """ Row-major cells. """
        return self._cells

    @property
    def colmajor(self):
        """ Column-major cells. """
        return self.T._cells

    def at(self, i, j):
        """ Cell at row i, column j. """
        i = int(i)
        j = int(j)
        if i < 0:
            i += shape[0]
        if j < 0:
            j += shape[1]
        assert i in range(shape[0]) and j in range(shape[1]), "oob"
        return self._cells[shape[1]*i + j]

    class _SubmatrixGetter:
        def __init__(self, matrix):
            self._matrix = matrix
        def __getitem__(self, ij):
            assert isinstance(ij, tuple)
            assert len(ij) == 2
            def process(k, length):
                if isinstance(k, slice):
                    conv = lambda x: x if x is None else int(x)
                    k = slice(conv(k.start), conv(k.stop), conv(k.step))
                    return tuple(range(*k.indices(length)))
                if iterable(k):
                    return k
                return (int(k), )
            i, j = ij
            rs = process(i, shape[0])
            cs = process(j, shape[1])
            cells = [self._matrix.at(r, c) for r in rs for c in cs]
            shp = len(rs), len(cs)
            return Matrix[field, shp](cells)
    @property
    def sub(self):
        """ Submatrix of specified rows and columns. """
        return _SubmatrixGetter(self)


    def __and__(lhs, rhs):
        """ Dot product between two vectors. """
        assert isinstance(rhs, Matrix), "must dot-product with a vector"
        assert field == rhs.field, "matrices must be over the same field"
        assert lhs.isvec, "cannot dot product a non-vector"
        assert rhs.isvec, "cannot dot product a non-vector"
        assert len(lhs) == len(rhs), "cannot operate on different sized vectors"
        dot = field.zero
        for i in range(len(lhs)):
            dot += lhs[i] * rhs[i]
        return dot

    def __or__(lhs, rhs):
        """ Cross product between two three-element vectors. """
        assert isinstance(rhs, Matrix), "must cross-product with a vector"
        assert field == rhs.field, "matrices must be over the same field"
        assert lhs.isvec, "cannot cross product a non-vector"
        assert rhs.isvec, "cannot cross product a non-vector"
        assert len(lhs) == 3, "cannot cross product a non-3D-vector"
        assert len(rhs) == 3, "cannot cross product a non-3D-vector"
        ax, ay, az = lhs._cells
        bx, by, bz = rhs._cells
        cells = (ay*bz - az*by, az*bx - ax*bz, ax*by - ay*bx)
        shp = (3, 1) # column vector takes precedence.
        if lhs.shape == (1, 3) and rhs.shape == (1, 3):
            shp = (1, 3)
        return Matrix[field, shp](cells)

    def __matmul__(lhs, rhs):
        """ Matrix multiplication. """
        assert isinstance(rhs, Matrix), "must be matrix multiply with a matrix"
        assert shape[1] == rhs.shape[0], ("incorrect size for matrix "
                f"multiplication ({'x'.join(map(str, lhs.shape))} times "
                f"{'x'.join(map(str, rhs.shape))})")
        assert field == rhs.field, "matrices must be over the same field"
        shp = shape[0], rhs.shape[1]
        rows = lhs.rows
        cols = rhs.cols
        cells = [rows[i] & cols[j] for i in range(shp[0]) for j in range(shp[1])]
        return Matrix[field, shp](cells)

    def __xor__(m, exp):
        """ Matrix power. """
        try:
            exp = int(exp)
        except Exception:
            pass
        assert isinstance(exp, int), "must use an integer power"
        assert m.issquare, "matrix must be square to multiply with itself"
        if exp < 0:
            return m.inv ^ (-exp)
        power = m.eye
        running = m
        while True:
            if (exp & 1):
                power @= running
            exp >>= 1
            if not exp:
                break
            running @= running
        return power

    def __call__(m, *values):
        """ Shorthand for: m @ vstack(*values) """
        return m @ vstack(*values)


    @property
    def det(self):
        """ Matrix determinant. """
        assert self.issquare, "cannot find determinant of a non-square matrix"

        if self.isempty:
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
            det = field.zero
            for j in range(size):
                subcells = submatrix(cells, size, 0, j)
                subsize = size - 1
                subdet = determinant(subcells, subsize)
                if j & 1:
                    det -= cells[j] * subdet
                else:
                    det += cells[j] * subdet
            return det

        return determinant(self._cells, shape[0])

    @property
    def trace(self):
        """ Matrix trace. """
        assert self.issquare, "cannot find trace of a non-square matrix"
        trace = field.zero
        for i in range(shape[0]):
            trace += self.at(i, i)
        return trace

    @property
    def mag(self):
        """ Vector magnitude. """
        assert self.isvec, "cannot find magnitude of a non-vector"
        return (self & self).sqrt


    @property
    def rref(self):
        """ Reduced-row echelon form. """

        def row_swap(shape, cells, row1, row2):
            num_cols = shape[1]
            for j in range(num_cols):
                k1 = row1 * num_cols + j
                k2 = row2 * num_cols + j
                cells[k1], cells[k2] = cells[k2], cells[k1]

        def row_mul(shape, cells, row, by):
            num_cols = shape[1]
            for j in range(num_cols):
                cells[row*num_cols + j] *= by

        def row_add(shape, cells, src, by, dst):
            num_cols = shape[1]
            for i in range(num_cols):
                src_k = src*num_cols + i
                dst_k = dst*num_cols + i
                cells[dst_k] += by * cells[src_k]

        cells = list(self._cells)

        num_rows, num_cols = shape
        lead = 0
        for r in range(num_rows):
            if lead >= num_cols:
                break

            i = r
            while cells[num_cols*i + lead] == field.zero:
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
            if pivot_value != field.zero:
                row_mul(shape, cells, r, ~pivot_value)

            for i in range(num_rows):
                if i != r:
                    row_lead_value = cells[num_cols*i + lead]
                    if row_lead_value != field.zero:
                        row_add(shape, cells, r, -row_lead_value, i)

            lead += 1

        return Matrix[field, shape](cells)

    @property
    def pivots(self):
        """ rref pivot column indices. """
        rref = self.rref
        return tuple(i for i, c in enumerate(rref.cols)
                if 1 == sum(x != field.zero for x in c)
                and 1 == sum(x == field.one for x in c))

    @property
    def nonpivots(self):
        """ rref non-pivot column indices. """
        return tuple(j for j in range(shape[1]) if j not in self.pivots)


    @property
    def colspace(self):
        """ Basis for column space (as column vectors). """
        return tuple(self.cols[p] for p in self.pivots)

    @property
    def rowspace(self):
        """ Basis for row space (as column vectors). """
        rref = self.rref
        nonzeros = [i for i, r in enumerate(rref.rows)
                if any(x != field.zero for x in r)]
        return tuple(rref.rows[i].T for i in nonzeros)

    @property
    def nullspace(self):
        """ Basis for null space (as column vectors). """
        sys = self.rref # implied zero-vec augment.
        def find_first_one(xs):
            for i, x in enumerate(xs):
                if x == field.one:
                    return i
            return None
        pivotat = [find_first_one(sys.rows[i]) for i in range(shape[0])]
        basis = [[field.zero]*shape[1] for _ in sys.nonpivots]
        for n, j in enumerate(sys.nonpivots):
            for i in range(shape[0]):
                if pivotat[i] is None or pivotat[i] > j:
                    basis[n][j] = 1
                    break
                basis[n][pivotat[i]] = -sys.at(i, j)
        return tuple(Matrix[field, (shape[1], 1)](x) for x in basis)


    @classmethod
    def _cast(cls, obj, for_obj=None):
        if isinstance(obj, field):
            return cls([obj] * prod(shape))
        return super(Matrix[field, shape], cls)._cast(obj, for_obj)

    @classmethod
    def _zero(cls):
        cells = [field.zero] * prod(shape) if shape[0] else []
        return cls(cells)
    @classmethod
    def _one(cls):
        one = field.one
        cells = [field.one] * prod(shape) if shape[0] else []
        return cls(cells)

    def __abs__(self): # make abs element-wise.
        return type(self)(abs(x) for x in self._cells)
    @classmethod
    def _add(cls, a, b):
        return cls(x + y for x, y in zip(a._cells, b._cells))
    @classmethod
    def _neg(cls, a):
        return cls(-x for x in a._cells)
    @classmethod
    def _mul(cls, a, b):
        return cls(x * y for x, y in zip(a._cells, b._cells))
    @classmethod
    def _rec(cls, a):
        return cls(~x for x in a._cells)
    @classmethod
    def _exp(cls, a):
        return cls(x.exp for x in a._cells)
    @classmethod
    def _log(cls, a):
        return cls(x.log for x in a._cells)

    @classmethod
    def _eq_zero(cls, a):
        return all(x == field.zero for x in a._cells)

    @classmethod
    def _hashof(cls, a):
        return hash(a._cells)

    def __repr__(self):
        if self.isempty:
            return "[ ]"

        if self._print_zero_as_dot:
            def s(x):
                eqz = False
                if not s.failed:
                    try:
                        eqz = (x == field.zero)
                    except NotImplementedError:
                        s.failed = True
                        pass
                return "." if eqz else repr(x)
            s.failed = False
        else:
            s = repr

        multiline = not (self._print_colvec_flat and self.isvec)
        max_len = max(len(s(x)) for x in self._cells)
        padded = (max_len > 3)
        rows = []
        for k, x in enumerate(self._cells):
            if (k % shape[1]) == 0:
                rows.append([])
            rows[-1].append(f"{s(x):>{max_len}}")
        join = lambda x: "  ".join(x) if padded else " ".join(x)
        wrap = lambda x: f"[ {x} ]" if padded else f"[{x}]"
        str_rows = (wrap(join(row)) for row in rows)
        return ("\n" if multiline else "").join(str_rows)


def concat(*rows):
    """ Concatenates the given matrices or elements as:
        concat(([0], [1]), ([2], [3])) -> [1,2][2,3]
    """
    cells = []
    shape = [0, 0]
    for row in rows:
        height = None
        xs = []
        if isinstance(row, Matrix):
            raise ValueError("must give a list of matrices for each row")
        for x in row:
            if not isinstance(x, Matrix):
                x = Matrix[type(x), (1, 1)]([x])
            if height is None:
                height = x.shape[0]
            if height != x.shape[0]:
               raise ValueError("inconsistent vertical concat size")
            xs.append(x)

        rowcells = []
        for row in range(height):
            for x in xs:
                rowcells.extend(x.rows[row])
        assert len(rowcells) % height == 0
        width = len(rowcells) // height
        if not shape[1]:
            shape[1] = width
        if width != shape[1]:
            raise ValueError("inconsistent horizontal concat size")
        cells.extend(rowcells)
        shape[0] += height

    field = fieldof(cells)
    return Matrix[field, tuple(shape)](cells)

def vstack(*xs):
    """ Vertically concatenates the given matrices or elements. """
    return concat(*([x] for x in xs))

def hstack(*xs):
    """ Horizontally concatenates the given matrices or elements. """
    return concat(xs)

def rep(x, rows, cols=1):
    """
    Repeats the given matrix or element the given number of times for each
    direction.
    """
    rows = int(rows)
    cols = int(cols)
    return concat(*[[x] * cols for _ in range(rows)])

def diag(*xs):
    """
    Creates a matrix with a diagonal of the given elements and zeros elsewhere.
    """
    if len(xs) == 1 and iterable(xs[0]):
        return diag(*xs[0])
    field = fieldof(xs)
    n = len(xs)
    cells = [field.zero] * (n*n) if n else []
    for i in range(n):
        cells[i*n + i] = xs[i]
    return Matrix[field, (n, n)](cells)

def eye(n, *, field):
    """ Identity matrix, of the given size. """
    return Matrix[field, (n, n)].eye
def zeros(rows, cols=None, *, field):
    """ Zero-filled matrix, defaulting to square if only one size given. """
    cols = rows if cols is None else cols
    return Matrix[field, (rows, cols)].zero
def ones(rows, cols=None, *, field):
    """ One-filled matrix, defaulting to square if only one size given. """
    cols = rows if cols is None else cols
    return Matrix[field, (rows, cols)].one


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
    print_attr("M.field", "Cells are of this field.")
    print_attr("M.shape", "(row count, column count)")

    cls = Matrix[Field, (1, 1)]
    attrs = [(name, attr) for name, attr in vars(cls).items()
            if attr.__doc__ is not None
            and name != "__module__"
            and name != "__doc__"
            and name != "template"
            and name not in Matrix.params]

    func_ops = {
        "__abs__": "abs",
        "__iter__": "iter",
        "__len__": "len",
        "__bool__": "bool",
        "__int__": "int",
        "__float__": "float",
        "__complex__": "complex",
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
        elif callable(attr):
            sig = inspect.signature(attr)
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
        sig = inspect.signature(func)
        sig = str(sig)
        sig = sig[1:-1]
        if sig.endswith(", *, field"):
            sig = sig[:-len(", *, field")]
        name = f"{func.__name__}({sig})"
        print_attr(name, func.__doc__)
