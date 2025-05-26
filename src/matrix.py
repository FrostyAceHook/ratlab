from math import prod

from field import Field
from util import classproperty, immutable, templated


# Matrices.
@templated(parents=Field, decorators=immutable)
def Matrix(field, shape):
    """ Fixed-sized two-dimensional sequence of elements. """
    if not isinstance(field, type) or not issubclass(field, Field):
        raise TypeError("field must be a field class")
    if issubclass(field, Matrix):
        raise TypeError("mate a matrix of matrices? calm down")
    if not isinstance(shape, tuple) or len(shape) != 2:
        raise TypeError("shape must be a tuple of (numrows, numcols)")
    shape = (int(shape[0]), int(shape[1]))
    if not all(isinstance(x, int) for x in shape):
        raise TypeError("must have integer numrows or numcols")
    if prod(shape) <= 0 or shape[0] < 0:
        raise TypeError("num elements <=0")

    def __init__(self, cells, print_colvec_flat=True, print_zero_as_dot=True):
        # cells = flattened, with contiguous rows.
        try:
            cells = tuple(cells)
        except Exception:
            raise TypeError("cells must be a flattened sequence")
        if not all(isinstance(x, field) for x in cells):
            raise TypeError("cells must be of given field")
        if len(cells) != prod(shape):
            raise TypeError("cells must match flattened shape")

        self.cells = cells
        self.print_colvec_flat = print_colvec_flat
        self.print_zero_as_dot = print_zero_as_dot

    @property
    def rows(self):
        """ Tuple of row vectors. """
        if not self:
            return tuple()
        num_rows, num_cols = shape
        if num_rows == 1:
            return (self, )
        vec_type = Matrix[field, (1, num_cols)]
        return tuple(vec_type(self.at(i, j) for j in range(num_cols))
                                            for i in range(num_rows))

    @property
    def cols(self):
        """ Tuple of column vectors. """
        if not self:
            return tuple()
        num_rows, num_cols = shape
        if num_cols == 1:
            return (self, )
        vec_type = Matrix[field, (num_rows, 1)]
        return tuple(vec_type(self.at(i, j) for i in range(num_rows))
                                            for j in range(num_cols))

    @classproperty
    def isvec(self):
        """ Is row vector or column vector? """
        return shape[0] == 1 or shape[1] == 1

    @classproperty
    def issquare(self):
        """ Is square matrix? """
        return shape[0] == shape[1]

    @property
    def isdiag(self):
        """ Is diagonal matrix? (only diagonal is non-zero) """
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
        cells = [None] * math.prod(shp)
        for i in range(shape[0]):
            for j in range(shape[1]):
                cells[shp[1]*j + i] = self.at(i, j)
        return Matrix[field, shp](cells)

    @property
    def inv(self):
        """ Matrix inverse. """
        assert self.issquare, "cannot invert a non-square matrix"
        assert self.det != 0, "cannot invert a non-invertible matrix"
        size = shape[0]
        aug = hstack(self, self.one)
        aug = aug.rref
        inverse_cols = aug.cols[size:]
        return hstack(*inverse_cols)

    @property
    def diag(self):
        """ Matrix diagonal (as column vector). """
        cells = [self.at(i, i) for i in range(min(shape))]
        return Matrix[field, (len(cells), 1)](cells)


    def __iter__(self):
        """ Row-major cell iterate. """
        return iter(self.cells)

    def __len__(self):
        """ Number of cells. """
        return len(self.cells)

    def at(self, i, j=None):
        """ Cell at row i, column j. Only need one ordinate if vector. """
        assert isinstance(i, int)
        if j is None:
            assert self.isvec, "specifiy both coordinates for a matrix cell"
            if i < 0:
                i += len(self)
            assert i in range(len(self)), "oob"
            return self.cells[i]
        assert isinstance(j, int)
        if i < 0:
            i += shape[0]
        if j < 0:
            j += shape[1]
        assert i in range(shape[0]) and j in range(shape[1]), "oob"
        return self.cells[shape[1]*i + j]

    def __getitem__(self, ij):
        """ Submatrix of specified rows and columns. """
        assert isinstance(ij, tuple)
        assert len(ij) == 2
        def process(k, length):
            if isinstance(k, int):
                return (k, )
            if isinstance(k, slice):
                return tuple(range(*k.indices(length)))
            return k # hope is iterable
        i, j = ij
        rs = process(i, shape[0])
        cs = process(j, shape[1])
        cells = [self.at(r, c) for r in rs for c in cs]
        shp = len(rs), len(cs)
        return Matrix[field, shp](cells)


    def __and__(lhs, rhs):
        """ Dot product between two vectors. """
        assert isinstance(rhs, Matrix), "must dot-product with a vector"
        assert field == rhs.field, "matrices must be over the same field"
        assert lhs.isvec, "cannot dot product a non-vector"
        assert rhs.isvec, "cannot dot product a non-vector"
        assert len(lhs) == len(rhs), "cannot operate on different sized vectors"
        return field.sumof(x*y for x, y in zip(lhs, rhs))

    def __or__(lhs, rhs):
        """ Cross product between two three-element vectors. """
        assert isinstance(rhs, Matrix), "must cross-product with a vector"
        assert field == rhs.field, "matrices must be over the same field"
        assert lhs.isvec, "cannot cross product a non-vector"
        assert rhs.isvec, "cannot cross product a non-vector"
        assert len(lhs) == 3, "cannot cross product a non-3D-vector"
        assert len(rhs) == 3, "cannot cross product a non-3D-vector"
        ax, ay, az = lhs.cells
        bx, by, bz = rhs.cells
        cells = (ay*bz - az*by, az*bx - ax*bz, ax*by - ay*bx)
        return Matrix[field, (3, 1)](cells)

    def __matmul__(lhs, rhs):
        """ Matrix multiplication. """
        assert isinstance(rhs, Matrix), "must be matrix multiply with a matrix"
        assert field == rhs.field, "matrices must be over the same field"
        assert shape[1] == rhs.shape[0], "incorrect size for matrix multiplication"
        shp = shape[0], rhs.shape[1]
        rows = lhs.rows
        cols = rhs.cols
        cells = [None] * prod(shp)
        for i in range(shp[0]):
            for j in range(shp[1]):
                cells[shp[1]*i + j] = rows[i] & cols[j]
        return Matrix[field, shp](cells)

    def __xor__(matrix, exp):
        """ Matrix power. """
        try:
            exp = int(exp)
        except Exception:
            pass
        assert isinstance(exp, int), "must use an integer power"
        assert matrix.issquare, "matrix must be square to multiply with itself"
        if exp < 0:
            return matrix.inv ^ (-exp)
        power = matrix.one
        running = matrix
        while exp:
            if (exp & 1):
                power @= running
            exp >>= 1
            running @= running
        return power

    def __call__(matrix, *values):
        """ Shorthand for: matrix @ vstack(*values) """
        return matrix @ vstack(*values)


    @property
    def det(self):
        """ Matrix determinant. """
        assert self.issquare, "cannot find determinant of a non-square matrix"

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

        return determinant(self.cells, shape[0])

    @property
    def trace(self):
        """ Matrix trace. """
        assert self.issquare, "cannot find trace of a non-square matrix"
        return field.prodof(self.at(i, i) for i in range(shape[0]))

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

        cells = list(self.cells)

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
        pivotat = [field.find(field.one, sys.rows[i]) for i in range(shape[0])]
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
        return super()._cast(obj, for_obj)

    @classmethod
    def _zero(cls):
        zero = field.zero
        cells = [zero] * prod(shape)
        return cls(cells)
    @classmethod
    def _one(cls):
        if not cls.issquare:
            raise NotImplementedError()
        zero = field.zero
        one = field.one
        cells = [zero] * prod(shape)
        n = shape[0]
        for i in range(n):
            cells[i*n + i] = one
        return cls(cells)

    def __abs__(self):
        return Matrix[field, shape](abs(x) for x in self.cells)
    @classmethod
    def _add(cls, a, b):
        return cls(x + y for x, y in zip(a, b))
    @classmethod
    def _neg(cls, a):
        return cls(-x for x in a)
    @classmethod
    def _mul(cls, a, b):
        return cls(x * y for x, y in zip(a, b))
    @classmethod
    def _rec(cls, a):
        return cls(~x for x in a)
    @classmethod
    def _exp(cls, a):
        return cls(x.exp for x in a)
    @classmethod
    def _log(cls, a):
        return cls(x.log for x in a)

    @classmethod
    def _eq_zero(cls, a):
        return all(x == field.zero for x in a)

    @classmethod
    def _hashof(cls, a):
        return hash(a.cells)

    def __repr__(self):
        if self.print_zero_as_dot:
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

        multiline = not (self.print_colvec_flat and self.isvec)
        max_len = max(len(s(x)) for x in self.cells)
        padded = (max_len > 3)
        rows = []
        for k, x in enumerate(self.cells):
            if (k % shape[1]) == 0:
                rows.append([])
            rows[-1].append(f"{s(x):>{max_len}}")
        join = lambda x: "  ".join(x) if padded else " ".join(x)
        wrap = lambda x: f"[ {x} ]" if padded else f"[{x}]"
        str_rows = (wrap(join(row)) for row in rows)
        return ("\n" if multiline else "").join(str_rows)


def eye(field, n):
    return Matrix[field, (n, n)].one

def diag(*elts):
    field = Field.fieldof(elts)
    n = len(elts)
    cells = [field.zero] * (n*n)
    for i in range(n):
        cells[i*n + i] = elts[i]
    return Matrix[field, (n, n)](cells)

def concat(*rows):
    cells = []
    shape = [0, 0]
    for row in rows:
        height = None
        elts = []
        for elt in row:
            if not isinstance(elt, Matrix):
                elt = Matrix[type(elt), (1, 1)]([elt])
            if height is None:
                height = elt.shape[0]
            if height != elt.shape[0]:
               raise ValueError("inconsistent vertical concat size")
            elts.append(elt)

        rowcells = []
        for row in range(height):
            for elt in elts:
                rowcells.extend(elt[row, :])
        assert len(rowcells) % height == 0
        width = len(rowcells) // height
        if not shape[1]:
            shape[1] = width
        if width != shape[1]:
            raise ValueError("inconsistent horizontal concat size")
        cells.extend(rowcells)
        shape[0] += height

    field = Field.fieldof(cells)
    return Matrix[field, tuple(shape)](cells)

def vstack(*elts):
    return concat(*([e] for e in elts))

def hstack(*elts):
    return concat(elts)
