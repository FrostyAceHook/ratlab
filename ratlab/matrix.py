import cmath as _cmath
import inspect as _inspect
import itertools as _itertools
import math as _math
import struct as _struct
import sys as _sys

from . import cons as _cons
from .util import (
    classconst as _classconst,
    immutable as _immutable,
    instconst as _instconst,
    iterable as _iterable,
    maybe_unpack as _maybe_unpack,
    objtname as _objtname,
    singleton as _singleton,
    templated as _templated,
    tname as _tname,
)

import numpy as _np


_SEQ_TYPES = (tuple, list)
def _maybe_unpack_seq(xs):
    return _maybe_unpack(xs, only_unpack=_SEQ_TYPES)

def _maybe_unpack_ints(xs):
    xs = _maybe_unpack(xs, dont_unpack=Matrix)
    if len(xs) == 1 and isinstance(xs[0], Matrix) and xs[0].isvec:
        return tuple(xs[0].tofield(int).numpyvec())
    return xs




class Field:
    """
    Base class for implementing operations over elements. Cannot do operations on
    entire matrices, only performs for single elements of those matrices.
    """
    def __new__(cls):
        raise TypeError("cannot instantiate fields")

    @classmethod
    def dtype(cls): # dtype to use for the backing numpy array.
        return _np.dtype(object)

    @classmethod
    def fromobj(cls, obj): # create a field element from the given object.
        raise TypeError(f"cannot create {_tname(cls)} from {_objtname(obj)}")

    @classmethod
    def to_bool(cls, a): # bool(a), BUT this should not be implemented by typical
                         # fields. it is preferred that a direct comparison must
                         # be made.
        raise TypeError(f"cannot cast {_tname(cls)} to bool")
    @classmethod
    def to_int(cls, a): # int(a)
        raise TypeError(f"cannot cast {_tname(cls)} to int")
    @classmethod
    def to_float(cls, a): # float(a)
        raise TypeError(f"cannot cast {_tname(cls)} to float")
    @classmethod
    def to_complex(cls, a): # complex(a)
        raise TypeError(f"cannot cast {_tname(cls)} to complex")

    @classmethod
    def consts(cls): # map of str to elements, all optional.
        return {}
        # The names below are all specially recognised, otherwise it may be
        # anything.
        # "__0__": 0, # additive identity.
        # "__1__": 1, # multiplicative identity.
        # "__i__": 1j, # imaginary unit.
        # "__e__": 2.71828, # euler's number.
        # "__pi__": 3.14159, # pi.
        # "__inf__": float("inf"), # positive real infinity.
        # "__nan__": float("nan"), # not-a-number.

    @classmethod
    def _get_const(cls, key, msg): # probs dont override.
        consts = cls.consts()
        if key not in consts:
            raise TypeError(f"{msg} over {_tname(cls)}")
        return consts[key]
    @_classconst
    def zero(cls): # probs dont override.
        return cls._get_const("__0__", "no additive identity (zero)")
    @_classconst
    def one(cls): # probs dont override.
        return cls._get_const("__1__", "no multiplicative identity (one)")
    @_classconst
    def i(cls): # probs dont override.
        return cls._get_const("__i__", "no imaginary unit (i)")
    @_classconst
    def e(cls): # probs dont override.
        return cls._get_const("__e__", "cannot represent euler's number (e)")
    @_classconst
    def pi(cls): # probs dont override.
        return cls._get_const("__pi__", "cannot represent pi")
    @_classconst
    def inf(cls): # probs dont override.
        return cls._get_const("__inf__", "cannot represent infinity")
    @_classconst
    def nan(cls): # probs dont override.
        return cls._get_const("__nan__", "cannot represent not-a-number")


    # Mathematical operations, all of which must return something which this
    # field can still operate on.

    @classmethod
    def real(cls, a): # Re(a)
        raise TypeError(f"no concept of real-vs-imaginary over {_tname(cls)}")
    @classmethod
    def imag(cls, a): # Im(a)
        raise TypeError(f"no concept of real-vs-imaginary over {_tname(cls)}")
    @classmethod
    def abs(cls, a): # |a|
        raise TypeError(f"cannot do absolution over {_tname(cls)}")
    @classmethod
    def angle(cls, a): # arg(a) = atan2(Im(a) / Re(a))  (angle(0) = 0)
        raise TypeError(f"cannot do complex-angle over {_tname(cls)}")
    @classmethod
    def sgn(cls, a): # signum(a) = a / |a|  (signum(0) = 0)
        raise TypeError(f"cannot do signum function over {_tname(cls)}")

    @classmethod
    def add(cls, a, b): # a+b
        raise TypeError(f"cannot do addition over {_tname(cls)}")
    @classmethod
    def sub(cls, a, b): # a-b
        raise TypeError(f"cannot do subtraction over {_tname(cls)}")
    @classmethod
    def mul(cls, a, b): # a*b
        raise TypeError(f"cannot do multiplication over {_tname(cls)}")
    @classmethod
    def div(cls, a, b): # a/b
        raise TypeError(f"cannot do division over {_tname(cls)}")
    @classmethod
    def mod(cls, a, b): # positive result of a%b
        raise TypeError(f"cannot do modulus over {_tname(cls)}")
    @classmethod
    def power(cls, a, b): # a^b
        raise TypeError(f"cannot do powers over {_tname(cls)}")
    @classmethod
    def root(cls, a, b): # a^(1/b)
        raise TypeError(f"cannot do roots over {_tname(cls)}")
    @classmethod
    def log(cls, a, b): # log_a(b)
        raise TypeError(f"cannot do logarithms over {_tname(cls)}")

    @classmethod
    def floor(cls, a): # max(i in Z s.t. i <= a)
        raise TypeError(f"cannot do floor function over {_tname(cls)}")
    @classmethod
    def ceil(cls, a): # min(i in Z s.t. i >= a)
        raise TypeError(f"cannot do ceil function over {_tname(cls)}")
    @classmethod
    def trunc(cls, a): # floor(a) if a >= 0 else ceil(a)
        raise TypeError(f"cannot do truncate function over {_tname(cls)}")
    @classmethod
    def rint(cls, a): # b s.t. b in Z and |b| is maximised and
                      #        |a - b| <= |a - c| for all c in Z
        raise TypeError(f"cannot do round-to-int function over {_tname(cls)}")

    @classmethod
    def sin(cls, a): # sin(a)
        raise TypeError(f"cannot do trigonometric sine over {_tname(cls)}")
    @classmethod
    def cos(cls, a): # cos(a)
        raise TypeError(f"cannot do trigonometric cosine over {_tname(cls)}")
    @classmethod
    def tan(cls, a): # tan(a)
        raise TypeError(f"cannot do trigonometric tangent over {_tname(cls)}")
    @classmethod
    def csc(cls, a): # csc(a) = 1/sin(a)
        raise TypeError(f"cannot do trigonometric cosecant over {_tname(cls)}")
    @classmethod
    def sec(cls, a): # sec(a) = 1/cos(a)
        raise TypeError(f"cannot do trigonometric secant over {_tname(cls)}")
    @classmethod
    def cot(cls, a): # cot(a) = 1/tan(a)
        raise TypeError(f"cannot do trigonometric cotangent over {_tname(cls)}")

    @classmethod
    def asin(cls, a): # arcsin(a)
        raise TypeError("cannot do trigonometric inverse-sine over "
                f"{_tname(cls)}")
    @classmethod
    def acos(cls, a): # arccos(a)
        raise TypeError("cannot do trigonometric inverse-cosine over "
                f"{_tname(cls)}")
    @classmethod
    def atan(cls, a): # arctan(a)
        raise TypeError("cannot do trigonometric inverse-tangent over "
                f"{_tname(cls)}")
    @classmethod
    def acsc(cls, a): # arccsc(a) = arcsin(1/a)
        raise TypeError("cannot do trigonometric inverse-cosecant over "
                f"{_tname(cls)}")
    @classmethod
    def asec(cls, a): # arcsec(a) = arccos(1/a)
        raise TypeError("cannot do trigonometric inverse-secant over "
                f"{_tname(cls)}")
    @classmethod
    def acot(cls, a): # arccot(a) = arctan(1/a)
        raise TypeError("cannot do trigonometric inverse-cotangent over "
                f"{_tname(cls)}")

    @classmethod
    def sinh(cls, a): # sinh(a)
        raise TypeError("cannot do trigonometric hyperbolic-sine over "
                f"{_tname(cls)}")
    @classmethod
    def cosh(cls, a): # cosh(a)
        raise TypeError("cannot do trigonometric hyperbolic-cosine over "
                f"{_tname(cls)}")
    @classmethod
    def tanh(cls, a): # tanh(a)
        raise TypeError("cannot do trigonometric hyperbolic-tangent over "
                f"{_tname(cls)}")
    @classmethod
    def csch(cls, a): # csch(a) = 1/sinh(a)
        raise TypeError("cannot do trigonometric hyperbolic-cosecant over "
                f"{_tname(cls)}")
    @classmethod
    def sech(cls, a): # sech(a) = 1/cosh(a)
        raise TypeError("cannot do trigonometric hyperbolic-secant over "
                f"{_tname(cls)}")
    @classmethod
    def coth(cls, a): # coth(a) = 1/tanh(a)
        raise TypeError("cannot do trigonometric hyperbolic-cotangent over "
                f"{_tname(cls)}")

    @classmethod
    def asinh(cls, a): # arcsinh(a)
        raise TypeError("cannot do trigonometric inverse-hyperbolic-sine over "
                f"{_tname(cls)}")
    @classmethod
    def acosh(cls, a): # arccosh(a)
        raise TypeError("cannot do trigonometric inverse-hyperbolic-cosine over "
                f"{_tname(cls)}")
    @classmethod
    def atanh(cls, a): # arctanh(a)
        raise TypeError("cannot do trigonometric inverse-hyperbolic-tangent "
                f"over {_tname(cls)}")
    @classmethod
    def acsch(cls, a): # arccsch(a) = arcsinh(1/a)
        raise TypeError("cannot do trigonometric inverse-hyperbolic-cosecant "
                f"over {_tname(cls)}")
    @classmethod
    def asech(cls, a): # arcsech(a) = arccosh(1/a)
        raise TypeError("cannot do trigonometric inverse-hyperbolic-secant over "
                f"{_tname(cls)}")
    @classmethod
    def acoth(cls, a): # arccoth(a) = arctanh(1/a)
        raise TypeError("cannot do trigonometric inverse-hyperbolic-cotangent "
                f"over {_tname(cls)}")

    @classmethod
    def atan2(cls, y, x): # quadrant-aware arctan(y/x)
        raise TypeError("cannot do quadrant-aware trigonometric inverse-tangent "
                f"over {_tname(cls)}")

    @classmethod
    def diff(cls, y, x): # (d/dx y)
        raise TypeError(f"cannot do derivatives over {_tname(cls)}")
    @classmethod
    def intt(cls, y, x): # (int y dx)
        raise TypeError(f"cannot do integration over {_tname(cls)}")
    @classmethod
    def def_intt(cls, y, x, a, b): # (int(a..b) y dx)
        raise TypeError(f"cannot do definite integration over {_tname(cls)}")

    # End mathematical operations.


    # Field also needs to facilitate the manipulation of comparison results.
    # Fields which implement the following methods generally should not implement
    # the mathematical operations (as it doesn't make logical sense). Note that
    # this exclusivity is not enforced.

    @classmethod
    def always(cls, a): # is always true? must return bool.
        raise TypeError(f"no concept of always true over {_tname(cls)}")
    @classmethod
    def anytime(cls, a): # is true at any point? must return bool.
        raise TypeError(f"no concept of anytime true over {_tname(cls)}")

    @classmethod
    def logical_not(cls, a): # not a
        raise TypeError(f"cannot do logical-not over {_tname(cls)}")
    @classmethod
    def logical_and(cls, a, b): # a and b
        raise TypeError(f"cannot do logical-and over {_tname(cls)}")
    @classmethod
    def logical_or(cls, a, b): # a or b
        raise TypeError(f"cannot do logical-or over {_tname(cls)}")

    # End comparison result operations.


    # Comparing operations, all of which must return an element operable on by
    # `cls.comparison_field()`. Note that both mathematical and comparison-result
    # fields can/should implement comparison within themselves.

    @classmethod
    def comparison_field(cls): # field to use for the result of comparisons.
        raise TypeError(f"cannot do comparison over {_tname(cls)}")

    @classmethod
    def eq(cls, a, b): # a == b
        raise TypeError(f"cannot do equality over {_tname(cls)}")
    @classmethod
    def ne(cls, a, b): # a != b
        raise TypeError(f"cannot do inequality over {_tname(cls)}")
    @classmethod
    def lt(cls, a, b): # a < b
        raise TypeError(f"cannot do less-than over {_tname(cls)}")
    @classmethod
    def le(cls, a, b): # a <= b
        raise TypeError(f"cannot do less-than-or-equal-to over {_tname(cls)}")

    # End comparing operations.


    # Generally only applicable for floating point:
    @classmethod
    def isinf(cls, a): # does a represent an infinite value? must return bool.
        raise TypeError(f"no concept of infinity over {_tname(cls)}")
    @classmethod
    def isnan(cls, a): # does a represent a not-a-number? must return bool.
        raise TypeError(f"no concept of not-a-number over {_tname(cls)}")


    @classmethod
    def issame(cls, a, b): # a is identical to b? must return bool.
        raise TypeError(f"cannot check identical over {_tname(cls)}")

    @classmethod
    def hashed(cls, a): # hash(a)
        raise TypeError(f"cannot do hashing over {_tname(cls)}")

    @classmethod
    def stringify(cls, a, short): # repr(a), with short+long form
        raise TypeError(f"cannot stringify over {_tname(cls)}")








    # Ok i lied, field does actually implement the methods which operate on
    # entire matrices, HOWEVER, the functionality of these methods should never
    # change based on the field. These exist only to allow for more efficient
    # execution over certain fields, typically when the backend can be
    # implemented with numpy. Some guarantees are:
    # - operations over more than 1 matrix are already cast to same field (and
    #       shape if the operation is element-wise)
    # - operations which require certain shapes are already satisfied (i.e.
    #       vector-only ops, 2d matrix ops, square ops, have already been
    #       checked (though note that vectors may have any ndim)).

    @classmethod
    def _mat_numpyarr_dflt(cls, m, totype):
        tofield = toField(totype)
        if tofield is cls:
            raise RuntimeError("missing overload for "
                    f".numpyarr({totype.__name__})")
        return tofield._mat_fromfield(m).numpyarr(totype)

    # These numpyarr should only be implemented if its faster (or if the field
    # is int/float/complex and we need an array of int/float/complex, but you
    # don't need to worry about that). Note they need to be equivalent to the
    # above dflt impl. If genuinely implementing, you absolutely cannot use the
    # totype .fromfield method, and you must return an array of the form:
    # @classmethod
    # def _mat_numpyarr_int(cls, m):
    #     # CANNOT call any `field.fromfield(m)` methods
    #     return _np.empty(shape=m.shape.tonumpy, dtype=np.int64)
    # @classmethod
    # def _mat_numpyarr_float(cls, m):
    #     # CANNOT call any `field.fromfield(m)` methods
    #     return _np.empty(shape=m.shape.tonumpy, dtype=np.float64)
    # @classmethod
    # def _mat_numpyarr_complex(cls, m):
    #     # CANNOT call any `field.fromfield(m)` methods
    #     return _np.empty(shape=m.shape.tonumpy, dtype=np.complex128)

    @classmethod
    def _mat_fromfield(cls, m):
        # Note this method is allowed to use `_mat_numpyarr_int/float/complex`,
        # but must check for their existance. It is not allowed to use
        # `_mat_numpyarr_dflt` or `m.fromfield` (obviously lmao).
        fromobj = _np.vectorize(cls.fromobj, otypes=[cls.dtype()])
        return Matrix[cls, m.shape](fromobj(m._cells))

    @classmethod
    def _mat_eye(cls, M):
        if M.isempty: # dont call .zero/one if no need.
            return empty(M.field)
        if M.issingle:
            return M.one
        zero = M.field.zero
        one = M.field.one
        npshape = M.shape.tonumpy
        cells = _np.full(npshape, zero, dtype=cls.dtype())
        _np.fill_diagonal(cells, one)
        return M(cells)
    @classmethod
    def _mat_zeros(cls, M):
        if M.isempty: # dont call .zero if no need.
            return empty(M.field)
        zero = M.field.zero
        npshape = M.shape.tonumpy
        cells = _np.full(npshape, zero, dtype=cls.dtype())
        return M(cells)
    @classmethod
    def _mat_ones(cls, M):
        if M.isempty: # dont call .one if no need.
            return empty(M.field)
        one = M.field.one
        npshape = M.shape.tonumpy
        cells = _np.full(npshape, one, dtype=cls.dtype())
        return M(cells)
    @classmethod
    def _mat_full(cls, M, value):
        if M.isempty:
            return empty(M.field)
        val = value.obj
        npshape = M.shape.tonumpy
        cells = _np.full(npshape, val, dtype=cls.dtype())
        return M(cells)

    @classmethod
    def _mat_all(cls, m):
        return all(cls.to_bool(x) for x in m._cells)
    @classmethod
    def _mat_any(cls, m):
        return any(cls.to_bool(x) for x in m._cells)

    @classmethod
    def _mat_always(cls, m):
        return m._apply(cls.always, cls, m)
    @classmethod
    def _mat_anytime(cls, m):
        return m._apply(cls.anytime, cls, m)

    @classmethod
    def _mat_not(cls, m):
        return m._apply(cls.logical_not, cls, m)
    @classmethod
    def _mat_and(cls, m, o):
        return m._apply(cls.logical_and, cls, m, o)
    @classmethod
    def _mat_or(cls, m, o):
        return m._apply(cls.logical_or, cls, m, o)

    @classmethod
    def _mat_real(cls, m):
        if m.isempty:
            return m
        return m._apply(cls.real, cls, m)
    @classmethod
    def _mat_imag(cls, m):
        if m.isempty:
            return m
        return m._apply(cls.imag, cls, m)
    @classmethod
    def _mat_conj(cls, m):
        if m.isempty:
            return m
        return m.real - m.i * m.imag
    @classmethod
    def _mat_abs(cls, m):
        return m._apply(cls.abs, cls, m)
    @classmethod
    def _mat_angle(cls, m):
        return m._apply(cls.angle, cls, m)
    @classmethod
    def _mat_sgn(cls, m):
        return m._apply(cls.sgn, cls, m)

    @classmethod
    def _mat_neg(cls, m):
        if m.isempty: # dont call .zero if no need.
            return m
        sub = cls.sub
        zero = cls.zero
        return m._apply(lambda x: sub(zero, x), cls, m)
    @classmethod
    def _mat_add(cls, m, o):
        return m._apply(cls.add, cls, m, o)
    @classmethod
    def _mat_sub(cls, m, o):
        return m._apply(cls.sub, cls, m, o)
    @classmethod
    def _mat_mul(cls, m, o):
        return m._apply(cls.mul, cls, m, o)
    @classmethod
    def _mat_div(cls, m, o):
        return m._apply(cls.div, cls, m, o)
    @classmethod
    def _mat_mod(cls, m, o):
        return m._apply(cls.mod, cls, m, o)

    @classmethod
    def _mat_exp(cls, m):
        if m.isempty: # dont call .e if no need.
            return m
        power = cls.power
        base = cls.e
        return m._apply(lambda x: power(base, x), cls, m)
    @classmethod
    def _mat_exp2(cls, m):
        if m.isempty: # dont call .fromobj if no need.
            return m
        power = cls.power
        base = cls.fromobj(2)
        return m._apply(lambda x: power(base, x), cls, m)
    @classmethod
    def _mat_exp10(cls, m):
        if m.isempty: # dont call .fromobj if no need.
            return m
        power = cls.power
        base = cls.fromobj(10)
        return m._apply(lambda x: power(base, x), cls, m)
    @classmethod
    def _mat_pow(cls, m, o):
        return m._apply(cls.power, cls, m, o)

    @classmethod
    def _mat_ln(cls, m):
        if m.isempty: # dont call .e if no need.
            return m
        log = cls.log
        base = cls.e
        return m._apply(lambda x: log(base, x), cls, m)
    @classmethod
    def _mat_log2(cls, m):
        if m.isempty: # dont call .fromobj if no need.
            return m
        log = cls.log
        base = cls.fromobj(2)
        return m._apply(lambda x: log(base, x), cls, m)
    @classmethod
    def _mat_log10(cls, m):
        if m.isempty: # dont call .fromobj if no need.
            return m
        log = cls.log
        base = cls.fromobj(10)
        return m._apply(lambda x: log(base, x), cls, m)
    @classmethod
    def _mat_log(cls, m, base):
        return m._apply(cls.log, cls, base, m)

    @classmethod
    def _mat_sqrt(cls, m):
        if m.isempty: # dont call .fromobj if no need.
            return m
        root = cls.root
        base = cls.fromobj(2)
        return m._apply(lambda x: root(x, base), cls, m)
    @classmethod
    def _mat_cbrt(cls, m):
        if m.isempty: # dont call .fromobj if no need.
            return m
        root = cls.root
        base = cls.fromobj(3)
        return m._apply(lambda x: root(x, base), cls, m)
    @classmethod
    def _mat_root(cls, m, n):
        return m._apply(cls.root, cls, m, n)

    @classmethod
    def _mat_floor(cls, m):
        return m._apply(cls.floor, cls, m)
    @classmethod
    def _mat_ceil(cls, m):
        return m._apply(cls.ceil, cls, m)
    @classmethod
    def _mat_trunc(cls, m):
        return m._apply(cls.trunc, cls, m)
    @classmethod
    def _mat_rint(cls, m):
        return m._apply(cls.rint, cls, m)

    @classmethod
    def _mat_sin(cls, m):
        return m._apply(cls.sin, cls, m)
    @classmethod
    def _mat_cos(cls, m):
        return m._apply(cls.cos, cls, m)
    @classmethod
    def _mat_tan(cls, m):
        return m._apply(cls.tan, cls, m)
    @classmethod
    def _mat_csc(cls, m):
        return m._apply(cls.csc, cls, m)
    @classmethod
    def _mat_sec(cls, m):
        return m._apply(cls.sec, cls, m)
    @classmethod
    def _mat_cot(cls, m):
        return m._apply(cls.cot, cls, m)

    @classmethod
    def _mat_asin(cls, m):
        return m._apply(cls.asin, cls, m)
    @classmethod
    def _mat_acos(cls, m):
        return m._apply(cls.acos, cls, m)
    @classmethod
    def _mat_atan(cls, m):
        return m._apply(cls.atan, cls, m)
    @classmethod
    def _mat_acsc(cls, m):
        return m._apply(cls.acsc, cls, m)
    @classmethod
    def _mat_asec(cls, m):
        return m._apply(cls.asec, cls, m)
    @classmethod
    def _mat_acot(cls, m):
        return m._apply(cls.acot, cls, m)

    @classmethod
    def _mat_sinh(cls, m):
        return m._apply(cls.sinh, cls, m)
    @classmethod
    def _mat_cosh(cls, m):
        return m._apply(cls.cosh, cls, m)
    @classmethod
    def _mat_tanh(cls, m):
        return m._apply(cls.tanh, cls, m)
    @classmethod
    def _mat_csch(cls, m):
        return m._apply(cls.csch, cls, m)
    @classmethod
    def _mat_sech(cls, m):
        return m._apply(cls.sech, cls, m)
    @classmethod
    def _mat_coth(cls, m):
        return m._apply(cls.coth, cls, m)

    @classmethod
    def _mat_asinh(cls, m):
        return m._apply(cls.asinh, cls, m)
    @classmethod
    def _mat_acosh(cls, m):
        return m._apply(cls.acosh, cls, m)
    @classmethod
    def _mat_atanh(cls, m):
        return m._apply(cls.atanh, cls, m)
    @classmethod
    def _mat_acsch(cls, m):
        return m._apply(cls.acsch, cls, m)
    @classmethod
    def _mat_asech(cls, m):
        return m._apply(cls.asech, cls, m)
    @classmethod
    def _mat_acoth(cls, m):
        return m._apply(cls.acoth, cls, m)

    @classmethod
    def _mat_atan2(cls, y, x):
        return y._apply(cls.atan2, cls, y, x)

    @classmethod
    def _mat_torad(cls, m):
        if m.isempty: # dont call a bunch if no need.
            return m
        pi = cls.pi
        _180 = cls.fromobj(180)
        div = cls.div
        # maybe preverse largest subproducts.
        _180_pi = div(_180, pi)
        return m._apply(lambda x: div(x, _180_pi), cls, m)
    @classmethod
    def _mat_todeg(cls, m):
        if m.isempty: # dont call a bunch if no need.
            return m
        pi = cls.pi
        _180 = cls.fromobj(180)
        mul = cls.mul
        div = cls.div
        _180_pi = div(_180, pi)
        return m._apply(lambda x: mul(x, _180_pi), cls, m)

    @classmethod
    def _mat_diff(cls, y, x):
        return y._apply(y.field.diff, y.field, y, x)
    @classmethod
    def _mat_intt(cls, y, x):
        return y._apply(y.field.intt, y.field, y, x)
    @classmethod
    def _mat_def_intt(cls, y, x, a, b):
        return y._apply(y.field.def_intt, y.field, y, x, a, b)


    @classmethod
    def _mat_eq(cls, m, o):
        return m._apply(cls.eq, cls.comparison_field(), m, o)
    @classmethod
    def _mat_ne(cls, m, o):
        return m._apply(cls.ne, cls.comparison_field(), m, o)
    @classmethod
    def _mat_lt(cls, m, o):
        return m._apply(cls.lt, cls.comparison_field(), m, o)
    @classmethod
    def _mat_le(cls, m, o):
        return m._apply(cls.le, cls.comparison_field(), m, o)


    @classmethod
    def _mat_isinf(cls, m):
        return m._apply(cls.isinf, bool, m)
    @classmethod
    def _mat_isnan(cls, m):
        return m._apply(cls.isnan, bool, m)
    @classmethod
    def _mat_isfinite(cls, m):
        return ~cls._mat_isinf(m) & ~cls._mat_isnan(m)


    @classmethod
    def _mat_det(cls, m):
        if m.isempty:
            return m.one
        if m.issingle:
            return m # det(x) = x.

        mul = cls.mul
        add = cls.add
        sub = cls.sub

        # hardcode 2x2.
        if m.shape[0] == 2:
            a, b, c, d = cells.reshape(-1)
            return sub(mul(a, d), mul(b, c))

        def submatrix(cells, size, row):
            return cells[_np.arange(size) != row, 1:]

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

        return single(determinant(cells, m.shape[0]), field=cls, _nocast=True)

    @classmethod
    def _mat_trace(cls, m):
        return m.diag.summ

    @classmethod
    def _mat_norm(cls, m):
        if m.isempty:
            return m.zero
        absm = m.abs
        return (absm * absm).summ.sqrt

    @classmethod
    def _mat_dot(cls, m, o):
        if m.isempty:
            return m.zero
        # Reshape to column vector.
        m = m.reshape(m.size)
        o = o.reshape(o.size)
        # Conjugate left, then elementwise product, then sum.
        return (m.conj * o).summ

    @classmethod
    def _mat_cross(cls, m, o):
        # Use crossproduct formula.
        ax, ay, az = m.reshape(m.size)
        bx, by, bz = o.reshape(o.size)
        mul = cls.mul
        sub = cls.sub
        cells = _np.array([
                sub(mul(ay, bz), mul(az, by)),
                sub(mul(az, bx), mul(ax, bz)),
                sub(mul(ax, by), mul(ay, bx)),
            ], dtype=cls.dtype())
        # If both vectors are the same shape, keep it. Otherwise use colvec.
        newshape = m.shape if (m.shape == o.shape) else Shape(3)
        cells = cells.reshape(newshape.tonumpy)
        return Matrix[cls, newshape](cells)

    @classmethod
    def _mat_matmul(cls, m, o):
        newshape = Shape(m.shape[0], o.shape[1])
        mul = cls.mul
        add = cls.add
        cells = Matrix[cls, newshape].zeros._cells.copy()
        # blazingly fast new matrix multiplication algorithm scientists are
        # dubbing the "naive method" (i think it means really smart).
        for i in range(m.shape[0]):
            for j in range(o.shape[1]):
                for k in range(m.shape[1]):
                    prod = mul(m._cells[i, k], o._cells[k, j])
                    cells[i, j] = add(cells[i, j], prod)
        return Matrix[cls, newshape](cells)

    @classmethod
    def _mat_summ(cls, m, axis):
        if m.isempty:
            return m.zero if axis is ... else m
        return m._fold(cls.add, axis)
    @classmethod
    def _mat_prod(cls, m, axis):
        if m.isempty:
            return m.one if axis is ... else m
        return m._fold(cls.mul, axis)
    @classmethod
    def _mat_minn(cls, m, axis):
        lt = cls.lt
        return m._fold(lambda a, b: b if lt(b, a) else a, axis)
    @classmethod
    def _mat_maxx(cls, m, axis):
        lt = cls.lt
        return m._fold(lambda a, b: b if lt(a, b) else a, axis)

    @classmethod
    def _mat_mean(cls, m, axis):
        n = m.size if axis is ... else m.shape[axis]
        return m.summ_along(axis) / n
    @classmethod
    def _mat_geomean(cls, m, axis):
        n = m.size if axis is ... else m.shape[axis]
        return m.prod_along(axis).root(n)
    @classmethod
    def _mat_harmean(cls, m, axis):
        n = m.size if axis is ... else m.shape[axis]
        return n / (m.one / m).summ_along(axis)
    @classmethod
    def _mat_quadmean(cls, m, axis):
        n = m.size if axis is ... else m.shape[axis]
        return ((m * m).summ_along(axis) / n).sqrt

    @classmethod
    def _mat_rref(cls, m, imprecise=False):
        if m.isempty:
            return m

        zero = cls.zero
        one = cls.one
        add = cls.add
        sub = cls.sub
        mul = cls.mul
        div = cls.div
        eq = cls.eq
        ne = cls.ne
        eqz = lambda x: eq(zero, x)
        eqo = lambda x: eq(one, x)
        nez = lambda x: ne(zero, x)
        neo = lambda x: ne(one, x)
        neg = lambda a: sub(zero, a)
        rec = lambda a: div(one, a)

        cells = m._cells.copy()
        rows, cols = m.shape[0], m.shape[1]

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
            if nez(pivot_value):
                row_mul(r, rec(pivot_value))
            # Check its 1.
            pivot_value = cells[r, lead]
            if neo(pivot_value):
                raise ValueError(f"couldn't make cell =1, cell is: "
                        f"{repr(pivot_value)}")

            for i in range(rows):
                if i == r:
                    continue
                row_lead_value = cells[i, lead]
                if nez(row_lead_value):
                    row_add(i, r, neg(row_lead_value))
                # Check its 0.
                row_lead_value = cells[i, lead]
                if nez(row_lead_value):
                    raise ValueError("couldn't make cell =0, cell is: "
                            f"{repr(row_lead_value)}")

            lead += 1

        # If imprecise, cheeky collapse to exactly zero or one.
        if imprecise:
            @_np.vectorize(otypes=[cls.dtype()])
            def collapse(x):
                if eqz(x):
                    return zero
                if eqo(x):
                    return one
                return x
            cells = collapse(cells)

        return Matrix[cls, m.shape](cells)


    @classmethod
    def _mat_issame(cls, m, o):
        return m._apply(cls.issame, bool, m, o)

    @classmethod
    def _mat_hash(cls, m):
        return hash((m.shape, ) + tuple(cls.hashed(x) for x in m._cells))

    @classmethod
    def _mat_rep(cls, m, short, multiline):
        if m.isempty:
            return "my boy "*(not short) + "M.T."

        stringify = lambda x: cls.stringify(x, short)

        if m.issingle:
            return stringify(m.obj)

        if short and not m.isvec:
            # Shorten elements of zero to a single dot.
            def repme(x):
                try:
                    if cls.issame(cls.zero, x):
                        return "."
                except:
                    pass
                return stringify(x)
        else:
            repme = stringify

        # cheeky matrix of the reps to make access easy.
        reps = m._apply(repme, str, m)
        reps_nonctrl = reps._apply(_cons.nonctrl, str, reps)
        lens = reps_nonctrl._apply(len, int, reps_nonctrl)
        width = int(lens.maxx)
        return Field._mat_rep_helper(reps, short, width, m.lastaxis, multiline,
                allow_flat=True)

    @staticmethod
    def _mat_rep_helper(m, short, width, axis, multiline, allow_flat=False):
        # this method is a helper, and is only defined for matrices over str.
        newline = "\n" if multiline else ""
        if axis > 1:
            def tostr(x):
                x = Field._mat_rep_helper(x, short, width, axis - 1, multiline)
                x = x.split("\n")
                m = x[0] + "".join(f"\n  {line}" for line in x[1:])
                return "[ " + m + " ]"
            return newline.join(tostr(x) for x in m.along(axis))
        # 2d print.
        # Print col vecs as rows with marked transpose, if allowed.
        suffix = ""
        cells = m._cells
        if allow_flat and m.iscol:
            suffix = _cons.coloured(40, "ᵀ")
            cells = cells.T
        def rowstr(row):
            return [" " * (width - len(_cons.nonctrl(r))) + r for r in row]
        rows = [rowstr(row) for row in cells]
        padded = (not short) or (width > 3)
        join = lambda x: "  ".join(x) if padded else " ".join(x)
        wrap = lambda x: f"[ {x} ]" if padded else f"[{x}]"
        str_rows = (wrap(join(row)) for row in rows)
        return newline.join(str_rows) + suffix


class RealField(Field):
    @classmethod
    def real(cls, a):
        return a
    @classmethod
    def imag(cls, a):
        return cls.zero

    @classmethod
    def _mat_real(cls, m):
        return m
    @classmethod
    def _mat_imag(cls, m):
        return m.zeros
    @classmethod
    def _mat_conj(cls, m):
        return m



@_templated(parents=Field)
def _NonField(_base):
    # Wraps a non-field class in the field api, implementing some methods with
    # defaults. Shouldn't be used directly, use `toField`.

    if not isinstance(_base, type):
        raise TypeError(f"expected a type, got {_objtname(_base)}")
    if issubclass(_base, Field):
        raise TypeError(f"expected a non-field type, got {_tname(_base)}")
    # Dont make a field where the elements are themselves matrices.
    if issubclass(_base, Matrix):
        raise TypeError("mate a matrix of matrices? calm down")

    # Without inheriting from Field, we don't recognise many operations, and
    # primarily only exist to allow matrix iteration/access (which may still be
    # desirable).

    @classmethod
    def to_int(cls, a):
        if hasattr(a, "__int__"):
            return a.__int__()
        return super(_NonField[_base], cls).to_int(a)
    @classmethod
    def to_float(cls, a):
        if hasattr(a, "__float__"):
            return a.__float__()
        if hasattr(a, "__int__"):
            return float(a.__int__())
        return super(_NonField[_base], cls).to_float(a)
    @classmethod
    def to_complex(cls, a):
        if hasattr(a, "__complex__"):
            return a.__complex__()
        if hasattr(a, "__float__"):
            return complex(a.__float__())
        if hasattr(a, "__int__"):
            return complex(a.__int__())
        return super(_NonField[_base], cls).to_complex(a)

    @classmethod
    def hashed(cls, a):
        if hasattr(a, "__hash__"):
            return a.__hash__()
        return super(_NonField[_base], cls).hashed(a)
    @classmethod
    def stringify(cls, a, short):
        if hasattr(a, "__repr__"):
            return a.__repr__()
        return super(_NonField[_base], cls).stringify(a, short)

    return locals()

_NonField.tnamer = lambda F: f"<non-field: {_tname(F, quoted=False)}>"


@_singleton
class toField:
    """
    Returns a field-class-equivalent for the given kinda-field. If 'field' is
    actually a subclass of 'Field', returns it without change, otherwise
    returns an instantiation of '_NonField' which implements the api with
    some default behaviour.
    """

    def __init__(self):
        self._mapping = {}

    def __call__(self, field):
        if not isinstance(field, type):
            raise TypeError("expected a type for 'field', got "
                    f"{_objtname(field)}")
        if issubclass(field, Field):
            return field
        if field not in self._mapping:
            self._mapping[field] = _NonField[field]
        return self._mapping[field]

    def map_nonfield_to(self, fromtype, field):
        """
        When given `fromtype` in the future, instead of returning
        `_NonField[field]`, it will return `field`.
        """
        if not isinstance(fromtype, type):
            raise TypeError("expected a type for 'fromtype', got "
                    f"{_objtname(fromtype)}")
        if issubclass(fromtype, Field):
            raise ValueError("expected a non-field type for 'fromtype', got "
                    f"{_tname(fromtype)}")
        if not isinstance(field, type):
            raise TypeError("expected a type for 'field', got "
                    f"{_objtname(field)}")
        if not issubclass(field, Field):
            raise ValueError("expected a field type for 'field', got "
                    f"{_tname(field)}")
        if fromtype in self._mapping:
            raise RuntimeError("expected an unmapped type, got "
                    f"{_tname(fromtype)}")
        self._mapping[fromtype] = field



@_singleton
class lits:
    def __init__(self):
        self._field = None
        self._injects = {
            "e": (lambda field: Single[field].e),
            "pi": (lambda field: Single[field].pi),
            "inf": (lambda field: Single[field].inf),
            "nan": (lambda field: Single[field].nan),
        }

    def _is_overridden(self, space, name, field=...):
        assert name in self._injects
        if field is ...:
            field = self._field
        if name not in space:
            return False
        if field is None:
            return True
        got = space[name]
        try:
            expect = self._injects[name](field)
        except Exception:
            return True # assume nothing but the worst.
        if type(got) is not type(expect):
            return True
        try:
            if isinstance(expect, Matrix):
                return not expect.issame(got).alll
            else:
                return (expect != got).anyy
        except Exception:
            return True # assume nothing but the worst.

    def __call__(self, field, inject=True, *, space=None):
        """
        Sets the current/default field to the given field and injects constants
        such as 'e' and 'pi' into the globals.
        """
        if field is not None:
            field = toField(field)
        prev_field = self._field
        self._field = field
        if not inject:
            return
        # Inject constants into the space.
        if space is None:
            space = _get_space()
        for name, getter in self._injects.items():
            # Dont wipe vars the user has set.
            if self._is_overridden(space, name, field=prev_field):
                continue
            didset = False
            if field is not None:
                try:
                    space[name] = getter(field)
                    didset = True
                except Exception:
                    pass
            if not didset:
                space.pop(name, None)



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


"""
Cheeky rant about how matrix internals will interoperate with numpy, since we
have both different shaping logic and different memory layout expectations.

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
                    raise TypeError("expected integer dimension lengths, got "
                            f"{_objtname(l)}")
            if any(l < 0 for l in order):
                raise ValueError(f"dimension lengths cannot be negative, got: "
                        f"{order}")
            # Check ordering is logical.
            if len(set(order)) != len(order):
                raise ValueError(f"cannot repeat axes, got: {order}")
            if order and max(order) != len(order) - 1:
                assert max(order) > len(order) - 1
                raise ValueError(f"missing axis for swap, axis {max(order)} is "
                        "dangling")
        # Trim any trailing no-ops.
        while order and order[-1] == len(order) - 1:
            order = order[:-1]
        p._order = order
    @classmethod
    def tonumpy(P, ndim):
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
        return P(tuple(range(ndim - 1, 1, -1)) + (0, 1))
    @classmethod
    def fromnumpy(P, ndim):
        """
        From-numpy permutation, alias for 'P.tonumpy(ndim).inv'.
        """
        return P.tonumpy(ndim).inv

    @classmethod
    def tonumpyaxis(P, ndim, axis):
        """
        Numpy-equivalent axis to the given axis, for a shape of the given 'ndim'.
        """
        order = P.fromnumpy(ndim).order(ndim)
        if not isinstance(axis, int):
            raise TypeError(f"expected an integer axis, got {_objtname(axis)}")
        if axis < 0:
            raise ValueError(f"cannot have a negative axis, got: {axis}")
        if axis >= len(order):
            raise ValueError(f"axis {axis} is out of bounds for {len(order)} "
                    "dimensions")
        # ngl, i have no fucking idea why we have to use the fromnumpy permuter
        # here but it doesn't work otherwise. permutation stored in an order
        # tuple like this r strange.
        return order[axis]

    @classmethod
    def numpyT(P, ndim):
        """
        Numpy permutation tuple for a 2d transpose.
        """
        if not isinstance(ndim, int):
            raise TypeError("expected an integer number of dimensions, got "
                    f"{_objtname(ndim)}")
        if ndim < 0:
            raise ValueError("number of dimensions cannot be negative, got: "
                    f"{ndim}")
        if ndim < 2:
            raise ValueError("shouldn't have a numpy array with fewer than 2 "
                    "axes")
        return tuple(range(ndim - 2)) + (-1, -2)

    def order(p, ndim):
        """
        Tuple of permutation for the given number of dimensions.
        """
        if not isinstance(ndim, int):
            raise TypeError("expected an integer number of dimensions, got "
                    f"{_objtname(ndim)}")
        if ndim < 0:
            raise ValueError("number of dimensions cannot be negative, got: "
                    f"{ndim}")
        if ndim < p.ndim:
            raise ValueError("expected a number of dimensions >= this permuter's"
                    f"number of dimensions, got {ndim} while the permuter has "
                    f"{p.ndim}")
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
                "of dimensions use 'p.ndim')")

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
            ndim = max(p.ndim, seq.ndim)
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

    def __init__(s, *lens, size=None):
        """
        Creates a shape with the given axis lengths. If 'size' is given, implies
        a reshaping and allows '...' to occur once in the lengths, which causes
        that axis length to be automatically sized.
        """
        if size is not None:
            if not isinstance(size, int):
                raise TypeError("expected an integer size, got "
                        f"{_objtname(size)}")
            if size < 0:
                raise ValueError(f"size cannot be negative, got: {size}")
        if len(lens) == 1 and isinstance(lens[0], Shape):
            lens = lens[0]._lens
        else:
            lens = _maybe_unpack_ints(lens)
            ilens = [l for l in lens if l is not ...]
            for l in ilens:
                if not isinstance(l, int):
                    extra = "or '...' " * (size is not None)
                    raise TypeError(f"expected integer{extra} dimension "
                            f"lengths, got {_objtname(l)}")
            if any(l < 0 for l in ilens):
                raise ValueError("dimension lengths cannot be negative, got: "
                        f"{lens}")
            dotc = lens.count(...)
            if dotc > 0:
                if size is None:
                    raise ValueError("cannot use '...' when not resizing")
                if dotc > 1:
                    raise ValueError("cannot use '...' more than once")
                # Figure out ...
                perp = _math.prod(ilens)
                if size == 0:
                    dynamic = 0
                elif perp == 0:
                    dynamic = 0 # catch error later.
                elif (size % perp) > 0:
                    raise ValueError("unable to determine '...' length, "
                            f"stacking perpendicular sizes of {perp} will never "
                            f"fit {size}")
                else:
                    dynamic = size // perp
                lens = tuple(dynamic if l is ... else l for l in lens)
            # Collapse empty.
            if _math.prod(lens) == 0:
                lens = (0, )
            # Trim any trailing 1s (note this collapses single to ()).
            while lens and lens[-1] == 1:
                lens = lens[:-1]
        if size is not None and _math.prod(lens) != size:
            raise ValueError("cannot change size (no. elements) when reshaping, "
                    f"expected {size} size, got {_math.prod(lens)}")
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
    def fromnumpy(S, npshape):
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
                raise TypeError("expected integer numpy shape elements, got "
                        f"{_objtname(l)}")
        if any(l < 0 for l in npshape):
            raise ValueError("numpy shape elements cannot be negative, got: "
                    f"{npshape}")
        if npshape == (0, ):
            return S.empty
        if len(npshape) == 1:
            npshape += (1, ) # colvec.
        shape = Permuter.fromnumpy(len(npshape))(npshape)
        return S(shape)


    @_classconst
    def empty(S):
        """
        0x0 shape.
        """
        return S(0)
    @_classconst
    def single(S):
        """
        1x1 shape.
        """
        return S()

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
        return s._lens == (0, )
    @_instconst
    def issingle(s):
        """
        Is only one cell? (1x1)
        """
        return s._lens == ()
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
            return 0 if s.isempty else 1
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

    @_instconst
    def T(s):
        """
        Tranposed shape.
        """
        return s.withaxis(0, s[1]).withaxis(1, s[0])

    def check_broadcastable(s, o):
        if not isinstance(o, Shape):
            raise TypeError(f"expected a shape, got {_objtname(o)}")
        if s.isempty and o.isempty:
            return
        if s.isempty:
            raise ValueError("cannot broadcast from empty to non-empty, when "
                    f"attempting to broadcast {s} to {o}")
        if o.isempty:
            raise ValueError("cannot broadcast to empty from non-empty, when "
                    f"attempting to broadcast {s} to {o}")
        for axis in range(max(s.ndim, o.ndim)):
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




@_singleton
class NO_SEED: # :(
    def __repr__(self):
        return "NO_SEED"


# Matrix.
@_templated(decorators=_immutable)
def Matrix(field, shape):
    """
    Fixed-sized n-dimensional sequence of elements.
    """

    if not isinstance(field, type):
        raise TypeError(f"expected a type for field, got {_objtname(field)}")
    if not issubclass(field, Field):
        raise TypeError("expected a Field subclass for field, got "
                f"{_tname(field)} (which isn't)")
    if not isinstance(shape, Shape):
        raise TypeError(f"expected a Shape for shape, got {_objtname(shape)}")


    def __init__(m, cells=None):
        # All cells are stored in a numpy array. Note this array is data-
        # consistent, meaning numpy views it in the same way we do, but not
        # memory layout consistent, meaning we expect a different ravelled array
        # to what numpy actually uses. Theres a whole rant about this ealier in
        # the file.
        if not isinstance(cells, _np.ndarray):
            raise TypeError(f"expected numpy array, got {_objtname(cells)}")
        npshape = m.shape.tonumpy
        dtype = m.field.dtype()
        if cells.shape != npshape:
            raise ValueError(f"expected {npshape} shape, got {cells.shape}")
        if cells.dtype is not dtype:
            raise ValueError(f"expected {dtype} dtype, got {cells.dtype}")
        # Make it read-only.
        cells.flags["WRITEABLE"] = False
        m._cells = cells


    @classmethod
    def fromnumpy(M, arr):
        """
        Matrix over this field but of any shape from the given numpy array, alias
        for 'Matrix.fromnumpy(M.field, arr)'.
        """
        return Matrix.fromnumpy(M.field, arr)


    @classmethod
    def cast(M, *ms, broadcast=True):
        """
        Attempts to cast each object to a matrix over 'field', all with the same
        size. Note that the shape of this class is ignored. If 'broadcast' is
        false, shapes will be left unchanged.
        """
        ms = _maybe_unpack_seq(ms)
        if not ms:
            return ()

        # Cast to the correct field.
        ms = tuple(m if isinstance(m, Matrix) else single(m) for m in ms)
        ms = tuple(m.tofield(M.field) for m in ms)

        # If no broadcasting, don't do broadcasting, so the final result won't be
        # broadcast. Also if only one nothing to broadcast against.
        if not broadcast or len(ms) == 1:
            return ms

        # Handle empties.
        if all(m.isempty for m in ms):
            return ms
        if any(m.isempty for m in ms):
            raise TypeError("cannot operate with a mix of empty and non-empty "
                    "matrices")

        # Broadcast all to the same (largest) shape.
        ndim = max(m.ndim for m in ms)
        newshape = Shape(max(m.shape[axis] for m in ms) for axis in range(ndim))
        return tuple(m.broadcast(newshape) for m in ms)


    @_classconst
    def zero(M):
        """
        Single zero.
        """
        return single(M.field.zero, field=M.field, _nocast=True)
    @_classconst
    def one(M):
        """
        Single one.
        """
        return single(M.field.one, field=M.field, _nocast=True)
    @_classconst
    def i(M):
        """
        Single imaginary unit.
        """
        return single(M.field.i, field=M.field, _nocast=True)
    @_classconst
    def e(M):
        """
        Single euler's number (2.71828...).
        """
        return single(M.field.e, field=M.field, _nocast=True)
    @_classconst
    def pi(M):
        """
        Single pi (3.14159...).
        """
        return single(M.field.pi, field=M.field, _nocast=True)
    @_classconst
    def inf(M):
        """
        Single infinity.
        """
        return single(M.field.inf, field=M.field, _nocast=True)
    @_classconst
    def nan(M):
        """
        Single not-a-number.
        """
        return single(M.field.nan, field=M.field, _nocast=True)


    @_classconst
    def eye(M):
        """
        Identity matrix.
        """
        if not M.issquare:
            raise TypeError("only square matricies have an identity matrix, got "
                    f"{M.shape}")
        return M.field._mat_eye(M)
    @_classconst
    def zeros(M):
        """
        Zero-filled matrix.
        """
        return M.field._mat_zeros(M)
    @_classconst
    def ones(M):
        """
        One-filled matrix.
        """
        return M.field._mat_ones(M)
    @classmethod
    def full(M, value):
        """
        Matrix filled with the given value.
        """
        value, = M.cast(value)
        if not value.issingle:
            raise TypeError(f"expected single for fill value, got {value.shape}")
        return M.field._mat_full(M, value)

    @_classconst
    def size(M):
        """
        Number of cells.
        """
        return M.shape.size
    @_classconst
    def ndim(M):
        """
        Number of dimensions. Note empty has -1 dimensions and single has 0.
        """
        return M.shape.ndim
    @_classconst
    def lastaxis(M):
        """
        Index of the last axis with non-1 length. Returns 0 for empty and single.
        """
        return M.shape.lastaxis

    @_classconst
    def isempty(M):
        """
        Is empty? (0x0)
        """
        return M.shape.isempty
    @_classconst
    def issingle(M):
        """
        Is only one cell? (1x1)
        """
        return M.shape.issingle
    @_classconst
    def isvec(M):
        """
        Has at-most one axis with length >1? (empty and single count as vectors)
        """
        return M.shape.isvec
    @_classconst
    def iscol(M):
        """
        Is column vector? (empty and single count as column vectors)
        """
        return M.shape.iscol
    @_classconst
    def isrow(M):
        """
        Is row vector? (empty and single count as row vectors)
        """
        return M.shape.isrow
    @_classconst
    def issquare(M):
        """
        Is square matrix? (only 2D matrices can be square)
        """
        return M.shape.issquare


    def numpyarr(m, totype=None):
        """
        Read-only numpy nd-array of the matrix, of the shape 'm.shape.tonumpy'.
        'totype' dictates the dtype of the returned array:
            none uses the natural dtype of the field (typically 'object' unless
            the field is numeric);
            'int' casts to integer and returns dtype 'np.int64' (note this will
            fail on overflow);
            'float' casts to floating and returns dtype 'np.float64';
            'complex' casts to complex and returns dtype 'np.complex128'.

        WARNING: numpy uses different shape logic to matrices, so the returned
                 array is always at-least 2D (unless empty). To get a vector
                 (provided 'm' is a vector), use 'm.numpyvec(totype)'.
        """
        if totype is None:
            return m._cells
        if not isinstance(totype, type):
            raise TypeError("exepcted type for 'totype', got "
                    f"{_objtname(totype)}")
        if totype is not int and totype is not float and totype is not complex:
            raise ValueError("expected one of 'int', 'float', or 'complex' for "
                    f"'totype', got {_tname(totype)}")
        name = {int: "int", float: "float", complex: "complex"}[totype]
        dflt = m.field._mat_numpyarr_dflt
        func = getattr(m.field, f"_mat_numpyarr_{name}", dflt)
        return func(m)

    def numpyvec(m, totype=None):
        """
        Read-only numpy 1d-array of the vector. 'totype' dictates the dtype of
        the returned array, see 'm.numpyarr' for more info.
        """
        if not m.isvec:
            raise TypeError("expected vector to convert to numpy vector, got "
                    f"{m.shape}")
        return m.numpyarr(totype).reshape(-1)


    @_instconst
    def obj(m):
        """
        Cast a single to the object it contains.
        """
        if not m.issingle:
            raise TypeError("expected single for scalar cast to object, got "
                    f"{m.shape}")
        return m._cells[0, 0]

    def __bool__(m):
        """
        Cast a single to bool.
        """
        if not m.issingle:
            raise TypeError("expected single for scalar cast to bool, got "
                    f"{m.shape}")
        return m.field.to_bool(m.obj)
    def __int__(m):
        """
        Cast a single to int.
        """
        if not m.issingle:
            raise TypeError("expected single for scalar cast to int, got "
                    f"{m.shape}")
        return m.field.to_int(m.obj)
    def __float__(m):
        """
        Cast a single to float.
        """
        if not m.issingle:
            raise TypeError("expected single for scalar cast to float, got "
                    f"{m.shape}")
        return m.field.to_float(m.obj)
    def __complex__(m):
        """
        Cast a single to complex.
        """
        if not m.issingle:
            raise TypeError("expected single for scalar cast to complex, got "
                    f"{m.shape}")
        return m.field.to_complex(m.obj)


    def tofield(m, newfield):
        """
        Casts all cells to the given field.
        """
        newfield = toField(newfield)
        if m.field is newfield:
            return m
        return newfield._mat_fromfield(m)


    def reshape(m, *newshape):
        """
        Views the ravelled cells under the given shape. The size of the new shape
        must be equal to the size of the current shape (aka cannot change the
        number of elements). If '...' is given as an axis length, that axis is
        automatically sized (if possible) to accomodate all elements (note it may
        size to 0 if resizing empty).
        """
        if not newshape:
            # technically a "reshape to single" but thats pretty useless so in
            # the spirit of clearer code (since seeing x.reshape() kinda looks
            # like it sohuld just not change shape) we dont allow no args.
            raise TypeError("must specify a new shape")
        newshape = Shape(*newshape, size=m.size)
        # See the rant earlier in this file about numpy interoperation, short
        # answer is the backing array memory layout isnt what we expect so we got
        # work to do.
        if m.isvec and newshape.isvec:
            # If vector, it doesn't matter.
            cells = m._cells.reshape(newshape.tonumpy)
        elif m.ndim <= 2 and newshape.ndim <= 2:
            # If 2d, can use f-style ordering to get our memory layout.
            cells = m._cells.reshape(newshape.tonumpy, order="F")
        else:
            # For higher dimensions, easiest way to get the ravelled layout we
            # expect is to do 2d matrix transpose then read off c-style. Then we
            # gotta reshape into transposed and transpose back to our view. bob
            # the builder type shi.
            cells = m._cells
            if not m.isvec: # No need to transpose if vector.
                cells = cells.transpose(Permuter.numpyT(cells.ndim))
            if newshape.isvec: # No need to do extra transposes if vector output.
                cells = cells.reshape(newshape.tonumpy, order="C")
            else:
                cells = cells.reshape(newshape.T.tonumpy, order="C")
                cells = cells.transpose(Permuter.numpyT(cells.ndim))
        return Matrix[m.field, newshape](cells)


    @_instconst
    def ravel(m):
        """
        Vector of cells in natural iteration order (sequential axes, which is
        row-major), aka the flattened cells.
        """
        return m.reshape(Shape(m.size))


    def broadcast(m, *newshape):
        """
        Broadcasts the matrix to a new shape. Broadcasting allows axes which
        previously had a length of 1 to be "broadcast" to the (larger) requested
        length by repeating along that axis. Otherwise, the new axis length must
        be the same.
        """
        if not newshape: # give empty tuple for no-op
            raise TypeError("must specify a new shape")
        newshape = Shape(*newshape)
        m.shape.check_broadcastable(newshape)
        cells = _np.broadcast_to(m._cells, newshape.tonumpy)
        return Matrix[m.field, newshape](cells)


    def permute(m, *neworder):
        """
        Permutes the axes into the given order (like a transpose).
        """
        if not neworder: # give empty tuple for no-op
            raise TypeError("must specify a new order")
        permuter = Permuter(*neworder)
        # Empty and single are invariant under permutation.
        if m.isempty or m.issingle:
            return m
        # No-op permutation.
        if permuter.ndim == 0:
            return m
        newshape = permuter(m.shape)
        # Vector handled separately cause its easy (no data reordering).
        if m.isvec:
            cells = m._cells.reshape(newshape.tonumpy)
        # Handle 2d tranpose separate bc speed.
        elif m.ndim <= 2 and newshape.ndim <= 2:
            # only option here is 2d transpose.
            cells = m._cells.T
        else:
            cells = m._cells
            # Gotta make the missus happy. Numpy will never implicitly add
            # dimensions, to lets do that first.
            new_ndim = max(2, newshape.ndim)
            old_ndim = max(2, m.ndim)
            ndim = max(new_ndim, old_ndim)
            npshape = (1, ) * (new_ndim - old_ndim) + m.shape.tonumpy
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
        return Matrix[m.field, newshape](cells)


    def drop(m, *axes):
        """
        Removes the given axes from the shape, shifting other axes down. The
        lengths of these axes must be 0 or 1 (aka they must not have contributed
        to the size). If axes are repeated, additional occurences are ignored.
        """
        if not axes: # give empty tuple for no-op
            raise TypeError("must specify some axes")
        axes = _maybe_unpack_ints(axes)
        for axis in axes:
            if not isinstance(axis, int):
                raise TypeError(f"expected integer axes, got {_objtname(axis)}")
        if any(a < 0 for a in axes):
            raise ValueError(f"axes cannot be negative, got: {axes}")
        axes = set(axes)
        shape = m.shape
        # Reverse iteration to not change the indices of future axes as we remove
        # earlier ones.
        for axis in sorted(axes, reverse=True):
            shape = shape.dropaxis(axis)
        return m.reshape(shape)
    @_instconst
    def dropall(m):
        """
        Removes all axes with lengths 0 or 1, shifting other axes down.
        """
        axes = tuple(axis for axis in range(m.ndim) if m.shape[axis] <= 1)
        return m.drop(axes)


    def flip(m, *axes):
        """
        Reverses the order along the given axes. If any axes are repeated, it is
        treated as another flip (so even counts of axes do nothing).
        """
        if not axes: # give empty tuple for no-op
            raise TypeError("must specify some axes")
        axes = _maybe_unpack_ints(axes)
        for axis in axes:
            if not isinstance(axis, int):
                raise TypeError(f"expected integer axes, got {_objtname(axis)}")
        if any(a < 0 for a in axes):
            raise ValueError(f"axes cannot be negative, got: {axes}")
        if m.isempty:
            return m
        axes = {a for a in axes if (axes.count(a) % 2) == 1 and a < m.ndim}
        npaxes = tuple(Permuter.tonumpyaxis(m.ndim, a) for a in axes)
        cells = _np.flip(m._cells, npaxes)
        return Matrix[m.field, m.shape](cells)
    @_instconst
    def flipall(m):
        """
        Reverses the order along all axes.
        """
        axes = tuple(range(m.ndim))
        return m.flip(axes)
    @_instconst
    def flipud(m):
        """
        Flip columns, alias for 'm.flip(0)'.
        """
        return m.flip(0)
    @_instconst
    def fliplr(m):
        """
        Flip rows, alias for 'm.flip(1)'.
        """
        return m.flip(1)


    def tile(m, *counts, axis=None):
        """
        Tiles this matrix 'counts[axis]' times along each axis. If one bare
        integer is given for 'counts', 'axis' must be specified to tile along
        only that axis (to prevent this, wrap it in a tuple or give another 1).
        If tiling a vector, one bare integer may be given without specifying
        'axis' and it will tile along the vector's axis.
        """
        if len(counts) == 0:
            raise TypeError("must specify some counts")
        if len(counts) == 1 and isinstance(counts[0], int):
            if axis is None:
                if not m.isvec:
                    raise TypeError("must specify axis when tiling a non-vector")
                axis = m.lastaxis
            if not isinstance(axis, int):
                raise TypeError("expected an integer axis, got "
                        f"{_objtname(axis)}")
            if axis < 0:
                raise ValueError(f"axis cannot be negative, got: {axis}")
            counts = (1, ) * axis + (counts[0], )
            axis = None
        else:
            counts = _maybe_unpack_ints(counts)
        for count in counts:
            if not isinstance(count, int):
                raise TypeError("expected an integer count, got "
                        f"{_objtname(count)}")
        if any(count < 0 for count in counts):
            raise ValueError(f"counts cannot be negative, got: {counts}")
        if axis is not None:
            raise TypeError("cannot specify axis with multiple counts")
        if m.isempty:
            return m
        # Use shape as helper.
        counts = Shape(counts)
        if counts.isempty:
            return empty(m.field)
        if counts.issingle:
            return m
        # Reorder counts to the numpy ordering.
        ndim = max(2, m.ndim, counts.ndim)
        npcounts = Permuter.tonumpy(ndim)(counts)
        npcounttuple = tuple(npcounts[axis] for axis in range(ndim))
        cells = _np.tile(m._cells, npcounttuple)
        return m.fromnumpy(cells)


    def repeat(m, *counts, axis=None):
        """
        Repeats each element in this matrix 'counts[axis]' times along each axis.
        If one bare integer is given for 'counts', 'axis' must be specified to
        repeat along only that axis (to prevent this, wrap it in a tuple or give
        another 1). If repeating a vector, one bare integer may be given without
        specifying 'axis' and it will repeat along the vector's axis.
        """
        if len(counts) == 0:
            raise TypeError("must specify some counts")
        if len(counts) == 1 and isinstance(counts[0], int):
            if axis is None:
                if not m.isvec:
                    raise TypeError("must specify axis when repeating a non-"
                            "vector")
                axis = m.lastaxis
            if not isinstance(axis, int):
                raise TypeError("expected an integer axis, got "
                        f"{_objtname(axis)}")
            if axis < 0:
                raise ValueError(f"axis cannot be negative, got: {axis}")
            counts = (1, ) * axis + (counts[0], )
            axis = None
        else:
            counts = _maybe_unpack_ints(counts)
        for count in counts:
            if not isinstance(count, int):
                raise TypeError("expected an integer count, got "
                        f"{_objtname(count)}")
        if any(count < 0 for count in counts):
            raise ValueError(f"counts cannot be negative, got: {counts}")
        if axis is not None:
            raise TypeError("cannot specify axis with multiple counts")
        if m.isempty:
            return m
        # Use shape as helper.
        counts = Shape(counts)
        if counts.isempty:
            return empty(m.field)
        if counts.issingle:
            return m
        # Numpy repeat is slightly different to ours, as it allows repeat counts
        # to differ at each index and cant repeat along several axes at the same
        # time. So, we just repeat along each axis manually.
        cells = m._cells
        # Add axes since numpy wont do it for us.
        npshape = (1,) * (counts.ndim - cells.ndim) + cells.shape
        cells = cells.reshape(npshape)
        for axis in range(counts.ndim):
            if counts[axis] > 1:
                npaxis = Permuter.tonumpyaxis(cells.ndim, axis)
                cells = _np.repeat(cells, counts[axis], axis=npaxis)
        return m.fromnumpy(cells)


    def roll(m, *shifts, axis=None):
        """
        Circularly rotates elements by 'shifts[axis]' positions along each axis.
        If one bare integer is given for 'shifts', 'axis' must be specified to
        roll along only that axis (to prevent this, wrap it in a tuple or give
        another 0). If 'm' is a vector, one integer shift may be given without
        specifying 'axis' and it will roll along the vector axis.
        """
        if len(shifts) == 0:
            raise TypeError("must specify some shifts")
        if len(shifts) == 1 and isinstance(shifts[0], int):
            if axis is None:
                if not m.isvec:
                    raise TypeError("must specify axis when rolling a non-"
                            "vector")
                axis = m.lastaxis
            if not isinstance(axis, int):
                raise TypeError("expected an integer axis, got "
                        f"{_objtname(axis)}")
            if axis < 0:
                raise ValueError(f"axis cannot be negative, got: {axis}")
            shifts = (0, ) * axis + (shift[0], )
            axis = None
        else:
            shifts = _maybe_unpack_ints(shifts)
            for shift in shifts:
                if not isinstance(shift, int):
                    raise TypeError("expected an integer shift, got "
                            f"{_objtname(shift)}")
        if axis is not None:
            raise TypeError("cannot specify axis with multiple shifts")
        if m.isempty:
            return m
        # Roll each axis (only doing axes which we actually have).
        cells = m._cells
        for axis, shift in enumerate(shifts[:m.ndim]):
            shift %= m.shape[axis]
            if shift == 0:
                continue
            npaxis = Permuter.tonumpyaxis(m.ndim, axis)
            cells = _np.roll(cells, shift, npaxis)
        return Matrix[m.field, m.shape](cells)


    @_instconst
    def T(m):
        """
        Swaps the first two axes.
        """
        return m.permute(1, 0)


    @_instconst
    def diag(m):
        """
        Vector of the main diagonal, for 2D matrices.
        """
        return m.diagk(0)
    def diagk(m, k, emptyok=False):
        """
        Vector of the 'k'-th diagonal where k>0 is above the main diagonal and
        k<0 is below, for 2D matrices.
        """
        if m.ndim > 2:
            raise TypeError(f"only 2D matrices have a diagonal, got {m.shape}")
        if not isinstance(k, int):
            raise TypeError(f"expected an integer 'k', got {_objtname(k)}")
        lo = -m.shape[0] + 1
        hi = m.shape[1] - 1
        if k < lo or k > hi:
            if emptyok:
                return empty(m.field)
            raise TypeError(f"expected 'k' in {lo}..{hi}, got: {k}")
        if m.isempty or m.issingle:
            return m
        return m.fromnumpy(m._cells.diagonal(k))


    def along(m, *axes, drop=False):
        """
        Tuple of perpendicular matrices when travelling along the given axes
        sequentially. If 'drop', drops the axis travelled along in the returned
        matrices.
        """
        axes = _maybe_unpack_ints(axes)
        for axis in axes:
            if not isinstance(axis, int):
                raise TypeError(f"expected integer axes, got {_objtname(axis)}")
        if any(a < 0 for a in axes):
            raise ValueError(f"axes cannot be negative, got: {axes}")
        if len(set(axes)) != len(axes):
            raise ValueError(f"cannot repeat axes, got: {axes}")
        # Empty is empty.
        if m.isempty:
            return ()
        if not axes:
            return (m, )
        def get(x, axis, i):
            if axis >= x.ndim:
                return x
            idx = [slice(None)] * x.ndim
            idx[axis] = slice(i, i + 1)
            idx = tuple(idx)
            idx = Permuter.tonumpy(x.ndim)(idx)
            return x.at[idx]
        # Iterate each axis (reversed to give the correct order overall).
        traversed = (m, )
        for axis in reversed(axes):
            new = []
            for x in traversed:
                new.extend(get(x, axis, i) for i in range(x.shape[axis]))
            traversed = new
        # Drop if there are axes to drop.
        hangingaxes = set(range(min(axes), m.ndim)) - set(axes)
        if drop and hangingaxes:
            traversed = tuple(x.drop(axes) for x in traversed)
        return tuple(traversed)


    def group(m, *axes, drop=False):
        """
        Treats the product of the given axes as a group and returns a tuple of
        the iteration through all groups in the matrix in canonical order.
        """
        axes = _maybe_unpack_ints(axes)
        for axis in axes:
            if not isinstance(axis, int):
                raise TypeError(f"expected integer axes, got {_objtname(axis)}")
        if any(a < 0 for a in axes):
            raise ValueError(f"axes cannot be negative, got: {axes}")
        if len(set(axes)) != len(axes):
            raise ValueError(f"cannot repeat axes, got: {axes}")
        travel_axes = set(range(m.ndim)) - set(axes)
        return m.along(sorted(travel_axes), drop=drop)
    @_instconst
    def cols(m):
        """
        Tuple of all columns in canonical order.
        """
        return m.group(0)
    @_instconst
    def rows(m):
        """
        Tuple of all rows in canonical order.
        """
        return m.group(1)


    @_instconst
    def at(m):
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
        return _MatrixAt(m)


    def __iter__(m):
        """
        Vector-only cell iterate.
        """
        if not m.isvec:
            raise TypeError(f"only vectors have bare iteration, got {m.shape} "
                    "(use .along or .ravel for matrix iterate)")
        return (single(x, field=m.field, _nocast=True) for x in m._cells.ravel())

    def __len__(m):
        """
        Vector-only cell count, alias for 'm.size'.
        """
        if not m.isvec:
            raise TypeError(f"only vectors have bare length, got {m.shape} (use "
                    ".size for matrix cell count)")
        return m.size

    def __getitem__(m, i):
        """
        Vector-only cell access, supporting slices.
        """
        if not m.isvec:
            raise TypeError(f"only vectors have bare getitem, got {m.shape} "
                    "(use .at for matrix cell access)")
        if isinstance(i, int):
            if i < -m.size or i >= m.size:
                raise IndexError(f"index {i} out of bounds for vector of size "
                        f"{m.size}")
            if i < 0:
                i += m.size
            i = slice(i, i + 1)
        if not isinstance(i, slice):
            raise TypeError("expected integer or slice to index vector, got "
                    f"{_objtname(i)}")
        # Keep vector ndim (dont collapse to column).
        cells = m._cells.ravel().__getitem__(i)
        newshape = m.shape.withaxis(m.lastaxis, len(cells))
        cells = cells.reshape(newshape.tonumpy)
        return Matrix[m.field, newshape](cells)


    @_instconst
    def isdiag(m):
        """
        Is diagonal matrix? (square, and only main diagonal is non-zero)
        """
        if m.ndim > 2:
            raise TypeError(f"only 2D matrices can be diagonal, got {m.shape}")
        if not m.issquare:
            return False
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                if i == j:
                    continue
                if m.at[i, j] != m.zero:
                    return False
        return True

    @_instconst
    def isuppertri(m):
        """
        Is upper-triangular matrix? (square, and below main diagonal is zero)
        """
        if m.ndim > 2:
            raise TypeError(f"only 2D matrices can be triangular, got {m.shape}")
        if not m.issquare:
            return False
        for i in range(m.shape[0]):
            for j in range(i + 1, m.shape[1]):
                if m.at[i, j] != m.zero:
                    return False
        return True

    @_instconst
    def islowertri(m):
        """
        Is lower-triangular matrix? (square, and above main diagonal is zero)
        """
        if m.ndim > 2:
            raise TypeError(f"only 2D matrices can be triangular, got {m.shape}")
        if not m.issquare:
            return False
        for i in range(m.shape[0]):
            for j in range(i):
                if m.at[i, j] != m.zero:
                    return False
        return True

    @_instconst
    def isorthogonal(m):
        """
        Is orthogonal matrix? (transpose == inverse)
        """
        if m.ndim > 2:
            raise TypeError(f"only 2D matrices can be orthogonal, got {m.shape}")
        if not m.issquare:
            return False
        if m.det == m.zero:
            return False
        return (m.T == m.inv).alll
    @_instconst
    def isunitary(m):
        """
        Is unitary matrix? (conjugate transpose == inverse)
        """
        if m.ndim > 2:
            raise TypeError(f"only 2D matrices can be orthogonal, got {m.shape}")
        if not m.issquare:
            return False
        if m.det == m.zero:
            return False
        return (m.H == m.inv).alll

    @_instconst
    def issymmetric(m):
        """
        Is symmetric matrix? (square, and below main diagonal = above main
        diagonal)
        """
        if m.ndim > 2:
            raise TypeError(f"only 2D matrices can be symmetric, got {m.shape}")
        if not m.issquare:
            return False
        return (m.T == m).alll


    @classmethod
    def _apply(M, func, rfield, *xs, user=False, cast=False):
        # Expects func to operate on field elements and return rfield elements.
        # If `user`, expects a return of `field, numpy array`.

        # toField me.
        if rfield is not None:
            rfield = toField(rfield)
        # Check me.
        if not xs:
            if rfield is None:
                rfield = M.field
            return empty(rfield)
        # Cast me.
        if cast:
            xs = M.cast(*xs)
        # Shape me.
        xshape = xs[0].shape
        # Empty me (dujj).
        if xshape.isempty:
            if rfield is None:
                rfield = M.field
            return tuple(empty(rfield) for _ in xs)
        # Expose me (dujj).
        xs = [x._cells for x in xs]
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
                rfield = toField(type(first))
        # Npshape me.
        npshape = (xshape.size, )
        if user:
            npshape += rshape.tonumpy
        # Dtype me.
        dtype = rfield.dtype()
        # Preallocate me.
        cells = _np.empty(npshape, dtype)
        # Do me (dujj).
        cells[0] = first
        for i in range(1, npshape[0]):
            if not user:
                ret = func(*(x[i] for x in xs))
            else:
                ret_field, ret = func(*(x[i] for x in xs))
                assert isinstance(ret, _np.ndarray)
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
    def applyto(M, func, *xs, rfield=None):
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
            ys = [single(y, field=M.field) for y in ys]
            ret = func(*ys)
            # Validate me (dujj).
            if not isinstance(ret, Matrix):
                ret = single(ret)
            # Unpack to field and cells.
            return ret.field, ret._cells
        return M._apply(wrapped, rfield, *xs, user=True, cast=True)

    def apply(m, func, *os, rfield=None):
        """
        Alias for 'M.applyto(func, m, *os, rfield=rfield)'.
        """
        return m.applyto(func, m, *os, rfield=rfield)


    def _fold(m, func, axis=..., seed=NO_SEED, right=False):
        # Expects func operate on field elements and return field elements. If
        # `axis` is `...`, seed must be a field element, otherwise it must be a
        # broadcastable perpendicular matrix. func always operates on and returns
        # field elements.

        if right:
            f = lambda a, b: func(b, a)
            order = reversed
        else:
            f = func
            order = lambda x: x

        if axis is ...:
            # Iterate through the flat cells.
            flat = m._cells.reshape(-1)
            for x in order(flat):
                if seed is NO_SEED:
                    seed = x
                    continue
                seed = f(seed, x)
            return single(seed, field=m.field)

        perpshape = m.shape.withaxis(axis, 1)
        if seed is not NO_SEED:
            try:
                seed = seed.broadcast(perpshape)
            except ValueError as e:
                raise ValueError("cannot broadcast seed for folding along axis "
                        f"{axis}") from e
        # Iterate along axis.
        for x in order(m.along(axis)):
            if seed is NO_SEED:
                seed = x
                continue
            seed = x._apply(f, m.field, seed, x)
        if seed is NO_SEED:
            assert m.isempty
            return m
        return seed

    def fold(m, func, axis=..., seed=NO_SEED, right=False):
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
        is '...', this folding is performed along the ravelled array, otherwise
        it is performed along that axis in parallel. If 'm' is empty, returns
        'seed' (which must be specified).
        """
        if not callable(func):
            raise TypeError(f"expected callable 'func', got {_objtname(func)}")
        if axis is not ...:
            if not isinstance(axis, int):
                raise TypeError("expected an integer axis, got "
                        f"{_objtname(axis)}")
            if axis < 0:
                raise ValueError(f"axis cannot be negative, got: {axis}")
        if seed is not NO_SEED:
            seed, = m.cast(seed, broadcast=False)
            if axis is ...:
                if not seed.issingle:
                    raise TypeError("expected a single for 'seed' when folding "
                            f"over all elements, got {seed.shape}")
                seed = seed.obj
        def wrapped(a, b):
            a = single(a, field=m.field)
            b = single(b, field=m.field)
            c = func(a, b)
            if not isinstance(c, Matrix):
                c = single(c, field=m.field)
            if not c.issingle:
                raise TypeError("expected folding function to return a single, "
                        f"got {c.shape}")
            return c.obj
        return m._fold(wrapped, axis=axis, seed=seed, right=right)


    @_instconst
    def alll(m):
        """
        Are all elements true? Returns a bool, not a matrix.
        """
        return m.field._mat_all(m)
    @_instconst
    def anyy(m):
        """
        Are any elements true? Returns a bool, not a matrix.
        """
        return m.field._mat_any(m)

    @_instconst
    def always(m):
        """
        Element-wise "is always true"?
        """
        return m.field._mat_always(m)
    @_instconst
    def anytime(m):
        """
        Element-wise "is true at any point"?
        """
        return m.field._mat_anytime(m)

    def __invert__(m):
        """
        Element-wise logical-not.
        """
        return m.field._mat_not(m)
    def __and__(m, o, *, reverse=False):
        """
        Element-wise logical-and.
        """
        m, o = m.cast(m, o)
        if reverse:
            m, o = o, m
        return m.field._mat_and(m, o)
    def __rand__(m, o):
        return m.__and__(o, reverse=True)
    def __or__(m, o, *, reverse=False):
        """
        Element-wise logical-or.
        """
        m, o = m.cast(m, o)
        if reverse:
            m, o = o, m
        return m.field._mat_or(m, o)
    def __ror__(m, o):
        return m.__or__(o, reverse=True)


    @_instconst
    def real(m):
        """
        Element-wise take-real.
        """
        return m.field._mat_real(m)
    @_instconst
    def imag(m):
        """
        Element-wise take-imaginary.
        """
        return m.field._mat_imag(m)
    @_instconst
    def conj(m):
        """
        Element-wise complex conjugate.
        """
        return m.field._mat_conj(m)
    def __abs__(m):
        """
        Element-wise absolution.
        """
        return m.field._mat_abs(m)
    @_instconst
    def abs(m):
        """
        Element-wise absolution, alias for 'abs(m)'.
        """
        return m.__abs__()
    @_instconst
    def angle(m):
        """
        Element-wise complex-argument.
        """
        return m.field._mat_angle(m)
    @_instconst
    def sgn(m):
        """
        Element-wise signum function: 0 if (x == 0), otherwise x / |x|
        """
        return m.field._mat_sgn(m)

    def __pos__(m):
        """
        Element-wise NOTHING.
        """
        return m
    def __neg__(m):
        """
        Element-wise negation.
        """
        return m.field._mat_neg(m)

    def __add__(m, o, *, reverse=False):
        """
        Element-wise addition.
        """
        m, o = m.cast(m, o)
        if reverse:
            m, o = o, m
        return m.field._mat_add(m, o)
    def __radd__(m, o):
        return m.__add__(o, reverse=True)

    def __sub__(m, o, *, reverse=False):
        """
        Element-wise subtraction.
        """
        m, o = m.cast(m, o)
        if reverse:
            m, o = o, m
        return m.field._mat_sub(m, o)
    def __rsub__(m, o):
        return m.__sub__(o, reverse=True)

    def __mul__(m, o, *, reverse=False):
        """
        Element-wise multiplication (use '@' for matrix multiplication).
        """
        m, o = m.cast(m, o)
        if reverse:
            m, o = o, m
        return m.field._mat_mul(m, o)
    def __rmul__(m, o):
        return m.__mul__(o, reverse=True)

    def __truediv__(m, o, *, reverse=False):
        """
        Element-wise division.
        """
        m, o = m.cast(m, o)
        if reverse:
            m, o = o, m
        return m.field._mat_div(m, o)
    def __rtruediv__(m, o):
        return m.__truediv__(o, reverse=True)

    def __mod__(m, o, *, reverse=False):
        """
        Element-wise modulus.
        """
        m, o = m.cast(m, o)
        if reverse:
            m, o = o, m
        return m.field._mat_mod(m, o)
    def __rmod__(m, o):
        return m.__mod__(o, reverse=True)

    @_instconst
    def exp(m):
        """
        Element-wise natural exponential.
        """
        return m.field._mat_exp(m)
    @_instconst
    def exp2(m):
        """
        Element-wise base-2 exponential.
        """
        return m.field._mat_exp2(m)
    @_instconst
    def exp10(m):
        """
        Element-wise base-10 exponential.
        """
        return m.field._mat_exp10(m)
    def __pow__(m, o, *, reverse=False):
        """
        Element-wise power.
        """
        m, o = m.cast(m, o)
        if reverse:
            m, o = o, m
        return m.field._mat_pow(m, o)
    def __rpow__(m, o):
        return m.__pow__(o, reverse=True)

    @_instconst
    def ln(m):
        """
        Element-wise natural logarithm.
        """
        return m.field._mat_ln(m)
    @_instconst
    def log2(m):
        """
        Element-wise base-2 logarithm.
        """
        return m.field._mat_log2(m)
    @_instconst
    def log10(m):
        """
        Element-wise base-10 logarithm.
        """
        return m.field._mat_log10(m)
    def log(m, base):
        """
        Element-wise base-specified logarithm.
        """
        m, base = m.cast(m, base)
        return m.field._mat_log(m, base)

    @_instconst
    def sqrt(m):
        """
        Element-wise square root.
        """
        return m.field._mat_sqrt(m)
    @_instconst
    def cbrt(m):
        """
        Element-wise cube root.
        """
        return m.field._mat_cbrt(m)
    def root(m, n):
        """
        Element-wise nth root.
        """
        m, n = m.cast(m, n)
        return m.field._mat_root(m, n)

    @_instconst
    def floor(m):
        """
        Element-wise floor function.
        """
        return m.field._mat_floor(m)
    @_instconst
    def ceil(m):
        """
        Element-wise ceil function.
        """
        return m.field._mat_ceil(m)
    @_instconst
    def trunc(m):
        """
        Element-wise truncate function.
        """
        return m.field._mat_trunc(m)
    @_instconst
    def rint(m):
        """
        Element-wise round-to-integer function.
        """
        return m.field._mat_rint(m)

    @_instconst
    def sin(m):
        """
        Element-wise trigonometric sine.
        """
        return m.field._mat_sin(m)
    @_instconst
    def cos(m):
        """
        Element-wise trigonometric cosine.
        """
        return m.field._mat_cos(m)
    @_instconst
    def tan(m):
        """
        Element-wise trigonometric tangent.
        """
        return m.field._mat_tan(m)
    @_instconst
    def csc(m):
        """
        Element-wise trigonometric cosecant: csc(x) = 1/sin(x)
        """
        return m.field._mat_csc(m)
    @_instconst
    def sec(m):
        """
        Element-wise trigonometric secant: sec(x) = 1/cos(x)
        """
        return m.field._mat_sec(m)
    @_instconst
    def cot(m):
        """
        Element-wise trigonometric cotangent: cot(x) = 1/tan(x)
        """
        return m.field._mat_cot(m)

    @_instconst
    def asin(m):
        """
        Element-wise trigonometric inverse-sine:
            asin(x) = y, s.t. y in [-pi/2, pi/2] and sin(y) = x
        """
        return m.field._mat_asin(m)
    @_instconst
    def acos(m):
        """
        Element-wise trigonometric inverse-cosine:
            acos(x) = y, s.t. y in [0, pi] and cos(y) = x
        """
        return m.field._mat_acos(m)
    @_instconst
    def atan(m):
        """
        Element-wise trigonometric inverse-tangent:
            atan(x) = y, s.t. y in (-pi/2, pi/2) and tan(y) = x
        """
        return m.field._mat_atan(m)
    @_instconst
    def acsc(m):
        """
        Element-wise trigonometric inverse-cosecant: acsc(x) = asin(1/x)
        """
        return m.field._mat_acsc(m)
    @_instconst
    def asec(m):
        """
        Element-wise trigonometric inverse-secant: asec(x) = acos(1/x)
        """
        return m.field._mat_asec(m)
    @_instconst
    def acot(m):
        """
        Element-wise trigonometric inverse-cotangent: acot(x) = atan(1/x)
        """
        return m.field._mat_acot(m)

    @_instconst
    def sinh(m):
        """
        Element-wise trigonometric hyperbolic-sine.
        """
        return m.field._mat_sinh(m)
    @_instconst
    def cosh(m):
        """
        Element-wise trigonometric hyperbolic-cosine.
        """
        return m.field._mat_cosh(m)
    @_instconst
    def tanh(m):
        """
        Element-wise trigonometric hyperbolic-tangent.
        """
        return m.field._mat_tanh(m)
    @_instconst
    def csch(m):
        """
        Element-wise trigonometric hyperbolic-cosecant: csch(x) = 1/sinh(x)
        """
        return m.field._mat_csch(m)
    @_instconst
    def sech(m):
        """
        Element-wise trigonometric hyperbolic-secant: sech(x) = 1/cosh(x)
        """
        return m.field._mat_sech(m)
    @_instconst
    def coth(m):
        """
        Element-wise trigonometric hyperbolic-cotangent: coth(x) = 1/tanh(x)
        """
        return m.field._mat_coth(m)

    @_instconst
    def asinh(m):
        """
        Element-wise trigonometric inverse-hyperbolic-sine:
            asinh(x) = y, s.t. sinh(y) = x
        """
        return m.field._mat_asinh(m)
    @_instconst
    def acosh(m):
        """
        Element-wise trigonometric inverse-hyperbolic-cosine:
            acosh(x) = y, s.t. y >= 0 and cosh(y) = x
        """
        return m.field._mat_acosh(m)
    @_instconst
    def atanh(m):
        """
        Element-wise trigonometric inverse-hyperbolic-tangent:
            atanh(x) = y, s.t. tanh(y) = x
        """
        return m.field._mat_atanh(m)
    @_instconst
    def acsch(m):
        """
        Element-wise trigonometric inverse-hyperbolic-cosecant:
            acsch(x) = asinh(1/x)
        """
        return m.field._mat_acsch(m)
    @_instconst
    def asech(m):
        """
        Element-wise trigonometric inverse-hyperbolic-secant:
            asech(x) = acosh(1/x)
        """
        return m.field._mat_asech(m)
    @_instconst
    def acoth(m):
        """
        Element-wise trigonometric inverse-hyperbolic-cotangent:
            acoth(x) = atanh(1/x)
        """
        return m.field._mat_acoth(m)

    @_instconst
    def torad(m):
        """
        Converts degrees to radians, alias for 'm / (180/pi)'.
        """
        return m.field._mat_torad(m)
    @_instconst
    def todeg(m):
        """
        Converts radians to degrees, alias for 'm * (180/pi)'.
        """
        return m.field._mat_todeg(m)

    @_instconst
    def sind(m):
        """
        Alias for 'm.torad.sin'.
        """
        return m.torad.sin
    @_instconst
    def cosd(m):
        """
        Alias for 'm.torad.cos'.
        """
        return m.torad.cos
    @_instconst
    def tand(m):
        """
        Alias for 'm.torad.tan'.
        """
        return m.torad.tan
    @_instconst
    def cscd(m):
        """
        Alias for 'm.torad.csc'.
        """
        return m.torad.csc
    @_instconst
    def secd(m):
        """
        Alias for 'm.torad.sec'.
        """
        return m.torad.sec
    @_instconst
    def cotd(m):
        """
        Alias for 'm.torad.cot'.
        """
        return m.torad.cot

    @_instconst
    def asind(m):
        """
        Alias for 'm.asin.todeg'.
        """
        return m.asin.todeg
    @_instconst
    def acosd(m):
        """
        Alias for 'm.acos.todeg'.
        """
        return m.acos.todeg
    @_instconst
    def atand(m):
        """
        Alias for 'm.atan.todeg'.
        """
        return m.atan.todeg
    @_instconst
    def acscd(m):
        """
        Alias for 'm.acsc.todeg'.
        """
        return m.acsc.todeg
    @_instconst
    def asecd(m):
        """
        Alias for 'm.asec.todeg'.
        """
        return m.asec.todeg
    @_instconst
    def acotd(m):
        """
        Alias for 'm.acot.todeg'.
        """
        return m.acot.todeg


    def diff(y, x):
        """
        Element-wise derivative with respect to 'x'.
        """
        y, x = y.cast(y, x)
        return y.field._mat_diff(y, x)

    def intt(y, x, *bounds):
        """
        Element-wise integral with respect to 'x'. If bounds are provided,
        evaluates the definite integral.
        """
        bounds = _maybe_unpack(bounds) # unpack mats also.
        if len(bounds) == 0:
            y, x = y.cast(y, x)
            return y.field._mat_intt(y, x)
        if len(bounds) != 2:
            raise TypeError("expected 0 or 2 integration bounds, got "
                    f"{len(bounds)}")
        y, x, a, b = y.cast(y, x, *bounds)
        return y.field._mat_def_intt(y, x, a, b)


    def __eq__(m, o):
        """
        Element-wise equal-to.
        """
        m, o = m.cast(m, o)
        return m.field._mat_eq(m, o)
    def __ne__(m, o):
        """
        Element-wise not-equal-to.
        """
        m, o = m.cast(m, o)
        return m.field._mat_ne(m, o)
    def __lt__(m, o):
        """
        Element-wise less-than.
        """
        m, o = m.cast(m, o)
        return m.field._mat_lt(m, o)
    def __le__(m, o):
        """
        Element-wise less-than-or-equal-to.
        """
        m, o = m.cast(m, o)
        return m.field._mat_le(m, o)
    def __gt__(m, o):
        m, o = m.cast(m, o)
        return o.field._mat_lt(o, m)
    def __ge__(m, o):
        m, o = m.cast(m, o)
        return o.field._mat_le(o, m)

    def issame(m, o):
        """
        Element-wise identical check. Note this is different to '==' (which does
        equivalence, and may be different than identical values).
        """
        m, o = m.cast(m, o)
        return m.field._mat_issame(m, o)

    @_instconst
    def isreal(m):
        """
        Element-wise imaginary component =0.
        """
        return (m.imag == m.zero)
    @_instconst
    def isimag(m):
        """
        Element-wise real component =0.
        """
        return (m.real == m.zero)
    @_instconst
    def isinf(m):
        """
        Element-wise "is infinite"?
        """
        return m.field._mat_isinf(m)
    @_instconst
    def isnan(m):
        """
        Element-wise "is not-a-number"?
        """
        return m.field._mat_isnan(m)
    @_instconst
    def isfinite(m):
        """
        Element-wise "is not infinite and is not not-a-number"?
        """
        return m.field._mat_isfinite(m)

    @_instconst
    def inv(m):
        """
        Inverse matrix, for square 2D matrices.
        """
        if not m.issquare:
            raise TypeError(f"cannot invert a non-square matrix, got {m.shape}")
        if m.det == m.zero:
            raise ValueError("cannot invert a non-invertible matrix, got det=0")
        if m.isempty:
            return m
        aug = hstack(m, m.eye)
        aug = aug.rref
        return aug.at[:, m.shape[0]:]

    @_instconst
    def det(m):
        """
        Determinant, for 2D square matrices.
        """
        if m.ndim > 2:
            raise TypeError("only 2D matrices have a determinant, got "
                    f"{m.shape}")
        if not m.issquare:
            raise TypeError(f"only square matrices a determinant, got {m.shape}")
        return m.field._mat_det(m)

    @_instconst
    def trace(m):
        """
        Sum of diagonal elements, for 2D square matrices.
        """
        if m.ndim > 2:
            raise TypeError(f"only 2D matrices have a trace, got {m.shape}")
        if not m.issquare:
            raise TypeError(f"only square matrices have a trace, got {m.shape}")
        return m.field._mat_trace(m)

    @_instconst
    def H(m):
        """
        Hermitian transpose, alias for 'm.conj.T'.
        """
        return m.conj.T

    @_instconst
    def norm(m):
        """
        Vector euclidean norm (2-norm): sqrt(sum[abs(xi) ** 2])
        """
        if not m.isvec:
            raise TypeError(f"only vectors have a norm, got {m.shape}")
        return m.field._mat_norm(m)

    def dot(m, o):
        """
        Vector dot product, using Hermitian inner product: sum[conj(xi) * yi]
        """
        m, o = m.cast(m, o, broadcast=False)
        if not m.isvec or not o.isvec:
            raise TypeError(f"only vectors have a dot product, got {m.shape} "
                    f"and {o.shape}")
        if m.size != o.size:
            raise TypeError("only equal-length vectors have a dot product, "
                    f"got {m.shape} and {o.shape}")
        return m.field._mat_dot(m, o)

    def cross(m, o):
        """
        Vector 3-element cross product.
        """
        m, o = m.cast(m, o, broadcast=False)
        if not m.isvec or not o.isvec:
            raise TypeError(f"only vectors have a cross product, got {m.shape} "
                    f"and {o.shape}")
        if m.size != 3 or o.size != 3:
            raise TypeError("only 3-element vectors have a cross product, got "
                    f"{m.shape} and {o.shape}")
        return m.field._mat_cross(m, o)

    def __matmul__(m, o, *, reverse=False):
        """
        Matrix multiplication.
        """
        m, o = m.cast(m, o, broadcast=False)
        if reverse:
            m, o = o, m
        if m.ndim > 2 or o.ndim > 2:
            raise TypeError("only 2D matrices have matrix multiplication, got "
                    f"{m.shape} and {o.shape}")
        if m.shape[1] != o.shape[0]:
            raise TypeError("need equal inner dimension lengths for matrix "
                    f"multiplication, got {m.shape} @ {o.shape}")
        return m.field._mat_matmul(m, o)
    def __rmatmul__(m, o):
        return m.__matmul__(m, o, reverse=True)


    @_instconst
    def minn(m):
        """
        Minimum of all elements, alias for 'm.minn_along(...)'.
        """
        return m.minn_along(...)
    def minn_along(m, axis):
        """
        Minimum of the values along the given axis. If 'axis' is '...', returns
        the minimum over all elements. In the case of ties, the earlier occurence
        is kept.
        """
        if m.isempty:
            raise TypeError("cannot find minimum of empty")
        if axis is not ...:
            if not isinstance(axis, int):
                raise TypeError("expected an integer axis, got "
                        f"{_objtname(axis)}")
            if axis < 0:
                raise ValueError(f"axis cannot be negative, got: {axis}")
        return m.field._mat_minn(m, axis)

    @_instconst
    def maxx(m):
        """
        Maximum of all elements, alias for 'm.maxx_along(...)'.
        """
        return m.maxx_along(...)
    def maxx_along(m, axis):
        """
        Maximum of the values along the given axis. If 'axis' is '...', returns
        the maximum over all elements. In the case of ties, the earlier occurence
        is kept.
        """
        if m.isempty:
            raise TypeError("cannot find maximum of empty")
        if axis is not ...:
            if not isinstance(axis, int):
                raise TypeError("expected an integer axis, got "
                        f"{_objtname(axis)}")
            if axis < 0:
                raise ValueError(f"axis cannot be negative, got: {axis}")
        return m.field._mat_maxx(m, axis)

    @_instconst
    def summ(m):
        """
        Sum of all elements, alias for 'm.summ_along(...)'.
        """
        return m.summ_along(...)
    def summ_along(m, axis):
        """
        Additive sum of the values along the given axis. If 'axis' is '...',
        returns the sum over all elements.
        """
        if axis is not ...:
            if not isinstance(axis, int):
                raise TypeError("expected an integer axis, got "
                        f"{_objtname(axis)}")
            if axis < 0:
                raise ValueError(f"axis cannot be negative, got: {axis}")
        return m.field._mat_summ(m, axis)

    @_instconst
    def prod(m):
        """
        Product of all elements, alias for 'm.prod_along(...)'.
        """
        return m.prod_along(...)
    def prod_along(m, axis):
        """
        Multiplicative product of the values along the given axis. If 'axis' is
        '...', returns the product over all elements.
        """
        if axis is not ...:
            if not isinstance(axis, int):
                raise TypeError("expected an integer axis, got "
                        f"{_objtname(axis)}")
            if axis < 0:
                raise ValueError(f"axis cannot be negative, got: {axis}")
        return m.field._mat_prod(m, axis)

    @_instconst
    def mean(m):
        """
        Arithmetic mean of all elements, alias for 'm.mean_along(...)'.
        """
        return m.mean_along(...)
    def mean_along(m, axis):
        """
        Arithmetic mean of the values along the given axis (or along all elements
        if 'axis' is '...'): sum[xi] / n
        """
        if m.isempty:
            raise TypeError("cannot find arithmetic mean of empty")
        if axis is not ...:
            if not isinstance(axis, int):
                raise TypeError("expected an integer axis, got "
                        f"{_objtname(axis)}")
            if axis < 0:
                raise ValueError(f"axis cannot be negative, got: {axis}")
        return m.field._mat_mean(m, axis)
    @_instconst
    def ave(m):
        """
        Alias for 'm.mean'.
        """
        return m.mean
    def ave_along(m, axis):
        """
        Alias for 'm.mean_along(axis)'.
        """
        return m.mean_along(axis)

    @_instconst
    def geomean(m):
        """
        Geometric mean of all elements, alias for 'm.geomean_along(...)'.
        """
        return m.geomean_along(...)
    def geomean_along(m, axis):
        """
        Geometric mean of the values along the given axis (or along all elements
        if 'axis' is '...'): prod[xi] ** (1/n)
        """
        if m.isempty:
            raise TypeError("cannot find geometric mean of empty")
        if axis is not ...:
            if not isinstance(axis, int):
                raise TypeError("expected an integer axis, got "
                        f"{_objtname(axis)}")
            if axis < 0:
                raise ValueError(f"axis cannot be negative, got: {axis}")
        return m.field._mat_geomean(m, axis)

    @_instconst
    def harmean(m):
        """
        Harmonic mean of all elements, alias for 'm.harmean_along(...)'.
        """
        return m.harmean_along(...)
    def harmean_along(m, axis):
        """
        Harmonic mean of the values along the given axis (or along all elements
        if 'axis' is '...'): n / sum[1/xi]
        """
        if m.isempty:
            raise TypeError("cannot find harmonic mean of empty")
        if axis is not ...:
            if not isinstance(axis, int):
                raise TypeError("expected an integer axis, got "
                        f"{_objtname(axis)}")
            if axis < 0:
                raise ValueError(f"axis cannot be negative, got: {axis}")
        return m.field._mat_harmean(m, axis)

    @_instconst
    def quadmean(m):
        """
        Quadratic mean (root-mean-square) of all elements, alias for
        'm.quadmean_along(...)'.
        """
        return m.quadmean_along(...)
    def quadmean_along(m, axis):
        """
        Quadratic mean (root-mean-square) of the values along the given axis (or
        along all elements if 'axis' is '...'): sqrt(n * sum[xi**2])
        """
        if m.isempty:
            raise TypeError("cannot find quadratic mean of empty")
        if axis is not ...:
            if not isinstance(axis, int):
                raise TypeError("expected an integer axis, got "
                        f"{_objtname(axis)}")
            if axis < 0:
                raise ValueError(f"axis cannot be negative, got: {axis}")
        return m.field._mat_quadmean(m, axis)


    @_instconst
    def rref(m):
        """
        Reduced row echelon form, for 2D matrices.
        """
        if m.ndim > 2:
            raise TypeError(f"only 2D matrices can be rrefed, got {m.shape}")
        return m.field._mat_rref(m)

    @_instconst
    def pivots(m):
        """
        Tuple of RREF pivot column indices.
        """
        if m.isempty:
            return ()
        sys = m.rref
        zero = m.field.zero
        one = m.field.one
        eq = m.field.eq
        ne = m.field.ne
        # TODO: bool auto-promotion
        return tuple(i for i, c in enumerate(sys.cols)
                     if 1 == sum(ne(zero, x) for x in c)
                     and 1 == sum(eq(one, x) for x in c))

    @_instconst
    def nonpivots(m):
        """
        Tuple of RREF non-pivot column indices.
        """
        return tuple(j for j in range(m.shape[1]) if j not in m.pivots)

    @_instconst
    def colspace(m):
        """
        Tuple of basis vectors for column space.
        """
        return tuple(m.cols[p] for p in m.pivots)

    @_instconst
    def rowspace(m):
        """
        Tuple of basis vectors for row space.
        """
        if m.isempty:
            return ()
        sys = m.rref
        zero = m.field.zero
        ne = m.field.ne
        # TODO: bool auto-promotion
        nonzeros = (i for i, r in enumerate(sys.rows)
                    if any(ne(zero, x) for x in r))
        return tuple(sys.rows[i].T for i in nonzeros)

    @_instconst
    def nullspace(m):
        """
        Tuple of basis vectors for null space.
        """
        sys = m.rref # implied zero-vec augment.
        zero = m.field.zero
        one = m.field.one
        eq = m.field.eq
        sub = m.field.sub
        neg = lambda a: sub(zero, a)
        def find_first_one(xs):
            for i, x in enumerate(xs):
                if eq(one, x):
                    return i
            return None
        pivotat = tuple(find_first_one(row) for row in sys.rows)
        nonpivots = sys.nonpivots
        Col = Matrix[m.field, Shape(m.shape[1])]
        zeros = Col.zeros._cells
        basis = [zeros.copy() for _ in nonpivots]
        for n, j in enumerate(nonpivots):
            for i in range(m.shape[0]):
                if pivotat[i] is None or pivotat[i] > j:
                    basis[n][j] = one
                    break
                basis[n][pivotat[i]] = neg(sys._cells[i, j])
        return tuple(Col(x) for x in basis)


    def __hash__(m):
        return m.field._mat_hash(m)

    def __repr__(m, short=None, multiline=True):
        if short is None:
            short = doesdflt2short()
        short = not not short
        multiline = not not multiline
        return m.field._mat_rep(m, short, multiline)


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
            i = _np.array(i, dtype=object)
            if i.shape != (len(i), ):
                raise TypeError(f"expected a 1d sequence to index axis {axis}, "
                        f"got shape: {i.shape}")
            for j in i:
                if not isinstance(j, int):
                    raise TypeError("expected a sequence of integers to index "
                            f"axis {axis}, got {_objtname(j)}")
        if not isinstance(i, _np.ndarray):
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
        m = self._matrix
        if m.isempty:
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
        return m.fromnumpy(m._cells[ijk])
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
    field, shape = params
    field = toField(field)
    shape = Shape(shape)
    return field, shape


# Make a creator from a numpy array.
def _matrix_fromnumpy(field, arr):
    """
    Creates a matrix over the given field from the given numpy array.
    """
    shape = Shape.fromnumpy(arr.shape)
    arr = arr.reshape(shape.tonumpy)
    return Matrix[field, shape](arr)
Matrix.fromnumpy = _matrix_fromnumpy
del _matrix_fromnumpy



class Int(RealField):
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
        return float(a.__int__())
    @classmethod
    def to_complex(cls, a):
        return complex(a.__int__())

    @classmethod
    def consts(cls):
        return super().consts() | {
            "__0__": 0,
            "__1__": 1,
        }

    @classmethod
    def abs(cls, a):
        return abs(a)
    @classmethod
    def sgn(cls, a):
        raise (a > 0) - (a < 0)
    @classmethod
    def add(cls, a, b):
        return a + b
    @classmethod
    def sub(cls, a, b):
        return a - b
    @classmethod
    def mul(cls, a, b):
        return a * b
    @classmethod
    def div(cls, a, b):
        rem = (a % b)
        if rem:
            raise TypeError(f"expected integer division result over int, got: "
                    f"{a} / {b}")
        return a // b
    @classmethod
    def mod(cls, a, b):
        return a % b
    @classmethod
    def power(cls, a, b):
        if b < 0:
            if a == 0:
                raise ZeroDivisionError("0^(-x)")
            if a == 1:
                return 1
            raise TypeError(f"expected integer power result over int, got: "
                    f"{a} ** {b}")
        return a ** b
    @classmethod
    def root(cls, a, b):
        if b == 0:
            raise ZeroDivisionError("x^(1/0)")
        if b < 0:
            return cls.power(cls.root(a, -b), -1)
        raise NotImplementedError("iroot")
    @classmethod
    def log(cls, a, b):
        raise NotImplementedError("ilog")

    @classmethod
    def comparison_field(cls):
        return bool
    @classmethod
    def eq(cls, a, b):
        return a == b
    @classmethod
    def ne(cls, a, b):
        return a != b
    @classmethod
    def lt(cls, a, b):
        return a < b
    @classmethod
    def le(cls, a, b):
        return a <= b

    @classmethod
    def issame(cls, a, b):
        return a == b

    @classmethod
    def hashed(cls, a):
        return hash(a.__int__())

    @classmethod
    def stringify(cls, a, short):
        return _cons.pretty_number(a.__int__(), short)


    @classmethod
    def _mat_numpyarr_int(cls, m):
        cells = m._cells
        # Check for overflow.
        hi = (1 << 63) - 1
        lo = -(1 << 63)
        if ((cells < lo) | (cells > hi)).any():
            raise ValueError("cannot cast too-large value to int64, overflowed")
        return cells.astype(_np.int64)

    @classmethod
    def _mat_numpyarr_float(cls, m):
        cells = m._cells
        # Check for overflow.
        hif = _sys.float_info.max
        hi = int(hif) + int(_math.ulp(hif))//2 - 1
        lo = -hi
        if ((cells < lo) | (cells > hi)).any():
            raise ValueError("cannot cast too-large value to float64, "
                    "overflowed")
        return cells.astype(_np.float64)

    @classmethod
    def _mat_numpyarr_complex(cls, m):
        cells = m._cells
        # Same as float check.
        hif = _sys.float_info.max
        hi = int(hif) + int(_math.ulp(hif))//2 - 1
        lo = -hi
        if ((cells < lo) | (cells > hi)).any():
            raise ValueError("cannot cast too-large value to complex128, "
                    "overflowed")
        return cells.astype(_np.complex128)

    @classmethod
    def _mat_fromfield(cls, m):
        # Note we can't use numpy method since that uses 64b (instead of
        # unbounded) ints. However we will use the fields cast method.
        toobj = _np.vectorize(m.field.to_int, otypes=[cls.dtype()])
        return Matrix[cls, m.shape](toobj(m._cells))


    # Apparently numpy just uses python integers when given an object dtype for
    # these functions.
    @classmethod
    def _mat_eye(cls, M):
        cells = _np.eye(M.shape[0], dtype=cls.dtype())
        return M(cells)
    @classmethod
    def _mat_zeros(cls, M):
        cells = _np.zeros(M.shape.tonumpy, dtype=cls.dtype())
        return M(cells)
    @classmethod
    def _mat_ones(cls, M):
        cells = _np.ones(M.shape.tonumpy, dtype=cls.dtype())
        return M(cells)



def _float_eq(x, y, ulps=50):
    # Helper to return bool np array of the equality of the two given np float64
    # arrays. Automatically allows for 'ulps' of imprecision, except in some
    # cases like:
    # - nan always compares false
    # - 0.0 only compares equal to exactly 0.0 or (-0.0)
    # - must be exact:
    #   - inf
    #   - (-inf)
    #   - 1
    #   - (-1)
    # Note that the imprecision is allowed to extend around zero, and this leads
    # to some strange cases, like:
    # >> eps == -eps
    # True
    # >> eps == 0.0
    # False
    # But she'll be so right.

    absx = _np.abs(x)
    absy = _np.abs(y)

    # Mask places where we can use inexact comparison. use abs a bunch to speed
    # up memory access (maybe idk) since we already need it for =1 check.
    inexact = (_np.isfinite(absx) & _np.isfinite(absy))
    inexact &= ((absx != 0.0) & (absy != 0.0))
    inexact &= ((absx != 1.0) & (absy != 1.0))

    # Get the bits of each number, using "negative" bits for negative numbers
    # instead of the literal bits of the negative number.
    ux = absx.view(_np.uint64).copy()
    uy = absy.view(_np.uint64).copy()
    ux[x < 0.0] = -ux[x < 0.0]
    uy[y < 0.0] = -uy[y < 0.0]
    ux -= uy
    dif = _np.abs(ux, out=ux)

    # Make the comparison, adding eqs where they are close enough and exact isn't
    # required.
    eq = (x == y)
    eq[inexact] |= (dif <= ulps)[inexact]
    return eq

def _float_eq1(x, y, ulps=50):
    x = _np.array([a], dtype=float)
    y = _np.array([b], dtype=float)
    return _float_eq(x, y)[0]


class Float(RealField):
    @classmethod
    def dtype(cls):
        return _np.dtype(float)

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
            "__inf__": float("inf"),
            "__nan__": float("nan"),
        }

    @classmethod
    def abs(cls, a):
        return abs(a)
    @classmethod
    def sgn(cls, a):
        raise float((a > 0) - (a < 0))

    @classmethod
    def add(cls, a, b):
        return a + b
    @classmethod
    def sub(cls, a, b):
        return a - b
    @classmethod
    def mul(cls, a, b):
        return a * b
    @classmethod
    def div(cls, a, b):
        return a / b
    @classmethod
    def mod(cls, a, b):
        return a % b
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
    def floor(cls, a):
        return _math.floor(a)
    @classmethod
    def ceil(cls, a):
        return _math.ceil(a)
    @classmethod
    def trunc(cls, a):
        return _math.trunc(a)
    @classmethod
    def rint(cls, a):
        return _math.floor(a + 0.5) if (a > 0) else _math.ceil(a - 0.5)

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
    def csc(cls, a):
        return 1 / _math.sin(a)
    @classmethod
    def sec(cls, a):
        return 1 / _math.cos(a)
    @classmethod
    def cot(cls, a):
        return 1 / _math.tan(a)

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
    def acsc(cls, a):
        return _math.asin(1 / a)
    @classmethod
    def asec(cls, a):
        return _math.acos(1 / a)
    @classmethod
    def acot(cls, a):
        return _math.atan(1 / a)

    @classmethod
    def sinh(cls, a):
        return _math.sinh(a)
    @classmethod
    def cosh(cls, a):
        return _math.cosh(a)
    @classmethod
    def tanh(cls, a):
        return _math.tanh(a)
    @classmethod
    def csch(cls, a):
        return 1 / _math.sinh(a)
    @classmethod
    def sech(cls, a):
        return 1 / _math.cosh(a)
    @classmethod
    def coth(cls, a):
        return 1 / _math.tanh(a)

    @classmethod
    def asinh(cls, a):
        return _math.asinh(a)
    @classmethod
    def acosh(cls, a):
        return _math.acosh(a)
    @classmethod
    def atanh(cls, a):
        return _math.atanh(a)
    @classmethod
    def acsch(cls, a):
        return _math.asinh(1 / a)
    @classmethod
    def asech(cls, a):
        return _math.acosh(1 / a)
    @classmethod
    def acoth(cls, a):
        return _math.atanh(1 / a)

    @classmethod
    def atan2(cls, y, x):
        return _math.atan2(y, x)

    @classmethod
    def comparison_field(cls):
        return bool
    @classmethod
    def eq(cls, a, b, ulps=15):
        return _float_eq1(x, y)
    @classmethod
    def ne(cls, a, b):
        if _math.isnan(a) or _math.isnan(b):
            return False
        return not cls.eq(a, b)
    @classmethod
    def lt(cls, a, b):
        return a < b
    @classmethod
    def le(cls, a, b):
        # note this isn't the same as `a == b || a < b` over the field, since ==
        # allows imprecision.
        return a <= b

    @classmethod
    def isinf(cls, a):
        return _math.isinf(a)
    @classmethod
    def isnan(cls, a):
        return _math.isnan(a)

    @classmethod
    def issame(cls, a, b):
        # bytewise compare.
        return _struct.pack("d", a) == _struct.pack("d", b)

    @classmethod
    def hashed(cls, a):
        return hash(a.__float__())

    @classmethod
    def stringify(cls, a, short):
        return _cons.pretty_number(a.__float__(), short)



    @classmethod
    def _mat_numpyarr_int(cls, m):
        cells = m._cells
        # Check finite.
        if not _np.isfinite(cells).all():
            raise ValueError("cannot cast non-finite values to integer")
        # Check integer.
        if not (cells == _np.round(cells)).all():
            raise ValueError("cannot cast non-integer to integer")
        # Check bounds.
        hi = _math.nextafter(float((1 << 63) - 1), 0.0)
        lo = _math.nextafter(float(-(1 << 63)), 0.0)
        if ((cells < lo) | (cells > hi)).any():
            raise ValueError("cannot cast too-large value to int64, overflowed")
        return cells.astype(_np.int64)

    @classmethod
    def _mat_numpyarr_float(cls, m):
        cells = m._cells
        # justin caseme.
        if cells.dtype != _np.float64:
            cells = cells.astype(_np.float64)
        return cells

    @classmethod
    def _mat_numpyarr_complex(cls, m):
        return m._cells.astype(_np.complex128)

    @classmethod
    def _mat_fromfield(cls, m):
        # Take advantage of an overloading numpyarr float.
        if hasattr(m.field, "_mat_numpyarr_float"):
            cells = m.field._mat_numpyarr_float(m)
            return Matrix[cls, m.shape](cells)
        # Use the fields cast method.
        toobj = _np.vectorize(m.field.to_float, otypes=[cls.dtype()])
        return Matrix[cls, m.shape](toobj(m._cells))


    @classmethod
    def _mat_eye(cls, M):
        cells = _np.eye(M.shape[0], dtype=cls.dtype())
        return M(cells)
    @classmethod
    def _mat_zeros(cls, M):
        cells = _np.zeros(M.shape.tonumpy, dtype=cls.dtype())
        return M(cells)
    @classmethod
    def _mat_ones(cls, M):
        cells = _np.ones(M.shape.tonumpy, dtype=cls.dtype())
        return M(cells)
    @classmethod
    def _mat_full(cls, M, value):
        val = value.obj
        cells = _np.full(M.shape.tonumpy, val, dtype=cls.dtype())
        return M(cells)


    @classmethod
    def _mat_abs(cls, m):
        cells = m._cells.__abs__()
        return type(m)(cells)
    @classmethod
    def _mat_sgn(cls, m):
        cells = (m._cells < 0.0).astype(cls.dtype()) - (m._cells < 0.0)
        return type(m)(cells)

    @classmethod
    def _mat_neg(cls, m):
        cells = -m._cells
        return type(m)(cells)
    @classmethod
    def _mat_add(cls, m, o):
        cells = m._cells + o._cells
        return type(m)(cells)
    @classmethod
    def _mat_sub(cls, m, o):
        cells = m._cells - o._cells
        return type(m)(cells)
    @classmethod
    def _mat_mul(cls, m, o):
        cells = m._cells * o._cells
        return type(m)(cells)
    @classmethod
    def _mat_div(cls, m, o):
        cells = m._cells / o._cells
        return type(m)(cells)
    @classmethod
    def _mat_mod(cls, m, o):
        cells = m._cells % o._cells
        return type(m)(cells)

    @classmethod
    def _mat_exp(cls, m):
        cells = _np.exp(m._cells)
        return type(m)(cells)
    @classmethod
    def _mat_exp2(cls, m):
        cells = _np.exp2(m._cells)
        return type(m)(cells)
    @classmethod
    def _mat_exp10(cls, m):
        cells = 10 ** m._cells
        return type(m)(cells)
    @classmethod
    def _mat_pow(cls, m, o):
        cells = m._cells ** o._cells
        return type(m)(cells)

    @classmethod
    def _mat_ln(cls, m):
        cells = _np.log(m._cells)
        return type(m)(cells)
    @classmethod
    def _mat_log2(cls, m):
        cells = _np.log2(m._cells)
        return type(m)(cells)
    @classmethod
    def _mat_log10(cls, m):
        cells = _np.log10(m._cells)
        return type(m)(cells)
    @classmethod
    def _mat_log(cls, m, base):
        cells = _np.log(m) / _np.log(base)
        return type(m)(cells)

    @classmethod
    def _mat_sqrt(cls, m):
        cells = _np.sqrt(m._cells)
        return type(m)(cells)
    @classmethod
    def _mat_cbrt(cls, m):
        cells = _np.cbrt(m._cells)
        return type(m)(cells)
    @classmethod
    def _mat_root(cls, m, n):
        cells = m._cells ** (1 / n._cells)
        return type(m)(cells)

    @classmethod
    def _mat_floor(cls, m):
        cells = _np.floor(m._cells)
        return type(m)(cells)
    @classmethod
    def _mat_ceil(cls, m):
        cells = _np.ceil(m._cells)
        return type(m)(cells)
    @classmethod
    def _mat_trunc(cls, m):
        cells = _np.trunc(m._cells)
        return type(m)(cells)
    @classmethod
    def _mat_rint(cls, m):
        cells = _np.floor(m._cells + 0.5)
        cells[m._cells < 0] = _np.ceil(m._cells[m._cells < 0] - 0.5)
        return type(m)(cells)

    @classmethod
    def _mat_sin(cls, m):
        cells = _np.sin(m._cells)
        return type(m)(cells)
    @classmethod
    def _mat_cos(cls, m):
        cells = _np.cos(m._cells)
        return type(m)(cells)
    @classmethod
    def _mat_tan(cls, m):
        cells = _np.tan(m._cells)
        return type(m)(cells)
    @classmethod
    def _mat_csc(cls, m):
        cells = 1 / _np.sin(m._cells)
        return type(m)(cells)
    @classmethod
    def _mat_sec(cls, m):
        cells = 1 / _np.cos(m._cells)
        return type(m)(cells)
    @classmethod
    def _mat_cot(cls, m):
        cells = 1 / _np.tan(m._cells)
        return type(m)(cells)

    @classmethod
    def _mat_asin(cls, m):
        cells = _np.asin(m._cells)
        return type(m)(cells)
    @classmethod
    def _mat_acos(cls, m):
        cells = _np.acos(m._cells)
        return type(m)(cells)
    @classmethod
    def _mat_atan(cls, m):
        cells = _np.atan(m._cells)
        return type(m)(cells)
    @classmethod
    def _mat_acsc(cls, m):
        cells = _np.asin(1 / m._cells)
        return type(m)(cells)
    @classmethod
    def _mat_asec(cls, m):
        cells = _np.acos(1 / m._cells)
        return type(m)(cells)
    @classmethod
    def _mat_acot(cls, m):
        cells = _np.atan(1 / m._cells)
        return type(m)(cells)

    @classmethod
    def _mat_sinh(cls, m):
        cells = _np.sinh(m._cells)
        return type(m)(cells)
    @classmethod
    def _mat_cosh(cls, m):
        cells = _np.cosh(m._cells)
        return type(m)(cells)
    @classmethod
    def _mat_tanh(cls, m):
        cells = _np.tanh(m._cells)
        return type(m)(cells)
    @classmethod
    def _mat_csch(cls, m):
        cells = 1 / _np.sinh(m._cells)
        return type(m)(cells)
    @classmethod
    def _mat_sech(cls, m):
        cells = 1 / _np.cosh(m._cells)
        return type(m)(cells)
    @classmethod
    def _mat_coth(cls, m):
        cells = 1 / _np.tanh(m._cells)
        return type(m)(cells)

    @classmethod
    def _mat_asinh(cls, m):
        cells = _np.asinh(m._cells)
        return type(m)(cells)
    @classmethod
    def _mat_acosh(cls, m):
        cells = _np.acosh(m._cells)
        return type(m)(cells)
    @classmethod
    def _mat_atanh(cls, m):
        cells = _np.atanh(m._cells)
        return type(m)(cells)
    @classmethod
    def _mat_acsch(cls, m):
        cells = _np.asinh(1 / m._cells)
        return type(m)(cells)
    @classmethod
    def _mat_asech(cls, m):
        cells = _np.acosh(1 / m._cells)
        return type(m)(cells)
    @classmethod
    def _mat_acoth(cls, m):
        cells = _np.atanh(1 / m._cells)
        return type(m)(cells)

    @classmethod
    def _mat_atan2(cls, y, x):
        cells = _np.atan2(y._cells, x._cells)
        return type(y)(cells)

    @classmethod
    def _mat_torad(cls, m):
        cells = m._cells * (_math.pi / 180)
        return type(m)(cells)
    @classmethod
    def _mat_todeg(cls, m):
        cells = m._cells * (180 / _math.pi)
        return type(m)(cells)

    @classmethod
    def _mat_issame(cls, m, o):
        # byte-wise compare, but maintaining element size.
        cells = (m._cells.view(_np.void) == o._cells.view(_np.void))
        return Matrix[bool, m.shape](cells)
    @classmethod
    def _mat_eq(cls, m, o):
        cells = _float_eq(m._cells, o._cells)
        return Matrix[bool, m.shape](cells)
    @classmethod
    def _mat_ne(cls, m, o):
        cells = ~_float_eq(m._cells, o._cells)
        cells &= (~_np.isnan(m._cells) & ~_np.isnan(o._cells))
        return Matrix[bool, m.shape](cells)
    @classmethod
    def _mat_lt(cls, m, o):
        cells = (m._cells < o._cells)
        return Matrix[bool, m.shape](cells)
    @classmethod
    def _mat_le(cls, m, o):
        cells = (m._cells <= o._cells)
        return Matrix[bool, m.shape](cells)

    @classmethod
    def _mat_isinf(cls, m):
        cells = _np.isinf(m._cells)
        return Matrix[bool, m.shape](cells)
    @classmethod
    def _mat_isnan(cls, m):
        cells = _np.isnan(m._cells)
        return Matrix[bool, m.shape](cells)
    @classmethod
    def _mat_isfinite(cls, m):
        cells = _np.isfinite(m._cells)
        return Matrix[bool, m.shape](cells)

    @classmethod
    def _mat_det(cls, m):
        if m.isempty:
            return m.one
        val = _np.linalg.det(m._cells)
        return single(val, field=cls, _nocast=True)

    @classmethod
    def _mat_trace(cls, m):
        val = _np.trace(m._cells)
        return single(val, field=cls, _nocast=True)

    @classmethod
    def _mat_norm(cls, m):
        a = m._cells.reshape(-1)
        val = _np.linalg.norm(a)
        return single(val, field=cls, _nocast=True)

    @classmethod
    def _mat_dot(cls, m, o):
        if m.isempty:
            return m.zero
        a = m._cells.reshape(-1)
        b = o._cells.reshape(-1)
        val = _np.vdot(a, b)
        return single(val, field=cls, _nocast=True)

    @classmethod
    def _mat_cross(cls, m, o):
        a = m._cells.reshape(-1)
        b = o._cells.reshape(-1)
        val = _np.cross(a, b)
        return single(val, field=cls, _nocast=True)

    @classmethod
    def _mat_matmul(cls, m, o):
        newshape = Shape(m.shape[0], o.shape[1])
        cells = m._cells @ o._cells
        return Matrix[cls, newshape](cells)

    @classmethod
    def _mat_summ(cls, m, axis):
        if m.isempty:
            return m.zero if axis is ... else m
        # Note we ignore nans, bc who tf wants nans.
        if axis is ...:
            val = _np.nansum(m._cells)
            return single(val, field=cls, _nocast=True)
        if axis >= m.ndim:
            return m
        # Use numpy-equiv axis and dont wipe that axis once sumed along.
        npaxis = Permuter.tonumpyaxis(m.ndim, axis)
        cells = _np.nansum(m._cells, axis=npaxis, keepdims=True)
        return m.fromnumpy(cells)
    @classmethod
    def _mat_prod(cls, m, axis):
        if m.isempty:
            return m.one if axis is ... else m
        if axis is ...:
            val = _np.nanprod(m._cells)
            return single(val, field=cls, _nocast=True)
        if axis >= m.ndim:
            return m
        npaxis = Permuter.tonumpyaxis(m.ndim, axis)
        cells = _np.nanprod(m._cells, axis=npaxis, keepdims=True)
        return m.fromnumpy(cells)
    @classmethod
    def _mat_minn(cls, m, axis):
        if axis is ...:
            val = _np.nanmin(m._cells)
            return single(val, field=cls, _nocast=True)
        if axis >= m.ndim:
            return m
        npaxis = Permuter.tonumpyaxis(m.ndim, axis)
        cells = _np.nanmin(m._cells, axis=npaxis, keepdims=True)
        return m.fromnumpy(cells)

    @classmethod
    def _mat_maxx(cls, m, axis):
        if axis is ...:
            val = _np.nanmax(m._cells)
            return single(val, field=cls, _nocast=True)
        if axis >= m.ndim:
            return m
        npaxis = Permuter.tonumpyaxis(m.ndim, axis)
        cells = _np.nanmax(m._cells, axis=npaxis, keepdims=True)
        return m.fromnumpy(cells)

    @classmethod
    def _mat_rref(cls, m):
        return super()._mat_rref(m, imprecise=True)




class Complex(Field):
    @classmethod
    def dtype(cls):
        return _np.dtype(complex)

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
        if not (a.imag == 0.0):
            raise ValueError(f"cannot cast complex to int, got: {a}")
        if not a.real.is_integer():
            raise ValueError(f"cannot cast non-integer to int, got: {a}")
        return a.real.__int__()
    @classmethod
    def to_float(cls, a):
        if not (a.imag == 0.0):
            raise ValueError(f"cannot cast complex to float, got: {a}")
        return a.real.__float__()
    @classmethod
    def to_complex(cls, a):
        return a.__complex__()

    @classmethod
    def consts(cls):
        return super().consts() | {
            "__0__": complex(0.0),
            "__1__": complex(1.0),
            "__i__": 1j,
            "__e__": complex(_math.e),
            "__pi__": complex(_math.pi),
            "__inf__": complex("inf"),
            "__nan__": complex("nan+nanj"),
        }

    @classmethod
    def real(cls, a):
        return complex(a.real)
    @classmethod
    def imag(cls, a):
        return complex(a.imag)
    @classmethod
    def abs(cls, a):
        return complex(abs(a))
    @classmethod
    def angle(cls, a):
        if a == 0:
            return complex(0)
        return complex(_math.atan2(a.imag, a.real))
    @classmethod
    def sgn(cls, a):
        if a == 0:
            return complex(0)
        return a / abs(a)

    @classmethod
    def add(cls, a, b):
        return a + b
    @classmethod
    def sub(cls, a, b):
        return a - b
    @classmethod
    def mul(cls, a, b):
        return a * b
    @classmethod
    def div(cls, a, b):
        return a / b
    @classmethod
    def mod(cls, a, b):
        if a.imag == 0.0 and b.imag == 0.0:
            return a % b
        raise TypeError(f"cannot do modulus over complex values, got: {a}, and: "
                f"{b}")
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
    def floor(cls, a):
        return _math.floor(a.real) + 1j * _math.floor(a.imag)
    @classmethod
    def ceil(cls, a):
        return _math.ceil(a.real) + 1j * _math.ceil(a.imag)
    @classmethod
    def trunc(cls, a):
        return _math.trunc(a.real) + 1j * _math.trunc(a.imag)
    @classmethod
    def rint(cls, a):
        rnd = lambda x: _math.floor(x + 0.5) if (x > 0) else _math.ceil(x - 0.5)
        return rnd(a.real) + 1j * rnd(a.imag)

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
    def csc(cls, a):
        return 1 / _cmath.sin(a)
    @classmethod
    def sec(cls, a):
        return 1 / _cmath.cos(a)
    @classmethod
    def cot(cls, a):
        return 1 / _cmath.tan(a)

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
    def acsc(cls, a):
        return _cmath.asin(1 / a)
    @classmethod
    def asec(cls, a):
        return _cmath.acos(1 / a)
    @classmethod
    def acot(cls, a):
        return _cmath.atan(1 / a)

    @classmethod
    def sinh(cls, a):
        return _cmath.sinh(a)
    @classmethod
    def cosh(cls, a):
        return _cmath.cosh(a)
    @classmethod
    def tanh(cls, a):
        return _cmath.tanh(a)
    @classmethod
    def csch(cls, a):
        return 1 / _cmath.sinh(a)
    @classmethod
    def sech(cls, a):
        return 1 / _cmath.cosh(a)
    @classmethod
    def coth(cls, a):
        return 1 / _cmath.tanh(a)

    @classmethod
    def asinh(cls, a):
        return _cmath.asinh(a)
    @classmethod
    def acosh(cls, a):
        return _cmath.acosh(a)
    @classmethod
    def atanh(cls, a):
        return _cmath.atanh(a)
    @classmethod
    def acsch(cls, a):
        return _cmath.asinh(1 / a)
    @classmethod
    def asech(cls, a):
        return _cmath.acosh(1 / a)
    @classmethod
    def acoth(cls, a):
        return _cmath.atanh(1 / a)

    @classmethod
    def atan2(cls, y, x):
        if y.imag == 0.0 and x.imag == 0.0:
            return complex(_math.atan2(y.real, x.real))
        raise TypeError("cannot perform quadrant-aware atan on complex values, "
                f"got: {y}, and: {x}")

    @classmethod
    def comparison_field(cls):
        return bool
    @classmethod
    def eq(cls, a, b, ulps=15):
        return _float_eq1(a.real, b.real) and _float_eq1(a.imag, b.imag)
    @classmethod
    def ne(cls, a, b):
        if _cmath.isnan(a) or _cmath.isnan(b):
            return False
        return not cls.eq(a, b)
    @classmethod
    def lt(cls, a, b):
        if a.imag == 0.0 and b.imag == 0.0:
            return a.real < b.real
        raise TypeError(f"cannot order complex values, got: {a}, and: {b}")
    @classmethod
    def le(cls, a, b):
        if a.imag == 0.0 and b.imag == 0.0:
            return a.real <= b.real
        raise TypeError(f"cannot order complex values, got: {a}, and: {b}")

    @classmethod
    def isinf(cls, a):
        return _cmath.isinf(a)
    @classmethod
    def isnan(cls, a):
        return _cmath.isnan(a)

    @classmethod
    def issame(cls, a, b):
        realeq = (_struct.pack("d", a.real) == _struct.pack("d", b.real))
        imageq = (_struct.pack("d", a.imag) == _struct.pack("d", b.imag))
        return realeq and imageq

    @classmethod
    def hashed(cls, a):
        return hash(a.__complex__())

    @classmethod
    def stringify(cls, a, short):
        return _cons.pretty_number(a.__complex__(), short)



    @classmethod
    def _mat_numpyarr_int(cls, m):
        cells = m._cells
        # Check its all real.
        if not (cells.imag == 0.0).all():
            raise ValueError("cannot cast complex values to integer, got non-"
                    "zero imaginary components")
        # Forward to floating.
        cells = cells.real
        return Matrix[Float, m.shape](cells).numpyarr(int)

    @classmethod
    def _mat_numpyarr_float(cls, m):
        cells = m._cells
        # Check its all real.
        if not (cells.imag == 0.0).all():
            raise ValueError("cannot cast complex values to floating, got non-"
                    "zero imaginary components")
        cells = cells.real
        # justin caseme.
        if cells.dtype != _np.float64:
            cells = cells.astype(_np.float64)
        return cells

    @classmethod
    def _mat_numpyarr_complex(cls, m):
        cells = m._cells
        # justin caseme.
        if cells.dtype != _np.complex128:
            cells = cells.astype(_np.complex128)
        return cells

    @classmethod
    def _mat_fromfield(cls, m):
        # Take advantage of an overloading numpyarr complex.
        if hasattr(m.field, "_mat_numpyarr_complex"):
            cells = m.field._mat_numpyarr_complex(m)
            return Matrix[cls, m.shape](cells)
        # Use the fields cast method.
        toobj = _np.vectorize(m.field.to_complex, otypes=[cls.dtype()])
        return Matrix[cls, m.shape](toobj(m._cells))


    @classmethod
    def _mat_eye(cls, M):
        cells = _np.eye(M.shape[0], dtype=cls.dtype())
        return M(cells)
    @classmethod
    def _mat_zeros(cls, M):
        cells = _np.zeros(M.shape.tonumpy, dtype=cls.dtype())
        return M(cells)
    @classmethod
    def _mat_ones(cls, M):
        cells = _np.ones(M.shape.tonumpy, dtype=cls.dtype())
        return M(cells)
    @classmethod
    def _mat_full(cls, M, value):
        val = value.obj
        cells = _np.full(M.shape.tonumpy, val, dtype=cls.dtype())
        return M(cells)


    @classmethod
    def _mat_real(cls, m):
        # again, taking real collapses to float64.
        cells = m._cells.real.astype(cls.dtype())
        return type(m)(cells)
    @classmethod
    def _mat_imag(cls, m):
        cells = m._cells.imag.astype(cls.dtype())
        return type(m)(cells)
    @classmethod
    def _mat_conj(cls, m):
        cells = _np.conj(m._cells)
        return type(m)(cells)

    @classmethod
    def _mat_abs(cls, m):
        # numpy abs returns float64.
        cells = m._cells.__abs__().astype(cls.dtype())
        return type(m)(cells)
    @classmethod
    def _mat_sgn(cls, m):
        cells = m._cells.copy()
        nonz = m._cells[m._cells != 0]
        cells[m._cells != 0] = nonz / nonz.__abs__()
        return type(m)(cells)

    @classmethod
    def _mat_neg(cls, m):
        cells = -m._cells
        return type(m)(cells)
    @classmethod
    def _mat_add(cls, m, o):
        cells = m._cells + o._cells
        return type(m)(cells)
    @classmethod
    def _mat_sub(cls, m, o):
        cells = m._cells - o._cells
        return type(m)(cells)
    @classmethod
    def _mat_mul(cls, m, o):
        cells = m._cells * o._cells
        return type(m)(cells)
    @classmethod
    def _mat_div(cls, m, o):
        cells = m._cells / o._cells
        return type(m)(cells)
    @classmethod
    def _mat_mod(cls, m, o):
        a = m._cells
        b = o._cells
        if (a.imag == 0.0).all() and (b.imag == 0.0).all():
            cells = (a.real % b.real).astype(cls.dtype())
            return type(m)(cells)
        raise ValueError("cannot do modulus over complex values, got non-zero "
                "imaginary components")

    @classmethod
    def _mat_exp(cls, m):
        cells = _np.exp(m._cells)
        return type(m)(cells)
    @classmethod
    def _mat_exp2(cls, m):
        cells = _np.exp2(m._cells)
        return type(m)(cells)
    @classmethod
    def _mat_exp10(cls, m):
        cells = 10 ** m._cells
        return type(m)(cells)
    @classmethod
    def _mat_pow(cls, m, o):
        cells = m._cells ** o._cells
        return type(m)(cells)

    @classmethod
    def _mat_ln(cls, m):
        cells = _np.log(m._cells)
        return type(m)(cells)
    @classmethod
    def _mat_log2(cls, m):
        cells = _np.log2(m._cells)
        return type(m)(cells)
    @classmethod
    def _mat_log10(cls, m):
        cells = _np.log10(m._cells)
        return type(m)(cells)
    @classmethod
    def _mat_log(cls, m, base):
        cells = _np.log(m) / _np.log(base)
        return type(m)(cells)

    @classmethod
    def _mat_sqrt(cls, m):
        cells = _np.sqrt(m._cells)
        return type(m)(cells)
    @classmethod
    def _mat_cbrt(cls, m):
        cells = _np.cbrt(m._cells)
        return type(m)(cells)
    @classmethod
    def _mat_root(cls, m, n):
        cells = m._cells ** (1 / n._cells)
        return type(m)(cells)

    @classmethod
    def _mat_floor(cls, m):
        cells = _np.floor(m._cells.real) + 1j * _np.floor(m._cells.imag)
        return type(m)(cells)
    @classmethod
    def _mat_ceil(cls, m):
        cells = _np.ceil(m._cells.real) + 1j * _np.ceil(m._cells.imag)
        return type(m)(cells)
    @classmethod
    def _mat_trunc(cls, m):
        cells = _np.trunc(m._cells.real) + 1j * _np.trunc(m._cells.imag)
        return type(m)(cells)
    @classmethod
    def _mat_rint(cls, m):
        def rnd(x):
            y = _np.floor(x + 0.5)
            y[x < 0] = _np.ceil(x[x < 0] - 0.5)
            return y
        cells = rnd(m._cells.real) + 1j * rnd(m._cells.imag)
        return type(m)(cells)

    @classmethod
    def _mat_sin(cls, m):
        cells = _np.sin(m._cells)
        return type(m)(cells)
    @classmethod
    def _mat_cos(cls, m):
        cells = _np.cos(m._cells)
        return type(m)(cells)
    @classmethod
    def _mat_tan(cls, m):
        cells = _np.tan(m._cells)
        return type(m)(cells)
    @classmethod
    def _mat_csc(cls, m):
        cells = 1 / _np.sin(m._cells)
        return type(m)(cells)
    @classmethod
    def _mat_sec(cls, m):
        cells = 1 / _np.cos(m._cells)
        return type(m)(cells)
    @classmethod
    def _mat_cot(cls, m):
        cells = 1 / _np.tan(m._cells)
        return type(m)(cells)

    @classmethod
    def _mat_asin(cls, m):
        cells = _np.asin(m._cells)
        return type(m)(cells)
    @classmethod
    def _mat_acos(cls, m):
        cells = _np.acos(m._cells)
        return type(m)(cells)
    @classmethod
    def _mat_atan(cls, m):
        cells = _np.atan(m._cells)
        return type(m)(cells)
    @classmethod
    def _mat_acsc(cls, m):
        cells = _np.asin(1 / m._cells)
        return type(m)(cells)
    @classmethod
    def _mat_asec(cls, m):
        cells = _np.acos(1 / m._cells)
        return type(m)(cells)
    @classmethod
    def _mat_acot(cls, m):
        cells = _np.atan(1 / m._cells)
        return type(m)(cells)

    @classmethod
    def _mat_sinh(cls, m):
        cells = _np.sinh(m._cells)
        return type(m)(cells)
    @classmethod
    def _mat_cosh(cls, m):
        cells = _np.cosh(m._cells)
        return type(m)(cells)
    @classmethod
    def _mat_tanh(cls, m):
        cells = _np.tanh(m._cells)
        return type(m)(cells)
    @classmethod
    def _mat_csch(cls, m):
        cells = 1 / _np.sinh(m._cells)
        return type(m)(cells)
    @classmethod
    def _mat_sech(cls, m):
        cells = 1 / _np.cosh(m._cells)
        return type(m)(cells)
    @classmethod
    def _mat_coth(cls, m):
        cells = 1 / _np.tanh(m._cells)
        return type(m)(cells)

    @classmethod
    def _mat_asinh(cls, m):
        cells = _np.asinh(m._cells)
        return type(m)(cells)
    @classmethod
    def _mat_acosh(cls, m):
        cells = _np.acosh(m._cells)
        return type(m)(cells)
    @classmethod
    def _mat_atanh(cls, m):
        cells = _np.atanh(m._cells)
        return type(m)(cells)
    @classmethod
    def _mat_acsch(cls, m):
        cells = _np.asinh(1 / m._cells)
        return type(m)(cells)
    @classmethod
    def _mat_asech(cls, m):
        cells = _np.acosh(1 / m._cells)
        return type(m)(cells)
    @classmethod
    def _mat_acoth(cls, m):
        cells = _np.atanh(1 / m._cells)
        return type(m)(cells)

    @classmethod
    def _mat_atan2(cls, y, x):
        b = y._cells
        a = x._cells
        if not ((a.imag == 0.0).all() and (b.imag == 0.0).all()):
            raise ValueError("cannot do quadrant-aware atan over complex "
                    "values, got non-zero imaginary components")
        cells = _np.atan2(b.real, a.real)
        # complex me.
        cells = cells.astype(cls.dtype())
        return type(y)(cells)

    @classmethod
    def _mat_torad(cls, m):
        cells = m._cells * (_math.pi / 180)
        return type(m)(cells)
    @classmethod
    def _mat_todeg(cls, m):
        cells = m._cells * (180 / _math.pi)
        return type(m)(cells)

    @classmethod
    def _mat_issame(cls, m, o):
        # byte-wise compare, but maintaining element size.
        cells = (m._cells.view(_np.void) == o._cells.view(_np.void))
        return Matrix[bool, m.shape](cells)
    @classmethod
    def _mat_eq(cls, m, o):
        mre = m._cells.real
        mim = m._cells.imag
        ore = o._cells.real
        oim = o._cells.imag
        cells = (_float_eq(mre, ore) & _float_eq(mim, oim))
        return Matrix[bool, m.shape](cells)
    @classmethod
    def _mat_ne(cls, m, o):
        mre = m._cells.real
        mim = m._cells.imag
        ore = o._cells.real
        oim = o._cells.imag
        cells = ((~_float_eq(mre, ore)) | (~_float_eq(mim, oim)))
        cells &= ((~_np.isnan(m._cells)) & (~_np.isnan(o._cells)))
        return Matrix[bool, m.shape](cells)
    @classmethod
    def _mat_lt(cls, m, o):
        mim = m._cells.imag
        oim = o._cells.imag
        if not ((mim == 0.0).all() and (oim == 0.0).all()):
            raise TypeError("cannot order complex (values were not all real)")
        mre = m._cells.real
        ore = o._cells.real
        cells = (mre < ore)
        return Matrix[bool, m.shape](cells)
    @classmethod
    def _mat_le(cls, m, o):
        mim = m._cells.imag
        oim = o._cells.imag
        if not ((mim == 0.0).all() and (oim == 0.0).all()):
            raise TypeError("cannot order complex (values were not all real)")
        mre = m._cells.real
        ore = o._cells.real
        cells = (mre <= ore)
        return Matrix[bool, m.shape](cells)

    @classmethod
    def _mat_isinf(cls, m):
        cells = _np.isinf(m._cells)
        return Matrix[bool, m.shape](cells)
    @classmethod
    def _mat_isnan(cls, m):
        cells = _np.isnan(m._cells)
        return Matrix[bool, m.shape](cells)
    @classmethod
    def _mat_isfinite(cls, m):
        cells = _np.isfinite(m._cells)
        return Matrix[bool, m.shape](cells)

    @classmethod
    def _mat_det(cls, m):
        if m.isempty:
            return m.one
        val = _np.linalg.det(m._cells)
        return single(val, field=cls, _nocast=True)

    @classmethod
    def _mat_trace(cls, m):
        val = _np.trace(m._cells)
        return single(val, field=cls, _nocast=True)

    @classmethod
    def _mat_norm(cls, m):
        a = m._cells.reshape(-1)
        val = _np.linalg.norm(a)
        return single(val, field=cls, _nocast=True)

    @classmethod
    def _mat_dot(cls, m, o):
        if m.isempty:
            return m.zero
        a = m._cells.reshape(-1)
        b = o._cells.reshape(-1)
        val = _np.vdot(a, b)
        return single(val, field=cls, _nocast=True)

    @classmethod
    def _mat_cross(cls, m, o):
        a = m._cells.reshape(-1)
        b = o._cells.reshape(-1)
        val = _np.cross(a, b)
        return single(val, field=cls, _nocast=True)

    @classmethod
    def _mat_matmul(cls, m, o):
        newshape = Shape(m.shape[0], o.shape[1])
        cells = m._cells @ o._cells
        return Matrix[cls, newshape](cells)

    @classmethod
    def _mat_summ(cls, m, axis):
        if m.isempty:
            return m.zero if axis is ... else m
        # Note we ignore nans, bc who tf wants nans.
        if axis is ...:
            val = _np.nansum(m._cells)
            return single(val, field=cls, _nocast=True)
        if axis >= m.ndim:
            return m
        # Use numpy-equiv axis and dont wipe that axis once sumed along.
        npaxis = Permuter.tonumpyaxis(m.ndim, axis)
        cells = _np.nansum(m._cells, axis=npaxis, keepdims=True)
        return m.fromnumpy(cells)
    @classmethod
    def _mat_prod(cls, m, axis):
        if m.isempty:
            return m.one if axis is ... else m
        if axis is ...:
            val = _np.nanprod(m._cells)
            return single(val, field=cls, _nocast=True)
        if axis >= m.ndim:
            return m
        npaxis = Permuter.tonumpyaxis(m.ndim, axis)
        cells = _np.nanprod(m._cells, axis=npaxis, keepdims=True)
        return m.fromnumpy(cells)
    @classmethod
    def _mat_minn(cls, m, axis):
        if axis is ...:
            val = _np.nanmin(m._cells)
            return single(val, field=cls, _nocast=True)
        if axis >= m.ndim:
            return m
        npaxis = Permuter.tonumpyaxis(m.ndim, axis)
        cells = _np.nanmin(m._cells, axis=npaxis, keepdims=True)
        return m.fromnumpy(cells)

    @classmethod
    def _mat_maxx(cls, m, axis):
        if axis is ...:
            val = _np.nanmax(m._cells)
            return single(val, field=cls, _nocast=True)
        if axis >= m.ndim:
            return m
        npaxis = Permuter.tonumpyaxis(m.ndim, axis)
        cells = _np.nanmax(m._cells, axis=npaxis, keepdims=True)
        return m.fromnumpy(cells)

    @classmethod
    def _mat_rref(cls, m):
        return super()._mat_rref(m, imprecise=True)


class Bool(Field):
    @classmethod
    def dtype(cls):
        return _np.dtype(bool)

    @classmethod
    def fromobj(cls, x):
        if isinstance(x, bool):
            return x.__bool__()
        # dont use default __bool__ for int/float/complex, instead force a
        # comparison.
        if isinstance(x, (int, float, complex)):
            raise TypeError(f"cannot create {_tname(cls)} implicitly from "
                    f"{_objtname(x)}, you must use an explicit comparison")
        if hasattr(x, "__bool__"):
            return x.__bool__()
        return super().fromobj(x)

    @classmethod
    def to_bool(cls, a):
        return a.__bool__()
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
            "__0__": False,
            "__1__": True,
        }

    @classmethod
    def always(cls, a):
        return a
    @classmethod
    def anytime(cls, a):
        return a
    @classmethod
    def logical_not(cls, a):
        return not a
    @classmethod
    def logical_and(cls, a, b):
        return a and b
    @classmethod
    def logical_or(cls, a, b):
        return a or b

    @classmethod
    def comparison_field(cls):
        return bool # shi man thats me.
    @classmethod
    def eq(cls, a, b):
        return a == b
    @classmethod
    def ne(cls, a, b):
        return a != b
    @classmethod
    def lt(cls, a, b):
        return a < b
    @classmethod
    def le(cls, a, b):
        return a <= b

    @classmethod
    def issame(cls, a, b):
        return a == b

    @classmethod
    def hashed(cls, a):
        return hash(a.__bool__())

    @classmethod
    def stringify(cls, a, short):
        return _cons.pretty_number(a.__bool__(), short)



    @classmethod
    def _mat_numpyarr_int(cls, m):
        return m._cells.astype(_np.int64)
    @classmethod
    def _mat_numpyarr_float(cls, m):
        return m._cells.astype(_np.float64)
    @classmethod
    def _mat_numpyarr_complex(cls, m):
        return m._cells.astype(_np.complex128)

    @classmethod
    def _mat_eye(cls, M):
        cells = _np.eye(M.shape[0], dtype=cls.dtype())
        return M(cells)
    @classmethod
    def _mat_zeros(cls, M):
        cells = _np.zeros(M.shape.tonumpy, dtype=cls.dtype())
        return M(cells)
    @classmethod
    def _mat_ones(cls, M):
        cells = _np.ones(M.shape.tonumpy, dtype=cls.dtype())
        return M(cells)
    @classmethod
    def _mat_full(cls, M, value):
        val = value.obj
        cells = _np.full(M.shape.tonumpy, val, dtype=cls.dtype())
        return M(cells)

    @classmethod
    def _mat_all(cls, m):
        return bool(m._cells.all())
    @classmethod
    def _mat_any(cls, m):
        return bool(m._cells.any())

    @classmethod
    def _mat_always(cls, m):
        return m
    @classmethod
    def _mat_anytime(cls, m):
        return m

    @classmethod
    def _mat_not(cls, m):
        cells = (~m._cells)
        return type(m)(cells)
    @classmethod
    def _mat_and(cls, m, o):
        cells = (m._cells & o._cells)
        return type(m)(cells)
    @classmethod
    def _mat_or(cls, m, o):
        cells = (m._cells | o._cells)
        return type(m)(cells)




# Map the python types to their proper field.
toField.map_nonfield_to(int, Int)
toField.map_nonfield_to(float, Float)
toField.map_nonfield_to(complex, Complex)
toField.map_nonfield_to(bool, Bool)



class Single:
    def __getitem__(self, field):
        return Matrix[field, Shape.single]
Single = Single()
Single.__doc__ = """
Refers to the single (1x1) matrix type over the given field. Note this is not a
proper templated class, only a thin wrapper.
"""

def single(x, *, field=None, _nocast=False):
    """
    Single (1x1) matrix of the given object.
    """
    if _nocast:
        assert field is not None
        xfield = field
    else:
        xfield = toField(type(x))
    cells = _np.empty((1, 1), dtype=xfield.dtype())
    cells[0, 0] = x
    mat = Single[xfield](cells)
    if not _nocast and field is not None:
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
    field = toField(field)
    cells = _np.array([], dtype=field.dtype())
    return Empty[field](cells)

def isempty(x):
    """
    True iff 'x' is a matrix with no cells.
    """
    return isinstance(x, Matrix) and x.isempty



def castall(ms, broadcast=True, *, field=None):
    """
    When given an sequence of values/matrices, returns them as matrices cast to
    the same field and (optionally) broadcast to the same shape.
    """
    if not _iterable(ms):
        raise TypeError(f"expected an iterable 'ms', got {_objtname(ms)}")
    if not isinstance(ms, (list, tuple)):
        ms = tuple(ms)
    if not ms:
        return ()
    for m in ms:
        if isinstance(m, Matrix):
            Mat = type(m)
            break
    else:
        Mat = Single[_get_field(field)]
    return Mat.cast(ms, broadcast=broadcast)



def drop(m, *axes, field=None):
    """
    Alias for 'm.drop(*axes)' if given axes, otherwise alias for 'm.dropall'.
    """
    m, = castall([m], field=field)
    if len(axes) == 0:
        return m.dropall
    return m.drop(*axes)
def flip(m, *axes, field=None):
    """
    Alias for 'm.flip(*axes)' if given axes, otherwise alias for 'm.flipall'.
    """
    m, = castall([m], field=field)
    if len(axes) == 0:
        return m.flipall
    return m.flip(*axes)
def flipud(m, *, field=None):
    """
    Alias for 'm.flipud'.
    """
    m, = castall([m], field=field)
    return m.flipud
def fliplr(m, *, field=None):
    """
    Alias for 'm.fliplr'.
    """
    m, = castall([m], field=field)
    return m.fliplr
def tile(m, *counts, axis=None, field=None):
    """
    Alias for 'm.tile(*counts, axis=axis)'.
    """
    m, = castall([m], field=field)
    return m.tile(*counts, axis=axis)
def repeat(m, *counts, axis=None, field=None):
    """
    Alias for 'm.repeat(*counts, axis=axis)'.
    """
    m, = castall([m], field=field)
    return m.repeat(*counts, axis=axis)
def roll(m, *shifts, axis=None, field=None):
    """
    Alias for 'm.roll(*shifts, axis=axis)'.
    """
    m, = castall([m], field=field)
    return m.roll(*shifts, axis=axis)

def inv(m, *, field=None):
    """
    Alias for 'm.inv'.
    """
    m, = castall([m], field=field)
    return m.inv
def det(m, *, field=None):
    """
    Alias for 'm.det'.
    """
    m, = castall([m], field=field)
    return m.det
def trace(m, *, field=None):
    """
    Alias for 'm.trace'.
    """
    m, = castall([m], field=field)
    return m.trace
def norm(m, *, field=None):
    """
    Alias for 'm.norm'.
    """
    m, = castall([m], field=field)
    return m.norm
def dot(m, y, *, field=None):
    """
    Alias for 'm.dot(y)'.
    """
    m, y = castall([m, y], field=field, broadcast=False)
    return m.dot(y)
def cross(m, y, *, field=None):
    """
    Alias for 'm.cross(y)'.
    """
    m, y = castall([m, y], field=field, broadcast=False)
    return m.cross(y)

def alll(*ms, field=None):
    """
    Alias for 'ms[0].alll and ms[1].alll and ...'.
    """
    ms = _maybe_unpack_seq(ms)
    ms = castall(ms, field=field)
    return all([m.alll for m in ms]) # dont shortcircuit.
def anyy(*ms, field=None):
    """
    Alias for 'ms[0].anyy or ms[1].anyy or ...'.
    """
    ms = _maybe_unpack_seq(ms)
    ms = castall(ms, field=field)
    return any([m.anyy for m in ms]) # dont shortcircuit.
def always(m, *, field=None):
    """
    Alias for 'm.always'.
    """
    m, = castall([m], field=field)
    return m.always
def anytime(m, *, field=None):
    """
    Alias for 'm.anytime'.
    """
    m, = castall([m], field=field)
    return m.anytime

def real(m, *, field=None):
    """
    Alias for 'm.real'.
    """
    m, = castall([m], field=field)
    return m.real
def imag(m, *, field=None):
    """
    Alias for 'm.imag'.
    """
    m, = castall([m], field=field)
    return m.imag
def conj(m, *, field=None):
    """
    Alias for 'm.conj'.
    """
    m, = castall([m], field=field)
    return m.conj
def angle(m, *, field=None):
    """
    Alias for 'm.angle'.
    """
    m, = castall([m], field=field)
    return m.angle
def sgn(m, *, field=None):
    """
    Alias for 'm.sgn'.
    """
    m, = castall([m], field=field)
    return m.sgn

def sqrt(m, *, field=None):
    """
    Alias for 'm.sqrt'.
    """
    m, = castall([m], field=field)
    return m.sqrt
def cbrt(m, *, field=None):
    """
    Alias for 'm.cbrt'.
    """
    m, = castall([m], field=field)
    return m.cbrt
def root(base, m, *, field=None):
    """
    Alias for 'm.root(base)'.
    """
    m, = castall([m], field=field)
    return m.root(base)
def exp(m, *, field=None):
    """
    Alias for 'm.exp'.
    """
    m, = castall([m], field=field)
    return m.exp
def exp2(m, *, field=None):
    """
    Alias for 'm.exp2'.
    """
    m, = castall([m], field=field)
    return m.exp2
def exp10(m, *, field=None):
    """
    Alias for 'm.exp10'.
    """
    m, = castall([m], field=field)
    return m.exp10
def ln(m, *, field=None):
    """
    Alias for 'm.ln'.
    """
    m, = castall([m], field=field)
    return m.ln
def log2(m, *, field=None):
    """
    Alias for 'm.log2'.
    """
    m, = castall([m], field=field)
    return m.log2
def log10(m, *, field=None):
    """
    Alias for 'm.log10'.
    """
    m, = castall([m], field=field)
    return m.log10
def log(base, m, *, field=None):
    """
    Alias for 'm.log(base)'.
    """
    m, = castall([m], field=field)
    return m.log(base)
def floor(m, *, field=None):
    """
    Alias for 'm.floor'.
    """
    m, = castall([m], field=field)
    return m.floor
def ceil(m, *, field=None):
    """
    Alias for 'm.ceil'.
    """
    m, = castall([m], field=field)
    return m.ceil
def trunc(m, *, field=None):
    """
    Alias for 'm.trunc'.
    """
    m, = castall([m], field=field)
    return m.trunc
def rint(m, *, field=None):
    """
    Alias for 'm.rint'.
    """
    m, = castall([m], field=field)
    return m.rint
def sin(m, *, field=None):
    """
    Alias for 'm.sin'.
    """
    m, = castall([m], field=field)
    return m.sin
def cos(m, *, field=None):
    """
    Alias for 'm.cos'.
    """
    m, = castall([m], field=field)
    return m.cos
def tan(m, *, field=None):
    """
    Alias for 'm.tan'.
    """
    m, = castall([m], field=field)
    return m.tan
def csc(m, *, field=None):
    """
    Alias for 'm.csc'.
    """
    m, = castall([m], field=field)
    return m.csc
def sec(m, *, field=None):
    """
    Alias for 'm.sec'.
    """
    m, = castall([m], field=field)
    return m.sec
def cot(m, *, field=None):
    """
    Alias for 'm.cot'.
    """
    m, = castall([m], field=field)
    return m.cot
def asin(m, *, field=None):
    """
    Alias for 'm.asin'.
    """
    m, = castall([m], field=field)
    return m.asin
def acos(m, *, field=None):
    """
    Alias for 'm.acos'.
    """
    m, = castall([m], field=field)
    return m.acos
def atan(m, *, field=None):
    """
    Alias for 'm.atan'.
    """
    m, = castall([m], field=field)
    return m.atan
def acsc(m, *, field=None):
    """
    Alias for 'm.acsc'.
    """
    m, = castall([m], field=field)
    return m.acsc
def asec(m, *, field=None):
    """
    Alias for 'm.asec'.
    """
    m, = castall([m], field=field)
    return m.asec
def acot(m, *, field=None):
    """
    Alias for 'm.acot'.
    """
    m, = castall([m], field=field)
    return m.acot
def sinh(m, *, field=None):
    """
    Alias for 'm.sinh'.
    """
    m, = castall([m], field=field)
    return m.sinh
def cosh(m, *, field=None):
    """
    Alias for 'm.cosh'.
    """
    m, = castall([m], field=field)
    return m.cosh
def tanh(m, *, field=None):
    """
    Alias for 'm.tanh'.
    """
    m, = castall([m], field=field)
    return m.tanh
def csch(m, *, field=None):
    """
    Alias for 'm.csch'.
    """
    m, = castall([m], field=field)
    return m.csch
def sech(m, *, field=None):
    """
    Alias for 'm.sech'.
    """
    m, = castall([m], field=field)
    return m.sech
def coth(m, *, field=None):
    """
    Alias for 'm.coth'.
    """
    m, = castall([m], field=field)
    return m.coth
def asinh(m, *, field=None):
    """
    Alias for 'm.asinh'.
    """
    m, = castall([m], field=field)
    return m.asinh
def acosh(m, *, field=None):
    """
    Alias for 'm.acosh'.
    """
    m, = castall([m], field=field)
    return m.acosh
def atanh(m, *, field=None):
    """
    Alias for 'm.atanh'.
    """
    m, = castall([m], field=field)
    return m.atanh
def acsch(m, *, field=None):
    """
    Alias for 'm.acsch'.
    """
    m, = castall([m], field=field)
    return m.acsch
def asech(m, *, field=None):
    """
    Alias for 'm.asech'.
    """
    m, = castall([m], field=field)
    return m.asech
def acoth(m, *, field=None):
    """
    Alias for 'm.acoth'.
    """
    m, = castall([m], field=field)
    return m.acoth
def atan2(y, x, *, field=None):
    """
    Quadrant-aware 'atan(y / x)'.
    """
    y, x = castall([y, x], field=field)
    return y.field._mat_atan2(y, x)
def torad(degrees, *, field=None):
    """
    Alias for 'degrees.torad'.
    """
    m, = castall([degrees], field=field)
    return m.torad
def todeg(radians, *, field=None):
    """
    Alias for 'radians.todeg'.
    """
    m, = castall([radians], field=field)
    return m.todeg
def sind(m, *, field=None):
    """
    Alias for 'm.sind'.
    """
    m, = castall([m], field=field)
    return m.sind
def cosd(m, *, field=None):
    """
    Alias for 'm.cosd'.
    """
    m, = castall([m], field=field)
    return m.cosd
def tand(m, *, field=None):
    """
    Alias for 'm.tand'.
    """
    m, = castall([m], field=field)
    return m.tand
def cscd(m, *, field=None):
    """
    Alias for 'm.cscd'.
    """
    m, = castall([m], field=field)
    return m.cscd
def secd(m, *, field=None):
    """
    Alias for 'm.secd'.
    """
    m, = castall([m], field=field)
    return m.secd
def cotd(m, *, field=None):
    """
    Alias for 'm.cotd'.
    """
    m, = castall([m], field=field)
    return m.cotd
def asind(m, *, field=None):
    """
    Alias for 'm.asind'.
    """
    m, = castall([m], field=field)
    return m.asind
def acosd(m, *, field=None):
    """
    Alias for 'm.acosd'.
    """
    m, = castall([m], field=field)
    return m.acosd
def atand(m, *, field=None):
    """
    Alias for 'm.atand'.
    """
    m, = castall([m], field=field)
    return m.atand
def acscd(m, *, field=None):
    """
    Alias for 'm.acscd'.
    """
    m, = castall([m], field=field)
    return m.acscd
def asecd(m, *, field=None):
    """
    Alias for 'm.asecd'.
    """
    m, = castall([m], field=field)
    return m.asecd
def acotd(m, *, field=None):
    """
    Alias for 'm.acotd'.
    """
    m, = castall([m], field=field)
    return m.acotd
def atand2(y, x, *, field=None):
    """
    Quadrant-aware 'atand(y / x)'.
    """
    return atan2(y, x, field=field).todeg
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
def issame(x, y, *, field=None):
    """
    Alias for 'x.issame(y)'.
    """
    x, y = castall([x, y], field=field)
    return x.issame(y)
def isreal(m, *, field=None):
    """
    Alias for 'm.isreal'.
    """
    m, = castall([m], field=field)
    return m.isreal
def isimag(m, *, field=None):
    """
    Alias for 'm.isimag'.
    """
    m, = castall([m], field=field)
    return m.isimag
def isinf(m, *, field=None):
    """
    Alias for 'm.isinf'.
    """
    m, = castall([m], field=field)
    return m.isinf
def isnan(m, *, field=None):
    """
    Alias for 'm.isnan'.
    """
    m, = castall([m], field=field)
    return m.isnan
def isfinite(m, *, field=None):
    """
    Alias for 'm.isfinite'.
    """
    m, = castall([m], field=field)
    return m.isfinite


def _cast_for_fold(ms, axis, field):
    if len(ms) == 1 and not isinstance(ms[0], _SEQ_TYPES):
        m, = castall(ms, field=field)
        if axis is None:
            axis = ...
    else:
        ms = _maybe_unpack_seq(ms)
        ms = castall(ms, field=field)
        if axis is not None:
            raise TypeError("cannot specify axis with multiple matrices")
        axis = ms[0].ndim
        m = stack(axis, ms)
    return m, axis

def minn(*ms, axis=None, field=None):
    """
    Alias for 'm.minn_along(axis)' if given one bare matrix, otherwise broadcasts
    then stacks the given matrices and takes the minimum along the stacked axis.
    """
    m, axis = _cast_for_fold(ms, axis, field)
    return m.minn_along(axis)
def maxx(*ms, axis=None, field=None):
    """
    Alias for 'm.maxx_along(axis)' if given one bare matrix, otherwise broadcasts
    then stacks the given matrices and takes the maximum along the stacked axis.
    """
    m, axis = _cast_for_fold(ms, axis, field)
    return m.maxx_along(axis)
def summ(*ms, axis=None, field=None):
    """
    Alias for 'm.summ_along(axis)' if given one bare matrix, otherwise broadcasts
    then stacks the given matrices and takes the sum along the stacked axis.
    """
    m, axis = _cast_for_fold(ms, axis, field)
    return m.summ_along(axis)
def prod(*ms, axis=None, field=None):
    """
    Alias for 'm.prod_along(axis)' if given one bare matrix, otherwise broadcasts
    then stacks the given matrices and takes the product along the stacked axis.
    """
    m, axis = _cast_for_fold(ms, axis, field)
    return m.prod_along(axis)
def mean(*ms, axis=None, field=None):
    """
    Alias for 'm.mean_along(axis)' if given one bare matrix, otherwise broadcasts
    then stacks the given matrices and takes the mean along the stacked axis.
    """
    m, axis = _cast_for_fold(ms, axis, field)
    return m.mean_along(axis)
def ave(*ms, axis=None, field=None):
    """
    Alias for 'mean(*ms, axis=axis)'.
    """
    return mean(*ms, axis=axis, field=field)
def geomean(*ms, axis=None, field=None):
    """
    Alias for 'm.geomean_along(axis)' if given one bare matrix, otherwise
    broadcasts then stacks the given matrices and takes the geometric mean along
    the stacked axis.
    """
    m, axis = _cast_for_fold(ms, axis, field)
    return m.geomean_along(axis)
def harmean(*ms, axis=None, field=None):
    """
    Alias for 'm.harmean_along(axis)' if given one bare matrix, otherwise
    broadcasts then stacks the given matrices and takes the harmonic mean along
    the stacked axis.
    """
    m, axis = _cast_for_fold(ms, axis, field)
    return m.harmean_along(axis)
def quadmean(*ms, axis=None, field=None):
    """
    Alias for 'm.quadmean_along(axis)' if given one bare matrix, otherwise
    broadcasts then stacks the given matrices and takes the quadratic mean along
    the stacked axis.
    """
    m, axis = _cast_for_fold(ms, axis, field)
    return m.quadmean_along(axis)
def logmean(x, y, *, field=None):
    """
    Logarithmic mean of 'x' and 'y': (x - y) / ln(x / y)
    """
    x, y = castall([x, y], field=field)
    if not alll(x > 0, y > 0):
        raise ValueError("expected strictly positive values for logmean")
    def f(a, b):
        if a == 0 or b == 0:
            return a.zero
        if a == b:
            return a
        return (a - b) / (a / b).ln
    return x.apply(f, y)


def short(m, *, field=None):
    """
    Prints a short string representation of 'm'.
    """
    m, = castall([m], field=field)
    print(m.__repr__(short=True))

def long(m, *, field=None):
    """
    Prints a long string representation of 'm'.
    """
    m, = castall([m], field=field)
    print(m.__repr__(short=False))

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


def stack(axis, *ms, field=None):
    """
    Stacks the given matrices along the given axis.
    """
    ms = _maybe_unpack_seq(ms)
    ms = castall(ms, field=field, broadcast=False)
    field = _get_field(field, ms)
    if not isinstance(axis, int):
        raise TypeError(f"expected an integer axis, got {_objtname(axis)}")
    if axis < 0:
        raise ValueError(f"axis cannot be negative, got: {axis}")

    # Stacking empty do NOTHIGN and stacking nothign do EMPTY.
    if not ms:
        return empty(field)
    if all(m.isempty for m in ms):
        return empty(field)

    # Huge stack of one item.
    if len(ms) == 1:
        return ms[0]

    # Check perpendicular sizes.
    perpshape = ms[0].shape.withaxis(axis, 1)
    for m in ms:
        if m.shape.withaxis(axis, 1) != perpshape:
            raise TypeError(f"expected a perpendicular shape of {perpshape} for "
                    f"stacking, got {m.shape}")
    # Get all cells with the right number of dimensions.
    ndim = max(ms[0].ndim, axis + 1)
    ys = [x._cells for x in ms]
    ys = [y.reshape((1, ) * (ndim - y.ndim) + y.shape) for y in ys]
    npaxis = Permuter.tonumpyaxis(ndim, axis)
    cells = _np.concatenate(ys, axis=npaxis)
    return Matrix.fromnumpy(field, cells)
def vstack(*ms, field=None):
    """
    Vertically concatenates the given matrices.
    """
    return stack(0, *ms, field=field)
def hstack(*ms, field=None):
    """
    Horizontally concatenates the given matrices.
    """
    return stack(1, *ms, field=field)
def matstack(*ms, field=None):
    """
    Concatenates the given matrices along the third axis (index 2).
    """
    return stack(2, *ms, field=field)

def ravel(*ms, field=None):
    """
    Concatenated vector of the ravelled cells of all given matrices.
    """
    ms = _maybe_unpack_seq(ms)
    ms = castall(ms, field=field, broadcast=False)
    field = _get_field(field, ms)
    if len(ms) == 0:
        return empty(field)
    cells = _np.concatenate([m.ravel.numpyvec() for m in ms])
    return Matrix.fromnumpy(field, cells)

def tovec(*ms, field=None):
    """
    Concatenated column vector of the given matrices, values, or iterables.
    """
    # dont maybe unpack ms lmao.
    concat = []
    for m in ms:
        if isinstance(m, Matrix) or not _iterable(m):
            m = [m]
        for x in m:
            if not isinstance(x, Matrix):
                x, = castall([x], field=field, broadcast=False)
            field = _get_field(field, [x])
            if not x.isvec:
                raise TypeError("expected iterable to contain vectors to "
                        f"concatenate into column, got {x.shape}")
            concat.append(x.numpyvec())
    field = _get_field(field)
    if len(concat) == 0:
        return empty(field)
    cells = _np.concatenate(concat)
    return Matrix.fromnumpy(field, cells)


def eye(n, *, field=None):
    """
    2D identity matrix of the given size.
    """
    field = _get_field(field)
    if not isinstance(n, int):
        raise TypeError(f"expected an integer size, got {_objtname(n)}")
    if n < 0:
        raise ValueError(f"size cannot be negative, got: {n}")
    return Matrix[field, Shape(n, n)].eye

def zeros(*lens, field=None):
    """
    Zero-filled matrix with the given axis lengths.
    """
    field = _get_field(field)
    shape = Shape(*lens)
    return Matrix[field, shape].zeros

def ones(*lens, field=None):
    """
    One-filled matrix with the given axis lengths.
    """
    field = _get_field(field)
    shape = Shape(*lens)
    return Matrix[field, shape].ones

def full(value, *lens, field=None):
    """
    Matrix filled with the given value and the given axis lengths.
    """
    value, = castall([value], field=field)
    field = _get_field(field, [value])
    shape = Shape(*lens)
    return Matrix[field, shape].full(value)

def diag(*ms, k=None, emptyok=False, field=None):
    """
    If given a matrix, returns the 'k'-th diagonal of that matrix (defaulting to
    the main diagonal). Otherwise, concatenates all given vectors and returns a
    square matrix with that vector as the 'k'-th diagonal.
    """
    if k is not None and not isinstance(k, int):
        raise TypeError(f"expected an integer 'k', got {_objtname(k)}")
    ms = _maybe_unpack_seq(ms)
    ms = castall(ms, field=field, broadcast=False)
    field = _get_field(field, ms)
    if len(ms) == 1:
        m = ms[0]
        if not m.isvec:
            if k is None:
                k = 0
            return m.diagk(k, emptyok=emptyok)
    m = tovec(*ms, field=field)
    if m.isempty:
        # Not giving k also allows empty.
        if emptyok or k is None:
            return m
        raise ValueError("cannot create a matrix with an empty 'k'-th diag")
    # Create the main diagonal only first.
    if m.issingle:
        cells = m._cells
    else:
        cells = _np.full((m.size, m.size), field.zero, dtype=field.dtype())
        _np.fill_diagonal(cells, m.numpyvec())
    if k is not None and k != 0:
        # If we didn't ask for main diagonal, pad it out.
        if k > 0:
            padby = ((0, k), (k, 0))
        else:
            padby = ((-k, 0), (0, -k))
        cells = _np.pad(cells, padby, "constant", constant_values=field.zero)
    return Matrix.fromnumpy(field, cells)

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
    x = tuple(x0 + step * i for i in range(n))
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
    return linspace(x0.ln, x1.ln, n, field=field).exp


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
    Ys = Y.along(0, drop=True)
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
    from .engine import KW_PREV

    if long is None:
        long = not doesdflt2short()

    # Trim down to the matrix variables.
    space = _get_space()
    mspace = {k: v for k, v in space.items() if isinstance(v, Matrix)}
    # Dont include the "last result" variable.
    mspace.pop(KW_PREV, None)
    # Dont include default injected vars.
    for name in lits._injects:
        if not lits._is_overridden(mspace, name):
            mspace.pop(name, None)

    if not mspace:
        print(_cons.coloured(245, "no matrix variables."))
    for name, value in mspace.items():
        pre = _cons.coloured([208, 161], [name, " = "])
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
        if c in ".+-*/%=&|@<>^~:!":
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
        name = _cons.coloured(cols, txts)
        print(_cons.entry(name, desc, width=width, pwidth=22, lead=2))

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
        "__ne__": "!=",
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
        if sig.endswith(", _nocast=False"):
            sig = sig[:-len(", _nocast=False")]
        if sig.endswith(", *, field=None"):
            sig = sig[:-len(", *, field=None")]
        elif sig.endswith("*, field=None"):
            sig = sig[:-len("*, field=None")]
        elif sig.endswith(", field=None"):
            sig = sig[:-len(", field=None")]
        expr = f"{name}({sig})"
        print_entry(expr, doc)
