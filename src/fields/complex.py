import cmath
import math
import struct

import matrix
from util import coloured, classconst, immutable, tname


@immutable
class Complex(matrix.Field):
    def __init__(self, re=0.0, im=0.0, *, isnan=False):
        if not isinstance(re, float):
            raise TypeError(f"expected float for re, got {tname(type(re))}")
        if not isinstance(im, float):
            raise TypeError(f"expected float for im, got {tname(type(im))}")
        if not isinstance(isnan, bool):
            raise TypeError(f"expected bool for isnan, got {tname(type(isnan))}")
        if isnan or re != re or im != im:
            isnan = True
            re = float("nan")
            im = float("nan")
        self.isnan = isnan
        self._re = re
        self._im = im


    @classmethod
    def from_int(cls, x):
        return cls(float(x))
    @classmethod
    def from_float(cls, x):
        return cls(x)
    @classmethod
    def from_complex(cls, x):
        return cls(x.real, x.imag)

    @classmethod
    def to_int(cls, a):
        if a.isnan:
            raise ValueError("cannot cast nan to int")
        if a._im:
            raise ValueError(f"cannot cast non-real to int, got: {a}")
        if not a._re.is_integer():
            raise ValueError(f"cannot cast non-integer to int, got: {a}")
        return int(a._re)
    @classmethod
    def to_float(cls, a):
        if a.isnan:
            return float("nan")
        if a._im:
            raise ValueError(f"cannot cast non-real to float, got: {a}")
        return a._re
    @classmethod
    def to_complex(cls, a):
        if a.isnan:
            return complex("nan+nanj")
        return complex(a._re, a._im)

    @classconst
    def exposes(cls):
        return {"isnan": bool}

    @classconst
    def zero(cls):
        return cls(0.0)
    @classconst
    def one(cls):
        return cls(1.0)

    @classmethod
    def add(cls, a, b):
        if a.isnan or b.isnan:
            return cls(isnan=True)
        return cls(a._re + b._re, a._im + b._im)
    @classmethod
    def sub(cls, a, b):
        if a.isnan or b.isnan:
            return cls(isnan=True)
        return cls(a._re - b._re, a._im - b._im)
    @classmethod
    def absolute(cls, a):
        if a.isnan:
            return cls(isnan=True)
        return cls(math.sqrt(a._re * a._re + a._im * a._im))

    @classmethod
    def mul(cls, a, b):
        if a.isnan or b.isnan:
            return cls(isnan=True)
        # (a.re + i a.im)(b.re + i b.im)
        # = a.re b.re - a.im b.im + i (a.im b.re + a.re b.im)
        re = a._re * b._re - a._im * b._im
        im = a._im * b._re + a._re * b._im
        return cls(re, im)
    @classmethod
    def div(cls, a, b):
        if a.isnan or b.isnan:
            return cls(isnan=True)
        # (a.re + i a.im)/(b.re + i b.im)
        #
        #   (a.re b.re + a.im b.im) + i (a.im b.re - a.re b.im)
        # = ---------------------------------------------------
        #                 (b.re b.re + b.im b.im)
        re = a._re * b._re + a._im * b._im
        im = a._im * b._re - a._re * b._im
        den = b._re * b._re + b._im * b._im
        if den == 0.0:
            raise ZeroDivisionError("x/0")
        re /= den
        im /= den
        return cls(re, im)

    @classmethod
    def power(cls, a, b):
        if a.isnan or b.isnan:
            return cls(isnan=True)
        a = complex(a._re, a._im)
        b = complex(b._re, b._im)
        return cls.from_complex(a ** b)
    @classmethod
    def root(cls, a, b):
        if a.isnan or b.isnan:
            return cls(isnan=True)
        if b._re == 0.0 and b._im == 0.0:
            raise ZeroDivisionError("x ^ (1/0)")
        a = complex(a._re, a._im)
        b = complex(b._re, b._im)
        return cls.from_complex(a ** (1 / b))
    @classmethod
    def log(cls, a, b):
        if a.isnan or b.isnan:
            return cls(isnan=True)
        a = complex(a._re, a._im)
        b = complex(b._re, b._im)
        return cls.from_complex(cmath.log(b) / cmath.log(a))

    @classmethod
    def sin(cls, a):
        if a.isnan:
            return cls(isnan=True)
        a = complex(a._re, a._im)
        return cls.from_complex(cmath.sin(a))
    @classmethod
    def cos(cls, a):
        if a.isnan:
            return cls(isnan=True)
        a = complex(a._re, a._im)
        return cls.from_complex(cmath.cos(a))
    @classmethod
    def tan(cls, a):
        if a.isnan:
            return cls(isnan=True)
        a = complex(a._re, a._im)
        return cls.from_complex(cmath.tan(a))

    @classmethod
    def asin(cls, a):
        if a.isnan:
            return cls(isnan=True)
        a = complex(a._re, a._im)
        return cls.from_complex(cmath.asin(a))
    @classmethod
    def acos(cls, a):
        if a.isnan:
            return cls(isnan=True)
        a = complex(a._re, a._im)
        return cls.from_complex(cmath.acos(a))
    @classmethod
    def atan(cls, a):
        if a.isnan:
            return cls(isnan=True)
        a = complex(a._re, a._im)
        return cls.from_complex(cmath.atan(a))
    @classmethod
    def atan2(cls, y, x):
        if y.isnan or x.isnan:
            return cls(isnan=True)
        if y._im != 0.0 or x._im != 0.0:
            raise ValueError(f"cannot use atan2 with non-real, got: {y}, and: "
                    f"{x}")
        return cls(math.atan2(y._re, x._re))

    @classmethod
    def conj(cls, a):
        if a.isnan:
            return cls(isnan=True)
        return cls(a._re, -a._im)

    @classmethod
    def issame(cls, a, b):
        if a.isnan or b.isnan:
            return a.isnan == b.isnan
        return a._re == b._re and a._im == b._im
    @classmethod
    def eq(cls, a, b):
        if a.isnan or b.isnan:
            return a.isnan == b.isnan
        def iseq(x, y, ulps=15):
            if math.isinf(x) or math.isinf(y):
                return x == y
            # dont let zero and one be equal unless exact.
            if x == 0.0 or y == 0.0 or abs(x) == 1.0 or abs(y) == 1.0:
                return x == y
            # we do tricks around here (c my beloved).
            toint = lambda z: struct.unpack("=q", struct.pack("=d", z))[0]
            ux = toint(abs(x))
            uy = toint(abs(y))
            if (x < 0.0) != (y < 0.0):
                return ux + uy <= ulps
            return abs(ux - uy) <= ulps
        return iseq(a._re, b._re) and iseq(a._im, b._im)
    @classmethod
    def lt(cls, a, b):
        if a.isnan or b.isnan:
            return a.isnan < b.isnan
        if a._im or b._im: # complex is unorderable.
            raise ValueError(f"cannot order non-real, got: {a}, and: {b}")
        return a._re < b._re

    @classmethod
    def hashed(cls, a):
        return hash((a._re, a._im))

    @classmethod
    def rep(cls, a, short):
        if a.isnan:
            return "nan"
        re = a._re
        im = a._im
        plus = coloured(161, "+")
        minus = coloured(161, "-")
        def repn(n):
            if n == 0.0: # -0.0 -> 0.0
                n = 0.0
            if n < 0.0:
                return minus + repn(-n)
            s = f"{n:.6g}" if short else repr(n)
            # "xxx.0" -> "xxx"
            if s.endswith(".0"):
                s = s[:-2]
            if "e" not in s:
                return coloured(135, s)
            # "xxx.xe-0y" -> "xxx.xe-y"
            # "xxx.xe+0y" -> "xxx.xey"
            s = s.replace("e-0", "e-")
            s = s.replace("e+", "e")
            s = s.replace("e0", "e")
            return coloured(135, s)
            # if you wanna highlight the e, maybe try this but i couldnt find a
            # nice colour lmao.
            if "e-" in s:
                parts = s.split("e-")
                parts.insert(1, "e-")
            else:
                parts = s.split("e")
                parts.insert(1, "e")
            if len(parts) != 3:
                raise Exception(f"weirdly split string {parts}")
            return coloured([135, 105, 135], parts)

        sep = plus
        i = coloured(38, "i")

        if im == 0.0:
            im_s = ""
        elif im == 1.0:
            im_s = i
        else:
            if im == -1.0:
                sep = minus
                im_s = ""
            elif im < 0.0:
                sep = minus
                im_s = repn(-im)
            else:
                im_s = repn(im)
            im_s = f"{im_s}{i}"

        if re == 0.0:
            re_s = ""
        else:
            re_s = repn(re)

        if not re_s and not im_s:
            return repn(0.0)

        if not im_s:
            return re_s

        if not re_s:
            if sep == plus:
                sep = ""
            return f"{sep}{im_s}"

        if not short:
            sep = f" {sep} "
        return f"{re_s}{sep}{im_s}"
