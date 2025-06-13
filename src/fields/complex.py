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
        self._isnan = isnan
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
        if a._isnan:
            raise NotImplementedError()
        if a._im:
            raise NotImplementedError()
        if math.isinf(a._re) or a._re != int(a._re):
            raise NotImplementedError()
        return int(a._re)
    @classmethod
    def to_float(cls, a):
        if a._isnan:
            return float("nan")
        if a._im:
            raise NotImplementedError()
        return a._re
    @classmethod
    def to_complex(cls, a):
        if a._isnan:
            return complex("nan+nanj")
        return complex(a._re, a._im)

    @classconst
    def zero(cls):
        return cls(0.0)
    @classconst
    def one(cls):
        return cls(1.0)

    @classmethod
    def add(cls, a, b):
        if a._isnan or b._isnan:
            return cls(isnan=True)
        return cls(a._re + b._re, a._im + b._im)
    @classmethod
    def sub(cls, a, b):
        if a._isnan or b._isnan:
            return cls(isnan=True)
        return cls(a._re - b._re, a._im - b._im)
    @classmethod
    def absolute(cls, a):
        if a._isnan:
            return cls(isnan=True)
        return cls(math.sqrt(a._re * a._re + a._im * a._im))

    @classmethod
    def mul(cls, a, b):
        if a._isnan or b._isnan:
            return cls(isnan=True)
        # (a.re + i a.im)(b.re + i b.im)
        # = a.re b.re - a.im b.im + i (a.im b.re + a.re b.im)
        re = a._re * b._re - a._im * b._im
        im = a._im * b._re + a._re * b._im
        return cls(re, im)
    @classmethod
    def div(cls, a, b):
        if a._isnan or b._isnan:
            return cls(isnan=True)
        # (a.re + i a.im)/(b.re + i b.im)
        #
        #   (a.re b.re + a.im b.im) + i (a.im b.re - a.re b.im)
        # = ---------------------------------------------------
        #                 (b.re b.re + b.im bim)
        re = a._re * b._re + a._im * b._im
        im = a._im * b._re - a._re * b._im
        den = b._re * b._re + b._im * b._im
        re /= den
        im /= den
        return cls(re, im)

    @classmethod
    def power(cls, a, b):
        if a._isnan or b._isnan:
            return cls(isnan=True)
        def exp(x):
            # e^(a.re + i a.im)
            # = e^a.re e^(i a.im)
            # = e^a.re (cos(a.im) + i sin(a.im))
            re = math.exp(x._re) * math.cos(x._im)
            im = math.exp(x._re) * math.sin(x._im)
            return cls(re, im)
        # a^b = e^(ln(a) b)
        lna = cls.log(cls(math.e), a)
        return exp(cls.mul(lna, b))
    @classmethod
    def root(cls, a, b):
        if a._isnan or b._isnan:
            return cls(isnan=True)
        return cls.power(a, cls.div(cls.one, b))
    @classmethod
    def log(cls, a, b):
        if a._isnan or b._isnan:
            return cls(isnan=True)
        def ln(x):
            # ln(a)
            # = ln(mag(a)) + i arg(a)  [principal branch]
            # = 0.5 ln(mag(a)^2) + i arg(a)
            sqrmag = x._re * x._re + x._im * x._im
            re = 0.5 * math.log(sqrmag)
            im = math.atan2(x._im, x._re)
            return cls(re, im)
        # log_a(b) = ln(b) / ln(a)
        return cls.div(ln(b), ln(a))

    @classmethod
    def sin(cls, a):
        return cls.from_complex(cmath.sin(complex(a._re, a._im)))
    @classmethod
    def cos(cls, a):
        return cls.from_complex(cmath.cos(complex(a._re, a._im)))
    @classmethod
    def tan(cls, a):
        return cls.from_complex(cmath.tan(complex(a._re, a._im)))

    @classmethod
    def asin(cls, a):
        return cls.from_complex(cmath.asin(complex(a._re, a._im)))
    @classmethod
    def acos(cls, a):
        return cls.from_complex(cmath.acos(complex(a._re, a._im)))
    @classmethod
    def atan(cls, a):
        return cls.from_complex(cmath.atan(complex(a._re, a._im)))
    @classmethod
    def atan2(cls, y, x):
        y = complex(y._re, y._im)
        x = complex(x._re, x._im)
        return cls.from_complex(cmath.atan2(y, x))

    @classmethod
    def conj(cls, a):
        if a._isnan:
            return cls(isnan=True)
        return cls(a._re, -a._im)

    @classmethod
    def eq(cls, a, b):
        if a._isnan or b._isnan:
            return a._isnan == b._isnan
        def iseq(x, y, ulps=15):
            if math.isinf(x) or math.isinf(y):
                return x == y
            # we do tricks around here (c my beloved).
            def toint(z):
                u, = struct.unpack("=q", struct.pack("=d", z))
                return u ^ (u >> 63)
            ux = toint(abs(x))
            uy = toint(abs(y))
            if (x < 0.0) != (y < 0.0):
                return ux + uy <= ulps
            return abs(ux - uy) <= ulps
        return iseq(a._re, b._re) and iseq(a._im, b._im)
    @classmethod
    def lt(cls, a, b):
        if a._isnan or b._isnan:
            return a._isnan < b._isnan
        if a._im or b._im: # complex is unorderable.
            raise NotImplementedError()
        return a._re < b._re

    @classmethod
    def issame(cls, a, b):
        if a._isnan or b._isnan:
            return a._isnan == b._isnan
        return a._re == b._re and a._im == b._im

    @classmethod
    def hashed(cls, a):
        return hash((a._re, a._im))

    @classmethod
    def rep(cls, a, short):
        if a._isnan:
            return "nan"
        re = a._re
        im = a._im
        def repn(n):
            if n == 0.0: # -0.0 -> 0.0
                n = 0.0
            s = f"{n:.6g}" if short else repr(n)
            # "xxx.0" -> "xxx"
            if s.endswith(".0"):
                s = s[:-2]
            # "xxx.xe-0y" -> "xxx.xe-y"
            if "e" in s:
                if len(s) > 2 and s[-2] == "0":
                    s = s[:-2] + s[-1]
            return coloured(135, s)

        pos = coloured(161, "+")
        neg = coloured(161, "-")
        sep = pos
        i = coloured(38, "i")

        if im == 0.0:
            im_s = ""
        elif im == 1.0:
            im_s = i
        else:
            if im == -1.0:
                sep = neg
                im_s = ""
            elif im < 0.0:
                sep = neg
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
            if sep == pos:
                sep = ""
            return f"{sep}{im_s}"

        if not short:
            sep = f" {sep} "
        return f"{re_s}{sep}{im_s}"
