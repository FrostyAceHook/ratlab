import math
import struct

import matrix
from util import classconst, immutable, tname


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
        self.re = re
        self.im = im


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
            raise NotImplementedError()
        if a.im:
            raise NotImplementedError()
        if math.isinf(a.re) or a.re != int(a.re):
            raise NotImplementedError()
        return int(a.re)
    @classmethod
    def to_float(cls, a):
        if a.isnan:
            return float("nan")
        if a.im:
            raise NotImplementedError()
        return a.re
    @classmethod
    def to_complex(cls, a):
        if a.isnan:
            return complex("nan+nanj")
        return complex(a.re, a.im)

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
        return cls(a.re + b.re, a.im + b.im)
    @classmethod
    def sub(cls, a, b):
        if a.isnan or b.isnan:
            return cls(isnan=True)
        return cls(a.re - b.re, a.im - b.im)
    @classmethod
    def absolute(cls, a):
        if a.isnan:
            return cls(isnan=True)
        return cls(math.sqrt(a.re * a.re + a.im * a.im))

    @classmethod
    def mul(cls, a, b):
        if a.isnan or b.isnan:
            return cls(isnan=True)
        # (a.re + i a.im)(b.re + i b.im)
        # = a.re b.re - a.im b.im + i (a.im b.re + a.re b.im)
        re = a.re * b.re - a.im * b.im
        im = a.im * b.re + a.re * b.im
        return cls(re, im)
    @classmethod
    def div(cls, a, b):
        if a.isnan or b.isnan:
            return cls(isnan=True)
        # (a.re + i a.im)/(b.re + i b.im)
        #
        #   (a.re b.re + a.im b.im) + i (a.im b.re - a.re b.im)
        # = ---------------------------------------------------
        #                 (b.re b.re + b.im bim)
        re = a.re * b.re + a.im * b.im
        im = a.im * b.re - a.re * b.im
        den = b.re * b.re + b.im * b.im
        re /= den
        im /= den
        return cls(re, im)

    @classmethod
    def power(cls, a, b):
        if a.isnan or b.isnan:
            return cls(isnan=True)
        def exp(x):
            # e^(a.re + i a.im)
            # = e^a.re e^(i a.im)
            # = e^a.re (cos(a.im) + i sin(a.im))
            re = math.exp(x.re) * math.cos(x.im)
            im = math.exp(x.re) * math.sin(x.im)
            return cls(re, im)
        # a^b = e^(ln(a) b)
        lna = cls.log(cls(math.e), a)
        return exp(cls.mul(lna, b))
    @classmethod
    def root(cls, a, b):
        if a.isnan or b.isnan:
            return cls(isnan=True)
        return cls.power(a, cls.div(cls.one, b))
    @classmethod
    def log(cls, a, b):
        if a.isnan or b.isnan:
            return cls(isnan=True)
        def ln(x):
            # ln(a)
            # = ln(mag(a)) + i arg(a)  [principal branch]
            # = 0.5 ln(mag(a)^2) + i arg(a)
            sqrmag = x.re * x.re + x.im * x.im
            re = 0.5 * math.log(sqrmag)
            im = math.atan2(x.im, x.re)
            return cls(re, im)
        # log_a(b) = ln(b) / ln(a)
        return cls.div(ln(b), ln(a))

    @classmethod
    def eq(cls, a, b):
        if a.isnan or b.isnan:
            return a.isnan == b.isnan
        def iseq(x, y, ulps=5):
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
        return iseq(a.re, b.re) and iseq(a.im, b.im)
    @classmethod
    def lt(cls, a, b):
        if a.isnan or b.isnan:
            return a.isnan < b.isnan
        if a.im or b.im: # complex is unorderable.
            raise NotImplementedError()
        return a.re < b.re

    @classmethod
    def hashed(cls, a):
        return hash((a.re, a.im))

    @classmethod
    def rep(cls, a, short):
        if a.isnan:
            return "nan"
        re = a.re
        im = a.im
        def repn(n):
            if n == 0.0:
                return "0"
            s = f"{n:.6g}" if short else repr(n)
            # "xxx.0" -> "xxx"
            if s.endswith(".0"):
                s = s[:-2]
            # "xxx.xe-0y" -> "xxx.xe-y"
            if "e" in s:
                if len(s) > 2 and s[-2] == "0":
                    s = s[:-2] + s[-1]
            return s

        sep = "+"

        if im == 0.0:
            im_s = ""
        elif im == 1.0:
            im_s = "i"
        else:
            if im == -1.0:
                sep = "-"
                im_s = ""
            elif im < 0.0:
                sep = "-"
                im_s = repn(-im)
            else:
                im_s = repn(im)
            im_s = f"{im_s}i"

        if re == 0.0:
            re_s = ""
        else:
            re_s = repn(re)

        if not re_s and not im_s:
            return "0"

        if not im_s:
            return re_s

        if not re_s:
            if sep == "+":
                sep = ""
            return f"{sep}{im_s}"

        if not short:
            sep = f" {sep} "
        return f"{re_s}{sep}{im_s}"
