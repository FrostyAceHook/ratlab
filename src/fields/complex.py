import math

import matrix
from fields.real import Real
from util import classconst, immutable
from matrix import Single, single


@immutable
class Complex(matrix.Field):
    def __init__(self, re=single(Real()), im=single(Real()), *, isnan=False):
        if not isinstance(re, Single[Real]):
            raise TypeError("real value must be a Real")
        if not isinstance(im, Single[Real]):
            raise TypeError("imaginary value must be a Real")
        if not isinstance(isnan, bool):
            raise TypeError("isnan must be a bool")
        if re.isnan or im.isnan:
            isnan = True
        self.isnan = isnan
        if not isnan:
            self.re = re
            self.im = im


    @classmethod
    def from_int(cls, x):
        return cls(Single[Real].cast(x))
    @classmethod
    def from_float(cls, x):
        return cls(Single[Real].cast(x))
    @classmethod
    def from_complex(cls, x):
        re = Single[Real].cast(x.real)
        im = Single[Real].cast(x.imag)
        return cls(re, im)

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
        return cls(single(Real(0.0)))
    @classconst
    def one(cls):
        return cls(single(Real(1.0)))

    @classmethod
    def add(cls, a, b):
        return cls(a.re + b.re, a.im + b.im)
    @classmethod
    def sub(cls, a, b):
        return cls(a.re - b.re, a.im - b.im)
    @classmethod
    def absolute(cls, a):
        return cls((a.re * a.re + a.im * a.im).sqrt)

    @classmethod
    def mul(cls, a, b):
        # (a.re + i a.im)(b.re + i b.im)
        # = a.re b.re - a.im b.im + i (a.im b.re + a.re b.im)
        re = a.re * b.re - a.im * b.im
        im = a.im * b.re + a.re * b.im
        return cls(re, im)
    @classmethod
    def div(cls, a, b):
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
        def exp(x):
            # e^(a.re + i a.im)
            # = e^a.re * e^(i a.im)
            # = e^a.re * (cos(a.im) + i sin(a.im))
            re = x.re.exp * x.im.cos
            im = x.re.exp * x.im.sin
            return cls(re, im)
        # a^b = e^(ln(a) * b)
        return exp(a.ln * b)
    @classmethod
    def root(cls, a, b):
        return cls.power(a, cls.div(cls.one, b))
    @classmethod
    def log(cls, a, b):
        def ln(x):
            # ln(a)
            # = ln(mag(a)) + i arg(a)  [principal branch]
            sqrmag = x.re * x.re + x.im * x.im
            re = 0.5 * sqrmag.ln
            im = Real.atan2(x.im, x.re)
            return cls(re, im)
        # log_a(b) = ln(b) / ln(a)
        return ln(b) / ln(a)

    @classmethod
    def eq(cls, a, b):
        return a.re == b.re and a.im == b.im
    @classmethod
    def lt(cls, a, b):
        if a.im or b.im: # complex is unorderable.
            raise NotImplementedError()
        return a.re < b.re

    @classmethod
    def hashed(cls, a):
        return hash((a.re, a.im))

    def _repr(self, islong):
        if self.isnan:
            return "nan"
        re = self.re
        im = self.im
        rep = Single[Real].repr_long if islong else Single[Real].repr_short

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
                im_s = rep(-im)
            else:
                im_s = rep(im)
            is_digits = all(map(str.isdigit, im_s))
            im_s = f"{im_s}i" if is_digits else f"i∙{im_s}"

        if re == 0.0:
            re_s = ""
        else:
            re_s = rep(re)

        if not re_s and not im_s:
            return "0"

        if not im_s:
            return re_s

        if not re_s:
            if sep == "+":
                sep = ""
            return f"{sep}{im_s}"

        return f"{re_s} {sep} {im_s}"

    @classmethod
    def repr_short(cls, a):
        return a._repr(False)
    @classmethod
    def repr_long(cls, a):
        return a._repr(True)
