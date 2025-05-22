from math import atan2

import field
from util import immutable


def _fp_mul(a, b):
    assert isinstance(a, float)
    assert isinstance(b, float)
    if a != a or b != b:
        return float("nan")
    if a == 0.0 or b == 0.0: # avoid 0*inf = nan
        return 0.0
    return a * b

def _fp_isnan(x):
    assert isinstance(x, float)
    return x != x


@immutable
class Num(field.Field):
    def __init__(self, re=0.0, im=0.0, *, isnan=False):
        if isinstance(re, int):
            re = float(re)
        if isinstance(im, int):
            im = float(im)
        if not isinstance(re, float):
            raise TypeError("real value must be a float")
        if not isinstance(im, float):
            raise TypeError("imaginary value must be a float")
        if not isinstance(isnan, bool):
            raise TypeError("isnan must be a bool")
        if _fp_isnan(re) or _fp_isnan(im):
            isnan = True
        if re == 0.0:
            re = 0.0
        if im == 0.0:
            im = 0.0
        self.isnan = isnan
        if not isnan:
            self.re = re
            self.im = im


    @classmethod
    def _cast(cls, obj, for_obj):
        if not isinstance(obj, complex):
            try:
                obj = complex(obj)
            except Exception:
                pass
        if isinstance(obj, complex):
            isnan = _fp_isnan(obj.real) or _fp_isnan(obj.imag)
            return cls(obj.real, obj.imag, isnan=isnan)
        super()._cast(obj, for_obj)

    @classmethod
    def _zero(cls):
        return cls(0)
    @classmethod
    def _one(cls):
        return cls(1)

    @classmethod
    def _add(cls, a, b):
        if a.isnan or b.isnan:
            return cls(isnan=True)
        return cls(a.re + b.re, a.im + b.im)
    @classmethod
    def _neg(cls, a):
        if a.isnan:
            return cls(isnan=True)
        return cls(-a.re, -a.im)
    @classmethod
    def _mul(cls, a, b):
        if a.isnan or b.isnan:
            return cls(isnan=True)
        # (a.re + i a.im)(b.re + i b.im)
        # = a.re b.re - a.im b.im + i(a.im b.re + a.re b.im)
        re = _fp_mul(a.re, b.re) - _fp_mul(a.im, b.im)
        im = _fp_mul(a.im, b.re) + _fp_mul(a.re, b.im)
        return cls(re, im)
    @classmethod
    def _rec(cls, a):
        if a.isnan:
            return cls(isnan=True)
        if a.re == 0.0 and a.im == 0.0:
            raise ZeroDivisionError("1/0")
        # 1/(a.re + i a.im)
        # = (a.re - i a.im)/(a.re^2 + a.im^2)
        d = _fp_mul(a.re, a.re) + _fp_mul(a.im, a.im)
        return cls(a.re / d, -a.im / d)
    @classmethod
    def _exp(cls, a):
        if a.isnan:
            return cls(isnan=True)
        # e^(a.re + i a.im)
        # = e^a.re * e^(i a.im)
        # = e^a.re * (cos(a.im) + i sin(a.im))
        re = _fp_mul(field.exp(a.re), field.cos(a.im))
        im = _fp_mul(field.exp(a.re), field.sin(a.im))
        return cls(re, im)
    @classmethod
    def _log(cls, a):
        if a.isnan:
            return cls(isnan=True)
        # log(a)
        # = log(|a|) + i arg(a)  [principal branch]
        absa = _fp_mul(a.re, a.re) + _fp_mul(a.im, a.im)
        re = _fp_mul(0.5, field.log(absa))
        im = atan2(a.im, a.re)
        return cls(re, im)

    @classmethod
    def _eq_zero(cls, a):
        if a.isnan:
            return False
        return a.re == 0.0 and a.im == 0.0
    @classmethod
    def _lt_zero(cls, a):
        if a.isnan: # nan is unorderable.
            raise NotImplementedError()
        if a.im: # complex is unorderable.
            raise NotImplementedError()
        return a.re < 0.0

    @classmethod
    def _floatof(cls, a):
        if a.isnan:
            return float("nan")
        if a.im:
            raise NotImplementedError()
        return float(a.re)
    @classmethod
    def _complexof(cls, a):
        if a.isnan:
            return complex("nan+nanj")
        return complex(a.re, a.im)

    @classmethod
    def _hashof(cls, a):
        return hash((a.re, a.im))

    def __repr__(self):
        if self.isnan:
            return "nan"
        re = self.re
        im = self.im
        disp = lambda x: f"{(0.0 if x == 0.0 else x):.5g}"

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
                im_s = disp(-im)
            else:
                im_s = disp(im)
            is_digits = all(map(str.isdigit, im_s))
            im_s = f"{im_s}i" if is_digits else f"i∙{im_s}"

        if re == 0.0:
            re_s = ""
        else:
            re_s = disp(re)

        if not re_s and not im_s:
            return "0"

        if not im_s:
            return re_s

        if not re_s:
            if sep == "+":
                sep = ""
            return f"{sep}{im_s}"

        return f"{re_s} {sep} {im_s}"
