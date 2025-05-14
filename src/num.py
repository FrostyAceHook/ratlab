from field import Field
from immutable import immutable
import maths


@immutable
class Num(Field):
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
        if maths.isnan(re) or maths.isnan(im):
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
    def cast(cls, obj, for_obj=None):
        if not isinstance(obj, complex):
            try:
                obj = complex(obj)
            except Exception:
                pass
        if isinstance(obj, complex):
            isnan = maths.isnan(obj.real) or maths.isnan(obj.imag)
            return cls(obj.real, obj.imag, isnan=isnan)
        raise NotImplementedError()

    @classmethod
    def zero(cls):
        return cls(0)
    @classmethod
    def one(cls):
        return cls(1)
    @classmethod
    def inf(cls):
        return cls(float("inf"))
    @classmethod
    def nan(cls):
        return cls(isnan=True)

    @classmethod
    def add(cls, a, b):
        if a.isnan or b.isnan:
            return cls.nan()
        return cls(a.re + b.re, a.im + b.im)
    @classmethod
    def neg(cls, a):
        if a.isnan:
            return cls.nan()
        return cls(-a.re, -a.im)
    @classmethod
    def mul(cls, a, b):
        if a.isnan or b.isnan:
            return cls.nan()
        # (a.re + i a.im)(b.re + i b.im)
        # = a.re b.re - a.im b.im + i(a.im b.re + a.re b.im)
        re = maths.mul(a.re, b.re) - maths.mul(a.im, b.im)
        im = maths.mul(a.im, b.re) - maths.mul(a.re, b.im)
        return cls(re, im)
    @classmethod
    def rec(cls, a):
        if a.isnan:
            return cls.nan()
        if a.re == 0.0 and a.im == 0.0:
            raise ZeroDivisionError("1/0")
        # 1/(a.re + i a.im)
        # = (a.re - i a.im)/(a.re^2 + a.im^2)
        d = maths.mul(a.re, a.re) + maths.mul(a.im, a.im)
        return cls(a.re / d, -a.im / d)
    @classmethod
    def exp(cls, a):
        if a.isnan:
            return cls.nan()
        # e^(a.re + i a.im)
        # = e^a.re * e^(i a.im)
        # = e^a.re * (cos(a.im) + i sin(a.im))
        re = maths.mul(maths.exp(a.re), maths.cos(a.im))
        im = maths.mul(maths.exp(a.re), maths.sin(a.im))
        return cls(re, im)
    @classmethod
    def log(cls, a):
        if a.isnan:
            return cls.nan()
        # log(a)
        # = log(|a|) + i arg(a)  [principal branch]
        absa = maths.mul(a.re, a.re) + maths.mul(a.im, a.im)
        re = maths.mul(0.5, maths.log(absa))
        im = maths.atan2(a.im, a.re)
        return cls(re, im)

    @classmethod
    def eq_zero(cls, a):
        if a.isnan:
            return False
        return a.re == 0.0 and a.im == 0.0
    @classmethod
    def lt_zero(cls, a):
        if a.isnan: # nan is unorderable.
            raise NotImplementedError()
        if a.im: # complex is unorderable.
            raise NotImplementedError()
        return a.re < 0.0

    def floatof(a):
        if a.isnan:
            return float("nan")
        if a.im:
            raise NotImplementedError()
        return float(a.re)
    def complexof(a):
        if a.isnan:
            return complex("nan+nanj")
        return complex(a.re, a.im)

    def hashof(a):
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
