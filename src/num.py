import math
from field import Field
from immutable import immutable


def mul(a, b):
    if a != a or b != b:
        return float("nan")
    if a == 0.0 or b == 0.0: # avoid 0*inf = nan
        return 0.0
    return a * b

def log(x):
    if x != x:
        return float("nan")
    if x == 0.0:
        return -float("inf")
    return math.log(x)

def exp(x):
    if x != x:
        return float("nan")
    return math.exp(x)

def cos(x):
    if x != x:
        return float("nan")
    if math.isinf(x):
        return float("nan")
    pi = math.pi
    lookup = {pi/2: 0.0, -pi/2: 0.0, pi: -1.0, -pi: -1.0, 2*pi: 1.0}
    return lookup.get(x, math.cos(x))

def sin(x):
    assert x == x
    if math.isinf(x):
        return float("nan")
    pi = math.pi
    lookup = {pi/2: 1.0, -pi/2: -1.0, pi: 0.0, -pi: 0.0, 2*pi: 0.0}
    return lookup.get(x, math.sin(x))



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
            raise TypeError("isnan must be true false")
        if math.isnan(re) or math.isnan(im):
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
    def zero(cls):
        return Num(0)
    @classmethod
    def one(cls):
        return Num(1)
    @classmethod
    def inf(cls):
        return Num(float("inf"))
    @classmethod
    def nan(cls):
        return Num(isnan=True)

    def cast(self, obj):
        if isinstance(obj, (int, float)):
            return Num(obj)
        if isinstance(obj, complex):
            isnan = math.isnan(obj.real) or math.isnan(obj.imag)
            return Num(obj.real, obj.imag, isnan=isnan)
        raise NotImplementedError()

    def add(a, b):
        if a.isnan or b.isnan:
            return Num.nan()
        return Num(a.re + b.re, a.im + b.im)
    def neg(a):
        if a.isnan:
            return Num.nan()
        return Num(-a.re, -a.im)
    def mul(a, b):
        if a.isnan or b.isnan:
            return Num.nan()
        # (a.re + i a.im)(b.re + i b.im)
        # = a.re b.re - a.im b.im + i(a.im b.re + a.re b.im)
        re = mul(a.re, b.re) - mul(a.im, b.im)
        im = mul(a.im, b.re) - mul(a.re, b.im)
        return Num(re, im)
    def rec(a):
        if a.isnan:
            return Num.nan()
        if a.re == 0.0 and a.im == 0.0:
            raise ZeroDivisionError("1/0")
        # 1/(a.re + i a.im)
        # = (a.re - i a.im)/(a.re^2 + a.im^2)
        d = mul(a.re, a.re) + mul(a.im, a.im)
        return Num(a.re / d, -a.im / d)
    def exp(a):
        if a.isnan:
            return Num.nan()
        # e^(a.re + i a.im)
        # = e^a.re * e^(i a.im)
        # = e^a.re * (cos(a.im) + i sin(a.im))
        re = mul(exp(a.re), cos(a.im))
        im = mul(exp(a.re), sin(a.im))
        return Num(re, im)
    def log(a):
        if a.isnan:
            return Num.nan()
        # log(a)
        # = log(|a|) + i arg(a)  [principal branch]
        re = mul(0.5, log(mul(a.re, a.re) + mul(a.im, a.im)))
        im = math.atan2(a.im, a.re)
        return Num(re, im)

    def eq_zero(a):
        if a.isnan:
            return False
        return a.re == 0.0 and a.im == 0.0
    def lt_zero(a):
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
