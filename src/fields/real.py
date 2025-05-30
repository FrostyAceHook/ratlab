import math

import matrix
from util import classconst, immutable



@immutable
class Real(matrix.Field):
    def __init__(self, v=0.0):
        if not isinstance(v, float):
            raise TypeError("value must be a float")
        self._v = v
        self.isnan = (v != v)

    @classmethod
    def from_int(cls, x):
        return cls(float(x))
    @classmethod
    def from_float(cls, x):
        return cls(x)
    @classmethod
    def from_complex(cls, x):
        if x.imag != 0.0:
            raise NotImplementedError()
        return cls(x.real)

    @classmethod
    def to_int(cls, a):
        if math.isinf(a._v) or a._v != int(a._v):
            raise NotImplementedError()
        return int(a._v)
    @classmethod
    def to_float(cls, a):
        return a._v
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
        return cls(a._v + b._v)
    @classmethod
    def sub(cls, a, b):
        return cls(a._v - b._v)
    @classmethod
    def absolute(cls, a):
        return cls(abs(a._v))

    @classmethod
    def mul(cls, a, b):
        return cls(a._v * b._v)
    @classmethod
    def div(cls, a, b):
        return cls(a._v / b._v)

    @classmethod
    def power(cls, a, b):
        return cls(a._v ** b._v)
    @classmethod
    def log(cls, a, b):
        return cls(math.log(b._v) / math.log(a._v))

    @classmethod
    def sin(cls, a):
        return cls(math.sin(a._v))
    @classmethod
    def cos(cls, a):
        return cls(math.cos(a._v))
    @classmethod
    def tan(cls, a):
        return cls(math.tan(a._v))

    @classmethod
    def asin(cls, a):
        return cls(math.asin(a._v))
    @classmethod
    def acos(cls, a):
        return cls(math.acos(a._v))
    @classmethod
    def atan(cls, a):
        return cls(math.atan(a._v))

    @classconst
    def yes(cls):
        return True
    @classmethod
    def eq(cls, a, b):
        return a._v == b._v
    @classmethod
    def lt(cls, a, b):
        return a._v < b._v

    @classmethod
    def hashed(cls, a):
        return hash(a._v)

    @classmethod
    def repr_short(cls, a):
        return "0" if a._v == 0.0 else f"{a._v:.6g}"
    @classmethod
    def repr_long(cls, a):
        return "0" if a._v == 0.0 else repr(a._v)
