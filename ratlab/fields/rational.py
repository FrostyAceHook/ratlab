import math

import matrix
from util import tname, classconst, immutable, simplest_ratio


@immutable
class Rational(matrix.Field):
    def __init__(self, numerator, denominator=1):
        if not isinstance(numerator, int):
            raise TypeError("expected an integer numerator, got "
                    f"{tname(type(numerator))}")
        if not isinstance(denominator, int):
            raise TypeError("expected an integer denominator, got "
                    f"{tname(type(denominator))}")
        if denominator < 0:
            numerator = -numerator
            denominator = -denominator
        if denominator == 0:
            raise ZeroDivisionError("x/0")
        gcd = math.gcd(numerator, denominator)
        # note gcd(0,x)==x, therefore collapses 0/1
        self._nu = numerator // gcd
        self._de = denominator // gcd
    @property
    def numer(s):
        """
        Element-wise numerator.
        """
        return type(s)(s._nu)
    @property
    def denom(s):
        """
        Element-wise denominator.
        """
        return type(s)(s._de)


    @classmethod
    def from_int(cls, x):
        return cls(x)
    @classmethod
    def from_float(cls, x):
        return cls(*simplest_ratio(x))
    @classmethod
    def from_complex(cls, x):
        if x.imag != 0.0:
            raise NotImplementedError()
        return cls(*simplest_ratio(x.real))

    @classmethod
    def to_int(cls, a):
        if a._de != 1:
            raise NotImplementedError()
        return a._nu
    @classmethod
    def to_float(cls, a):
        return a._nu / a._de
    @classmethod
    def to_complex(cls, a):
        return complex(a._nu / a._de)

    @classconst
    def zero(cls):
        return cls(0)
    @classconst
    def one(cls):
        return cls(1)

    @classconst
    def exposes(cls):
        return {"numer": cls, "denom": cls}

    @classmethod
    def add(cls, a, b):
        return cls(a._nu * b._de + b._nu * a._de, a._de * b._de)
    @classmethod
    def sub(cls, a, b):
        return cls(a._nu * b._de - b._nu * a._de, a._de * b._de)
    @classmethod
    def absolute(cls, a):
        return cls(abs(a._nu), a._de)

    @classmethod
    def mul(cls, a, b):
        return cls(a._nu * b._nu, a._de * b._de)
    @classmethod
    def div(cls, a, b):
        return cls(a._nu * b._de, a._de * b._nu)

    @classmethod
    def issame(cls, a, b):
        return a._nu == b._nu and a._de == b._de
    @classmethod
    def lt(cls, a):
        return a._nu * b._de < b._nu * a._de

    @classmethod
    def hashed(cls, a):
        return hash((a._nu, a._de))

    @classmethod
    def rep(cls, a, short):
        s = str(a._nu) + (a._de != 1) * f"/{a._de}"
        if short and len(s) > 10:
            s = f"≈{(a._nu / a._de):.6g}"
        return s

    def exp_as_string(self):
        if self._de == 1:
            if self._nu == 1:
                return ""
            superscripts = {
                "-": "⁻", "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴",
                "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹",
            }
            return "".join(superscripts[c] for c in str(self._nu))
        lookup = {
            Rational(1, 2): "½", Rational(1, 3): "⅓", Rational(2, 3): "⅔",
            Rational(1, 4): "¼", Rational(3, 4): "¾", Rational(1, 5): "⅕",
            Rational(2, 5): "⅖", Rational(3, 5): "⅗", Rational(4, 5): "⅘",
            Rational(1, 6): "⅙", Rational(5, 6): "⅚", Rational(1, 8): "⅛",
            Rational(3, 8): "⅜", Rational(5, 8): "⅝", Rational(7, 8): "⅞",
        }
        if self in lookup:
            s = lookup[self]
        elif (-self) in lookup:
            s = "-" + lookup[-self]
        else:
            s = f"({repr(self)})"
        return "^" + s
