import math

import matrix
from numerical_methods import simplest_ratio
from util import classconst, immutable


@immutable
class Rational(matrix.Field):
    def __init__(self, numerator, denominator=1):
        if not isinstance(numerator, int):
            raise TypeError("numerator must be an int")
        if not isinstance(denominator, int):
            raise TypeError("denominator must be an int")
        if denominator < 0:
            numerator = -numerator
            denominator = -denominator
        if denominator == 0:
            raise ZeroDivisionError("x/0")
        gcd = math.gcd(numerator, denominator)
        # note gcd(0,x)==x, therefore collapses 0/1
        self.nu = numerator // gcd
        self.de = denominator // gcd


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
        if a.de != 1:
            raise NotImplementedError()
        return a.nu
    @classmethod
    def to_float(cls, a):
        return a.nu / a.de
    @classmethod
    def to_complex(cls, a):
        return complex(a.nu / a.de)

    @classconst
    def zero(cls):
        return cls(0, 1)
    @classconst
    def one(cls):
        return cls(1, 1)

    @classmethod
    def add(cls, a, b):
        return cls(a.nu * b.de + b.nu * a.de, a.de * b.de)
    @classmethod
    def sub(cls, a, b):
        return cls(a.nu * b.de - b.nu * a.de, a.de * b.de)
    @classmethod
    def absolute(cls, a):
        return cls(abs(a.nu), a.de)

    @classmethod
    def mul(cls, a, b):
        return cls(a.nu * b.nu, a.de * b.de)
    @classmethod
    def div(cls, a, b):
        return cls(a.nu * b.de, a.de * b.nu)

    @classmethod
    def power(cls, a, b):
        raise NotImplementedError("haven don it yet")
    @classmethod
    def root(cls, a, b):
        raise NotImplementedError("haven don it yet")
    @classmethod
    def log(cls, a, b):
        raise NotImplementedError("haven don it yet")

    @classmethod
    def eq(cls, a, b):
        return a.nu == b.nu and a.de == b.de
    @classmethod
    def lt(cls, a):
        return a.nu * b.de < b.nu * a.de

    @classmethod
    def hashed(cls, a):
        return hash((a.nu, a.de))

    @classmethod
    def repr_short(cls, a):
        s = cls.repr_long(a)
        return f"≈{(a.nu / a.de):.6g}" if len(s) > 10 else s

    @classmethod
    def repr_long(cls, a):
        return str(a.nu) + (a.de != 1) * f"/{a.de}"

    def exp_as_string(self):
        if self.de == 1:
            if self.nu == 1:
                return ""
            superscripts = {
                "-": "⁻", "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴",
                "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹",
            }
            return "".join(superscripts[c] for c in str(self.nu))
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
