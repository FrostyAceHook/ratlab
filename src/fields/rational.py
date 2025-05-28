import math

import field
from numerical_methods import simplest_ratio
from util import immutable


@immutable
class Rational(field.Field):
    def __init__(self, numerator, denominator):
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
    def _cast(cls, obj, for_obj):
        if not isinstance(obj, float):
            try:
                obj = float(obj)
            except Exception:
                pass
        if isinstance(obj, float):
            return cls(*simplest_ratio(obj))
        return super()._cast(obj, for_obj)

    @classmethod
    def _zero(cls):
        return cls(0, 1)
    @classmethod
    def _one(cls):
        return cls(1, 1)

    @classmethod
    def _add(cls, a, b):
        return cls(a.nu * b.de + b.nu * a.de, a.de * b.de)
    @classmethod
    def _neg(cls, a):
        return cls(-a.nu, a.de)
    @classmethod
    def _mul(cls, a, b):
        return cls(a.nu * b.nu, a.de * b.de)
    @classmethod
    def _rec(cls, a):
        return cls(a.de, a.nu)

    @classmethod
    def _eq_zero(cls, a):
        return a.nu == 0
    @classmethod
    def _lt_zero(cls, a):
        return a.nu < 0

    @classmethod
    def _intof(cls, a):
        if a.de != 1:
            raise NotImplementedError()
        return a.nu
    @classmethod
    def _floatof(cls, a):
        return a.nu / a.de

    @classmethod
    def _hashof(cls, a):
        return hash((a.nu, a.de))

    def __repr__(self):
        return str(self.nu) + (self.de != 1) * f"/{self.de}"

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
