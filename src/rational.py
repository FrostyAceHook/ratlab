import maths
from field import Field
from immutable import immutable


@immutable
class Rational(Field):
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
        gcd = maths.gcd(numerator, denominator)
        # note gcd(0,x)==x, therefore collapses 0/1
        self.nu = numerator // gcd
        self.de = denominator // gcd

    @classmethod
    def mapping(cls):
        return {
            int: (lambda x: Rational.zero().cast(x)),
            float: (lambda x: Rational.zero().cast(x)),
        }
    def cast(self, obj):
        if isinstance(obj, int):
            return Rational(obj, 1)
        if isinstance(obj, float):
            return Rational(*maths.simplest_ratio(obj))
        raise NotImplementedError()

    @classmethod
    def zero(cls):
        return Rational(0, 1)
    @classmethod
    def one(cls):
        return Rational(1, 1)

    def add(a, b):
        return Rational(a.nu * b.de + b.nu * a.de, a.de * b.de)
    def neg(a):
        return Rational(-a.nu, a.de)
    def mul(a, b):
        return Rational(a.nu * b.nu, a.de * b.de)
    def rec(a):
        return Rational(a.de, a.nu)

    def eq_zero(a):
        return a.nu == 0
    def lt_zero(a):
        return a.nu < 0

    def floatof(a):
        return a.nu / a.de

    def hashof(a):
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
