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

    def eq_zero(a, b):
        return a.nu == 0
    def lt_zero(a):
        return a.nu < 0

    def floatof(a):
        return a.nu / a.de

    def hashof(a):
        return hash((a.nu, a.de))

    def __repr__(self):
        return str(self.nu) + (self.de != 1) * f"/{self.de}"
