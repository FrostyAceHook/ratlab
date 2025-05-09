from types import GeneratorType

class Field:
    # override me:

    @classmethod
    def zero(cls): # additive identity.
        raise NotImplementedError()
    @classmethod
    def one(cls): # multiplicative identity.
        raise NotImplementedError()

    def cast(self, obj): # returns a type(self)
        raise NotImplementedError()

    def add(a, b): # a+b
        raise NotImplementedError()
    def neg(a): # -a
        raise NotImplementedError()
    def mul(a, b): # a*b
        raise NotImplementedError()
    def rec(a): # 1/a
        raise NotImplementedError()
    def exp(a): # e^a.
        raise NotImplementedError()
    def log(a): # ln(a)
        raise NotImplementedError()

    def eq_zero(a): # a == 0
        raise NotImplementedError()
    def lt_zero(a): # a < 0
        raise NotImplementedError()

    def intof(a): # int(a)
        raise NotImplementedError()
    def floatof(a): # float(a)
        raise NotImplementedError()
    def complexof(a): # complex(a)
        raise NotImplementedError()

    def hashof(a): # hash(a)
        raise NotImplementedError()


    # helpers (don touch but can look):

    @classmethod
    def sumof(cls, *xs):
        r = cls.zero()
        for x in xs:
            if isinstance(x, (tuple, list, set, GeneratorType)):
                r += cls.sumof(*x)
            else:
                r += x
        return r

    @classmethod
    def prodof(cls, *xs):
        r = cls.one()
        for x in xs:
            if isinstance(x, (tuple, list, set, GeneratorType)):
                r *= cls.sumof(*x)
            else:
                r *= x
        return r

    @classmethod
    def find(cls, x, ys, find_true=True):
        for i, y in enumerate(ys):
            if (x == y) == find_true:
                return i
        return None

    def root(self, n):
        if not isinstance(n, int):
            raise TypeError(".root is for integer roots, use `^` for "
                    "exponentiation")
        if n == 0:
            raise ZeroDivisionError("x^(1/0)")
        if n < 0:
            return ~self.root(-n)
        one = type(self).one()
        nth = ~type(self).sumof(one for _ in range(n))
        return self ^ nth
    @property
    def sqrt(self):
        return self.root(2)
    @property
    def cbrt(self):
        return self.root(3)



    # don look:

    def _cast(self, obj):
        if isinstance(obj, type(self)):
            return obj
        return self.cast(obj)

    def _apply(a, b, func):
        if isinstance(b, (tuple, list, set, GeneratorType)):
            return [a._apply(c, func) for c in b]
        else:
            b = a._cast(b)
            return func(a, b)

    def __pos__(self):
        return self
    def __neg__(self):
        return self.neg()
    def __invert__(self):
        if self.eq_zero():
            raise ZeroDivisionError("~0")
        return self.rec()
    def __abs__(self):
        return self.neg() if self.lt_zero() else self

    def __add__(self, other):
        return self._apply(other, lambda a, b: a.add(b))
    def __radd__(self, other):
        return self._apply(other, lambda a, b: b.add(a))
    def __sub__(self, other):
        return self._apply(other, lambda a, b: a.add(b.neg()))
    def __rsub__(self, other):
        return self._apply(other, lambda a, b: b.add(a.neg()))
    def __mul__(self, other):
        return self._apply(other, lambda a, b: a.mul(b))
    def __rmul__(self, other):
        return self._apply(other, lambda a, b: b.mul(a))
    def __truediv__(self, other):
        return self._apply(other, lambda a, b: a.mul(b.rec()))
    def __rtruediv__(self, other):
        return self._apply(other, lambda a, b: b.mul(a.rec()))
    def __xor__(self, other): # exponentiaton.
        # x^y == e^(ln(x) * y)
        # x == 0 case must be handled by class internals.
        def xor(a, b):
            if a.eq_zero() and b.eq_zero():
                raise ZeroDivisionError("0^0")
            ab = (a.log() * b).exp()
            if a.eq_zero() and not ab.eq_zero():
                raise ZeroDivisionError("0^non-positive")
            return ab
        return self._apply(other, xor)
    def __rxor__(self, other):
        def rxor(a, b):
            if a.eq_zero() and b.eq_zero():
                raise ZeroDivisionError("0^0")
            ba = (b.log() * a).exp()
            if b.eq_zero() and not ba.eq_zero():
                raise ZeroDivisionError("0^non-positive")
            return ba
        return self._apply(other, rxor)

    def __pow__(self, other): # repeated multiplication (or division).
        if not isinstance(other, int):
            raise TypeError("** is for repeated multiplication, use ^ for "
                    "exponentiation")
        if other < 0:
            return (self ** (-other)).rec()
        e = other
        p = self
        r = type(self).one()
        # iterative square incorporation.
        while e:
            if e & 1:
                r *= p
            p *= p
            e >>= 1
        return r

    def __eq__(self, other):
        # (self == other) iff (self - other == 0)
        return self._apply(other, lambda a, b: (a - b).eq_zero())
    def __lt__(self, other):
        other = self._cast(other)
        # (self < other) iff (self - other < 0)
        return self._apply(other, lambda a, b: (a - b).lt_zero())
    def __ne__(self, other):
        return self._apply(other, lambda a, b: not (a == b))
    def __le__(self, other):
        return self._apply(other, lambda a, b: (a < b) or (a == b))
    def __gt__(self, other):
        return self._apply(other, lambda a, b: (b < a))
    def __ge__(self, other):
        return self._apply(other, lambda a, b: (b < a) or (a == b))

    def __bool__(self):
        return not self.eq_zero()
    def __int__(self):
        return self.intof()
    def __float__(self):
        return self.floatof()
    def __complex__(self):
        return self.complexof()

    def __hash__(self):
        return self.hashof()
