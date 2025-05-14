from types import GeneratorType

class Field:
    # override me:

    @classmethod
    def zero(cls): # additive identity.
        raise NotImplementedError()
    @classmethod
    def one(cls): # multiplicative identity.
        raise NotImplementedError()

    @classmethod
    def cast(cls, obj, for_obj=None): # returns a cls version of obj
        raise NotImplementedError()

    @classmethod
    def add(cls, a, b): # a+b
        raise NotImplementedError()
    @classmethod
    def neg(cls, a): # -a
        raise NotImplementedError()
    @classmethod
    def mul(cls, a, b): # a*b
        raise NotImplementedError()
    @classmethod
    def rec(cls, a): # 1/a
        raise NotImplementedError()
    @classmethod
    def exp(cls, a): # e^a
        raise NotImplementedError()
    @classmethod
    def log(cls, a): # ln(a)
        raise NotImplementedError()

    @classmethod
    def eq_zero(cls, a): # a == 0
        raise NotImplementedError()
    @classmethod
    def eq_one(cls, a): # a == 1, only needed if non-additive type
        return (a - cls.one()).eq_zero()
    @classmethod
    def lt_zero(cls, a): # a < 0
        raise NotImplementedError()
    @classmethod
    def lt_one(cls, a): # a < 1, only needed if non-additive type
        return (a - cls.one()).lt_zero()

    def intof(a): # int(a)
        raise NotImplementedError()
    def floatof(a): # float(a)
        return float(a.intof())
    def complexof(a): # complex(a)
        return complex(a.floatof())

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

    @classmethod
    def _cast(cls, obj, for_obj=None):
        if isinstance(obj, cls):
            return obj
        return cls.cast(obj, for_obj=for_obj)

    @classmethod
    def _apply(cls, a, b, func):
        if isinstance(b, (tuple, list, set, GeneratorType)):
            return [cls._apply(a, c, func) for c in b]
        else:
            b = cls._cast(b, for_obj=a)
            return func(a, b)

    def __pos__(s):
        return s
    def __neg__(s):
        return type(s).neg(s)
    def __invert__(s):
        if type(s).eq_zero(s):
            raise ZeroDivisionError("~0")
        return type(s).rec(s)
    def __abs__(s):
        return type(s).neg(s) if s < type(s).zero() else s

    def __add__(s, o):
        return type(s)._apply(s, o, lambda a, b: type(s).add(a, b))
    def __radd__(s, o):
        return type(s)._apply(s, o, lambda a, b: type(s).add(b, a))
    def __sub__(s, o):
        return type(s)._apply(s, o, lambda a, b: type(s).add(a, type(s).neg(b)))
    def __rsub__(s, o):
        return type(s)._apply(s, o, lambda a, b: type(s).add(b, type(s).neg(a)))
    def __mul__(s, o):
        return type(s)._apply(s, o, lambda a, b: type(s).mul(a, b))
    def __rmul__(s, o):
        return type(s)._apply(s, o, lambda a, b: type(s).mul(b, a))
    def __truediv__(s, o):
        return type(s)._apply(s, o, lambda a, b: type(s).mul(a, type(s).rec(b)))
    def __rtruediv__(s, o):
        return type(s)._apply(s, o, lambda a, b: type(s).mul(b, type(s).rec(a)))
    def __pow__(s, o):
        # x^y == e^(ln(x) * y)
        # x == 0 case must be handled by class internals.
        cls = type(s)
        def xor(a, b):
            def eq_zero(x):
                try:
                    return x == type(x).zero()
                except NotImplementedError:
                    return False
            if eq_zero(a) and eq_zero(b):
                raise ZeroDivisionError("0^0")
            ab = cls.exp(cls.log(a) * b)
            if eq_zero(a) and not eq_zero(ab):
                raise ZeroDivisionError("0^non-positive")
            return ab
        return type(s)._apply(s, o, xor)
    def __rpow__(s, o):
        cls = type(s)
        def rxor(a, b):
            def eq_zero(x):
                try:
                    return x == cls.zero()
                except NotImplementedError:
                    return False
            if eq_zero(a) and eq_zero(b):
                raise ZeroDivisionError("0^0")
            ba = cls.exp(cls.log(b) * a)
            if eq_zero(b) and not eq_zero(ba):
                raise ZeroDivisionError("0^non-positive")
            return ba
        return type(s)._apply(s, o, rxor)

    def __eq__(s, o):
        # (s == o) iff (s - o == 0)
        # however can use (s / o == 1), provided o is non-zero.
        cls = type(s)
        def eq_zero(a, b):
            try:
                return cls.eq_zero(a - b)
            except NotImplementedError:
                try:
                    if cls.eq_zero(b):
                        return cls.eq_zero(a)
                except:
                    # assume b is non-zero.
                    pass
                return cls.eq_one(a / b)
        return type(s)._apply(s, o, eq_zero)
    def __lt__(s, o):
        # (s < o) iff (s - o < 0)
        # note cannot use (s / o < 0) since may be divving negative.
        return type(s)._apply(s, o, lambda a, b: type(s).lt_zero(a - b))
    def __ne__(s, o):
        return type(s)._apply(s, o, lambda a, b: not (a == b))
    def __le__(s, o):
        return type(s)._apply(s, o, lambda a, b: (a < b) or (a == b))
    def __gt__(s, o):
        return type(s)._apply(s, o, lambda a, b: (b < a))
    def __ge__(s, o):
        return type(s)._apply(s, o, lambda a, b: (b < a) or (a == b))

    def __bool__(s):
        try:
            x = type(s).zero()
        except NotImplementedError:
            x = type(s).one()
        return s != x
    def __int__(s):
        return s.intof()
    def __float__(s):
        return s.floatof()
    def __complex__(s):
        return s.complexof()

    def __hash__(s):
        return s.hashof()
