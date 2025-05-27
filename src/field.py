import math
from math import pi

from util import iterable, classproperty, tname


class Field:
    # PUBLIC:

    @classmethod
    def cast(cls, obj, for_obj=None):
        if isinstance(obj, cls):
            return obj
        return cls._cast(obj, for_obj)

    @classproperty
    def zero(cls): # additive identity.
        return cls._zero()
    @classproperty
    def one(cls): # multiplicative identity.
        return cls._one()

    @property
    def neg(a): # -a
        return -a
    @property
    def rec(a): # 1/a
        return ~a
    @property
    def abs(a): # |a|
        return abs(a)
    @property
    def exp(a): # e^a
        return type(a)._exp(a)
    @property
    def log(a): # ln(a)
        return type(a)._log(a)
    @property
    def sqrt(a):
        return a.root(2)
    @property
    def cbrt(a):
        return a.root(3)

    def root(self, n):
        if isinstance(n, float):
            raise TypeError(".root is for integer roots, use `**` for "
                    "exponentiation")
        n = int(n)
        if n == 0:
            raise ZeroDivisionError("x^(1/0)")
        if n < 0:
            return ~self.root(-n)
        try:
            nth = type(self).cast(n, for_obj=self)
        except NotImplementedError:
            # hand crank this bitch.
            one = type(self).one
            nth = type(self).zero
            for _ in range(n):
                nth += one
        nth = ~nth
        return self ** nth



    # OVERRIDE ME:

    @classmethod
    def _cast(cls, obj, for_obj): # returns a cls version of obj
        raise NotImplementedError(f"'{tname(type(obj))}' cannot be cast to "
                f"'{tname(cls)}'")

    @classmethod
    def _zero(cls): # additive identity.
        raise NotImplementedError()
    @classmethod
    def _one(cls): # multiplicative identity.
        raise NotImplementedError()

    @classmethod
    def _add(cls, a, b): # a+b
        raise NotImplementedError()
    @classmethod
    def _neg(cls, a): # -a
        raise NotImplementedError()
    @classmethod
    def _mul(cls, a, b): # a*b
        raise NotImplementedError()
    @classmethod
    def _rec(cls, a): # 1/a
        raise NotImplementedError()
    @classmethod
    def _exp(cls, a): # e^a
        raise NotImplementedError()
    @classmethod
    def _log(cls, a): # ln(a)
        raise NotImplementedError()

    @classmethod
    def _eq_zero(cls, a): # a == 0
        raise NotImplementedError()
    @classmethod
    def _eq_one(cls, a): # a == 1, only needed if non-additive type
        return cls._eq_zero(a - cls.one)
    @classmethod
    def _lt_zero(cls, a): # a < 0
        raise NotImplementedError()
    @classmethod
    def _lt_one(cls, a): # a < 1, only needed if non-additive type
        return cls._lt_zero(a - cls.one)

    @classmethod
    def _intof(cls, a): # int(a)
        raise NotImplementedError()
    @classmethod
    def _floatof(cls, a): # float(a)
        return float(cls._intof(a))
    @classmethod
    def _complexof(cls, a): # complex(a)
        return complex(cls._floatof(a))

    @classmethod
    def _hashof(cls, a): # hash(a)
        raise NotImplementedError()




    # don look:

    @classmethod
    def _do(cls, a, b, func):
        b = cls.cast(b, for_obj=a)
        return func(a, b)

    def __pos__(s):
        return s
    def __neg__(s):
        return type(s)._neg(s)
    def __invert__(s):
        if type(s)._eq_zero(s):
            raise ZeroDivisionError("~0")
        return type(s)._rec(s)
    def __abs__(s):
        return type(s)._neg(s) if s < type(s)._zero() else s

    def __add__(s, o):
        return type(s)._do(s, o, lambda a, b: type(s)._add(a, b))
    def __radd__(s, o):
        return type(s)._do(s, o, lambda a, b: type(s)._add(b, a))
    def __sub__(s, o):
        return type(s)._do(s, o, lambda a, b: type(s)._add(a, type(s)._neg(b)))
    def __rsub__(s, o):
        return type(s)._do(s, o, lambda a, b: type(s)._add(b, type(s)._neg(a)))
    def __mul__(s, o):
        return type(s)._do(s, o, lambda a, b: type(s)._mul(a, b))
    def __rmul__(s, o):
        return type(s)._do(s, o, lambda a, b: type(s)._mul(b, a))
    def __truediv__(s, o):
        return type(s)._do(s, o, lambda a, b: type(s)._mul(a, type(s)._rec(b)))
    def __rtruediv__(s, o):
        return type(s)._do(s, o, lambda a, b: type(s)._mul(b, type(s)._rec(a)))
    def __pow__(s, o):
        # x^y == e^(ln(x) * y)
        # x == 0 case must be handled by class internals.
        cls = type(s)
        def xor(a, b):
            def eq_zero(x):
                try:
                    return cls._eq_zero(x)
                except NotImplementedError:
                    return False
            if eq_zero(a) and eq_zero(b):
                raise ZeroDivisionError("0^0")
            ab = cls._exp(cls._log(a) * b)
            if eq_zero(a) and not eq_zero(ab):
                raise ZeroDivisionError("0^non-positive")
            return ab
        return type(s)._do(s, o, xor)
    def __rpow__(s, o):
        cls = type(s)
        def rxor(a, b):
            def eq_zero(x):
                try:
                    return cls._eq_zero(x)
                except NotImplementedError:
                    return False
            if eq_zero(a) and eq_zero(b):
                raise ZeroDivisionError("0^0")
            ba = cls._exp(cls._log(b) * a)
            if eq_zero(b) and not eq_zero(ba):
                raise ZeroDivisionError("0^non-positive")
            return ba
        return type(s)._do(s, o, rxor)

    def __eq__(s, o):
        # (s == o) iff (s - o == 0)
        # however can use (s / o == 1), provided o is non-zero.
        cls = type(s)
        def eq_zero(a, b):
            try:
                return cls._eq_zero(a - b)
            except NotImplementedError:
                try:
                    if cls._eq_zero(b):
                        return cls._eq_zero(a)
                except Exception:
                    # assume b is non-zero.
                    pass
                return cls._eq_one(a / b)
        return type(s)._do(s, o, eq_zero)
    def __lt__(s, o):
        # (s < o) iff (s - o < 0)
        # note cannot use (s / o < 0) since may be divving negative.
        return type(s)._do(s, o, lambda a, b: type(s)._lt_zero(a - b))
    def __ne__(s, o):
        return type(s)._do(s, o, lambda a, b: not (a == b))
    def __le__(s, o):
        return type(s)._do(s, o, lambda a, b: (a < b) or (a == b))
    def __gt__(s, o):
        return type(s)._do(s, o, lambda a, b: (b < a))
    def __ge__(s, o):
        return type(s)._do(s, o, lambda a, b: (b < a) or (a == b))

    def __bool__(s):
        try:
            return not type(s)._eq_zero(s)
        except NotImplementedError:
            return not type(s)._eq_one(s)
    def __int__(s):
        return type(s)._intof(s)
    def __float__(s):
        return type(s)._floatof(s)
    def __complex__(s):
        return type(s)._complexof(s)

    def __hash__(s):
        return type(s)._hashof(s)


def fieldof(*xs):
    if len(xs) == 1 and iterable(xs[0]):
        return fieldof(*xs[0])
    field = None
    for x in xs:
        if field is None:
            field = type(x)
            if not issubclass(field, Field):
                raise TypeError("invalid field")
        elif not isinstance(x, field):
            raise TypeError("inconsistent field")
    return Field if field is None else field


def summ(*xs, field):
    if len(xs) == 1 and iterable(xs[0]):
        return summ(*xs[0], field=field)
    if not xs:
        return field.zero
    assert field == fieldof(xs)
    r = field.zero
    for x in xs:
        r += x
    return r

def prod(*xs, field):
    if len(xs) == 1 and iterable(xs[0]):
        return prod(*xs[0], field=field)
    if not xs:
        return field.one
    assert field == fieldof(xs)
    r = field.one
    for x in xs:
        r *= x
    return r

def ave(*xs, field):
    if len(xs) == 1 and iterable(xs[0]):
        return ave(*xs[0], field=field)
    if not xs:
        raise ValueError("cannot average no elements")
    assert field == fieldof(xs)
    r = field.zero
    i = field.zero
    for x in xs:
        r += x
        i += field.one
    return r / i


def sqrt(x):
    if isinstance(x, Field):
        return x.sqrt
    x = float(x)
    if x != x:
        return x
    return math.sqrt(x)

def cbrt(x):
    if isinstance(x, Field):
        return x.cbrt
    x = float(x)
    if x != x:
        return x
    return math.cbrt(x)

def log(x):
    if isinstance(x, Field):
        return x.log
    x = float(x)
    if x != x:
        return x
    if x == 0.0:
        return -float("inf")
    return math.log(x)

def exp(x):
    if isinstance(x, Field):
        return x.exp
    x = float(x)
    if x != x:
        return x
    return math.exp(x)

def cos(x):
    y = float(x)
    if y != y:
        pass
    elif math.isinf(y):
        y = float("nan")
    else:
        lookup = {pi/2: 0.0, -pi/2: 0.0, pi: -1.0, -pi: -1.0, 2*pi: 1.0}
        y = lookup.get(y, math.cos(y))
    if isinstance(x, Field):
        return type(x).cast(y, for_obj=x)
    return y

def sin(x):
    y = float(x)
    if y != y:
        pass
    elif math.isinf(y):
        y = float("nan")
    else:
        lookup = {pi/2: 1.0, -pi/2: -1.0, pi: 0.0, -pi: 0.0, 2*pi: 0.0}
        y = lookup.get(y, math.sin(y))
    if isinstance(x, Field):
        return type(x).cast(y, for_obj=x)
    return y
