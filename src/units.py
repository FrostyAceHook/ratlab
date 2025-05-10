from enum import Enum

import maths
from field import Field
from immutable import immutable
from rational import Rational


class Scale(Enum):
    NANO = (1e-9, "n")
    MICRO = (1e-6, "μ")
    MILLI = (1e-3, "m")
    NONE = (1, "")
    KILO = (1e3, "k")
    MEGA = (1e6, "M")
    GIGA = (1e9, "G")

    @property
    def magnitude(self):
        return self._value_[0]
    @property
    def letter(self):
        return self._value_[1]


@immutable
class Unit(Field):
    BASE = ("m", "g", "s", "A", "K")

    def can_scale(self):
        if self in Unit.COMPOUNDS:
            return True
        return len(self.bases) == 1 and self.bases[0][1].de == 1


    def __init__(self, bases=(), *, logged=False, const=None):
        # field behaves as multiplicative only, however .

        had = set()
        cache = []
        for k, v in bases:
            if not isinstance(k, str):
                raise TypeError("unit bases must be strings")
            if k not in Unit.BASE:
                raise ValueError(f"invalid base unit {repr(k)}")
            if k in had:
                raise ValueError(f"repeated base unit {repr(k)}")
            had.add(k)
            if isinstance(v, (int, float)):
                v = Rational.zero().cast(v)
            if not isinstance(v, Rational):
                raise TypeError("unit powers must be rationals")
            if v:
                cache.append((k, v))
        cache = sorted(cache, key=lambda x: Unit.BASE.index(x[0]))

        if isinstance(const, (int, float)):
            const = Rational.zero().cast(const)
        if const is not None and not isinstance(const, Rational):
            raise TypeError("const must be none or a rational")
        self.const = const
        if const is not None and (cache or logged):
            raise ValueError("cannot specify units if const")

        self.bases = tuple(cache)
        self.logged = logged

    def cast(self, obj):
        if isinstance(obj, (int, float, Rational)):
            return Unit(const=obj)
        raise NotImplementedError()

    @classmethod
    def zero(cls):
        return Unit(logged=True)
    @classmethod
    def one(cls):
        return Unit()

    def add(a, b):
        isints = (a.const is not None) + (b.const is not None)
        if isints == 1:
            raise NotImplementedError("cannot add const and units")
        if isints == 2:
            return Unit(const=a.const + b.const)

        if not a.logged or not b.logged:
            raise NotImplementedError("cannot add non-logged units")
        bases = {k: v for k, v in a.bases}
        for name, power in b.bases:
            if name in bases:
                bases[name] += power
            else:
                bases[name] = power
        return Unit(bases.items(), logged=True)

    def neg(a):
        if a.const is not None:
            return Unit(const=-a.const)
        if not a.logged:
            raise NotImplementedError("cannot neg non-logged units")
        return Unit([(k, -v) for k, v in a.bases], logged=True)

    def mul(a, b):
        isints = (a.const is not None) + (b.const is not None)
        if isints == 1:
            if b.logged:
                a, b = b, a
            if not a.logged:
                raise NotImplementedError("cannot mul const and non-logged "
                        "units")
            return Unit([(k, v*b.const) for k, v in a.bases], logged=True)
        if isints == 2:
            return Unit(const=a.const * b.const)

        if a.logged or b.logged:
            raise NotImplementedError("cannot mul logged units")
        bases = {k: v for k, v in a.bases}
        for name, power in b.bases:
            if name in bases:
                bases[name] += power
            else:
                bases[name] = power
        return Unit(bases.items())

    def rec(a):
        if a.const is not None:
            return Unit(const=~a.const)
        if a.logged:
            raise NotImplementedError("cannot rec logged units")
        return Unit([(k, -v) for k, v in a.bases])

    def exp(a):
        if a.const is not None:
            raise NotImplementedError("cannot exp const")
        if not a.logged:
            raise NotImplementedError("cannot exp non-logged units")
        return Unit(a.bases, logged=False)
    def log(a):
        if a.const is not None:
            raise NotImplementedError("cannot log const")
        if a.logged:
            raise NotImplementedError("cannot double-log units")
        return Unit(a.bases, logged=True)

    def eq_zero(a):
        if a.const is not None:
            return a.const == 0
        if not a.logged:
            raise NotImplementedError("cannot compare non-logged units with "
                    "zero")
        return not a.bases
    def eq_one(a):
        if a.const is not None:
            return a.const == 1
        if a.logged:
            raise NotImplementedError("cannot compare logged units with one")
        return not a.bases

    def hashof(a):
        return hash((a.bases, a.logged))

    def __repr__(self):
        if self.const is not None:
            return f"units({self.const})"

        wrap = lambda s: f"log({s})" if self.logged else s

        if not self.bases:
            return wrap("unitless")

        # Alias common units.
        if self in Unit.COMPOUNDS:
            return wrap(Unit.COMPOUNDS[self])

        parts = []
        for name, power in self.bases:
            if parts:
                if power < 0:
                    power = -power
                    parts.append("/")
                else:
                    parts.append("∙")
            parts.append(f"{name}{power.exp_as_string()}")

        return wrap("".join(parts))

Unit.none = Unit([])



@immutable
class Quantity(Field):
    @property
    def bare(self):
        return Quantity(self.value, self.unit, isbare=True)
    @property
    def united(self):
        return Quantity(self.value, self.unit, isbare=False)

    @property
    def unitless(self):
        if self.unit != Unit.none:
            raise ValueError("value has units")
        return self.value

    def ideal_scale(self):
        if not self.unit.can_scale():
            return Scale.NONE
        value = abs(self.value)
        if value == 0.0:
            return Scale.NONE
        scales = sorted(Scale, key=lambda x: x.magnitude)
        mags = [x.magnitude for x in scales]
        # Handle the fact that "mm^2" is consider "(mm)^2" and not "m (m^2)".
        if len(self.unit.bases) == 1:
            power = self.unit.bases[0][1]
            if power.de == 1:
                mags = [x ** power.nu for x in mags]
        # Find the best scale.
        for scale, mag in zip(reversed(scales), reversed(mags)):
            if 1.0 <= value / mag:
                return scale
        return scales[0]

    def display_scaled(self, scale, prec=5):
        if scale != Scale.NONE and not self.unit.can_scale():
            raise ValueError("unit cannot be scaled")
        mag = scale.magnitude
        # Handle the fact that "mm^2" is consider "(mm)^2" and not "m (m^2)".
        if len(self.unit.bases) == 1:
            power = self.unit.bases[0][1]
            if power.de == 1:
                mag **= power.nu
        # Display the thing.
        value = f"{self.value / mag:.{prec}g}"
        unit = scale.letter + repr(self.unit)
        if self.isbare:
            return f"{value} ¿{unit}?"
        else:
            return f"{value} {unit}"


    def __init__(self, value=0.0, unit=Unit.none, *, loggedunits=False,
            isbare=False):
        if isinstance(value, int):
            value = float(value)
        if not isinstance(value, float):
            raise TypeError("value must be a float")
        if not isinstance(unit, Unit):
            raise TypeError("unit must be a Unit")
        if not isinstance(loggedunits, bool):
            raise TypeError("loggedunits must be a bool")
        if not isinstance(isbare, bool):
            raise TypeError("isbare must be a bool")
        self.value = value
        self.unit = unit
        self.loggedunits = loggedunits
        self.isbare = isbare

    @classmethod
    def zero(cls):
        return Quantity(0, Unit.none)
    @classmethod
    def one(cls):
        return Quantity(1, Unit.none)

    def cast(self, obj):
        if isinstance(obj, (int, float)):
            return Quantity(obj, Unit.none)
        raise NotImplementedError()

    def add(a, b):
        # ignore units if value is 0.
        if b.value == 0 and b.unit == Unit.none and not b.isbare:
            return a
        if a.value == 0 and a.unit == Unit.none and not a.isbare:
            return b
        # handle logged units (which only work when adding unitless).
        if b.loggedunits: # make a logged units.
            c = a
            a = b
            b = c
        if a.loggedunits:
            if (b.unit != Unit.none and not b.bare) or b.loggedunits:
                raise ValueError("can only add unitless with logged units")
            return Quantity(a.value + b.value, a.unit, loggedunits=True,
                    isbare=a.isbare)
        # handle bare.
        if a.isbare + b.isbare == 1:
            raise ValueError("quantites must be both bare or both united")
        unit = a.unit if a.isbare else a.unit
        if not a.isbare and a.unit != b.unit:
            raise ValueError(f"units do not agree ('{a.unit}' vs '{b.unit}')")
        return Quantity(a.value + b.value, unit, isbare=a.isbare)
    def neg(a):
        if a.isbare:
            return Quantity(-a.value, a.unit, isbare=True)
        if a.loggedunits:
            return Quantity(-a.value, a.unit, loggedunits=True)
        return Quantity(-a.value, a.unit)
    def mul(a, b):
        if a.isbare + b.isbare == 1:
            raise ValueError("quantites must be both bare or both united")
        if a.isbare:
            return Quantity(a.value * b.value, a.unit, isbare=True)
        if a.loggedunits or b.loggedunits:
            raise ValueError("cannot multiply logged units")
        return Quantity(a.value * b.value, a.unit * b.unit)
    def rec(a):
        if a.isbare:
            return Quantity(1.0 / a.value, a.unit, isbare=True)
        if a.loggedunits:
            raise ValueError("cannot reciprocate logged units")
        return Quantity(1.0 / a.value, Unit.none / a.unit)
    def exp(a):
        if a.isbare:
            return Quantity(maths.exp(a.value), a.unit, isbare=True)
    def log(a):
        if not a.isbare:
            raise ValueError("cannot logarithm united value (use bare)")
        return Quantity(maths.log(a.value), a.unit, isbare=True)

    def eq_zero(a):
        return a.value == 0.0
    def lt_zero(a):
        return a.value < 0.0

    def floatof(a):
        return a.value

    def hashof(a):
        return hash((a.value, a.unit))

    def __repr__(a):
        return a.display_scaled(a.ideal_scale())


class Toleranced:
    @property
    def bounds(self):
        return (self.lsl, self.usl)

    def __init__(self, nominal, lsl, usl):
        if usl < lsl:
            raise ValueError("usl cannot be greater than lsl")
        self.nominal = nominal
        self.lsl = lsl
        self.usl = usl

    def __repr__(self):
        scale = self.nominal.ideal_scale()
        l = self.lsl - self.nominal
        u = self.usl - self.nominal
        l = "+"*(l >= 0) + l.display_scaled(scale)
        u = "+"*(u >= 0) + u.display_scaled(scale)
        return f"{self.nominal.display_scaled(scale)} ({l}) ({u})"


m = Quantity(1, Unit([("m", 1)]))
cm = 1e-2 * m
mm = 1e-3 * m
um = 1e-6 * m
nm = 1e-9 * m
km = 1e3 * m

m2 = m ** 2
cm2 = cm ** 2
mm2 = mm ** 2
um2 = um ** 2
nm2 = nm ** 2
km2 = km ** 2

m3 = m ** 3
cm3 = cm ** 3
mm3 = mm ** 3
um3 = um ** 3
nm3 = nm ** 3
km3 = km ** 3

kg = Quantity(1e3, Unit([("g", 1)]))
g = 1e-3 * kg
mg = 1e-6 * kg
ug = 1e-9 * kg
ng = 1e-12 * kg

s = Quantity(1, Unit([("s", 1)]))
ms = 1e-3 * s
min = 60 * s
hr = 60*60 * s

s2 = s ** 2
ms2 = ms ** 2

A = Quantity(1, Unit([("A", 1)]))

K = Quantity(1, Unit([("K", 1)]))

N = kg * m / s2
kN = 1e3 * N

Pa = N / m2
kPa = 1e3 * Pa
MPa = 1e6 * Pa
GPa = 1e9 * Pa

J = N * m
mJ = 1e-3 * J
kJ = 1e3 * J
MJ = 1e6 * J

W = J / s
mW = 1e-3 * W
kW = 1e3 * W
MW = 1e6 * W

V = W / A
mV = 1e-3 * V
kV = 1e3 * V
MV = 1e6 * V

Ohm = V / A
mOhm = 1e-3 * Ohm
kOhm = 1e3 * Ohm
MOhm = 1e6 * Ohm


Unit.COMPOUNDS = {
    N.unit: "N", Pa.unit: "Pa", J.unit: "J", W.unit: "W", V.unit: "V",
    Ohm.unit: "Ω"
}
