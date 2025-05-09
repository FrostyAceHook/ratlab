from enum import Enum
from field import Field
from immutable import immutable
import maths



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
class Unit:
    BASE = ("m", "g", "s", "A", "K")

    def can_scale(self):
        return len(self.bases) == 1 or self in Unit.COMPOUNDS


    def __init__(self, bases):
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
            if not isinstance(v, int):
                raise TypeError("unit powers must be integers")
            if v:
                cache.append((k, v))
        cache = sorted(cache, key=lambda x: Unit.BASE.index(x[0]))
        self.bases = tuple(cache)

    def __repr__(self):
        if self == Unit.none:
            return "unitless"

        # Alias common units.
        if self in Unit.COMPOUNDS:
            return Unit.COMPOUNDS[self]

        parts = []
        for name, power in self.bases:
            if parts:
                if power < 0:
                    power = -power
                    parts.append("/")
                else:
                    parts.append("∙")

            exp = f"^{power}" * (power != 1)
            parts.append(f"{name}{exp}")

        return "".join(parts)

    def __hash__(self):
        return hash(self.bases)


    def __mul__(a, b):
        if not isinstance(b, Unit):
            return NotImplemented
        bases = {k: v for k, v in a.bases}
        for name, power in b.bases:
            if name in bases:
                bases[name] += power
            else:
                bases[name] = power
        return Unit(bases.items())

    def __truediv__(a, b):
        if not isinstance(b, Unit):
            return NotImplemented
        bases = {k: v for k, v in a.bases}
        for name, power in b.bases:
            if name in bases:
                bases[name] -= power
            else:
                bases[name] = -power
        return Unit(bases.items())

    def __pow__(a, exp):
        if isinstance(exp, float):
            if int(exp) != exp:
                return NotImplemented
            exp = int(exp)
        if not isinstance(exp, int):
            return NotImplemented
        return Unit({name: power * exp for name, power in a.bases})

    def __eq__(a, b):
        if not isinstance(b, Unit):
            return NotImplemented
        return a.bases == b.bases

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
        scales = sorted(Scale, key=lambda x: x.magnitude)
        for scale in reversed(scales):
            if 1.0 <= value / scale.magnitude:
                return scale
        return scales[0]

    def display_scaled(self, scale, prec=5):
        if scale != Scale.NONE and not self.unit.can_scale():
            raise ValueError("unit cannot be scaled")
        value = f"{self.value / scale.magnitude:.{prec}g}"
        unit = scale.letter + repr(self.unit)
        if self.isbare:
            return f"{value} ¿{unit}?"
        else:
            return f"{value} {unit}"


    def __init__(self, value=0.0, unit=Unit.none, *, isbare=False):
        if isinstance(value, int):
            value = float(value)
        if not isinstance(value, float):
            raise TypeError("value must be a float")
        if not isinstance(unit, Unit):
            raise TypeError("unit must be a Unit")
        if not isinstance(isbare, bool):
            raise TypeError("isbare must be a bool")
        self.value = value
        self.unit = unit
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
        if a.isbare + b.isbare == 1:
            raise ValueError("both quantites must be bare or united")
        unit = a.unit if a.isbare else a.unit
        if not a.isbare and a.unit != b.unit:
            raise ValueError(f"units do not agree ('{a.unit}' vs '{b.unit}')")
        return Quantity(a.value + b.value, unit, isbare=a.isbare)
    def neg(a):
        return Quantity(-a.value, a.unit, isbare=a.isbare)
    def mul(a, b):
        if a.isbare + b.isbare == 1:
            raise ValueError("both quantites must be bare or united")
        unit = a.unit if a.isbare else a.unit * b.unit
        return Quantity(a.value * b.value, unit, isbare=a.isbare)
    def rec(a):
        unit = a.unit if a.isbare else Unit.none / a.unit
        return Quantity(1.0 / a.value, unit, isbare=a.isbare)
    def exp(a):
        if not a.isbare:
            raise ValueError("cannot exponentiate united value (use bare)")
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
