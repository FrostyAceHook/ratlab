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
                    if parts[-1][-1].isdigit():
                        parts.append(" ")

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

Unit.m = Unit([("m", 1)])
Unit.m2 = Unit.m * Unit.m
Unit.m3 = Unit.m2 * Unit.m

Unit.g = Unit([("g", 1)])
Unit.g2 = Unit.g * Unit.g
Unit.g3 = Unit.g2 * Unit.g

Unit.s = Unit([("s", 1)])
Unit.s2 = Unit.s * Unit.s
Unit.s3 = Unit.s2 * Unit.s

Unit.A = Unit([("A", 1)])
Unit.A2 = Unit.A * Unit.A
Unit.A3 = Unit.A2 * Unit.A

Unit.K = Unit([("K", 1)])
Unit.K2 = Unit.K * Unit.K
Unit.K3 = Unit.K2 * Unit.K

Unit.m_s = Unit.m / Unit.s
Unit.m_s2 = Unit.m_s / Unit.s

Unit.N = Unit.g * Unit.m_s2
Unit.Pa = Unit.N / Unit.m2

Unit.J = Unit.N * Unit.m
Unit.W = Unit.J / Unit.s

Unit.C = Unit.A * Unit.s
Unit.V = Unit.g * Unit.m2 / Unit.s3 / Unit.A
Unit.Ohm = Unit.g * Unit.m2 / Unit.s3 / Unit.A2

Unit.COMPOUNDS = {Unit.N: "N", Unit.Pa: "Pa", Unit.J: "J", Unit.W: "W",
    Unit.C: "C", Unit.V: "V", Unit.Ohm: "Ω"}



@immutable
class Quantity(Field):
    @property
    def bare(self):
        return Quantity(self.value, self.unit, True)
    @property
    def united(self):
        return Quantity(self.value, self.unit, False)

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


    def __init__(self, value, unit, *, isbare=False):
        if isinstance(value, int):
            value = float(value)
        if not isinstance(value, float):
            raise TypeError("value must be a float")
        if not isinstance(unit, Unit):
            raise TypeError("unit must be a Unit")
        if not isinstance(isbare, bool):
            raise TypeError("isbare must be a bool")
        if maths.isnan(value):
            raise ValueError("nan quantity")
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
        if b.value == 0 and b.unit == Unit.none and not b.bare:
            return a
        if a.value == 0 and a.unit == Unit.none and not a.bare:
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

m = Quantity(1, Unit.m)
cm = Quantity(1e-2, Unit.m)
mm = Quantity(1e-3, Unit.m)
um = Quantity(1e-6, Unit.m)
nm = Quantity(1e-9, Unit.m)
km = Quantity(1e3, Unit.m)

kg = Quantity(1e3, Unit.g)
g = Quantity(1, Unit.g)
mg = Quantity(1e3, Unit.g)
ug = Quantity(1e-6, Unit.g)
ng = Quantity(1e-9, Unit.g)

s = Quantity(1, Unit.s)
ms = Quantity(1e-3, Unit.s)
min = Quantity(60, Unit.s)
hr = Quantity(60*60, Unit.s)

A = Quantity(1, Unit.A)

K = Quantity(1, Unit.K)


N = Quantity(1, Unit.N)
kN = Quantity(1e3, Unit.N)

Pa = Quantity(1, Unit.Pa)
kPa = Quantity(1e3, Unit.Pa)
MPa = Quantity(1e6, Unit.Pa)
GPa = Quantity(1e9, Unit.Pa)

J = Quantity(1, Unit.J)
mJ = Quantity(1e-3, Unit.J)
kJ = Quantity(1e3, Unit.J)
MJ = Quantity(1e6, Unit.J)

W = Quantity(1, Unit.W)
mW = Quantity(1e-3, Unit.W)
kW = Quantity(1e3, Unit.W)
MW = Quantity(1e6, Unit.W)

V = Quantity(1, Unit.V)
mV = Quantity(1e-3, Unit.V)
kV = Quantity(1e3, Unit.V)
MV = Quantity(1e6, Unit.V)

Ohm = Quantity(1, Unit.Ohm)
mOhm = Quantity(1e-3, Unit.Ohm)
kOhm = Quantity(1e3, Unit.Ohm)
MOhm = Quantity(1e6, Unit.Ohm)
