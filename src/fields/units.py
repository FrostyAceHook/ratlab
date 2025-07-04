# from enum import Enum

# import matrix
# from fields.real import Real
# from fields.rational import Rational
# from util import immutable


# class Scale(Enum):
#     NANO = (1e-9, "n")
#     MICRO = (1e-6, "μ")
#     MILLI = (1e-3, "m")
#     NONE = (1, "")
#     KILO = (1e3, "k")
#     MEGA = (1e6, "M")
#     GIGA = (1e9, "G")

#     @property
#     def magnitude(self):
#         return self._value_[0]
#     @property
#     def letter(self):
#         return self._value_[1]


# @immutable
# class Unit(matrix.Field):
#     BASE = ("m", "kg", "s", "A", "K")

#     def __init__(self, bases=()):
#         had = set()
#         cache = []
#         for k, v in bases:
#             if not isinstance(k, str):
#                 raise TypeError("unit bases must be strings")
#             if k not in Unit.BASE:
#                 raise ValueError(f"invalid base unit {repr(k)}")
#             if k in had:
#                 raise ValueError(f"repeated base unit {repr(k)}")
#             had.add(k)
#             if isinstance(v, (int, float)):
#                 v = Rational.cast(v)
#             if not isinstance(v, Rational):
#                 raise TypeError("unit powers must be rationals")
#             if v:
#                 cache.append((k, v))
#         cache = sorted(cache, key=lambda x: Unit.BASE.index(x[0]))
#         self._bases = tuple(cache)

#     @classconst
#     def one(cls):
#         return Unit()

#     @classmethod
#     def mul(cls, a, b):
#         bases = {k: v for k, v in a._bases}
#         for name, power in b._bases:
#             if name in bases:
#                 bases[name] += power
#             else:
#                 bases[name] = power
#         return Unit(bases.items())
#     @classmethod
#     def div(cls, a, b):
#         bases = {k: v for k, v in a._bases}
#         for name, power in b._bases:
#             if name in bases:
#                 bases[name] -= power
#             else:
#                 bases[name] = -power
#         return Unit(bases.items())

#     def __pow__(a, exp):
#         if isinstance(exp, int):
#             exp = Rational.from_int(exp)
#         elif isinstance(exp, float):
#             exp = Rational.from_float(exp)
#         if not isinstance(exp, Rational):
#             raise NotImplementedError()
#         return Unit([(k, v * exp) for k, v in a._bases])

#     @classmethod
#     def eq(cls, a, b):
#         return not (a / b)._bases

#     @classmethod
#     def hashed(cls, a):
#         return hash(a._bases)



# @immutable
# class Quantity(field.Field):
#     @property
#     def bare(self):
#         return type(self)(self._value, self._unit, logged=self._logged,
#                 isbare=True)
#     @property
#     def united(self):
#         return type(self)(self._value, self._unit, logged=self._logged,
#                 isbare=False)

#     @property
#     def unitless(self):
#         if self._unit and not self._isbare:
#             raise ValueError("value has units")
#         return self._value

#     def can_scale(self):
#         if self._logged:
#             return False
#         return self._unit in [x._unit for x, _ in type(self).SCALEABLE]

#     def ideal_scale(self):
#         if not self.can_scale():
#             return Scale.NONE
#         value = abs(self._value)
#         if value == 0.0:
#             return Scale.NONE
#         # Scale down by the offset of the displayed unit.
#         base = [x for x, _ in type(self).SCALEABLE if x._unit == self._unit]
#         base = base[0]
#         scales = [[x, x.magnitude] for x in Scale]
#         scales = sorted(scales, key=lambda x: x[1])
#         # Find the best scale.
#         for scale, mag in reversed(scales):
#             if 1.0 <= value / base._value / mag:
#                 return scale
#         return scales[0]

#     def display_scaled(self, scale, prec=5):
#         value = self._value
#         value /= scale.magnitude
#         if self.can_scale():
#             # Scale down by the offset of the displayed unit.
#             base = [x for x, _ in type(self).SCALEABLE if x._unit == self._unit]
#             base = base[0]
#             value /= base._value
#         elif scale != Scale.NONE:
#             raise NotImplementedError("unit cannot be scaled")
#         value = f"{value:.{prec}g}"
#         unit = scale.letter + type(self)._unit_repr(self._unit)
#         if self._logged:
#             unit = f"+ log({unit})"
#         if self._isbare:
#             return f"{value} ¿{unit}?"
#         else:
#             return f"{value} {unit}"

#     def unitsof(a, b):
#         if a._isbare or b._isbare:
#             return False
#         return a._unit == b._unit and a._logged == b._logged

#     def __init__(self, value=matrix.single(Real()), unit=matrix.single(Unit()),
#                 *, logged=False, isbare=False):
#         if isinstance(value, int):
#             value = float(value)
#         if not isinstance(value, float):
#             raise TypeError("value must be a float")
#         if not isinstance(unit, Unit):
#             raise TypeError("unit must be a Unit")
#         if not isinstance(logged, bool):
#             raise TypeError("logged must be a bool")
#         if not isinstance(isbare, bool):
#             raise TypeError("isbare must be a bool")
#         if not unit:
#             logged = False
#             # log(unitless) == unitless
#         self._value = value
#         self._unit = unit
#         self._logged = logged
#         self._isbare = isbare

#     @classmethod
#     def from_int(cls, x):
#         return cls(matrix.single(Real.from_int(x)))
#     @classmethod
#     def from_float(cls, x):
#         return cls(x)
#     @classmethod
#     def from_complex(cls, x):
#         if x.imag != 0.0:
#             raise NotImplementedError()
#         return cls(x.real)

#     @classmethod
#     def to_int(cls, a):
#         if a.isnan:
#             raise NotImplementedError()
#         if a.im:
#             raise NotImplementedError()
#         if math.isinf(a.re) or a.re != int(a.re):
#             raise NotImplementedError()
#         return int(a.re)
#     @classmethod
#     def to_float(cls, a):
#         if a.isnan:
#             return float("nan")
#         if a.im:
#             raise NotImplementedError()
#         return a.re
#     @classmethod
#     def to_complex(cls, a):
#         if a.isnan:
#             return complex("nan+nanj")
#         return complex(a.re, a.im)

#     @classmethod
#     def _cast(cls, obj, for_obj):
#         if not isinstance(obj, float):
#             try:
#                 obj = float(obj)
#             except Exception:
#                 pass
#         if isinstance(obj, float):
#             return cls(obj)
#         return super()._cast(obj, for_obj)

#     @classmethod
#     def _zero(cls):
#         return cls(0)
#     @classmethod
#     def _one(cls):
#         return cls(1)

#     @classmethod
#     def _add(cls, a, b):
#         # ignore units if value is 0.
#         if b._value == 0 and not b._unit and not b._isbare:
#             return a
#         if a._value == 0 and not a._unit and not a._isbare:
#             return b
#         # handle bare.
#         if a._isbare + b._isbare:
#             if (a._isbare or not a._unit) + (b._isbare or not b._unit) == 1:
#                 raise NotImplementedError()
#             if not a._isbare and b._isbare:
#                 a, b = b, a
#             return cls(a._value + b._value, a._unit, logged=a._logged,
#                     isbare=True)
#         # handle logged.
#         if a._logged + b._logged:
#             if (a._logged or not a._unit) + (b._logged or not b._unit) == 1:
#                 raise NotImplementedError()
#             return cls(a._value + b._value, a._unit * b._unit, logged=True)
#         # normal.
#         if a._unit != b._unit:
#             raise NotImplementedError("units do not agree "
#                     f"('{cls._unit_repr(a._unit)}' vs "
#                     f"'{cls._unit_repr(b._unit)}')")
#         return cls(a._value + b._value, a._unit)
#     @classmethod
#     def _neg(cls, a):
#         if a._isbare:
#             return cls(-a._value, a._unit, isbare=True)
#         if a._logged:
#             return cls(-a._value, ~a._unit, logged=True)
#         return cls(-a._value, a._unit)
#     @classmethod
#     def _mul(cls, a, b):
#         # handle bare.
#         if a._isbare + b._isbare:
#             if (a._isbare or not a._unit) + (b._isbare or not b._unit) == 1:
#                 raise NotImplementedError()
#             if not a._isbare and b._isbare:
#                 a, b = b, a
#             return cls(a._value * b._value, a._unit, logged=a._logged,
#                     isbare=True)
#         # handle logged.
#         if a._logged + b._logged == 2:
#             raise NotImplementedError()
#         if a._logged + b._logged == 1:
#             if b._logged:
#                 a, b = b, a
#             if b._unit:
#                 raise NotImplementedError("cannot raise a unit to a unit")
#             return cls(a._value*b._value, a._unit**b._value, logged=True)
#         return cls(a._value * b._value, a._unit * b._unit)
#     @classmethod
#     def _rec(cls, a):
#         if a._isbare:
#             return cls(1.0 / a._value, a._unit, isbare=True)
#         if a._logged:
#             raise NotImplementedError()
#         return cls(1.0 / a._value, Unit() / a._unit)
#     @classmethod
#     def _exp(cls, a):
#         if a._isbare:
#             return cls(field.exp(a._value), a._unit, isbare=True)
#         if a._logged or not a._unit:
#             return cls(field.exp(a._value), a._unit, logged=False)
#         raise NotImplementedError()
#     @classmethod
#     def _log(cls, a):
#         if a._isbare:
#             return cls(field.log(a._value), a._unit, isbare=True)
#         if a._logged:
#             raise NotImplementedError()
#         return cls(field.log(a._value), a._unit, logged=True)

#     @classmethod
#     def _eq_zero(cls, a):
#         return a._value == 0.0
#     @classmethod
#     def _lt_zero(cls, a):
#         return a._value < 0.0

#     @classmethod
#     def _intof(cls, a):
#         if a._value != a._value:
#             raise NotImplementedError()
#         if math.isinf(a._value) or a._value != int(a._value):
#             raise NotImplementedError()
#         return int(a._value)
#     @classmethod
#     def _floatof(cls, a):
#         return a._value

#     @classmethod
#     def _hashof(cls, a):
#         return hash((a._value, a._unit))

#     def __repr__(a):
#         return a.display_scaled(a.ideal_scale())

#     @classmethod
#     def _unit_repr(cls, unit):
#         if not unit._bases:
#             return "unitless"

#         for disp in [l for x, l in cls.SCALEABLE if x._unit == unit]:
#             return disp

#         parts = []
#         for name, power in unit._bases:
#             if parts:
#                 if power < 0:
#                     power = -power
#                     parts.append("/")
#                 else:
#                     parts.append("∙")
#             parts.append(f"{name}{power.exp_as_string()}")
#         return "".join(parts)



# class Toleranced:
#     @property
#     def bounds(self):
#         return (self.lsl, self.usl)

#     def __init__(self, nominal, lsl, usl):
#         if usl < lsl:
#             raise ValueError("usl cannot be greater than lsl")
#         self.nominal = nominal
#         self.lsl = lsl
#         self.usl = usl

#     def __repr__(self):
#         scale = self.nominal.ideal_scale()
#         l = self.lsl - self.nominal
#         u = self.usl - self.nominal
#         l = "+"*(l >= 0) + l.display_scaled(scale)
#         u = "+"*(u >= 0) + u.display_scaled(scale)
#         return f"{self.nominal.display_scaled(scale)} ({l}) ({u})"

#     """

# x = 2 * metre
# log(x) = log(2 * metre)
# log(x) = log(2) + log(metre)

# y = 3 * metre
# log(x) + log(y) = log(2) + log(metre) + log(3) + log(metre)
# log(x) + log(y) = log(2*3) + log(metre*metre)
# log(x * y) = log(2*3 * metre*metre)
# x * y = 6 * metre^2

# notice how this is correct. now, what would it mean to multiple logged units?
# `log(x) * log(y)` represents what in terms of plain x and y?

# log(x) * y
# = (log(x) + log(xU)) (y * yU)
# = y yU log(x) + y yU log(xU)
# = log(x ^ (y yU)) + log(xU ^ (y yU))
# this is nonsensical, unless yU is 1:
# = log(x ^ y) + log(xU ^ y)
# = log(x^y * xU^y)

# log(x) * log(y)
# = (log(x) + log(xU) (log(y) + log(yU))
# = log(x) log(y) + log(x) log(yU) + log(y) log(xU) + log(xU) log(yU)
# again nonsensical, requires xU or yU to be 1:

#     """


# m = Quantity(1, Unit([("m", 1)]))
# cm = 1e-2 * m
# mm = 1e-3 * m
# um = 1e-6 * m
# nm = 1e-9 * m
# km = 1e3 * m

# # im gunna munt.
# In = 0.254 * m
# Ft = 12 * In

# m2 = m ** 2
# cm2 = cm ** 2
# mm2 = mm ** 2
# um2 = um ** 2
# nm2 = nm ** 2
# km2 = km ** 2

# m3 = m ** 3
# cm3 = cm ** 3
# mm3 = mm ** 3
# um3 = um ** 3
# nm3 = nm ** 3
# km3 = km ** 3

# mL = cm3
# L = 1e3 * mL

# kg = Quantity(1, Unit([("kg", 1)]))
# g = 1e-3 * kg
# mg = 1e-6 * kg
# ug = 1e-9 * kg
# ng = 1e-12 * kg
# tonne = 1e3 * kg

# s = Quantity(1, Unit([("s", 1)]))
# ms = 1e-3 * s
# min = 60 * s
# hr = 60*60 * s

# s2 = s ** 2
# ms2 = ms ** 2

# A = Quantity(1, Unit([("A", 1)]))

# K = Quantity(1, Unit([("K", 1)]))

# N = kg * m / s2
# kN = 1e3 * N

# Pa = N / m2
# kPa = 1e3 * Pa
# MPa = 1e6 * Pa
# GPa = 1e9 * Pa

# Nm = N * m

# J = N * m
# mJ = 1e-3 * J
# kJ = 1e3 * J
# MJ = 1e6 * J

# W = J / s
# mW = 1e-3 * W
# kW = 1e3 * W
# MW = 1e6 * W

# V = W / A
# mV = 1e-3 * V
# kV = 1e3 * V
# MV = 1e6 * V

# Ohm = V / A
# mOhm = 1e-3 * Ohm
# kOhm = 1e3 * Ohm
# MOhm = 1e6 * Ohm


# Quantity.SCALEABLE = [
#     (m, "m"), (g, "g"), (s, "s"), (A, "A"), (K, "K"), (N, "N"), (Pa, "Pa"),
#     (J, "J"), (W, "W"), (V, "V"), (Ohm, "Ω"),
# ]
