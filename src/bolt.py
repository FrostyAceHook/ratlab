import units as u
from maths import pi


class Bolt:
    @classmethod
    def metric(cls, code, grade="12.9"):
        """
        Returns the `Bolt` object from the given metric code and grade. `code`
        must be "M3", "M4", etc. (an M* code) and `grade` must be one of "4.6",
        "8.8", "10.9", or "12.9".
        """
        DIMENSIONS = {
            # code: (major diameter, minor diameter, stress area),
            #https://www.accu.co.uk/p/117-iso-metric-thread-dimensions
            "M3":  ( 3.0*u.mm,  2.459*u.mm, 5.03*u.mm**2),
            "M4":  ( 4.0*u.mm,  3.242*u.mm, 8.78*u.mm**2),
            "M5":  ( 5.0*u.mm,  4.134*u.mm, 14.2*u.mm**2),
            "M6":  ( 6.0*u.mm,  4.917*u.mm, 20.1*u.mm**2),
            "M7":  ( 7.0*u.mm,  5.917*u.mm, 28.9*u.mm**2),
            "M8":  ( 8.0*u.mm,  6.647*u.mm, 36.6*u.mm**2),
            "M10": (10.0*u.mm,  8.376*u.mm, 58.0*u.mm**2),
            "M12": (12.0*u.mm, 10.106*u.mm, 84.3*u.mm**2),
            "M14": (14.0*u.mm, 11.835*u.mm,  115*u.mm**2),
            "M16": (16.0*u.mm, 13.835*u.mm,  157*u.mm**2),
            "M18": (18.0*u.mm, 15.394*u.mm,  192*u.mm**2),
            "M20": (20.0*u.mm, 17.294*u.mm,  245*u.mm**2),
            "M22": (22.0*u.mm, 19.294*u.mm,  303*u.mm**2),
            "M24": (24.0*u.mm, 20.752*u.mm,  353*u.mm**2),
            "M27": (27.0*u.mm, 23.752*u.mm,  459*u.mm**2),
            "M30": (30.0*u.mm, 26.211*u.mm,  561*u.mm**2),
            "M33": (33.0*u.mm, 29.211*u.mm,  694*u.mm**2),
            "M36": (36.0*u.mm, 31.670*u.mm,  817*u.mm**2),
            "M39": (39.0*u.mm, 34.670*u.mm,  976*u.mm**2),
        }
        STRENGTHS = {
            # grade: (tensile strength, yield strength, shear strength),
            # assuming yield shear strength = 0.58 yield strength
            #https://roymech.org/Useful_Tables/Matter/shear_tensile.html
            "4.6":  ( 400*u.MPa,  400*u.MPa*0.6,  400*u.MPa*0.6*0.58),
            "8.8":  ( 800*u.MPa,  800*u.MPa*0.8,  800*u.MPa*0.8*0.58),
            "10.9": (1000*u.MPa, 1000*u.MPa*0.9, 1000*u.MPa*0.9*0.58),
            "12.9": (1200*u.MPa, 1200*u.MPa*0.9, 1200*u.MPa*0.9*0.58),
            # 4.6 is shocking mate my fingerd be stronger.
        }
        if not isinstance(code, str):
            raise TypeError("non-string metric bolt code")
        if not isinstance(grade, str):
            raise TypeError("non-string metric bolt grade")
        if code not in DIMENSIONS:
            raise ValueError(f"invalid metric bolt code {repr(code)}")
        if grade not in STRENGTHS:
            raise ValueError(f"invalid metric bolt grade {repr(grade)}")
        return cls(code, grade, *DIMENSIONS[code], *STRENGTHS[grade])

    def axial_sf(self, axial_force):
        """
        Calculates the safety factor of the bolt under the given axial force,
        where failure is any yield.
        """
        if not isinstance(axial_force, u.Quantity):
            raise TypeError("axial force must be a united quantity")
        if not axial_force.unitsof(u.N):
            raise TypeError("axial force must be a force")
        axial_stress = axial_force / self.stress_area
        return (self.yield_strength / axial_stress).unitless

    def shear_sf(self, shear_force, on_shank=False):
        """
        Calculates the safety factor of the bolt under the given shear force,
        where failure is any yield. This models the bolt as a simple pin of
        steel, ignoring the effects of pre-load (which would strengthen the join,
        so this is a worst-case estimate).
        """
        if not isinstance(shear_force, u.Quantity):
            raise TypeError("shear force must be a united quantity")
        if not shear_force.unitsof(u.N):
            raise TypeError("shear force must be a force")
        # Use the diameter of the metal which is being sheared.
        d = self.major_diameter if on_shank else self.minor_diameter
        A = pi * d**2 / 4
        shear_stress = shear_force / A
        return (self.shear_strength / shear_stress).unitless

    def sf(self, axial_force, shear_force, on_shank=False):
        """
        Returns the safety factor of the bolt under the given forces, which is
        the minimum of the axial and shear safety factors.
        """
        axial_sf = self.axial_sf(axial_force)
        shear_sf = self.shear_sf(shear_force, on_shank=on_shank)
        return min(axial_sf, shear_sf)


    def __init__(self, code, grade, major_diameter, minor_diameter, stress_area,
            tensile_strength, yield_strength, shear_strength):
        # note shear strength is sometimes called "yield shear strength".
        if not isinstance(code, str):
            raise TypeError("code must be a string")
        if not isinstance(grade, str):
            raise TypeError("grade must be a string")
        if not isinstance(major_diameter, u.Quantity):
            raise TypeError("major diameter must be a united quantity")
        if not isinstance(minor_diameter, u.Quantity):
            raise TypeError("minor diameter must be a united quantity")
        if not isinstance(stress_area, u.Quantity):
            raise TypeError("stress area must be a united quantity")
        if not isinstance(tensile_strength, u.Quantity):
            raise TypeError("tensile strength must be a united quantity")
        if not isinstance(yield_strength, u.Quantity):
            raise TypeError("yeild strength must be a united quantity")
        if not isinstance(shear_strength, u.Quantity):
            raise TypeError("shear strength must be a united quantity")
        if not major_diameter.unitsof(u.m):
            raise ValueError("major diameter must be a length")
        if not minor_diameter.unitsof(u.m):
            raise ValueError("minor diameter must be a length")
        if not stress_area.unitsof(u.m2):
            raise ValueError("stress area must be an area")
        if not tensile_strength.unitsof(u.Pa):
            raise ValueError("tensile strength must be a pressure")
        if not yield_strength.unitsof(u.Pa):
            raise ValueError("yeild strength must be a pressure")
        if not shear_strength.unitsof(u.Pa):
            raise ValueError("shear strength must be a pressure")
        self.code = code
        self.grade = grade
        self.major_diameter = major_diameter
        self.minor_diameter = minor_diameter
        self.stress_area = stress_area
        self.tensile_strength = tensile_strength
        self.yield_strength = yield_strength
        self.shear_strength = shear_strength

    def __repr__(self):
        return f"{self.code} grade {self.grade} bolt"
