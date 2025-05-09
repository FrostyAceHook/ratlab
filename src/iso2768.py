import units as u


class iso2768:
    GEOMETRIC = ("",) + tuple("fmcv")
    FEATURE = ("",) + tuple("HKL")

    def __init__(self, geometric = "", feature = ""):
        if not isinstance(geometric, str):
            raise TypeError("non-string geometric tolerance class")
        if not isinstance(feature, str):
            raise TypeError("non-string feature tolerance class")
        if geometric not in iso2768.GEOMETRIC:
            raise ValueError("invalid geometric tolerance class "
                    f"{repr(geometric)}")
        if feature not in iso2768.FEATURE:
            raise ValueError(f"invalid feature tolerance class {repr(feature)}")
        if not geometric and not feature:
            raise ValueError("specify either geometric or feature tolerance "
                    "class")
        self._geometric = iso2768.GEOMETRIC.index(geometric) - 1
        self._feature = iso2768.FEATURE.index(feature) - 1

    def __repr__(self):
        g = iso2768.GEOMETRIC[self._geometric + 1]
        f = iso2768.FEATURE[self._feature + 1]
        return f"ISO 2768-{g}{f}"

    def _check_length(self, name, length):
        if isinstance(length, int):
            length = float(length)
        if isinstance(length, u.Quantity):
            if length.unit != u.m.unit and not length.isbare:
                raise ValueError(f"{name} must be a length")
            length = length.value
        if not isinstance(length, float):
            raise TypeError(f"{name} must be a number")
        length = abs(length)
        length *= u.m
        return length

    def _lookup(self, cls, cutoffs, table, nominal):
        for i, cutoff in enumerate(cutoffs):
            if nominal <= cutoff:
                tol = table[cls][i]
                break
        lsl = nominal - tol
        usl = nominal + tol
        return u.Toleranced(nominal, lsl, usl)


    def linear(self, nominal):
        if self._geometric == -1:
            raise ValueError("must specify a geometric tolerance class")

        nominal = self._check_length("nominal", nominal)

        # unsupported cases.
        if nominal <= 0.5*u.mm:
            raise ValueError("must explicitly specify tolerances for lengths "
                    "this small")
        if nominal > 4000*u.mm:
            raise ValueError("must explicitly specify tolerances for lengths "
                    "this large")
        if self._geometric == 0 and nominal > 2000*u.mm:
            raise ValueError("under 'f' class, must explicitly specify "
                    "tolerances for lengths this large")
        if self._geometric == 3 and nominal <= 3*u.mm:
            raise ValueError("under 'v' class, must explicitly specify "
                    "tolerances for lengths this small")

        nan = float("nan")
        cutoffs = u.mm * [3, 6, 30, 120, 400, 1000, 2000, 4000]
        table = u.mm * [
            [0.05, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50,  nan],
            [0.10, 0.10, 0.20, 0.30, 0.50, 0.80, 1.20, 2.00],
            [0.20, 0.30, 0.50, 0.80, 1.20, 2.00, 3.00, 4.00],
            [ nan, 0.50, 1.00, 1.50, 2.50, 4.00, 6.00, 8.00],
        ]
        return self._lookup(self._geometric, cutoffs, table, nominal)

    def flatness(self, nominal_A, nominal_B = 0*u.m):
        if self._feature == -1:
            raise ValueError("must specify a feature tolerance class")

        nominal_A = self._check_length("nominal_A", nominal_A)
        nominal_B = self._check_length("nominal_B", nominal_B)
        nominal = max(nominal_A, nominal_B)

        # unsupported cases.
        if nominal > 3000*u.mm:
            raise ValueError("must explicitly specify tolerances for lengths "
                    "this large")

        cutoffs = u.mm * [10, 30, 100, 300, 1000, 3000]
        table = u.mm * [
            [0.02, 0.05, 0.10, 0.20, 0.30, 0.40],
            [0.05, 0.10, 0.20, 0.40, 0.60, 0.80],
            [0.10, 0.20, 0.40, 0.80, 1.20, 1.60],
        ]
        return self._lookup(self._feature, cutoffs, table, nominal)

    def perpendicularity(self, nominal_A, nominal_B = 0*u.m):
        if self._feature == -1:
            raise ValueError("must specify a feature tolerance class")

        nominal_A = self._check_length("nominal_A", nominal_A)
        nominal_B = self._check_length("nominal_B", nominal_B)
        nominal = max(nominal_A, nominal_B)

        # unsupported cases.
        if nominal > 3000*u.mm:
            raise ValueError("must explicitly specify tolerances for lengths "
                    "this large")

        cutoffs = u.mm * [100, 300, 1000, 3000]
        table = u.mm * [
            [0.2, 0.3, 0.4, 0.5],
            [0.4, 0.6, 0.8, 1.0],
            [0.8, 1.0, 1.5, 2.0],
        ]
        return self._lookup(self._feature, cutoffs, table, nominal)

    def parallelism(self, nominal_A, nominal_B = 0*u.m):
        # identical to flatness?
        return self.flatness(nominal_A, nominal_B)


iso2768.H = iso2768(feature="H")
iso2768.K = iso2768(feature="K")
iso2768.L = iso2768(feature="L")

iso2768.f = iso2768(geometric="f")
iso2768.fH = iso2768(geometric="f", feature="H")
iso2768.fK = iso2768(geometric="f", feature="K")
iso2768.fL = iso2768(geometric="f", feature="L")

iso2768.m = iso2768(geometric="m")
iso2768.mH = iso2768(geometric="m", feature="H")
iso2768.mK = iso2768(geometric="m", feature="K")
iso2768.mL = iso2768(geometric="m", feature="L")

iso2768.c = iso2768(geometric="c")
iso2768.cH = iso2768(geometric="c", feature="H")
iso2768.cK = iso2768(geometric="c", feature="K")
iso2768.cL = iso2768(geometric="c", feature="L")

iso2768.v = iso2768(geometric="v")
iso2768.vH = iso2768(geometric="v", feature="H")
iso2768.vK = iso2768(geometric="v", feature="K")
iso2768.vL = iso2768(geometric="v", feature="L")
