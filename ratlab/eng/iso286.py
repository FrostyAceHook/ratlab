import fields.units as u


class iso286:
    FITS = tuple("ABCDEFGHJKMNPRSTUVXYZabcdefghjkmnprstuvxyz")
    TOLERANCES = ("01", ) + tuple(map(str, range(19)))

    COMMON_HOLE_BASIS_SPECS = """
            Hole              Clearance fits
        ,----------,----------------------------------,
        |       H6 |                         g5   h5  |
        |       H7 |                     f6 [g6] [h6] |
        |       H8 |                 e7 [f7]     [h7] |
        | H8 or H9 |             d8 [e8] f8       h8  |
        |      H10 |  b9    c9  [d9] e9          [h9] |
        |      H11 | [b11] [c11] d10              h10 |
        '----------'----------------------------------'

         Hole   Transition fits         Inteference fits
        ,----,--------------------,---------------------------,
        | H6 |  js5   k5  m5      | n5 p5                     |
        | H7 | [js6] [k6] m6 [n6] |   [p6] [r6] [s6] t6 u6 x6 |
        | H8 |  js7   k7  m7      |              s7     u7    |
        '----'--------------------'---------------------------'
    """

    COMMON_SHAFT_BASIS_SPECS = """
           Shaft             Clearance fits
        ,----------,---------------------------------,
        |       h5 |                       G6   H6   |
        |       h6 |                   F7 [G7] [H7]  |
        | h7 or h9 |               E8 [F8]     [H8]  |
        | h8 or h9 |           D9 [E9] F9      [H9]  |
        |       h9 |      C10 [D10]            [H10] |
        |       h9 | [B11]                           |
        '----------'---------------------------------'

        Shaft   Transition fits         Inteference fits
        ,----,--------------------,---------------------------,
        | h5 |  JS6   K6  M6      | N6 P6                     |
        | h6 | [JS7] [K7] M7 [N7] |   [P7] [R7] [S7] T7 U7 X7 |
        '----'--------------------'---------------------------'
    """


    # @property
    # def fit_type(self):
    #     lsl, usl = self.


    def __init__(self, spec):
        if not isinstance(spec, str):
            raise TypeError("non-string specifiction")
        fit = ""
        while spec and spec[0].isalpha():
            fit += spec[0]
            spec = spec[1:]
        tol = spec
        if fit not in iso286.FITS:
            raise ValueError(f"invalid fit class {repr(fit)}")
        if tol not in iso286.TOLERANCES:
            raise ValueError(f"invalid tolerance class {repr(tol)}")
        self._fit = fit
        self._tol = tol

    def __repr__(self):
        return f"ISO 286-{self._fit}{self._tol}"


    def hole(self, nominal_shaft):
        if self._fit.islower():
            raise ValueError("requires hole tolerance class")
        pass

    def shaft(self, nominal_hole):
        if self._fit.isupper():
            raise ValueError("requires shaft tolerance class")
        pass
