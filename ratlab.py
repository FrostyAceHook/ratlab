
# Hack src/ into the path so we can import.
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), "src"))


# Standard ratlab modules.

import math # just useful.

import matrix
from matrix import *

import util

from numerical_methods import *

from plot import *

from fields.rational import Rational
from fields.real import Real
from fields.complex import Complex
import fields.units as u

# from eng import *

import syntax as _syntax


def _main():
    # Print big beautiful logo (the creation of which put Stuart Inc. trillions
    # of dollars further into debt).
    # gradient from cyan to gold to cyan.
    col = [   51,   80, 121, 114, 155,  190,  226,  220]
    top = [",--", "--", "-", "-", "-", "--", "--", "--"]
    bot = ["'--", "--", "-", "-", "-", "--", "--", "--"]
    rat = ["|", " ", " ", " ", " ", "RAT", "LAB", "® S",
           "tua", "rt ", "Inc", ".", " ", " ", " ", "|"]
    col += col[::-1]
    top += [x[::-1] for x in top[::-1]]
    bot += [x[::-1] for x in bot[::-1]]
    print(util.coloured(col, top))
    print(util.coloured(col, rat))
    print(util.coloured(col, bot))

    # Public globals as initial variable space.
    space = {k: v for k, v in globals().items() if not k.startswith("_")}
    # And set an initial field.
    lits(Complex, space=space)

    args = _sys.argv[1:]

    # If no files, default to cli.
    if not args:
        args = ["-"]

    # Read and execute each input file, treating "-" as a cli.
    for i, path in enumerate(args):
        if path.strip() == "-":
            _syntax.run_cli(space)
        else:
            _syntax.run_file(space, path, i < len(args) - 1)

if __name__ == "__main__":
    _main()
