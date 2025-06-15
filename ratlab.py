
# Hack src/ into the path so we can import.
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), "src"))


# Standard ratlab modules.

import math # just useful.

import matrix
from matrix import *

import util

from plot import *

from fields.rational import Rational
from fields.complex import Complex
# import fields.units as u

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

    args = _sys.argv[1:]

    # Help msg.
    lowerargs = [x.lower() for x in args]
    if any(m in lowerargs for m in ["-h", "/h", "--help", "?", "-?", "/?"]):
        def wrap_string(s):
            return util.entry(s, width=76, lead=0) + "\n"
        def code_colour(s):
            if "".join([x.strip() for x in s]).isdigit():
                return 135
            if s.strip() in _syntax._KEYWORDS:
                return 161
            if s.strip() == ">>":
                return 73
            if s.lstrip().startswith("#"):
                return 245
            return -1
        msg = ""

        txts = ["usage: ", "ratlab ", "[path | -]...\n"]
        cols = [       73,       220,                80]
        msg += "\n"
        msg += util.coloured(cols, txts)
        msg += "\n"
        msg += wrap_string("Executes the given scripts sequentially and with a "
                           "shared variable space. If '-' is encountered as a "
                           "path, an interactive console is started. If no "
                           "arguments are given, starts a console.")


        txts = [
            ">> ", "[", "1", ", ", "2", ", ", "3", "][", "4", ", ", "5", ", ",
                "6", "]", " # a 2x3 matrix\n",
            "[", "1 2 3", "]\n",
            "[", "4 5 6", "]\n",
            ">> ", "lst", "[", "1", ", ", "2", ", ", "3", "]", " # a list\n",
            "[", "1", ", ", "2", ", ", "3", "]\n",
        ]
        msg += "\n"
        msg += wrap_string("Ratlab is essentially Python with pre-loaded "
                           "modules and added syntax for matrices:")
        msg += util.coloured(map(code_colour, txts), txts)


        txts = []
        for name, func in _syntax._COMMANDS.items():
            doc = " ".join(func.__doc__.split())
            txts.append(">> ")
            txts.append(name)
            txts.append(" " * (8 - len(name)) + f" # {doc}\n")
        msg += "\n"
        msg += wrap_string("When within the console, there are several commands "
                           "which can be used by typing their bare name:")
        msg += util.coloured(map(code_colour, txts), txts)

        print(msg, end="")
        quit()

    # If no files, default to cli.
    if not args:
        args = ["-"]

    # Public globals as initial variable space.
    space = {k: v for k, v in globals().items() if not k.startswith("_")}
    # And set an initial field (modifying the `space` and not our globals).
    exec("lits(Complex)", space, space)

    # Read and execute each input file, treating "-" as a cli.
    for i, path in enumerate(args):
        if path.strip() == "-":
            _syntax.run_console(space)
        else:
            _syntax.run_file(space, path, i < len(args) - 1)

if __name__ == "__main__":
    _main()
