
# Hack src/ into the path so we can import. Note this also allows the user code
# to import from here.
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import syntax
import util

def main():
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

    args = sys.argv[1:]
    cfargs = [x.casefold() for x in args]

    # Help msg.
    help_flags = ["-h", "/h", "--help", "?", "-?", "/?"]
    if any(flag.casefold() in cfargs for flag in help_flags):
        def wrap_string(s):
            return util.entry(s, width=76, lead=0) + "\n"
        def code_colour(s):
            if "".join(x.strip() for x in s).isdigit():
                return 135
            red = (syntax.KEYWORDS | set("+-*/=")) - {syntax.KW_PREV}
            if s.strip() in red:
                return 161
            if s.strip().isalpha():
                return 208
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
            ">> ", syntax.KW_LIST, "[", "1", ", ", "2", ", ", "3", "]",
                " # a list\n",
            "[", "1", ", ", "2", ", ", "3", "]\n",
            ">> ", "[ [", "1", "][", "2", "], [", "3", "][", "4", "] ]",
                " # hstack two 2-element vectors (2x1 matrices)\n",
            "[", "1 3", "]\n",
            "[", "2 4", "]\n",
        ]
        msg += "\n"
        msg += wrap_string("Ratlab is essentially Python with pre-loaded "
                           "modules and added syntax for matrices:")
        msg += util.coloured(map(code_colour, txts), txts)

        txts = []
        for name, func in syntax.COMMANDS.items():
            doc = " ".join(func.__doc__.split())
            txts.append(">> ")
            txts.append(name)
            txts.append(" " * (8 - len(name)) + f" # {doc}\n")
        msg += "\n"
        msg += wrap_string("When within the console, there are several commands "
                           "which can be used by typing their bare name (or by "
                           "calling them as nullary functions):")
        msg += util.coloured(map(code_colour, txts), txts)

        txts = [
            ">> ", "1", " + ", "2\n",
            "3\n",
            ">> ", syntax.KW_PREV, "\n",
            "3\n",
            ">> ", "x", " = ", "[", "3", " * ", "4", "]",
                " # a single (1x1) matrix\n",
            "x", " = ", "12\n",
            ">> ", syntax.KW_PREV, " + ", "3\n",
            "15\n"
        ]
        msg += "\n"
        msg += wrap_string("Additionally, the most recent result in the console "
                          f"is stored in the {repr(syntax.KW_PREV)} label.")
        msg += util.coloured(map(code_colour, txts), txts)

        print(msg, end="")
        sys.exit(0)

    # If no files, default to cli.
    if not args:
        args = ["-"]

    # Read and execute each input file, treating "-" as a cli.
    space = syntax.Space()
    for i, path in enumerate(args):
        if path.strip() == "-":
            syntax.run_console(space)
        else:
            syntax.run_file(space, path, i < len(args) - 1)

if __name__ == "__main__":
    main()
