
from . import cons as _cons
from . import engine as _engine


# Comically hardcoded. Please point and laugh.


def _big_beautiful_logo():
    # (the creation of which put Stuart Inc. trillions of dollars further into
    # debt).
    col = [   51,   80, 121, 114, 155,  190,  226,  220]
    top = [",--", "--", "-", "-", "-", "--", "--", "--"]
    bot = ["'--", "--", "-", "-", "-", "--", "--", "--"]
    rat = ["|", " ", " ", " ", " ", "RAT", "LAB", "® S",
           "tua", "rt ", "Inc", ".", " ", " ", " ", "|"]
    col += col[::-1]
    top += [x[::-1] for x in top[::-1]]
    bot += [x[::-1] for x in bot[::-1]]
    return (_cons.coloured(col, top) +
     "\n" + _cons.coloured(col, rat) +
     "\n" + _cons.coloured(col, bot))


def _usage(descriptions=True):
    def colour(s):
        if s.strip() == "ratlab":
            return 220
        if s.strip() == "usage:":
            return 73
        if s.strip() in {"|", "..."}:
            return 161
        if s.strip() in {"[", "]"}:
            return -1
        if s.strip().isalpha():
            return 208
        if "-" in s and all(c == "-" or c.isalpha() or not c.strip() for c in s):
            return 80
        return -1
    txts = [
        "usage: ", "ratlab",
        " [", "-h", "]",
        " [", "-x", " | ", "--bare-matrices", "]",
        " [", "PATH", "]", "..."
    ]
    msg = _cons.coloured(map(colour, txts), txts)
    if descriptions:
        txts = [
            "\n\noptions:\n",
            "  -h", ", ", "--help        ", "print this message and exit\n",
            "  -x", ", ", "--bare-lists  ", "makes bare [0,1,2] a list "
                                            "literal\n",
            "  --bare-matrices   ", "makes bare [0,1,2] a matrix literal "
                                    "(default)\n",
            "  PATH              ", "script path, or '-' for a console",
        ]
        msg += _cons.coloured(map(colour, txts), txts)
    return msg


def _help():
    def wrap_string(s):
        return _cons.entry(s, width=76, lead=0) + "\n"
    def colour_code(s):
        if "".join(x.strip() for x in s).isdigit():
            return 135
        red = (_engine.KEYWORDS | set("+-*/=")) - {_engine.KW_PREV}
        if s.strip() in red:
            return 161
        if s.strip().isalpha():
            return 208
        if s.strip() == ">>":
            return 73
        if s.lstrip().startswith("#"):
            return 245
        return -1

    msg = _usage(descriptions=True) + "\n"

    msg += "\n"
    msg += wrap_string("""
        Executes the given scripts sequentially and with a carry-over variable
        space. If '-' is encountered as a path, an interactive console is
        started. If no arguments are given, starts a console.
    """)

    msg += "\n"
    msg += wrap_string(f"""
        Ratlab is essentially Python with pre-loaded modules and added syntax for
        matrices. The syntax '{_engine.KW_LIST}[1,2,3]' creates a list and
        '{_engine.KW_MATRIX}[1,2,3]' creates a matrix. By default, bare
        square-brackets like '[1,2,3]' become matrices, but that can be stopped
        by giving '-x' or '--bare-lists' as an argument.
    """)
    txts = [
        ">> ", "[", "1", ", ", "2", ", ", "3", "][", "4", ", ", "5", ", ",
            "6", "]", " # a 2x3 matrix (under --bare-matrices)\n",
        "[", "1 2 3", "]\n",
        "[", "4 5 6", "]\n",
        ">> ", _engine.KW_LIST, "[", "1", ", ", "2", ", ", "3", "]",
            " # a list\n",
        "[", "1", ", ", "2", ", ", "3", "]\n",
        ">> ", _engine.KW_MATRIX, "[ [", "1", "][", "2", "], [", "3", "][", "4",
            "] ]", " # hstack two 2-element vectors (2x1 matrices)\n",
        "[", "1 3", "]\n",
        "[", "2 4", "]\n",
    ]
    msg += _cons.coloured(map(colour_code, txts), txts)

    msg += "\n"
    msg += wrap_string("""
        When within the console, there are several commands which can be used by
        typing their bare name (or by calling them as nullary functions):
    """)
    txts = []
    for name, func in _engine.COMMANDS.items():
        doc = " ".join(func.__doc__.split())
        txts.append(">> ")
        txts.append(name)
        txts.append(" " * (8 - len(name)) + f" # {doc}\n")
    msg += _cons.coloured(map(colour_code, txts), txts)
    msg += wrap_string("""
        Note that these commands are also keywords, and cannot be overwritten or
        used in a script.
    """)

    msg += "\n"
    msg += wrap_string(f"""
        In the console, the most recent result is stored in the
        {_cons.coloured(208, _engine.KW_PREV)} label.
    """)
    txts = [
        ">> ", "1", " + ", "2\n",
        "3\n",
        ">> ", _engine.KW_PREV, "\n",
        "3\n",
        ">> ", "x", " = ", "[", "3", " * ", "4", "]",
            " # a single (1x1) matrix\n",
        "x", " = ", "12\n",
        ">> ", _engine.KW_PREV, " + ", "3\n",
        "15\n"
    ]
    msg += _cons.coloured(map(colour_code, txts), txts)
    msg += wrap_string(f"""
        Note that {_cons.coloured(208, _engine.KW_PREV)} is also a keyword, and
        cannot be overwritten or used in a script.
    """)

    return msg.rstrip()


def run(args):
    # Leave a mark.
    print(_big_beautiful_logo())

    cfargs = [arg.casefold() for arg in args]

    # Check for help.
    help_flags = ["-h", "--help", "?", "-?", "/?"]
    if any(flag.casefold() in cfargs for flag in help_flags):
        print(_help())
        return

    # Worlds shittest argparse. Note it doesn't support chaining single letter
    # things.
    paths = []
    bare_lists = False
    for arg in args:
        if arg.casefold() in {"-x", "--bare-lists"}:
            bare_lists = True
            continue
        if arg.casefold() in {"--bare-matrices"}:
            bare_lists = False
            continue
        if arg != "-" and arg.startswith("-"):
            print(_usage(descriptions=False))
            txts = ["ratlab: error: ", "unrecognised option: ", repr(arg)]
            cols = [              124,                     203,        -1]
            print(_cons.coloured(cols, txts))
            return
        paths.append(arg)

    # If no files, default to cli.
    if not paths:
        paths = ["-"]

    # Read and execute each input file, treating "-" as a cli.
    ctx = _engine.Context(bare_lists=bare_lists)
    for i, path in enumerate(paths):
        if path.strip() == "-":
            ctx.run_console()
        else:
            ctx.run_file(path, i < len(args) - 1)
