import itertools as _itertools
import math as _math
import os as _os
import re as _re
import traceback as _traceback

from .util import (
    ilog10 as _ilog10,
    maybe_pack as _maybe_pack,
    objtname as _objtname,
)


# Hack to enable console escape codes.
_os.system("")

def coloured(cols, txts):
    """
    When given the colour 'cols' and text 'txts' arrays, prints each element of
    the text as its corresponding colour. If any text is already coloured, it is
    left as that colour. Colour codes can be found at:
    https://gist.github.com/fnky/458719343aabd01cfb17a3a4f7296797
    """
    cols = _maybe_pack(cols, aslist=True)
    txts = _maybe_pack(txts, aslist=True)
    if len(cols) != len(txts):
        raise TypeError("must give one colour for each piece of text, got "
                f"{len(cols)} colours and {len(txts)} texts")
    for col in cols:
        if not isinstance(col, int):
            raise TypeError("expected integer colour code, got "
                    f"{_objtname(col)}")
    for txt in txts:
        if not isinstance(txt, str):
            raise TypeError(f"expected string text, got {_objtname(txt)}")

    leading_off_code = _re.compile(r"^\x1B\[0m")
    leading_col_code = _re.compile(r"^\x1B\[38;5;([0-9]+)m")
    leading_ctrl_code = _re.compile(r"^\x1B\[[0-9;]*[A-Za-z]")
    col_code = lambda c: f"\x1B[38;5;{c}m" if c >= 0 else "\x1B[0m"

    # Decompose into a list of characters (without control codes) and which
    # colour they should be.
    chars = []
    codes = []
    full = "".join(txts)
    cur_col = -1
    while True:
        while True:
            # Check for an off code.
            match = leading_off_code.match(full)
            if match is not None:
                cur_col = -1
                full = full[len(match.group(0)):]
                continue
            # Check for a colour code.
            match = leading_col_code.match(full)
            if match is not None:
                cur_col = int(match.group(1))
                full = full[len(match.group(0)):]
                continue
            # Ignore any other code.
            match = leading_ctrl_code.match(full)
            if match is not None:
                full = full[len(match.group(0)):]
                continue
            break
        if not full:
            break
        codes.append(cur_col)
        chars.append(full[0])
        full = full[1:]

    # Construct the coloured string, only colouring where necessary.
    wants = [c for col, txt in zip(cols, txts)
               for c in [col] * len(nonctrl(txt))]
    assert len(chars) == len(codes)
    assert len(chars) == len(wants)
    for i, c in enumerate(codes):
        if c != -1:
            wants[i] = c
    # Reset to nothing at the end.
    chars.append("")
    wants.append(-1)

    segs = [] # funny
    prev = -1
    for w, c in zip(wants, chars):
        if w != prev:
            segs.append(col_code(w))
            prev = w
        segs.append(c)
    return "".join(segs)

def nonctrl(string):
    """
    Returns 'string' with all console control codes removed.
    """
    if not isinstance(string, str):
        raise TypeError(f"expected a str 'string', got {_objtname(string)}")
    control_code = _re.compile(r"\x1B\[[0-9;]*[A-Za-z]")
    return control_code.sub("", string)

def idxctrl(string, i):
    """
    Returns the correct index into 'string' which indexes the same character as
    'nonctrl(string)[i]'.
    """
    if not isinstance(string, str):
        raise TypeError(f"expected a str 'string', got {_objtname(string)}")
    if not isinstance(i, int):
        raise TypeError(f"expected an integer 'i', got {_objtname(i)}")
    if i >= len(string):
        return i
    if i < 0:
        raise IndexError(f"expected a positive index, got: {i}")
    leading_control_code = _re.compile(r"^(?:\x1B\[[0-9;]*[A-Za-z])+")
    missing = 0
    for _ in range(i + 1):
        new = leading_control_code.sub("", string)
        missing += len(string) - len(new)
        string = new[1:]
    return missing + i


def entry(name, desc=None, *, width=100, pwidth=20, lead=2):
    """
    Returns a string of the form "<name> .... <desc>", with total width as given
    and points stopping at 'pwidth' at the earliest. If 'desc' is none, just
    returns 'name' wrapped at 'width'. 'lead' spaces will be inserted before each
    line, to pad the string.
    """
    name = " ".join(name.split())
    parts = []
    pad_to = 0
    wrapme = ""
    first = True
    if desc is not None:
        desc = " ".join(desc.split())
        left = " " * lead + name + " .."
        left += "." * (pwidth - len(nonctrl(left))) + " "
        parts.append(left)
        pad_to = len(nonctrl(left))
        wrapme = desc
    else:
        first = False
        pad_to = lead
        wrapme = name

    while wrapme:
        line = wrapme[:idxctrl(wrapme, width - pad_to)]
        if len(nonctrl(line)) == width - pad_to and " " in line:
            line = line[:line.rindex(" ")]
        wrapme = wrapme[len(line):].lstrip()
        pad = " " * (0 if first else pad_to)
        parts.append(pad + line + "\n" * (not not wrapme))
        first = False
    return "".join(parts)



def pretty_number(x, short=True):
    """
    Coloured string representing the given number.
    """
    plus = coloured(161, "+")
    minus = coloured(161, "-")
    i = coloured(38, "i")
    col_num = 135

    if isinstance(x, bool):
        lookup = ["nogh", "yeagh"]
        if short:
            lookup = "NY"
        txt = lookup[x]
        return coloured(col_num, txt)

    if isinstance(x, int):
        if x < 0:
            return minus + pretty_number(-x, short=short)
        if x == 0:
            return coloured(col_num, "0")

        # Cant just use float since it may overflows/lose precision.
        threshold = 10

        # Find exact digit count.
        digits = 1 + _ilog10(x)

        # Print in full if short enough, or we gotta.
        full = not short or (digits <= threshold)

        # Fine to convert for small ints.
        if full:
            return coloured(col_num, str(x))

        # Otherwise, make it into a 6 digit mantissa and exponent.
        mantlen = 6
        exp = digits - 1

        # First cut down to mantlen+1 digits, then round, then remaining digits.
        dif = digits - mantlen - 1
        assert dif >= 0
        x //= pow(10, dif)
        x += 5 # round (most helpful research paper comment).
        if x >= pow(10, mantlen + 1): # may have added a digit.
            exp += 1
            x //= 10
        x //= 10

        # Get mant and exp string, trimming trailing zeros.
        mant = str(x)
        exp = str(exp)
        while len(mant) > 1 and mant[-1] == "0":
            mant = mant[:-1]
        if len(mant) > 1:
            mant = mant[0] + "." + mant[1:]

        # dujj.
        return coloured(col_num, f"{mant}e{exp}")

    if isinstance(x, float):
        x = complex(x)
    if not isinstance(x, complex):
        raise TypeError(f"expected int, float, or complex, got {_objtname(x)}")

    re = x.real
    im = x.imag
    def frep(n):
        if n != n:
            return coloured(col_num, "nan")
        if n == float("inf"):
            return coloured(col_num, "inf")
        if n == 0.0: # -0.0 -> 0.0
            n = 0.0
        if n < 0.0:
            return minus + frep(-n)
        s = f"{n:.6g}" if short else repr(n)
        # "xxx.0" -> "xxx"
        if s.endswith(".0"):
            s = s[:-2]
        if "e" not in s:
            return coloured(col_num, s)
        # "xxx.xe-0y" -> "xxx.xe-y"
        # "xxx.xe+0y" -> "xxx.xey"
        s = s.replace("e-0", "e-")
        s = s.replace("e+", "e")
        s = s.replace("e0", "e")
        s = s.replace(".0e", "e") # incase.
        return coloured(col_num, s)

    sep = plus

    if im == 0.0:
        im_s = ""
    elif im == 1.0:
        im_s = i
    else:
        if im == -1.0:
            sep = minus
            im_s = ""
        elif im < 0.0:
            sep = minus
            im_s = frep(-im)
        else:
            im_s = frep(im)
        im_s = f"{im_s}{i}"

    if re == 0.0:
        re_s = ""
    else:
        re_s = frep(re)

    if not re_s and not im_s:
        return frep(0.0)

    if not im_s:
        return re_s

    if not re_s:
        if sep == plus:
            sep = ""
        return f"{sep}{im_s}"

    return f"{re_s}{sep}{im_s}"


def pretty_exception(exc, callout=None, tb=True):
    """
    Coloured string of the given exception's message and (optionally) traceback.
    Prepended with "## <callout>:" if given.
    """
    def issquiggles(s):
        if s is None:
            return False
        if not s.strip():
            return False
        return all(c.strip() in {"", "~", "^"} for c in s)
    def colour(lines, i):
        line = lines[i]
        next_line = None if i == len(lines) - 1 else lines[i + 1]
        if not line.strip():
            # Blank.
            return ""
        elif line.startswith("Traceback "):
            # "Traceback (most recent call last):".
            return coloured(203, line)
        elif line.startswith("  File"):
            # Traceback file+line.
            if False and line.endswith(", in <module>"):
                # not a big fan of "in <module>", mostly bc it inconsistently
                # shows up, but ig leave it in for now.
                line = line[:len(", in <module>")]
            # regex me.
            with_in = r'(\s*File )(".*?")(, line )(\d+)(, in )(\S+)'
            without_in = r'(\s*File )(".*?")(, line )(\d+)'
            match = _re.match(with_in, line)
            if match is not None:
                txts = list(match.groups())
                cols = [7, 244, 255, 244, 7, 165, 7, 34]
            else:
                match = _re.match(without_in, line)
                if match is None:
                    return line
                txts = list(match.groups())
                cols = [7, 244, 255, 244, 7, 165]
            fname = txts.pop(1)
            if fname.startswith('"<') and fname.endswith('/console>"'):
                txts.insert(1, fname[:-len('console>"')])
                txts.insert(2, "console")
                txts.insert(3, '>"')
            elif fname.startswith('"<') and fname.endswith('>"'):
                txts.insert(1, fname[:2])
                txts.insert(2, fname[2:-2])
                txts.insert(3, fname[-2:])
            elif fname.startswith('"') and fname.endswith('"'):
                if _os.sep in fname:
                    folder, file = fname.rsplit(_os.sep, 1)
                    txts.insert(1, folder + _os.sep)
                    txts.insert(2, file[:-1])
                    txts.insert(3, file[-1])
                else:
                    txts.insert(1, fname[:1])
                    txts.insert(2, fname[1:-1])
                    txts.insert(3, fname[-1:])
            else:
                txts.insert(1, fname)
                cols.pop(3)
                cols.pop(2)
            return coloured(cols, txts)
        elif _re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*:', line) is not None:
            # Specific exception and its message.
            txts = line.split(":", 1)
            txts[0] += ":"
            return coloured([163, 171], txts)
        elif _re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', line) is not None:
            # Exception without a message.
            return coloured(163, line)
        elif issquiggles(line):
            # Markings.
            txts = ["".join(g) for _, g in _itertools.groupby(line)]
            cols = [{" ": -1, "~": 55, "^": 93}[x[0]] for x in txts]
            return coloured(cols, txts)
        # otherwise assume source code or something.

        # If the next line is squiggles, highlight the squiggled code.
        if issquiggles(next_line) and len(line) >= len(next_line):
            next_txts = ["".join(g) for _, g in _itertools.groupby(next_line)]
            cols = [{" ": -1, "~": 210, "^": 203}[x[0]] for x in next_txts]
            # cols.append(-1)
            # Group this line in the same manner as the next.
            lens = [len(s) for s in next_txts]
            txts = []
            i = 0
            for l in lens:
                txts.append(line[i:i + l])
                i += l
            if i < len(line):
                if cols[-1] == -1:
                    txts[-1] += line[i:]
                else:
                    cols.append(-1)
                    txts.append(line[i:])
            return coloured(cols, txts)

        # actually the "During handling of the ..." line also falls here.
        return line
    if not tb:
        etb = None
    else:
        etb = exc.__traceback__
        if etb is not None:
            # Cut the `exec` call from the traceback.
            etb = etb.tb_next
    tbe = _traceback.TracebackException(type(exc), exc, etb)
    old = "".join(tbe.format())
    oldlines = old.splitlines()
    lines = [colour(oldlines, i) for i in range(len(oldlines))]
    if callout is not None:
        lines.insert(0, coloured(124, f"## {callout}"))
    return "\n".join(lines)
