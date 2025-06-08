import ast as _ast
import codeop as _codeop
import contextlib as _contextlib
import linecache as _linecache
import io as _io
import itertools as _itertools
import os as _os
import re as _re
import traceback as _traceback
import warnings as _warnings
from pathlib import Path as _Path

import matrix as _matrix
from util import tname as _tname


def coloured(cols, txts):
    """
    When given the colour 'cols' and text 'txts' arrays, prints each element of
    the text as its corresponding colour. Colour codes can be found at:
    https://gist.github.com/fnky/458719343aabd01cfb17a3a4f7296797
    """
    if isinstance(cols, int):
        cols = (cols, )
    if isinstance(txts, str):
        txts = (txts, )
    if len(cols) != len(txts):
        raise TypeError("must give one colour for each piece of text, got "
                f"{len(cols)} colours and {len(txts)} texts")
    for col in cols:
        if not isinstance(col, int):
            raise TypeError("expected integer color code, got "
                    f"{_tname(type(col))}")
    for txt in txts:
        if not isinstance(txt, str):
            raise TypeError(f"expected string text, got {_tname(type(txt))}")
    def colour(c, t):
        if c < 0:
            return t
        if not t:
            return t
        return f"\x1B[38;5;{c}m{t}"
    segs = "".join(colour(c, t) for c, t in zip(cols, txts)) # funny
    return segs + "\x1B[0m" if segs else ""

# Hack to enable console escape codes.
_os.system("")



# Overview of syntax changes:
# - added 'lst' "keyword" to create list literals via `lst[1,2,3]`
# - any list literals become row matrices.
# - any subscripts on matrix literals append a row and are a matrix literal
#       (looks like [1,2][3,4] and includes [1,2][3,4][5,6]).
# - any literals within math expressions in matrix cells get cast to the current
#       field. note this only propagates via some ops. (looks like
#       [lit(1), lit(1) * lit(2) * function_call(1)], note the last 1 is not
#       cast (to get around this, you must wrap it in a matrix literal [1])).
# - if the entire parse is one expression and is not semi-colon terminated, wrap
#       it in a print-if-non-none.
# - override 'lits' to give the current space.


def _literal(x):
    field = _matrix._get_field(None)
    [x] = _matrix.Single[field].cast(x)
    return x

def _hstack(*xs):
    return _matrix.hstack(*xs)

def _vstack(mat, idx):
    if isinstance(idx, slice):
        raise TypeError("cannot use slices as matrix elements")
    if not isinstance(idx, tuple):
        idx = (idx, )
    newrow = _hstack(*idx)
    return _matrix.vstack(mat, newrow)

def _is_hvstack(node):
    if not isinstance(node, _ast.Call):
        return False
    func = node.func
    if not isinstance(func, _ast.Attribute):
        return False
    mod = func.value
    if not isinstance(mod, _ast.Name):
        return False
    if mod.id != "_syntax":
        return False
    name = func.attr
    return name == "_hstack" or name == "_vstack"

def _is_named(node, names):
    if not isinstance(node, _ast.Name):
        return False
    return node.id in names

def _ast_call(name, *args):
    return _ast.Call(
        func=_ast.Attribute(
            value=_ast.Name(id="_syntax", ctx=_ast.Load()),
            attr=name,
            ctx=_ast.Load()
        ),
        args=list(args),
        keywords=[],
    )

class _Transformer(_ast.NodeTransformer):
    def __init__(self):
        super().__init__()
        self.wrap_lits = False
        self.pierce_tuple = False
        self.keywords = ["lst", "_syntax"]

    def visit(self, node):
        was_wrapping_lits = self.wrap_lits

        propagate_to = (_ast.BoolOp, _ast.NamedExpr, _ast.BinOp, _ast.UnaryOp,
            _ast.Compare, _ast.IfExp, _ast.Constant,
            _ast.Tuple # also requires `pierce_tuple`.
        )
        if not isinstance(node, propagate_to):
            self.wrap_lits = False
        if isinstance(node, _ast.Tuple) and not self.pierce_tuple:
            self.wrap_lits = False
        self.pierce_tuple = False

        new_node = super().visit(node)

        self.wrap_lits = was_wrapping_lits

        return new_node

    def visit_Constant(self, node):
        if self.wrap_lits:
            return _ast_call("_literal", node)
        return node

    def visit_Name(self, node):
        # Ensure keywords aren't modified.
        if not isinstance(node.ctx, _ast.Load):
            if _is_named(node, self.keywords):
                raise SyntaxError(f"cannot modify keyword {repr(kw)}")
        return node

    def visit_Attribute(self, node):
        node = self.generic_visit(node)

        # Ensure keywords aren't modified.
        if not isinstance(node.ctx, _ast.Load):
            if _is_named(node, self.keywords):
                raise SyntaxError(f"cannot modify keyword {repr(kw)} attrs")
        return node

    def visit_List(self, node):
        # Only transform loads into matrices.
        if not isinstance(node.ctx, _ast.Load):
            return self.generic_visit(node)

        # All list literals are matrices.
        self.wrap_lits = True
        # If it's immediately wrapped in a tuple, unpack it to have the same
        # behaviour as subscripting:
        #  [1,2][(3,4)] == [1 2][3 4]
        #  [(1,2)][(3,4)] == error typically
        if len(node.elts) == 1 and isinstance(node.elts[0], _ast.Tuple):
            tpl = node.elts[0]
            if isinstance(tpl.ctx, _ast.Load): # ig check its load?
                node.elts = tpl.elts
        # Transform elements.
        node = self.generic_visit(node)
        # Make it a matrix.
        return _ast_call("_hstack", *node.elts)

    def visit_Subscript(self, node):
        what = "subscript"

        # Recurse to the object being subscripted.
        node.value = self.visit(node.value)
        val = node.value

        # See if we're trying to create a list.
        if _is_named(val, {"lst"}):
            what = "list"

        # See if we're subscripting a matrix literal.
        if _is_hvstack(val):
            what = "matrix"


        # Now that we know if this is matrix, we can recurse to slice children.
        self.wrap_lits = (what == "matrix")
        self.pierce_tuple = True # pierce one tuple.
        node.slice = self.visit(node.slice)

        if what == "list":
            # Make it a list, preserving context.
            if isinstance(node.slice, _ast.Tuple):
                elts = node.slice.elts[:] # unpack if tuple.
            else:
                elts = [node.slice]
            return _ast.List(elts=elts, ctx=node.ctx)

        if what == "matrix":
            # Ensure its a load.
            if not isinstance(node.ctx, _ast.Load):
                raise SyntaxError("cannot assign to a matrix literal")
            # Create a call to concat the row.
            return _ast_call("_vstack", node.value, node.slice)

        assert what == "subscript"
        return node

    def visit_Call(self, node):
        # Transform args.
        node = self.generic_visit(node)

        # See if it's a `lits` call.
        if not _is_named(node.func, {"lits"}):
            return node

        # Ensure theres only one arg given (and it is field).
        for kw in node.keywords:
            if kw.arg != "field":
                raise SyntaxError("lits() got an unexpected keyword argument: "
                        f"{repr(kw.arg)}")
        if len(node.args) > 1:
            raise SyntaxError("lits() takes 1 positional argument but "
                    f"{len(node.args)} were given")
        if node.args and node.keywords:
            raise SyntaxError("lits() got multiple values for argument 'field'")
        if not node.args and not node.keywords:
            raise SyntaxError("lits() missing 1 required positional argument: "
                    "'field'")

        # Add the current space as kwarg.
        space = _ast.Name(id="_space", ctx=_ast.Load())
        space_kwarg = _ast.keyword(arg="space", value=space)
        node.keywords.append(space_kwarg)

        return node

def _print_nonnone(x):
    if x is not None:
        print(repr(x))

def _parse(source, filename, feedback):
    # Parse the code.
    module = _ast.parse(source, filename)
    module = _Transformer().visit(module)

    # If requested, and a print-if-nonnone if the entire thing is an expression.
    body = module.body
    if feedback and len(body) == 1 and isinstance(body[0], _ast.Expr):
        # However, this can be circumvented by appending a semicolon.
        if not source.lstrip().endswith(";"):
            expr_node = body[0]
            tree = _ast.Expr(value=_ast_call("_print_nonnone", expr_node.value))
            module = _ast.Module(body=[tree], type_ignores=module.type_ignores)

    module = _ast.fix_missing_locations(module)
    return module

def _print_exc(exc, callout, tb=False):
    def colour(line):
        if not line.strip():
            # Blank.
            return ""
        elif line.startswith("Traceback "):
            # "Traceback (most recent call last):".
            return coloured(203, line)
        elif all(c in set(" ~^") for c in line):
            # Markings.
            txts = ["".join(g) for _, g in _itertools.groupby(line)]
            cols = [{" ": -1, "~": 55, "^": 93}[x[0]] for x in txts]
            return coloured(cols, txts)
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
            if fname[:2] == '"<' and fname[-2:] == '>"':
                txts.insert(1, fname)
                cols.pop(3)
                cols.pop(2)
            elif _os.sep not in fname:
                txts.insert(1, fname[0])
                txts.insert(2, fname[1:-1])
                txts.insert(3, fname[-1])
            else:
                folder, file = fname.rsplit(_os.sep, 1)
                txts.insert(1, folder + _os.sep)
                txts.insert(2, file[:-1])
                txts.insert(3, file[-1])
            return coloured(cols, txts)
        elif _re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*:', line) is not None:
            # Specific exception and its message.
            txts = line.split(":", 1)
            txts[0] += ":"
            return coloured([163, 171], txts)
        # otherwise assume source code or something.
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
    new = "\n".join(colour(line) for line in old.splitlines())
    print(coloured(124, f"## {callout}"))
    print(new)


def _execute(source, space, filename="<rat>", feedback=True):
    # Store cli source for nice error printing.
    if filename == "<rat>":
        lines = [line + "\n" for line in source.splitlines()]
        _linecache.cache[filename] = (len(source), None, lines, filename)
    try:
        # Transform and compile.
        parsed = _parse(source, filename=filename, feedback=feedback)
        compiled = compile(parsed, filename=filename, mode="exec")
    except Exception as e:
        _print_exc(e, "TYPO", tb=False)
        return False
    try:
        # Splice _syntax into this bitch.
        if "_syntax" not in space:
            space["_syntax"] = __import__(__name__)
        if space["_syntax"] is not __import__(__name__):
            raise RuntimeError("reserved '_syntax' set within variable space")
        # Splice the space into itself.
        if "_space" not in space:
            space["_space"] = space
        if space["_space"] is not space:
            raise RuntimeError("reserved '_space' set within variable space")
        # Execute. i dont remember now lmao but past me had issues when `locals`
        # wasnt also set to the global space. probably some shitting bug idk.
        exec(compiled, locals=space, globals=space)
    except Exception as e:
        _print_exc(e, "ERROR", tb=True)
        return False
    return True



def run_cli(space):
    """
    Starts a command-line interface which is basically just an embedded python
    interpreter. `space` should be a globals()-type dictionary of variables to
    expose (and it will be modified).
    """
    def get_input():
        source = ""
        while True:
            try:
                print(coloured(73, ".. " if source else ">> "), end="")
                line = input()
            except EOFError:
                break
            source += "\n"*(not not source) + line
            if not line:
                break
            try:
                # dont let the stupid compile command print to stderr.
                with _warnings.catch_warnings():
                    _warnings.simplefilter("ignore", SyntaxWarning)
                    with _contextlib.redirect_stderr(_io.StringIO()):
                        command = _codeop.compile_command(source)
            except Exception as e:
                break
            if command is not None:
                break
        return source

    leave = False
    while not leave:
        source = get_input()
        stripped = source.strip()
        if not stripped:
            continue
        if stripped == "quit":
            leave = True
            break
        if stripped == "cls":
            _os.system("cls")
            continue
        _execute(source, space)
    if leave:
        print(coloured([73, 80], ["-- ", "okie leaving."]))



def run_file(path, space):
    """
    Executes the file at the given path. `space` should be a globals()-type
    dictionary of variables to expose (and it will be modified).
    """
    path = _Path(path)

    def esc(path):
        s = str(path)
        # Always use forward slash separators for paths.
        if _os.name == "nt":
            s = s.replace("\\", "/")
        # Escape any control codes, using repr and trimming its quotes.
        for i in _itertools.chain(range(0x00, 0x20), range(0x7F, 0xA0)):
            s = s.replace(chr(i), repr(chr(i))[1:-1])
        quote = "\"" if ("'" in s) else "'"
        s = s.replace(quote, "\\" + quote)
        s = s.replace("\\", "\\\\")
        return quote + s + quote

    def coloured_esc(path):
        cols = [244, 255, 244]
        string = esc(path)
        if "/" not in string:
            txts = [string[0], string[1:-1], string[-1]]
        else:
            pre, aft = string.rsplit("/", 1)
            txts = [pre + "/", aft[:-1], aft[-1]]
        return cols, txts

    def query(before, path, after=""):
        if before:
            before = before + " "
        if after:
            after = " " + after
        txts = [before, after, ", ignore?", " (y/n): "]
        cols = [   203,   203,         203,        210]
        pcols, ptxts = coloured_esc(path)
        txts[1:1] = ptxts
        cols[1:1] = pcols
        msg = coloured(cols, txts)
        while True:
            response = input(msg).strip().casefold()
            if not response:
                continue
            if response in "yn":
                return response != "n"

    def error(before, path, after=""):
        if before:
            before = before + " "
        if after:
            after = " " + after
        txts = ["ratlab: error: ", before, after]
        cols = [              124,    203,   203]
        pcols, ptxts = coloured_esc(path)
        txts[2:2] = ptxts
        cols[2:2] = pcols
        print(coloured(cols, txts))
        quit()


    # Handle missing/invalid paths.
    bad = False
    if not path.exists():
        bad = True
        ignore = query("file", path, "doesn't exist")
    elif not path.is_file():
        bad = True
        ignore = query("path", path, "is not a file")
    if bad:
        if not ignore:
            error("missing file", path)
        return

    # Read and execute the file.
    with path.open("r", encoding="utf-8") as file:
        source = file.read()
        if not _execute(source, space, filename=str(path), feedback=False):
            ignore = query("exception in file", path)
            if not ignore:
                error("exception in file", path)
            return
