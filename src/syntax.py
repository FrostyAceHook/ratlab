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
from util import tname as _tname, coloured as _coloured



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


def _is_load(node):
    return isinstance(node.ctx, _ast.Load)

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

def _literal(x):
    field = _matrix._get_field(None)
    x, = _matrix.castall(x)
    return x

def _list_row(*xs):
    return _matrix.hstack(*xs)

def _list_col(*xs):
    return _matrix.vstack(*xs)

def _slice(matrix_literal, *sliced_by):
    append_me = _matrix.hstack(*sliced_by)
    return _matrix.vstack(matrix_literal, append_me)

def _is_matrix_literal(node):
    # strictly helper for subscript handler.
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
    return name == "_slice" or name == "_list_row" or name == "_list_col"

class _Transformer(_ast.NodeTransformer):
    def __init__(self, source, filename):
        super().__init__()
        self.source = source
        self.filename = filename
        self.wrap_lits = False
        self.row_vector = False
        self.pierce_tuple = False
        self.keywords = ["lst", "_syntax", "_space"]

    def syntaxerrorme(self, msg, node):
        exc = SyntaxError(msg)
        exc.filename = self.filename
        exc.lineno = node.lineno
        exc.offset = node.col_offset + 1 # 1-based.
        if getattr(node, "end_lineno", None) is not None:
            exc.end_lineno = node.end_lineno
        if getattr(node, "end_col_offset", None) is not None:
            exc.end_offset = node.end_col_offset + 1
        exc.text = self.source.splitlines()[node.lineno - 1]
        raise exc

    def visit(self, node):
        was_wrapping_lits = self.wrap_lits

        propagate_to = (_ast.BoolOp, _ast.NamedExpr, _ast.BinOp, _ast.UnaryOp,
            _ast.Compare, _ast.IfExp, _ast.Constant, _ast.Attribute,
            _ast.Tuple # also requires `pierce_tuple`.
        )
        if not isinstance(node, propagate_to):
            self.wrap_lits = False
        if isinstance(node, _ast.Tuple) and not self.pierce_tuple:
            self.wrap_lits = False
        self.pierce_tuple = False
        if not isinstance(node, _ast.List):
            self.row_vector = False

        new_node = super().visit(node)

        self.wrap_lits = was_wrapping_lits

        return new_node

    def visit_Constant(self, node):
        if self.wrap_lits:
            return _ast_call("_literal", node)
        return node

    def visit_Name(self, node):
        # Ensure keywords aren't modified.
        if not _is_load(node):
            if _is_named(node, self.keywords):
                self.syntaxerrorme(f"cannot modify keyword {repr(kw)} ", node)
        return node

    def visit_Attribute(self, node):
        node = self.generic_visit(node)

        # Ensure keywords aren't modified.
        if not _is_load(node):
            if _is_named(node, self.keywords):
                self.syntaxerrorme(f"cannot modify keyword {repr(kw)} "
                        "attributes", node)
        return node

    def visit_List(self, node):
        # Only transform loads into matrices.
        if not _is_load(node):
            return self.generic_visit(node)

        # If it's immediately wrapped in a tuple, unpack it to have the same
        # behaviour as subscripting:
        #  [1,2][(3,4)] == [1 2][3 4]
        #  [(1,2)][(3,4)] == error typically
        if len(node.elts) == 1 and isinstance(node.elts[0], _ast.Tuple):
            tpl = node.elts[0]
            if _is_load(tpl): # ig check its load?
                node.elts = tpl.elts
        # All list literals are matrices.
        self.wrap_lits = True
        # Pop whether to row vector or not.
        rowme = self.row_vector
        self.row_vector = False
        # Transform elements.
        node = self.generic_visit(node)
        # Make it a row vector if this literal is becoming a 2D matrix, otherwise
        # column vector.
        # NEVERMIND, its kinda ass. mainly for when concating matrices, like:
        #  x = [1,2]  (x = [1][2])
        #  [x, x]   (!= [1,1][2,2], it =[1][2][1][2])
        # funcname = "_list_row" if rowme else "_list_col"
        funcname = "_list_row"
        new_node = _ast_call(funcname, *node.elts)
        _ast.copy_location(new_node, node)
        return new_node

    def visit_Subscript(self, node):
        # Handle the thing we subscripting, so we can know if its a matrix
        # literal.
        self.row_vector = True
        node.value = self.visit(node.value)

        # Find what we're doing based on what we're subscripting.
        if _is_named(node.value, {"lst"}):
            what = "list" # create a list literal.
        elif _is_matrix_literal(node.value):
            what = "matrix" # append a row to a matrix literal.
        else:
            what = "normal" # idk normal subscript things.

        # Now can recurse to the new row/index.
        self.wrap_lits = (what == "matrix")
        self.pierce_tuple = True
        node.slice = self.visit(node.slice)

        if what == "list":
            # Make it a list, preserving context.
            if isinstance(node.slice, _ast.Tuple):
                elts = node.slice.elts[:] # unpack if tuple.
            else:
                elts = [node.slice]
            for elt in elts:
                elt.ctx = node.ctx
            new_node = _ast.List(elts=elts, ctx=node.ctx)
            _ast.copy_location(new_node, node)
            return new_node

        if what == "matrix":
            # Ensure its a load.
            if isinstance(node.ctx, _ast.Store):
                self.syntaxerrorme("cannot assign to a matrix literal", node)
            if isinstance(node.ctx, _ast.Del):
                self.syntaxerrorme("cannot delete a matrix literal", node)
            # Get all the slice elements.
            elts = node.slice
            if isinstance(elts, _ast.Tuple):
                if not _is_load(elts):
                    self.syntaxerrorme("how have you even done this", elts)
                elts = elts.elts
            else:
                elts = [elts]
            # Check no slice literals.
            for elt in elts:
                if isinstance(elt, _ast.Slice):
                    self.syntaxerrorme("cannot use slices as elements of a "
                            "matrix literal", elt)
            # Create a call to concat the row.
            new_node = _ast_call("_slice", node.value, *elts)
            _ast.copy_location(new_node, node)
            return new_node

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
                self.syntaxerrorme("lits() got an unexpected keyword argument: "
                        f"{repr(kw.arg)}", node)
        if len(node.args) > 1:
            self.syntaxerrorme("lits() takes 1 positional argument but "
                    f"{len(node.args)} were given", node)
        if node.args and node.keywords:
            self.syntaxerrorme("lits() got multiple values for argument 'field'",
                    node)
        if not node.args and not node.keywords:
            self.syntaxerrorme("lits() missing 1 required positional argument: "
                    "'field'", node)

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
    module = _Transformer(source, filename).visit(module)

    # If requested, and a print-if-nonnone if the entire thing is an expression.
    body = module.body
    if feedback and len(body) == 1 and isinstance(body[0], _ast.Expr):
        # However, this can be circumvented by appending a semicolon.
        if not source.lstrip().endswith(";"):
            expr_node = body[0]
            tree = _ast.Expr(value=_ast_call("_print_nonnone", expr_node.value))
            module = _ast.Module(body=[tree], type_ignores=module.type_ignores)

    _ast.fix_missing_locations(module)
    return module

def _print_exc(exc, callout, tb=False):
    def colour(line):
        if not line.strip():
            # Blank.
            return ""
        elif line.startswith("Traceback "):
            # "Traceback (most recent call last):".
            return _coloured(203, line)
        elif all(c in set(" ~^") for c in line):
            # Markings.
            txts = ["".join(g) for _, g in _itertools.groupby(line)]
            cols = [{" ": -1, "~": 55, "^": 93}[x[0]] for x in txts]
            return _coloured(cols, txts)
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
            return _coloured(cols, txts)
        elif _re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*:', line) is not None:
            # Specific exception and its message.
            txts = line.split(":", 1)
            txts[0] += ":"
            return _coloured([163, 171], txts)
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
    print(_coloured(124, f"## {callout}"))
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
        exec(compiled, space, space)
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
                print(_coloured(73, ".. " if source else ">> "), end="")
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
        print(_coloured([73, 80], ["-- ", "okie leaving."]))



def run_file(space, path, has_next):
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
        msg = _coloured(cols, txts)
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
        print(_coloured(cols, txts))
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
        if not has_next:
            return
        ignore = query("exception in file", path)
        if not ignore:
            error("exception in file", path)
        print(_coloured([73, 80], ["-- ", "okie continuing."]))
