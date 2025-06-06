import ast as _ast
import codeop as _codeop
import contextlib as _contextlib
import io as _io
import os as _os
import traceback as _traceback
import warnings as _warnings
from pathlib import Path as _Path

import matrix as _matrix

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


def _literal(x):
    if _matrix.lits.field is None:
        raise RuntimeError("specify a field using `lits`")
    if isinstance(x, _matrix.Matrix):
        return x
    Mat = _matrix.Matrix[_matrix.lits.field, (1, 1)]
    return Mat._cast(x)

def _hstack(xs):
    return _matrix.hstack(*xs)

def _vstack(mat, idx):
    if isinstance(idx, slice):
        raise TypeError("cannot use slices as matrix elements")
    if not isinstance(idx, tuple):
        idx = (idx, )
    newrow = _hstack(idx)
    return _matrix.vstack(mat, newrow)

def _is_hvstack(node):
    if not isinstance(node, _ast.Call):
        return False
    func = node.func
    if not isinstance(func, _ast.Attribute):
        return False
    # if not isinstance(func.ctx, _ast.Load):
    #     return False
    mod = func.value
    if not isinstance(mod, _ast.Name):
        return False
    # if not isinstance(mod.ctx, _ast.Load):
    #     return False
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
            if _is_name(node, self.keywords):
                raise SyntaxError(f"cannot modify keyword {repr(kw)} attrs")
        return node

    def visit_List(self, node):
        # All list literals are matrices.
        self.wrap_lits = True

        # Transform elements.
        node = self.generic_visit(node)

        # Make it a matrix.
        return _ast_call("_hstack", node)

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

def _print_nonnone(x):
    if x is not None:
        print(repr(x))

def _parse(source, name, feedback):
    # Parse the code.
    module = _ast.parse(source, name)
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


def _execute(source, space, feedback=False):
    try:
        parsed = _parse(source, name="<rat>", feedback=feedback)
        compiled = compile(parsed, "<rat>", "exec")
    except Exception as e:
        print("## TYPO:")
        _traceback.print_exception(type(e), e, None)
        return False
    try:
        # Splice _syntax into this bitch.
        if "_syntax" not in space:
            space["_syntax"] = __import__(__name__)
        if space["_syntax"] is not __import__(__name__):
            raise ValueError("reserved '_syntax' set within variable space")
        exec(compiled, space, space)
    except Exception as e:
        print("## ERROR:")
        _traceback.print_exception(type(e), e, e.__traceback__)
        return False
    return True



class cli:
    """ Command-line intepreter (not rly interface lmao). """

    _commands = {}
    def command(self, func, name):
        """
        Makes a function into a command (callable without brackets if that is the
        only text in the input, which invokes `func()`).
        """
        if name in self._commands:
            raise ValueError("command already defined under that name")
        self._commands[name] = func

    _queue = []
    def enqueue(self, code):
        """
        Queues `code` to-be executed as-if it was input when `cli(...)` is
        entered.
        """
        if not isinstance(code, str):
            raise ValueError("can only enqueue strings")
        self._queue.append(code)


    def start(self, space, msg=""):
        """
        Starts a command-line interface which is basically just an embedded
        python interpreter. `space` should be the `globals()` of the caller,
        `msg` is displayed upon entering the input loop.
        """
        if getattr(self, "_in_cli", False):
            raise TypeError("cannot enter cli while in cli")

        self._leave = False
        self._in_cli = True
        try:
            self._go(space, msg)
        finally:
            self._leave = False
            self._in_cli = False

    def _go(self, space, msg):
        def get_input():
            if self._queue:
                source = self._queue.pop(0)
                lines = source.split("\n")
                print(">>", lines[0])
                for line in lines[1:]:
                    print("..", line)
                return source

            source = ""
            while True:
                try:
                    line = input(">> " if not source else ".. ")
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

        if (msg := str(msg)):
            print(msg)

        while not self._leave:
            source = get_input()
            stripped = source.strip()
            if not stripped:
                continue
            if stripped in self._commands:
                try:
                    self._commands[stripped]()
                except Exception as e:
                    print("## ERROR:")
                    _traceback.print_exception(type(e), e, e.__traceback__)
                    continue
                continue
            _execute(source, space, feedback=True)
        if self._leave:
            print("-- okie leaving.")

cli = cli()

def _okie_we_leave():
    cli._leave = True
cli.command(_okie_we_leave, "quit")
cli.command(_okie_we_leave, "exit")
cli.command(_okie_we_leave, "return")
cli.command(lambda: _os.system("cls"), "cls")
cli.command(_matrix.mhelp, "mhelp")




def run_file(path, space):
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

    def query(msg):
        msg = f"{msg} (y/n/a): "
        if query.all:
            print(msg + "y")
            return True
        while True:
            response = input(msg).strip().casefold()
            if not response:
                continue
            if response in "yna":
                query.all = (response == "a")
                return response != "n"
    query.all = False

    # Handle missing/invalid paths.
    bad = False
    if not path.exists():
        bad = True
        ignore = query(f"file {esc(path)} doesn't exist, ignore?")
    elif not path.is_file():
        bad = True
        ignore = query(f"path {esc(path)} is not a file, ignore?")
    if bad:
        if not ignore:
            print("ratlab: error: missing file", esc(path))
            quit()
        return

    # Read and execute the file.
    with path.open("r", encoding="utf-8") as file:
        source = file.read()
        if not _execute(source, space):
            quit()
