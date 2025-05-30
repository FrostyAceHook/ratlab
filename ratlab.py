import ast as _ast
import codeop as _codeop
import contextlib as _contextlib
import io as _io
import os as _os
import sys as _sys
import traceback as _traceback
import warnings as _warnings
from pathlib import Path as _Path

# Hack src/ into the path so we can import.
_sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), "src"))
from src import *


class _cli:
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


    def cli(self, glbls, parse=None, msg=""):
        """
        Starts a command-line interface which is basically just an embedded
        python interpreter. `glbls` should be the `globals()` of the caller,
        `parse` should take a string and return an ast tree, `msg` is displayed
        upon entering the input loop.
        """
        if getattr(self, "_in_cli", False):
            raise TypeError("cannot enter cli while in cli")

        if parse is None:
            def parse(s):
                module = _ast.parse(s)
                module = _ast.fix_missing_locations(module)
                return module

        self._leave = False
        self._in_cli = True
        try:
            self._cli_loop(glbls, parse, msg)
        finally:
            self._leave = False
            self._in_cli = False

    def _cli_loop(self, glbls, parse, msg):
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
            try:
                parsed = parse(source)
                compiled = compile(parsed, "<cli>", "exec")
            except Exception as e:
                print("## TYPO:")
                _traceback.print_exception(type(e), e, None)
                continue
            try:
                exec(compiled, glbls, glbls)
            except Exception as e:
                print("## ERROR:")
                _traceback.print_exception(type(e), e, e.__traceback__)
                continue
        if self._leave:
            print("-- okie leaving.")

_cli = _cli()

def _okie_we_leave():
    _cli._leave = True
_cli.command(_okie_we_leave, "quit")
_cli.command(_okie_we_leave, "exit")
_cli.command(_okie_we_leave, "return")
_cli.command(lambda: _os.system("cls"), "cls")
_cli.command(matrix.mhelp, "mhelp")


# Sets the current field.
def lits(field):
    lits._field = field
lits._field = None


# Field-aware functions.

def eye(n):
    if lits._field is None:
        raise ValueError("specify a field using `lits`")
    return matrix.eye(n, field=lits._field)
def zeros(rows, cols=None):
    if lits._field is None:
        raise ValueError("specify a field using `lits`")
    return matrix.zeros(rows, cols, field=lits._field)
def ones(rows, cols=None):
    if lits._field is None:
        raise ValueError("specify a field using `lits`")
    return matrix.ones(rows, cols, field=lits._field)
def summ(*xs):
    if lits._field is None:
        raise ValueError("specify a field using `lits`")
    return matrix.summ(*xs, field=lits._field)
def prod(*xs):
    if lits._field is None:
        raise ValueError("specify a field using `lits`")
    return matrix.prod(*xs, field=lits._field)
def ave(*xs):
    if lits._field is None:
        raise ValueError("specify a field using `lits`")
    return matrix.ave(*xs, field=lits._field)
def minn(*xs):
    if lits._field is None:
        raise ValueError("specify a field using `lits`")
    return matrix.minn(*xs, field=lits._field)
def maxx(*xs):
    if lits._field is None:
        raise ValueError("specify a field using `lits`")
    return matrix.maxx(*xs, field=lits._field)


# Overview of syntax changes:
# - added 'lst' "keyword" to create list literals via `lst[1,2,3]`
# - any list literals become row matrices.
# - any subscripts on matrix literals append a row and are a matrix literal
#       (looks like [1,2][3,4] and includes [1,2][3,4][5,6]).
# - any literals within math expressions in matrix cells get cast to the current
#       field. note this only propagates via some ops. (looks like
#       [lit(1), lit(1) * lit(2) * function_call(1)], note the last 1 is not
#       cast (to get around this, you must wrap it in a matrix literal [1])).

def _wrapped_literal(x):
    if lits._field is None:
        raise ValueError("specify a field using `lits`")
    if isinstance(x, Matrix):
        return x
    Mat = Matrix[lits._field, (1, 1)]
    return Mat._cast(x)

def _wrapped_hstack(xs):
    return hstack(*xs)

def _wrapped_vstack(matrix, idx):
    if isinstance(idx, slice):
        raise TypeError("cannot use slices as matrix elements")
    if not isinstance(idx, tuple):
        idx = (idx, )
    newrow = _wrapped_hstack(idx)
    return vstack(matrix, newrow)

class _WrappedTransformer(_ast.NodeTransformer):
    def __init__(self):
        super().__init__()
        self.parents = []
        self.in_matrix = False

    def visit(self, node):
        self.parents.append(node)

        propagate_to = (_ast.BoolOp, _ast.NamedExpr, _ast.BinOp, _ast.UnaryOp,
                _ast.Compare, _ast.IfExp, _ast.Constant)
        was_in_matrix = self.in_matrix
        if not isinstance(node, propagate_to):
            self.in_matrix = False
        new_node = super().visit(node)
        self.in_matrix = was_in_matrix

        self.parents.pop()
        return new_node

    def visit_Constant(self, node):
        if self.in_matrix:
            return _ast.Call(
                func=_ast.Name(id="_wrapped_literal", ctx=_ast.Load()),
                args=[node],
                keywords=[],
            )
        return node

    def visit_Name(self, node):
        # dont touch if its an attribute.
        parent = self.parents[-2] if len(self.parents) >= 2 else None
        if parent is not None and isinstance(parent, _ast.Attribute):
            return node

        # Ensure `lst` is reserved.
        if node.id == "lst" and not isinstance(node.ctx, _ast.Load):
            raise SyntaxError("cannot modify 'lst' keyword")

        # Thats all rn. used to do more, may do more in the future.
        return node

    def visit_List(self, node):
        # All list literals are matrices.
        self.in_matrix = True

        # Transform its elements.
        self.generic_visit(node)

        # Make it a matrix.
        return _ast.Call(
            func=_ast.Name(id="_wrapped_hstack", ctx=_ast.Load()),
            args=[node],
            keywords=[]
        )

    def visit_Subscript(self, node):
        what = "subscript"

        # Recurse to the object being subscripted.
        node.value = self.visit(node.value)
        val = node.value

        # See if we're trying to create a list.
        if isinstance(val, _ast.Name) and val.id == "lst":
            what = "list"

        # See if we're subscripting a matrix literal.
        if isinstance(val, _ast.Call) and isinstance(val.func, _ast.Name):
            if val.func.id in {"_wrapped_hstack", "_wrapped_vstack"}:
                what = "matrix"


        # Now that we know if this is matrix, we can recurse to slice children.
        self.in_matrix = (what == "matrix")
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
            return _ast.Call(
                func=_ast.Name(id="_wrapped_vstack", ctx=_ast.Load()),
                args=[node.value, node.slice],
                keywords=[]
            )

        return node

def _wrapped_print(x):
    if x is not None:
        print(repr(x))

def _wrapped_parse(source):
    # Parse the code.
    module = _ast.parse(source, "<cli>")
    module = _WrappedTransformer().visit(module)

    # If the entire thing is an expression, add a print if non-none.
    if len(module.body) == 1 and isinstance(module.body[0], _ast.Expr):
        # However, this can be circumvented by appending a semicolon.
        if not source.lstrip().endswith(";"):
            expr_node = module.body[0]
            tree = _ast.Expr(
                value=_ast.Call(
                    func=_ast.Name(id="_wrapped_print", ctx=_ast.Load()),
                    args=[expr_node.value],
                    keywords=[],
                )
            )
            module = _ast.Module(body=[tree], type_ignores=[])

    module = _ast.fix_missing_locations(module)
    return module



if __name__ == "__main__":
    glbls = {k: v for k, v in globals().items()
            if k.lower().startswith("_wrapped") # need to be visible.
            or not k.startswith("_") # get rid of them.
        }

    # Read in any files entered.
    for path in _sys.argv[1:]:
        with util.readfile(path) as f:
            if f is util.MISSING:
                continue
            for line in f:
                if line[-1] == "\n":
                    line = line[:-1]
                _cli.enqueue(line)

    _cli.enqueue("lits(Real)")
    _cli.cli(glbls, _wrapped_parse,
        ",--------------------------,\n"
        "|    RATLAB® Stuart Inc.   |\n"
        "'--------------------------'\n"
        "  lits(field_cls) : changes literal types\n"
        "  [1,2][3,4] : matrix\n"
        "  lst[1,2,3] : list\n"
        "  mhelp : print matrix methods\n"
    )
