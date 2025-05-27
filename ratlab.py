import ast as _ast
import codeop as _codeop
import contextlib as _contextlib
import inspect as _inspect
import io as _io
import os as _os
import sys as _sys
import traceback as _traceback
import warnings as _warnings

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
            raise ValueError("must enqueue strings")
        if not code.lstrip():
            raise ValueError("cannot enqueue empty string")
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
                            command = codeop.compile_command(source)
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
def _char_check():
    for i in range(256*4 * 16):
        print("\n"*((i%24) == 0) + " "+chr(i)+" ", end="")
    print()
_cli.command(_char_check, "charcheck")


def _mhelp():
    def print_attr(name, desc):
        s = f"{name} .."
        s += "." * (15 - len(s))
        s += f" {desc.strip()}"
        print(s)

    print(f"Matrix - {Matrix.__doc__.strip()}")
    print_attr("field", "Cells are of this field.")
    print_attr("shape", "(row count, column count)")

    cls = Matrix[Field, (1, 1)]
    attrs = [(name, attr) for name, attr in vars(cls).items()
            if attr.__doc__ is not None
            and name != "__module__"
            and name != "__doc__"
            and name != "template"
            and name not in Matrix.params]
    for name, attr in attrs:
        if callable(attr):
            sig = _inspect.signature(attr)
            sig = str(sig)
            if sig.startswith("(self, "):
                sig = "(" + sig[len("(self, "):]
            elif sig.startswith("(self"):
                sig = "(" + sig[len("(self"):]
            name += sig
        print_attr(name, attr.__doc__)
_cli.command(_mhelp, "mhelp")


def lits(field):
    lits._field = field
lits._field = None


# current-field-aware identity matrix.
def eye(n):
    if lits._field is None:
        raise ValueError("specify a field using `lits`")
    return Matrix[lits._field, (n, n)].eye



def _wrapped_literal(x):
    if lits._field is not None:
        return lits._field.cast(x)
    return x

_WRAPPED_CONSTANTS = {
    "e": math.e,
    "pi": math.pi,
    "i": 1.0j,
    "inf": float("inf"),
    "cinf": complex("inf+infj"),
    "nan": float("nan"),
    "cnan": complex("nan+nanj")
}
def _wrapped_constant(x):
    if lits._field is None:
        return _WRAPPED_CONSTANTS[x]
    return lits._field.cast(_WRAPPED_CONSTANTS[x])

def _wrapped_matrix1row(elts):
    elts = tuple(elts)
    field = fieldof(elts)
    return Matrix[field, (1, len(elts))](elts)

def _wrapped_matrix(matrix, idx):
    if not isinstance(idx, tuple):
        idx = (idx, )
    newrow = _wrapped_matrix1row(idx)
    return vstack(matrix, newrow)

class _WrappedTransformer(_ast.NodeTransformer):
    def __init__(self):
        super().__init__()
        self.parents = []
    def visit(self, node):
        self.parents.append(node)
        new_node = super().visit(node)
        self.parents.pop()
        return new_node

    def visit_Constant(self, node):
        return _ast.Call(
            func=_ast.Name(id="_wrapped_literal", ctx=_ast.Load()),
            args=[node],
            keywords=[],
        )

    def visit_Name(self, node):
        # dont touch if its an attribute.
        parent = self.parents[-2] if len(self.parents) >= 2 else None
        if parent is not None and isinstance(parent, _ast.Attribute):
            return node

        # Might not be a recognised constant.
        if node.id not in _WRAPPED_CONSTANTS:
            return node

        # Ensure that its in a load context.
        if not isinstance(node.ctx, _ast.Load):
            raise SyntaxError("cannot modify constants")

        # Wrap it.
        return _ast.Call(
            func=_ast.Name(id="_wrapped_constant", ctx=_ast.Load()),
            args=[_ast.Constant(value=node.id)],
            keywords=[]
        )

    def visit_List(self, node):
        # Ensure children aren't skipped.
        self.generic_visit(node)

        # Make it a matrix.
        return _ast.Call(
            func=_ast.Name(id="_wrapped_matrix1row", ctx=_ast.Load()),
            args=[node],
            keywords=[]
        )

    def visit_Subscript(self, node):
        # Ensure children aren't skipped.
        self.generic_visit(node)

        # See if we're trying to create a list.
        if isinstance(node.value, _ast.Name):
            name = node.value
            if name.id == "lst" and isinstance(name.ctx, _ast.Load):
                # Make it a list.
                if isinstance(node.slice, _ast.Tuple):
                    elts = node.slice.elts[:]
                else:
                    elts = [node.slice]
                return _ast.List(elts=elts, ctx=_ast.Load())

        # Check if the value being subscripted is a matrix literal.
        if not isinstance(node.value, _ast.Call):
            return node
        if not isinstance(node.value.func, _ast.Name):
            return node
        func_name = node.value.func.id
        if func_name != "_wrapped_matrix1row" and func_name != "_wrapped_matrix":
            return node
        # Create a call to concat the row.
        return _ast.Call(
            func=_ast.Name(id="_wrapped_matrix", ctx=_ast.Load()),
            args=[node.value, node.slice],
            keywords=[]
        )

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
    _cli.enqueue("lits(Num)")
    _cli.cli(globals(), _wrapped_parse,
        ",--------------------------,\n"
        "|    RATLAB® Stuart Inc.   |\n"
        "'--------------------------'\n"
        "  lits(field_cls) : changes literal types\n"
        "  [1,2][3,4] : matrix\n"
        "  lst[1,2,3] : list\n"
        "  mhelp : print matrix methods\n"
    )
