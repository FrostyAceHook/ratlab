import ast as _ast
import codeop as _codeop
import os as _os
import sys as _sys
import traceback as _traceback

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
                _ast.fix_missing_locations(module)
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
def _char_check():
    for i in range(256*4 * 16):
        print("\n"*((i%24) == 0) + " "+chr(i)+" ", end="")
    print()
_cli.command(_char_check, "charcheck")



def literals(field):
    literals.field = field
literals.field = None


def _wrapped_literal(x):
    if literals.field is not None:
        try:
            return literals.field.cast(x)
        except NotImplementedError:
            pass
    return x

class _WrappedTransformer(_ast.NodeTransformer):
    def visit_Constant(self, node):
        return _ast.Call(
            func=_ast.Name(id="_wrapped_literal", ctx=_ast.Load()),
            args=[node],
            keywords=[],
        )

    # def visit_List(self, node):
    #     # Visit children in case they have nested lists.
    #     self.generic_visit(node)

    #     # Wrap the list in a function call: wrap_list([original_list])
    #     return _ast.Call(
    #         func=_ast.Name(id='wrap_list', ctx=_ast.Load()),
    #         args=[node],
    #         keywords=[]
    #     )

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

    _ast.fix_missing_locations(module)
    return module


if __name__ == "__main__":
    _cli.cli(globals(), _wrapped_parse,
        ",--------------------------,\n"
        "|    RATLAB® Stuart Inc.   |\n"
        "'--------------------------'\n"
        "  literals(field_cls) : changes literal types\n"
    )
