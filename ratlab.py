import ast as _ast
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


    def cli(self, glbls, code_to_tree, msg="", once=False):
        """
        Starts a command-line interface which is basically just an embedded
        python interpreter. `glbls` should be the `globals()` of the caller,
        `code_to_tree` should take a string and return an ast tree, `msg` is
        displayed upon entering the input loop and if `once` is true it will
        break from the input loop after a single (non-exceptional) input.
        """
        if getattr(self, "_in_cli", False):
            raise TypeError("cannot enter cli while in cli")

        self._leave = False
        self._in_cli = True
        try:
            self._cli_loop(glbls, code_to_tree, msg, once)
        finally:
            self._leave = False
            self._in_cli = False

    def _cli_loop(self, glbls, code_to_tree, msg, once):
        sent_input = False
        def get_input():
            nonlocal sent_input
            if self._queue:
                code = self._queue.pop(0).lstrip()
                lines = code.split("\n")
                print(f">> {lines[0]}")
                for line in lines[1:]:
                    print(f".. {line}")
                return code

            lines = []
            while True:
                try:
                    line = input(">> " if not lines else ".. ")
                except EOFError:
                    break

                if line.lstrip().startswith("#"):
                    # add a blank line if in a multiline block to align the line
                    # numbers.
                    if lines:
                        lines.append("")
                    continue

                escaped = line.endswith('\\')
                line = line[:-1] if escaped else line

                lines.append(line)
                if not escaped:
                    break

            sent_input = True
            return "\n".join(lines).strip()

        # log starting message.
        msg = str(msg)
        if msg:
            print(msg)

        while not self._leave:
            code = get_input()
            if not code:
                continue
            try:
                if code in self._commands:
                    self._commands[code]()
                else:
                    tree = code_to_tree(code)
                    compiled = compile(tree, "<ast>", "exec")
                    exec(compiled, glbls, glbls)
            except Exception:
                sent_input = False
                print("## ERROR:")
                print(_traceback.format_exc())

            # Only allow one input if requested.
            if once and sent_input:
                break
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
    literals.mapping = field.mapping()
literals.mapping = None


def _literal_int(x):
    if literals.mapping is None or int not in literals.mapping:
        return x
    return literals.mapping[int](x)
def _literal_float(x):
    if literals.mapping is None or float not in literals.mapping:
        return x
    return literals.mapping[float](x)
def _literal_complex(x):
    if literals.mapping is None or complex not in literals.mapping:
        return x
    return literals.mapping[complex](x)
def _literal_str(x):
    if literals.mapping is None or str not in literals.mapping:
        return x
    return literals.mapping[str](x)

class _Wrapped(_ast.NodeTransformer):
    def visit_Constant(self, node):
        if not isinstance(node.value, (int, float, complex, str)):
            return node
        func_name = f"_literal_{type(node.value).__name__}"
        return _ast.Call(
            func=_ast.Name(id=func_name, ctx=_ast.Load()),
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

def _print_if_nonnone(x):
    if x is not None:
        print(repr(x))

def _code_to_tree(code):
    # Parse the code.
    module = _ast.parse(code)
    module = _Wrapped().visit(module)

    # If the entire thing is an expression, add a print if non-none.
    if len(module.body) == 1 and isinstance(module.body[0], _ast.Expr):
        # However, this can be circumvented by appending a semicolon.
        if not code.lstrip().endswith(";"):
            expr_node = module.body[0]
            tree = _ast.Expr(
                value=_ast.Call(
                    func=_ast.Name(id="_print_if_nonnone", ctx=_ast.Load()),
                    args=[expr_node.value],
                    keywords=[],
                )
            )
            module = _ast.Module(body=[tree], type_ignores=[])

    _ast.fix_missing_locations(module)
    return module


if __name__ == "__main__":
    _cli.cli(globals(), _code_to_tree,
        ",--------------------------,\n"
        "|    RATLAB® Stuart Inc.   |\n"
        "'--------------------------'\n"
        "  multiline code : end line with a backslash\n"
        "  literals(field_cls) : changes literal types\n"
    )
