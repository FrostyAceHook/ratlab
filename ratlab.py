import ast
import traceback

# Hack src/ into the path so we can import.
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from src import *


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
        last_len = 0
        def log_raw(line, ending):
            nonlocal last_len

            if not isinstance(line, str):
                line = str(line)

            # Break multi-line logs into just one.
            if "\n" in line:
                # Handle everything but the last.
                for l in line.split("\n")[:-1]:
                    log_raw(l, "\n")
                # Now do the last one with the expected ending.
                line = line.rsplit("\n", 1)[1]

            this_len = len(line)
            if last_len > this_len:
                line += " "*(last_len - this_len)

            if ending == "\r":
                last_len = this_len
            else:
                last_len = 0

            print(line + ending, end="")

        def log_tmp(line):
            log_raw(line, "\r")
        def log(line):
            log_raw(line, "\n")

        sent_input = False
        def get_input():
            nonlocal sent_input
            if self._queue:
                code = self._queue.pop(0).lstrip()
                lines = code.split("\n")
                log(f">> {lines[0]}")
                for line in lines[1:]:
                    log(f".. {line}")
                return code

            lines = []
            while True:
                try:
                    log_tmp("")
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
            log(msg)

        while not self._leave:
            code = get_input()
            if not code:
                continue
            try:
                if code in {"quit", "exit", "return"}:
                    self._leave = True
                elif code == "cls":
                    _os.system("cls")
                else:
                    tree = code_to_tree(code)
                    compiled = compile(tree, "<ast>", "exec")
                    exec(compiled, glbls, glbls)
            except Exception as ex:
                sent_input = False
                log("## ERROR:")
                log(traceback.format_exc())

            # Only allow one input if requested.
            if once and sent_input:
                break
            if self._leave:
                log("-- okie leaving.")

cli = cli()

def _okie_we_leave():
    cli._leave = True
cli.command(_okie_we_leave, "quit")
cli.command(_okie_we_leave, "exit")
cli.command(_okie_we_leave, "return")
cli.command(lambda: _os.system("cls"), "cls")




def _literal_wrapper(literal):
    if _literal_wrapper.field is None:
        return literal
    return _literal_wrapper.field.zero().cast(literal)
_literal_wrapper.field = None

def set_field(field):
    _literal_wrapper.field = field

class LiteralsWrapped(ast.NodeTransformer):
    def visit_Constant(self, node):
        if isinstance(node.value, (int, float, complex)):
            return ast.Call(
                func=ast.Name(id="_literal_wrapper", ctx=ast.Load()),
                args=[node],
                keywords=[],
            )
        return node

    def visit_Num(self, node): # support python <3.8.
        return ast.Call(
            func=ast.Name(id="_literal_wrapper", ctx=ast.Load()),
            args=[node],
            keywords=[],
        )

def code_to_tree(code):
    try:
        # Try parsing in eval mode first to see if it's a single expression.
        expr_tree = ast.parse(code, mode="eval")
        expr_tree = LiteralsWrapped().visit(expr_tree)
        print_expr = ast.Expr(
            value=ast.Call(
                func=ast.Name(id='print', ctx=ast.Load()),
                args=[
                    ast.Call(
                        func=ast.Name(id='repr', ctx=ast.Load()),
                        args=[expr_tree.body],
                        keywords=[]
                    )
                ],
                keywords=[]
            )
        )
        module = ast.Module(body=[print_expr], type_ignores=[])
        ast.fix_missing_locations(module)
        return module
    except SyntaxError:
        # Otherwise treat as a statement.
        tree = ast.parse(code, mode="exec")
        tree = LiteralsWrapped().visit(tree)
        ast.fix_missing_locations(tree)
        return tree


if __name__ == "__main__":
    cli.cli(globals(), code_to_tree,
        ",--------------------------,\n"
        "|    RATLAB® Stuart Inc.   |\n"
        "'--------------------------'\n"
        "  multiline code : end line with a backslash\n"
        "  set_field(field_cls) : changes number literal types\n"
    )
