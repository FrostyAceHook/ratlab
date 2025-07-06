import ast as _ast
import codeop as _codeop
import contextlib as _contextlib
import io as _io
import itertools as _itertools
import linecache as _linecache
import os as _os
import re as _re
import sys as _sys
import threading as _threading
import uuid as _uuid
import warnings as _warnings
from importlib import abc as _importlib_abc
from importlib import import_module as _importlib_import
from importlib.util import spec_from_file_location as _importlib_spec
from pathlib import Path as _Path

from . import cons as _cons
from .util import objtname as _objtname



# Expensive initialisation (mostly importing) is done in the background. The only
# reason this is done is to provide a faster start-up when using the cli, since
# it is the only thing that benefits from threading (it calls `input()` which
# blocks). The background loader also has the role of specifying the initial
# objects in the space, via injects. No code will be executed until every
# "required" task is complete, but the other tasks are not waited on.

class _Failed:
    def __init__(self, msg):
        self._msg = msg
        self._name = None
        self._start_tossing = True
    def _txt(self):
        name = ""
        if self._name is not None:
            name = f" {repr(self._name)}"
        return f"never loaded{name}, because: {self._msg}"
    def _throw(self):
        raise RuntimeError(self._txt())
    def __repr__(self):
        return f"<{self._txt()}>"
    def __getattr__(self, name):
        if not isinstance(name, str):
            raise TypeError(f"expected string, got {_objtname(name)}")
        toss = object.__getattribute__(self, "_start_tossing")
        if toss:
            self._throw()
        raise AttributeError(f"{repr(type(self).__name__)} object has no "
                f"attribute {repr(name)}")

class _Task:
    BACKGROUND = 0
    REQUIRED = 1
    INJECTED = 2
    STAR_INJECTED = 3
    _LAST = 3

    def __init__(self, name, kind, func):
        if not isinstance(name, str):
            raise TypeError(f"expected string for 'name', got {_objtname(name)}")
        if not isinstance(kind, int):
            raise TypeError(f"expected integer for 'kind', got "
                    f"{_objtname(kind)}")
        if kind < 0 or kind > self._LAST:
            raise ValueError(f"expected a valid kind for 'kind', got: {kind}")
        if not callable(func):
            raise TypeError("expected callable for 'func', got "
                    f"{_objtname(func)}")
        self.name = name
        self.kind = kind
        self.func = func
        self.loaded = _threading.Event()
        self._ret = None

    @property
    def ret(self):
        if not self.loaded.is_set():
            self.loaded.wait()
        return self._ret

    def load(self):
        ret = self.func()
        if isinstance(ret, _Failed):
            ret._name = self.name
        self._ret = ret
        self.loaded.set()

class _Loader:
    def __init__(self, *tasks):
        for task in tasks:
            if not isinstance(task, _Task):
                raise TypeError("expected Task for each task, got "
                        f"{_objtname(task)}")
        if len({task.name for task in tasks}) != len(tasks):
            raise ValueError("got duplicate task names")
        self._tasks = {task.name: task for task in tasks}
        self._started = False
        self._finished_req = _threading.Event()
        self._finished = _threading.Event()

    def start(self):
        if self._started:
            return
        self._started = True
        def target():
            last_req = -1
            for i, task in enumerate(self._tasks.values()):
                if task.kind >= _Task.REQUIRED:
                    last_req = i
            if last_req == -1:
                self._finished_req.set()
            for i, task in enumerate(self._tasks.values()):
                task.load()
                if i == last_req:
                    self._finished_req.set()
            self._finished.set()
        _threading.Thread(target=target, daemon=True).start()

    def finish(self, nonreq=False):
        # justin caseme.
        self.start()
        event = self._finished if nonreq else self._finished_req
        if not event.is_set():
            event.wait()

    @property
    def injects(self):
        self.finish()
        things = []
        for task in self._tasks.values():
            if task.kind < _Task.INJECTED:
                continue
            thing = (task.name, task.kind == _Task.STAR_INJECTED, task.ret)
            things.append(thing)
        return things

    def __getitem__(self, name):
        if not isinstance(name, str):
            raise TypeError(f"expected string, got {_objtname(name)}")
        if name not in self._tasks:
            raise AttributeError(f"name {repr(name)} is not recognised")
        # If it hasn't been started yet, just start it now instead of throwing
        # or something.
        self.start()
        obj = self._tasks[name].ret
        if isinstance(obj, _Failed):
            obj._throw()
        return obj


def _task_import(path):
    def wrapped():
        try:
            return _importlib_import(path, package=__package__)
        except ImportError as e:
            e = str(e)
            if e[:1].isupper():
                e = e[:1].lower() + e[1:]
            return _Failed(f"failed to import module {repr(path)} ({e})")
    return wrapped

_bg = _Loader(
    _Task("math", _Task.INJECTED, _task_import("math")),
    _Task("np", _Task.INJECTED, _task_import("numpy")),
    _Task("plotting", _Task.STAR_INJECTED, _task_import(".plotting")),
    _Task("matrix", _Task.STAR_INJECTED, _task_import(".matrix")),
    _Task("sympy", _Task.BACKGROUND, _task_import("sympy")),
    _Task("pandas", _Task.BACKGROUND, _task_import("pandas")),
    _Task("scipy", _Task.BACKGROUND, _task_import("scipy")),
)
_bg.__doc__ = """
Holds the backgrounds tasks and the objects they return.
"""




# Functions exposed so that the transformed source can call them:

def _EXPOSED_literal(x):
    field = _bg["matrix"]._get_field(None)
    x, = _bg["matrix"].castall([x], field=field)
    return x

def _EXPOSED_hstack(*elements):
    return _bg["matrix"].hstack(*elements)

def _EXPOSED_vstack(matrix_literal, *sliced_by):
    append_me = _bg["matrix"].hstack(*sliced_by)
    return _bg["matrix"].vstack(matrix_literal, append_me)

def _EXPOSED_print(value, *assigns):
    # If no assigns, always print if non-none.
    if not assigns:
        if value is not None:
            print(repr(value))
        return
    # Otherwise its an assign, and we only print on matrix assign.
    if not isinstance(value, _bg["matrix"].Matrix):
        return
    txts = [y for x in assigns for y in [x, " = "]]
    cols = [208, 161] * len(assigns)
    pad = " " * sum(len(x) for x in txts)
    mat = repr(value).replace("\n", "\n" + pad)
    print(_cons.coloured(cols, txts) + mat)

def _EXPOSED_clear():
    """
    clear the screen
    """
    _os.system("cls")

class _History:
    def __init__(self):
        self.fnames = []
    def add(self, fname):
        self.fnames.append(fname)
    def clear(self):
        self.fnames.clear()
    def __getitem__(self, i):
        if not isinstance(i, slice):
            raise TypeError(f"expected slice, got {_objtname(i)}")
        lines = []
        for fname in self.fnames[i]:
            lines += _linecache.cache[fname][2]
        return "".join(lines).rstrip()
HISTORY = _History()
def _EXPOSED_history():
    """
    print this console's past inputs
    """
    print(HISTORY[:])

class _ExitConsoleException(Exception):
    pass
def _EXPOSED_quit():
    """
    exit this console
    """
    raise _ExitConsoleException()

KW_LIST = "lst"
KW_MATRIX = "mat"
KW_PREV = "ans"

COMMANDS = {"clear": _EXPOSED_clear, "history": _EXPOSED_history,
        "quit": _EXPOSED_quit}

_EXPOSED = {k: v for k, v in globals().items() if k.startswith("_EXPOSED")}
_DESOPXE = {v: k for k, v in _EXPOSED.items()} # reversed exposed.

KEYWORDS = {KW_LIST, KW_MATRIX, KW_PREV}
KEYWORDS |= set(COMMANDS.keys())
KEYWORDS |= set(_EXPOSED.keys())



def _add_globals(space):
    # Put all the injects.
    for name, star, value in _bg.injects:
        space[name] = value
        if not star:
            continue
        for subname, subvalue in vars(value).items():
            if subname.startswith("_"):
                continue
            space[subname] = subvalue

    # Set an initial field (modifying the `space` and not our globals).
    initial_field = _bg["matrix"].Complex
    _bg["matrix"].lits(initial_field, inject=True, space=space)

    # Splice every "exposed" object into the space.
    for name, value in _EXPOSED.items():
        if name in space:
            raise RuntimeError(f"? exposed {repr(name)} already set")
        space[name] = value



class _Transformer(_ast.NodeTransformer):
    def ast_call(self, func, *args):
        assert func in _DESOPXE
        return _ast.Call(
            func=_ast.Name(id=_DESOPXE[func], ctx=_ast.Load()),
            args=list(args),
            keywords=[],
        )
    def copyloc(self, dst, src):
        _ast.copy_location(dst, src)
        _ast.fix_missing_locations(dst)
        return dst
    def emptyloc(self, dst):
        _ast.fix_missing_locations(dst)
        return dst
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

    def is_load(self, node):
        return isinstance(node.ctx, _ast.Load)
    def is_named(self, node, names):
        if isinstance(names, str):
            names = {names}
        if not isinstance(node, _ast.Name):
            return False
        return node.id in names
    def is_matlit(self, node):
        if not isinstance(node, _ast.Call):
            return False
        names = {_EXPOSED_hstack.__name__, _EXPOSED_vstack.__name__}
        return self.is_named(node.func, names)


    def __init__(self, source, filename, console, bare_lists=False):
        super().__init__()
        self.source = source
        self.filename = filename
        self.console = console
        self.bare_lists = bare_lists
        self.wrap_lits = False
        self.pierce_tuple = False

    def console_things(self, body):
        if not body:
            return

        # Firstly, add the print of the last expression. Note this can be stopped
        # by appending another semicolon.
        last = body[-1]
        printme = False
        if not self.source.rstrip().endswith(";"):
            # Assigns should contain all variable names which were assigned to.
            # May be empty if its not an assignment.
            assigns = []
            # Do separate print on expr vs assign.
            if isinstance(last, (_ast.Assign, _ast.AugAssign)):
                if isinstance(last, _ast.AugAssign):
                    targets = [last.target]
                else:
                    targets = last.targets
                for x in targets:
                    if isinstance(x, _ast.Name):
                        name = _ast.Constant(value=x.id, ctx=_ast.Load())
                        assigns.append(name)
                        # Only print if we actually have a plain name.
                        printme = True
                    else:
                        name = _ast.Constant(value="...", ctx=_ast.Load())
                        assigns.append(name)
            elif isinstance(last, _ast.Expr):
                # Always print exprs.
                printme = True

            # Create the printing node.
            if printme:
                # Value of thing to print is always just var storing last result.
                prev = _ast.Name(id=KW_PREV, ctx=_ast.Load())
                print_node = self.ast_call(_EXPOSED_print, prev, *assigns)
                print_expr = _ast.Expr(print_node)
                self.emptyloc(print_expr)
                body.append(print_expr)


        # Add the prev result tracking.
        i = 0
        while i < len(body) - printme:
            node = body[i]
            prev = _ast.Name(id=KW_PREV, ctx=_ast.Store())
            if isinstance(node, _ast.Expr):
                # Make the expression an assign.
                new_node = _ast.Assign(targets=[prev], value=node.value)
                self.copyloc(new_node, node)
                body[i] = new_node
                i += 1
                continue
            if isinstance(node, _ast.Assign):
                # Add prev to the targets.
                self.copyloc(prev, body[i])
                body[i].targets.append(prev)
                i += 1
                continue
            if isinstance(node, _ast.AugAssign):
                # Add an assign to prev.
                new_node = _ast.Assign(targets=[prev], value=node.target)
                self.copyloc(new_node, node)
                body.insert(i + 1, new_node)
                i += 2
                continue
            # Otherwise, just set it to None.
            none = _ast.Constant(value=None, ctx=_ast.Load())
            new_node = _ast.Assign(targets=[prev], value=none)
            self.emptyloc(new_node)
            body.insert(i + 1, new_node)
            i += 2
            continue


    def cook(self):
        module = _ast.parse(self.source, self.filename)
        module = self.visit(module)
        # Console has additional changes.
        if self.console:
            self.console_things(module.body) # poor things.
            # Also add console inputs in the history.
            HISTORY.add(self.filename)
        return module


    def visit(self, node):
        was_wrapping_lits = self.wrap_lits

        propagate_to = (_ast.BoolOp, _ast.NamedExpr, _ast.BinOp, _ast.UnaryOp,
            _ast.Compare, _ast.IfExp, _ast.Constant,
            _ast.Tuple # also requires `pierce_tuple`.
            # _ast.Attribute, # maybe nice, but probably a little too extreme.
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
            new_node = self.ast_call(_EXPOSED_literal, node)
            return self.copyloc(new_node, node)
        return node

    def visit_Name(self, node):
        # Handle `last result` distinctly.
        if self.is_named(node, KW_PREV):
            if not self.console:
                self.syntaxerrorme("cannot reference Ratlab last-result keyword "
                        f"{repr(KW_PREV)} from outside console", node)
            if not self.is_load(node):
                self.syntaxerrorme(f"cannot directly modify {repr(KW_PREV)} (it "
                        "will automatically track the last result)", node)
        # Ensure keywords aren't modified.
        if not self.is_load(node):
            for kw in KEYWORDS:
                if self.is_named(node, kw):
                    self.syntaxerrorme("cannot modify Ratlab keyword "
                            f"{repr(kw)}", node)
        # Handle commands.
        for kw in COMMANDS:
            if self.is_named(node, kw):
                # Only available in console.
                if not self.console:
                    self.syntaxerrorme("cannot reference Ratlab command "
                            f"{repr(kw)} from outside console", node)
                # Make it the correct name.
                node.id = _DESOPXE[COMMANDS[kw]]
        # Nothing more to visit.
        return node

    def visit_Attribute(self, node):
        # No keywords have attributes.
        # TODO: kwprev kinda does have attributes.
        for kw in KEYWORDS:
            if self.is_named(node.value, kw):
                self.syntaxerrorme("cannot access attributes of Ratlab keyword "
                        f"{repr(kw)}", node)
        # Now do visiting.
        return self.generic_visit(node)

    def visit_Expr(self, node):
        # Visit first.
        node = self.generic_visit(node)
        # If the entire expr is an uncalled command, make it called.
        cmds = {_DESOPXE[func] for func in COMMANDS.values()}
        if self.is_named(node.value, cmds):
            new_node = _ast.Call(func=node.value, args=[], keywords=[])
            new_node = _ast.Expr(value=new_node)
            node = self.copyloc(new_node, node)
        return node

    def visit_List(self, node):
        # Only transform loads into matrices, leave set and del to keep syntax
        # like `[a,b] = 1,2` and `del [a,b]`.
        if not self.is_load(node):
            return self.generic_visit(node)

        # If we doing bare lists, don't transform to matrix.
        if self.bare_lists:
            return self.generic_visit(node)

        # If it's immediately wrapped in a tuple, unpack it to have the same
        # behaviour as subscripting:
        #  [1,2][(3,4)] == [1 2][3 4]
        #  [(1,2)][(3,4)] == error typically
        if len(node.elts) == 1 and isinstance(node.elts[0], _ast.Tuple):
            tpl = node.elts[0]
            if self.is_load(tpl): # ig check its load?
                node.elts = tpl.elts
        # Transform elements.
        self.wrap_lits = True
        node = self.generic_visit(node)
        # Make the hstack call.
        new_node = self.ast_call(_EXPOSED_hstack, *node.elts)
        return self.copyloc(new_node, node)

    def visit_Subscript(self, node):
        # Handle the thing we subscripting, so we can know if its a matrix
        # literal.
        node.value = self.visit(node.value)

        # Find what we're doing based on what we're subscripting.
        if self.is_named(node.value, KW_LIST):
            what = "list" # create a list literal.
        elif self.is_named(node.value, KW_MATRIX):
            what = "matrix literal" # create a matrix literal.
        elif self.is_matlit(node.value):
            what = "matrix row" # append a row to a matrix literal.
        else:
            what = "normal" # idk normal subscript things.

        # Now can recurse to the new row/index.
        self.wrap_lits = what.startswith("matrix")
        self.pierce_tuple = True
        node.slice = self.visit(node.slice)

        if what == "list":
            # Make it a list, preserving context.
            if isinstance(node.slice, _ast.Tuple):
                elts = node.slice.elts # unpack if tuple.
            else:
                elts = [node.slice]
            for elt in elts:
                elt.ctx = node.ctx
            new_node = _ast.List(elts=elts, ctx=node.ctx)
            return self.copyloc(new_node, node)

        if what.startswith("matrix"):
            # Ensure its a load.
            if isinstance(node.ctx, _ast.Store):
                self.syntaxerrorme("cannot assign to a matrix literal", node)
            if isinstance(node.ctx, _ast.Del):
                self.syntaxerrorme("cannot delete a matrix literal", node)
            # Elements already transformed, just need to get them.
            elts = node.slice
            if isinstance(elts, _ast.Tuple):
                if not self.is_load(elts):
                    self.syntaxerrorme("how have you even done this", elts)
                elts = elts.elts
            else:
                elts = [elts]
            # Check no slice literals.
            for elt in elts:
                if isinstance(elt, _ast.Slice):
                    self.syntaxerrorme("cannot use slices as elements of a "
                            "matrix literal", elt)
            # Make a row or make the literal.
            if what == "matrix literal":
                new_node = self.ast_call(_EXPOSED_hstack, *elts)
            elif what == "matrix row":
                new_node = self.ast_call(_EXPOSED_vstack, node.value, *elts)
            else:
                assert False
            return self.copyloc(new_node, node)

        return node



class _Loader(_importlib_abc.Loader):
    def __init__(self, fullname, path, kwargs):
        # Module name. not used by us but i think we need it.
        self.fullname = fullname
        # Source file path. lowkey used by us.
        self.path = path
        # Forwarded to transformer.
        self.__kwargs = kwargs

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        with open(self.path, "r", encoding="utf-8") as f:
            source = f.read()

        # Add the ratlab globals.
        _add_globals(module.__dict__)

        # Transform and execute.
        transformer = _Transformer(source, self.path, console=False,
                **self.__kwargs)
        parsed = transformer.cook()
        compiled = compile(parsed, filename=self.path, mode="exec")
        exec(compiled, module.__dict__)


class _PathFinder(_importlib_abc.MetaPathFinder):
    def __init__(self, **kwargs):
        # Forwarded to transformer.
        self.__kwargs = kwargs

    def find_spec(self, fullname, path, target=None):
        if path is None:
            searchme = _sys.path
            searchme.insert(0, _os.getcwd())
        else:
            searchme = path

        filename = fullname.rsplit(".")[-1]
        for base in searchme:
            # Check for script files.
            base = _Path(base)
            rat_path = base / f"{filename}.rat"
            if rat_path.is_file():
                loader = _Loader(fullname, rat_path, self.__kwargs)
                return _importlib_spec(fullname, rat_path, loader=loader)

            # Check for script folders/modules.
            package_path = base / filename
            init_path = package_path / "__init__.rat"
            if init_path.is_file():
                loader = _Loader(fullname, init_path, self.__kwargs)
                return _importlib_spec(fullname, init_path, loader=loader,
                        submodule_search_locations=[package_path])

        return None



class Context:
    def __init__(self, bare_lists=False):
        self._space_dict = None
        self._bare_lists = bare_lists
        self._importer_count = 0
        self._finder = _PathFinder(bare_lists=bare_lists)

    @property
    def _space(self):
        if not self._initialised:
            raise RuntimeError("space is not initialised")
        return self._space_dict

    @_contextlib.contextmanager
    def _use_importer(self):
        if not self._importer_count:
            _sys.meta_path.insert(0, self._finder)
        self._importer_count += 1
        try:
            yield
        finally:
            self._importer_count -= 1
            if not self._importer_count:
                _sys.meta_path.remove(self._finder)

    @property
    def _initialised(self):
        return self._space_dict is not None

    def _initialise(self):
        if self._initialised:
            return

        # Wait for background init (but dont wait the non-req tasks).
        _bg.finish(nonreq=False)

        # Make initial space important parameters.
        space = {
            "__name__": "__main__",
            # dont set __file__.
            "__doc__": None,
            "__package__": __package__,
            "__builtins__": __builtins__,
        }
        # Chuck the ratlab globals in.
        _add_globals(space)
        # Done.
        self._space_dict = space


    def _execute(self, source, filename=None):
        # For console if filename is none.
        console = (filename is None)
        if console:
            filename = f"<{_uuid.uuid4()}/console>"
            # Store source for some nice error printing.
            lines = [line + "\n" for line in source.splitlines()]
            _linecache.cache[filename] = (len(source), None, lines, filename)

        try:
            # Transform and compile.
            transformer = _Transformer(source, filename, console,
                    not not self._bare_lists)
            parsed = transformer.cook()
            compiled = compile(parsed, filename=filename, mode="exec")
        except Exception as e:
            print(_cons.pretty_exception(e, "TYPO", tb=False))
            return False
        try:
            # Enable the rat module importer.
            with self._use_importer():
                # Execute, with both locals and globals to simulate module-level
                # (filescope) code.
                exec(compiled, self._space, self._space)
        except Exception as e:
            if isinstance(e, _ExitConsoleException) and console:
                raise
            print(_cons.pretty_exception(e, "ERROR", tb=True))
            return False
        return True


    def run_console(self):
        """
        Starts a command-line interface which is basically just an embedded
        python interpreter.
        """
        def get_input():
            source = ""
            while True:
                print(_cons.coloured(73, ".. " if source else ">> "), end="")

                # Delay the start of background loading (since its a tad slow to
                # start up) as late as possible to reduce loading time.
                _bg.start()

                try:
                    line = input()
                except (EOFError, KeyboardInterrupt):
                    # they ctrl+c-ed my ass.
                    print()
                    raise _ExitConsoleException()

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

        first = True
        try:
            while True:
                source = get_input()

                # Delay space init until after first input to reduce loading
                # times.
                if first:
                    self._initialise()
                    # `last result` always starts with a value of none.
                    self._space[KW_PREV] = None
                    # History always starts cleared.
                    HISTORY.clear()

                    first = False

                self._execute(source)

        except _ExitConsoleException:
            print(_cons.coloured([73, 80], ["-- ", "okie leaving."]))



    def run_file(self, path, has_next):
        """
        Executes the file at the given path.
        """
        path = _Path(path)

        # Start bg loading asap (nothing to wait for, unlike console).
        _bg.start()

        # Initialise space asap (again, unlike console).
        self._initialise()

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
            txts = ["ratlab: ", before, after, ", ignore?", " (y/n): "]
            cols = [       124,    203,   203,         203,        210]
            pcols, ptxts = coloured_esc(path)
            txts[2:2] = ptxts
            cols[2:2] = pcols
            msg = _cons.coloured(cols, txts)
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
            print(_cons.coloured(cols, txts))
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
        if not self._execute(source, filename=str(path)):
            if not has_next:
                return
            ignore = query("exception in file", path)
            if not ignore:
                error("exception in file", path)
            print(_cons.coloured([73, 80], ["-- ", "okie continuing."]))
