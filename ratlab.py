
# Hack src/ into the path so we can import.
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


from brutil import cli
cli.cli(globals())
