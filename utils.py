from contextlib import contextmanager,redirect_stderr,redirect_stdout
from os import devnull
from typing import Dict
import pandas as pd
import numpy as np

# use this to suppress most of the unecessary polychord output, https://stackoverflow.com/questions/2125702/how-to-suppress-console-output-in-python
@contextmanager
def suppress_stdout_stderr():
    """A context manager that redirects stdout and stderr to devnull"""
    with open(devnull, 'w') as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)