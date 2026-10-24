"""State guards for the CPU tests.

Only what the tests themselves are responsible for restoring lives here;
keep this module tiny and free of frameworks.
"""

import contextlib
import random


@contextlib.contextmanager
def preserve_python_random_state():
    """Save and restore the global ``random`` module state.

    Restores on normal AND exceptional exit.
    """
    state = random.getstate()
    try:
        yield
    finally:
        random.setstate(state)
