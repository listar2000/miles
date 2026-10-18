"""Run a driver script as __main__, then hard-exit: a finished split primary can hang in object-store teardown.

    python -m miles_ext.disagg.run_exit train_async.py <args>
    python -m miles_ext.disagg.run_exit -m miles_ext.disagg.ondemand <args>
"""

import os
import runpy
import sys
import traceback

if __name__ == "__main__":
    module = sys.argv[1] == "-m"
    target = sys.argv[2] if module else sys.argv[1]
    sys.argv = [target, *sys.argv[3 if module else 2 :]]
    code = 0
    try:
        (
            runpy.run_module(target, run_name="__main__", alter_sys=True)
            if module
            else runpy.run_path(target, run_name="__main__")
        )
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else int(e.code is not None)
    except BaseException:
        traceback.print_exc()
        code = 1
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
