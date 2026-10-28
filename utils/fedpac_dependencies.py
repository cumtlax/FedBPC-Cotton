"""Probe optional native solvers out of process, before reading training data."""
from functools import lru_cache
import json
import os
import subprocess
import sys


@lru_cache(maxsize=1)
def check_fedpac_runtime():
    # Import failures in OSQP's binary extension can abort the interpreter;
    # try/except around `import cvxpy` cannot catch those faults.
    code = """
import json, sys, importlib.metadata
import cvxpy as cvx
import numpy as np
import torch
x = cvx.Variable(2)
p = cvx.Problem(cvx.Minimize(cvx.sum_squares(x)), [cvx.sum(x) == 1, x >= 0])
p.solve()
assert p.status in (cvx.OPTIMAL, cvx.OPTIMAL_INACCURATE)
assert x.value is not None and np.allclose(x.value, [0.5,0.5], atol=1e-4)
versions = {n: importlib.metadata.version(n) for n in ('cvxpy','numpy','torch')}
print('FEDPAC_RUNTIME=' + json.dumps(dict(python=sys.version, versions=versions,
                                         solver=p.solver_stats.solver_name)))
"""
    try:
        result = subprocess.run([sys.executable, "-X", "faulthandler", "-c", code],
                                capture_output=True, encoding="utf-8", errors="replace", timeout=30,
                                env=dict(os.environ, PYTHONIOENCODING="utf-8"))
    except subprocess.TimeoutExpired as error:
        raise RuntimeError("FedPAC solver preflight timed out; check the CVXPY/native-solver environment") from error
    if result.returncode:
        detail = (result.stderr or result.stdout)[-2500:]
        raise RuntimeError(
            f"FedPAC solver preflight failed (exit {result.returncode}). "
            "CVXPY/OSQP could not import and solve a tiny QP in this Python environment. "
            "Use a clean compatible environment and rerun the preflight; no algorithm fallback was applied.\n"
            + detail)
    lines = [s[len("FEDPAC_RUNTIME="):] for s in result.stdout.splitlines()
             if s.startswith("FEDPAC_RUNTIME=")]
    if len(lines) != 1:
        raise RuntimeError("FedPAC solver preflight returned no valid runtime record")
    return json.loads(lines[0])
