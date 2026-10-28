"""Cotton-only entry point for the paper's original comparison algorithms."""
import sys
from pathlib import Path
import argparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# A broken native extension may abort Python rather than raise ImportError.
# Configuration/help must remain usable without optional solver dependencies.
early = argparse.ArgumentParser(add_help=False)
early.add_argument("--algorithm")
early.add_argument("--show-config", default="false")
early_options, _ = early.parse_known_args()
if (early_options.algorithm == "FedPAC" and early_options.show_config.lower() != "true"
        and not any(flag in sys.argv for flag in ("--help", "-h"))):
    from utils.fedpac_dependencies import check_fedpac_runtime
    try:
        check_fedpac_runtime()
        # Keep the existing CVXPY-before-Torch loading order after a successful probe.
        import cvxpy
    except (ImportError, RuntimeError) as error:
        raise SystemExit(str(error))

from utils.cotton_baselines import main

if __name__ == "__main__":
    main()
