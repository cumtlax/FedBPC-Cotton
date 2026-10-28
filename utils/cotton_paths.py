"""Cotton output locations, independent of invocation cwd and image runs."""
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]
COTTON_EXPERIMENTS_ROOT = ROOT / "CottonExperiments"


def has_cli_option(argv, option):
    """Recognize both argparse spellings without overwriting explicit paths."""
    return any(value == option or value.startswith(option + "=") for value in argv)


def default_cotton_data_root():
    value = os.environ.get("COTTON_ROOT") or ROOT / "data/cotton"
    return Path(value).expanduser().absolute().resolve()


def cotton_output_root(value=None, *, from_cli=False):
    """CLI-relative paths use cwd; YAML-relative paths use the checkout.

    Explicit destinations are honored. No historical output is migrated.
    """
    if not value:
        return COTTON_EXPERIMENTS_ROOT.resolve()
    path = Path(value).expanduser()
    if not path.is_absolute() and not from_cli:
        path = ROOT / path
    return path.absolute().resolve()
