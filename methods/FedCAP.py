"""Run from repository root or methods/: python methods/FedCAP.py --help."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.fedcap_experiment import main

if __name__ == "__main__":
    main()
