# -*- coding: utf-8 -*-
"""FedCAP-Cotton one-command launcher.

Usage:
  bash scripts/run_final.sh METHOD [FIRST_SEED=1] [REPEATS=1] [FEATURE_SET=no-expression]

Examples:
  bash scripts/run_final.sh LocalOnly 1 1
  bash scripts/run_final.sh FedAvg 1 3
  bash scripts/run_final.sh FedCCFA 1 3
  bash scripts/run_final.sh FedCAP 2 1

Environment overrides are also accepted as flags:
  --device cpu --rounds 1 --clients 9 --sample-ratio 1.0 --test-eval true
  --output-root ... --data-root ... --dry-run

FedCAP-only ablation flags:
  --fedcap-config configs/FedCAP_CottonMain_NoSupport.yaml
  --fedcap-arg '--max-clusters 1' --fedcap-arg '--cotton-grouping source'
"""
import argparse
import os
import shlex
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.cotton_launch import (FEATURE_SETS, command_text, cotton_clients, participation_ratio,
                                 positive_int, run_logged, validate_data_root, validate_schedule)
BASELINES = (
    "FedAvg", "FedProx", "SCAFFOLD", "FedFM", "AdaptiveFedAvg", "Flash",
    "pFedMe", "Ditto", "FedRep", "FedBABU", "FedPAC", "IFCA",
    "LocalOnly", "Centralized", "FedAvgWeighted",
)
SUPPORTED = ("FedCAP", "FedCCFA", "FedDrift", *BASELINES)


def build_parser():
    p = argparse.ArgumentParser(description="FedCAP-Cotton fixed-protocol launcher")
    p.add_argument("method", choices=SUPPORTED)
    p.add_argument("first_seed", nargs="?", type=int, default=1)
    p.add_argument("repeats", nargs="?", type=int, default=1)
    p.add_argument("feature_set", nargs="?",
                   choices=FEATURE_SETS, default=os.environ.get("COTTON_FEATURE_SET", "no-expression"))
    p.add_argument("--device", default=os.environ.get("COTTON_DEVICE", "cuda"),
                   choices=("cpu", "cuda"))
    p.add_argument("--rounds", type=positive_int, default=os.environ.get("COTTON_ROUNDS", "100"))
    p.add_argument("--clients", type=cotton_clients, default=os.environ.get("COTTON_CLIENTS", "90"))
    p.add_argument("--sample-ratio", type=participation_ratio,
                   default=os.environ.get("COTTON_SAMPLE_RATIO", "0.2"))
    p.add_argument("--test-eval", default=os.environ.get("COTTON_TEST_EVAL", "false"),
                   choices=("true", "false"))
    p.add_argument("--fedcap-config",
                   help="FedCAP only: config path (default: configs/FedCAP_CottonMain.yaml)")
    p.add_argument("--fedcap-arg", action="append", default=None,
                   help="FedCAP only: extra argument string, repeatable; e.g. --fedcap-arg '--max-clusters 1'")
    p.add_argument("--data-root", default=os.environ.get("COTTON_ROOT", str(ROOT / "data" / "cotton")))
    p.add_argument("--output-root", help="Default: CottonExperiments/validation or final-test, matching --test-eval")
    p.add_argument("--dry-run", action="store_true")
    return p


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not (args.first_seed in (1, 2, 3) and args.repeats in (1, 2, 3)
            and 1 <= args.first_seed + args.repeats - 1 <= 3):
        raise SystemExit("Select a contiguous subset of seeds 1/2/3")
    try:
        if args.feature_set not in FEATURE_SETS:
            raise ValueError("Unsupported feature set")
        if args.method != "FedCAP" and (args.fedcap_config is not None or args.fedcap_arg):
            raise ValueError("--fedcap-config/--fedcap-arg are only supported by FedCAP")
        if args.fedcap_config is not None and not (ROOT / args.fedcap_config).is_file():
            raise ValueError("FedCAP config not found: " + args.fedcap_config)
        validate_schedule(args.clients, args.sample_ratio,
                          list(range(args.first_seed, args.first_seed + args.repeats)))
        data_root = validate_data_root(args.data_root)
    except ValueError as error:
        parser.error(str(error))
    phase = "final-test" if args.test_eval == "true" else "validation"
    output_root = (Path(args.output_root).expanduser().resolve() if args.output_root
                   else ROOT / "CottonExperiments" / phase)

    base = [sys.executable]
    if args.method == "FedCAP":
        fedcap_config = args.fedcap_config or "configs/FedCAP_CottonMain.yaml"
        base += ["methods/FedCAP.py", "--config", fedcap_config]
    elif args.method == "FedCCFA":
        base += ["methods/FedCAP.py", "--algorithm", "FedCCFA", "--dataset", "cotton30",
                 "--config", "configs/FedCCFA_CottonPaper.yaml"]
    elif args.method == "FedDrift":
        base += ["methods/FedCAP.py", "--algorithm", "FedDrift", "--dataset", "cotton30",
                 "--config", "configs/FedDrift_Cotton90.yaml"]
    else:
        config = ("configs/FedAvgWeighted_Cotton.yaml" if args.method == "FedAvgWeighted"
                  else "configs/CottonPaper_Baselines.yaml")
        base += ["methods/CottonBaselines.py", "--algorithm", args.method,
                 "--config", config]

    base += [
        "--cotton-root", str(data_root),
        "--cotton-feature-set", args.feature_set,
        "--cotton-test-eval", args.test_eval,
        "--cotton-cross-source-eval", "false",
        "--device", args.device,
        "--rounds", str(args.rounds),
        "--clients", str(args.clients),
        "--sample-ratio", str(args.sample_ratio),
    ]
    if args.method == "FedCAP" and args.fedcap_arg:
        for chunk in args.fedcap_arg:
            base += shlex.split(chunk)

    for i in range(args.repeats):
        seed = args.first_seed + i
        cmd = base + ["--seed", str(seed), "--output-root", str(output_root)]
        line = command_text(cmd)
        print("[cotton-final] method=%s seed=%d device=%s features=%s" % (
            args.method, seed, args.device, args.feature_set), flush=True)
        if args.dry_run:
            print("DRY-RUN", line)
            continue
        code, log_path = run_logged(cmd, cwd=ROOT, output_root=output_root,
                                    method=args.method, seed=seed)
        if code != 0:
            raise SystemExit("run failed (seed %d), see %s" % (seed, log_path))
    print("[cotton-final] done ->", output_root)


if __name__ == "__main__":
    main()
