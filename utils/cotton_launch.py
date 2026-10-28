"""Standard-library-only launch validation and exclusive UTF-8 console logs."""
import argparse
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys

FEATURE_SETS = ("all23", "no-expression", "sequence-structure",
                "sequence-structure-atac", "sequence-structure-chip")
DATA_FILES = ("clients/c1_cotton_nitrogen/train_ready.tsv",
              "clients/c2_cotton_mutant/train_ready.tsv",
              "clients/c3_cotton_salt/train_ready.tsv", "metadata/gene_families.tsv")


def positive_int(value):
    try:
        result = int(value)
    except (ValueError, TypeError):
        raise argparse.ArgumentTypeError("Expected a positive integer")
    if result < 1:
        raise argparse.ArgumentTypeError("Expected a positive integer")
    return result


def cotton_clients(value):
    result = positive_int(value)
    if result < 3 or result % 3:
        raise argparse.ArgumentTypeError("Cotton clients must be a positive multiple of three")
    return result


def participation_ratio(value):
    try:
        result = float(value)
    except (ValueError, TypeError):
        raise argparse.ArgumentTypeError("Expected a finite sample ratio in (0, 1]")
    if not math.isfinite(result) or not 0 < result <= 1:
        raise argparse.ArgumentTypeError("Expected a finite sample ratio in (0, 1]")
    return result


def validate_schedule(clients, ratio, seeds, experiments=()):
    if int(clients * ratio) < 1:
        raise ValueError("sample-ratio selects zero clients")
    if not seeds or any(seed < 0 for seed in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("Seeds must be nonnegative and unique")
    if len(set(experiments)) != len(experiments):
        raise ValueError("Experiments must be unique")


def validate_data_root(path):
    root = Path(path).expanduser().resolve()
    missing = [name for name in DATA_FILES if not (root / name).is_file()]
    if missing:
        raise ValueError(f"Cotton data missing under {root}: {missing}")
    return root


def command_text(command):
    return subprocess.list2cmdline(command) if os.name == "nt" else shlex.join(command)


def run_logged(command, *, cwd, output_root, method, seed):
    """Never truncate an older run; terminate a child if forwarding is interrupted."""
    console = Path(output_root) / "console"
    console.mkdir(parents=True, exist_ok=True)
    index = 1
    while True:
        log_path = console / f"{method}-s{seed}-{index:03d}.log"
        try:
            log = log_path.open("x", encoding="utf-8")
            break
        except FileExistsError:
            index += 1
    with log:
        log.write("[command] " + command_text(command) + "\n")
        log.flush()
        env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
        with subprocess.Popen(command, cwd=cwd, env=env, text=True, encoding="utf-8",
                              errors="replace", stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT) as process:
            try:
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    sys.stdout.write(line)
                    sys.stdout.flush()
                code = process.wait()
            except BaseException:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                raise
    return code, log_path
