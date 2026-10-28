"""Second audit: launch behavior and synthetic smoke coverage for comparison methods."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from test_idea_runtime import TinyDataset, synthetic_loader, config
from utils import cotton_baselines as baselines
from utils import fedcap_experiment as experiment
from utils.cotton_launch import run_logged


@pytest.fixture(scope="module", autouse=True)
def one_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def launch_module(relative):
    spec = importlib.util.spec_from_file_location("audit_" + Path(relative).stem, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_feddrift_actual_initial_cluster_and_clients_match_common_model(tmp_path):
    from entities.FedDrift import FedDriftClient, FedDriftServer
    from entities.FedCAP import FedCAPServer
    args = config("FedDrift_Cotton90.yaml")
    args["tensorboard_dir"] = str(tmp_path / "events")
    torch.manual_seed(37)
    clients = [FedDriftClient(i, args, TinyDataset(i // 2, i), TinyDataset(i // 2, i), i // 2)
               for i in range(6)]
    server = FedDriftServer(args)
    try:
        # Reproduce the original bug: the actual first cluster is an extra random model.
        assert not torch.equal(torch.nn.utils.parameters_to_vector(server.model.parameters()),
                               torch.nn.utils.parameters_to_vector(server.global_models[0].parameters()))
        experiment.synchronize_cotton_initialization(args, clients, server)
        expected = torch.nn.utils.parameters_to_vector(server.model.parameters())
        assert all(torch.equal(expected, torch.nn.utils.parameters_to_vector(c.model.parameters()))
                   for c in clients)
        assert torch.equal(expected, torch.nn.utils.parameters_to_vector(server.global_models[0].parameters()))
        torch.manual_seed(37)
        cap = FedCAPServer(config())
        assert torch.equal(expected, torch.nn.utils.parameters_to_vector(cap.initial.parameters()))
    finally:
        server.writer.close()


def test_common_initialization_can_be_disabled(tmp_path):
    from entities.FedDrift import FedDriftClient, FedDriftServer
    args = config("FedDrift_Cotton90.yaml")
    args.update(cotton_common_initialization=False, tensorboard_dir=str(tmp_path / "events"))
    clients = [FedDriftClient(i, args, TinyDataset(i // 2, i), TinyDataset(i // 2, i), i // 2)
               for i in range(6)]
    server = FedDriftServer(args)
    before = {k: v.clone() for k, v in server.global_models[0].state_dict().items()}
    try:
        experiment.synchronize_cotton_initialization(args, clients, server)
        assert all(torch.equal(v, server.global_models[0].state_dict()[k]) for k, v in before.items())
    finally:
        server.writer.close()


@pytest.mark.parametrize("algorithm", ["FedCCFA", "FedDrift"])
def test_main_comparison_methods_synthetic_end_to_end(algorithm, tmp_path, monkeypatch):
    monkeypatch.chdir(ROOT)
    monkeypatch.setattr(experiment, "distribute_cotton", synthetic_loader)
    argv = ["FedCAP.py", "--algorithm", algorithm, "--clients", "6", "--rounds", "2",
            "--sample-ratio", "1", "--device", "cpu", "--eval-interval", "1",
            "--output-root", str(tmp_path)]
    argv += (["--rep-epochs", "1", "--clf-epochs", "1", "--balanced-epochs", "1"]
             if algorithm == "FedCCFA" else ["--epochs", "1"])
    monkeypatch.setattr(sys, "argv", argv)
    experiment.main()
    summary = json.loads(next(tmp_path.rglob("summary-*.json")).read_text())
    assert summary["status"] == "completed" and not summary["cotton_test_evaluated"]
    provenance = json.loads(next(tmp_path.rglob("provenance-*.json")).read_text())
    assert provenance["initialization_policy"] == "actual_prediction_model_from_server_v2"


@pytest.mark.parametrize("algorithm", ["FedAvg", "FedProx", "SCAFFOLD", "FedFM", "AdaptiveFedAvg",
    "Flash", "pFedMe", "Ditto", "FedRep", "FedBABU", "IFCA", "LocalOnly", "Centralized"])
def test_other_comparison_methods_synthetic_end_to_end(algorithm, tmp_path, monkeypatch):
    monkeypatch.setattr(baselines, "distribute_cotton", synthetic_loader)
    args = baselines.resolve_config(baselines.parser().parse_args([
        "--algorithm", algorithm, "--clients", "6", "--rounds", "2", "--sample-ratio", "1",
        "--eval-interval", "1", "--device", "cpu", "--epochs", "1", "--rep-epochs", "1",
        "--clf-epochs", "1", "--output-root", str(tmp_path)]))
    summary = baselines.run(args)
    assert summary["status"] == "completed" and summary["evaluation_split"] == "validation"
    assert not summary["cotton_test_evaluated"]


def test_fedpac_native_solver_preloaded_in_isolated_process(tmp_path):
    if importlib.util.find_spec("cvxpy") is None:
        pytest.skip("FedPAC requires optional cvxpy installation")
    from utils.fedpac_dependencies import check_fedpac_runtime
    try:
        check_fedpac_runtime()
    except RuntimeError as error:
        pytest.skip("Native solver environment is unusable; preflight reports: " + str(error).splitlines()[0])
    code = f"""
import cvxpy
import sys
sys.path.insert(0, {str(ROOT)!r})
sys.path.insert(0, {str(ROOT / 'tests')!r})
import torch
torch.set_num_threads(1)
from test_idea_runtime import synthetic_loader
from utils import cotton_baselines as b
b.distribute_cotton = synthetic_loader
args = b.resolve_config(b.parser().parse_args([
    '--algorithm','FedPAC','--clients','6','--rounds','2','--sample-ratio','1',
    '--eval-interval','1','--device','cpu','--epochs','1','--rep-epochs','1',
    '--clf-epochs','1','--output-root',{str(tmp_path)!r}]))
result = b.run(args)
assert result['status'] == 'completed' and not result['cotton_test_evaluated']
"""
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True,
                            encoding="utf-8", errors="replace", timeout=60,
                            env=dict(os.environ, PYTHONIOENCODING="utf-8"))
    assert result.returncode == 0, result.stdout + result.stderr


def test_native_solver_fault_is_reported_before_inprocess_import(monkeypatch):
    from utils import fedpac_dependencies as deps
    deps.check_fedpac_runtime.cache_clear()
    failed = subprocess.CompletedProcess([], 3221225477, stdout="", stderr="native access violation")
    monkeypatch.setattr(deps.subprocess, "run", lambda *a, **k: failed)
    with pytest.raises(RuntimeError, match="preflight failed"):
        deps.check_fedpac_runtime()
    deps.check_fedpac_runtime.cache_clear()


def test_fedpac_configuration_does_not_load_optional_solver(tmp_path):
    result = subprocess.run([sys.executable, str(ROOT / "methods/CottonBaselines.py"),
                             "--algorithm", "FedPAC", "--show-config", "true", "--device", "cpu",
                             "--output-root", str(tmp_path / "unused")], cwd=ROOT,
                            capture_output=True, encoding="utf-8", errors="replace", timeout=30)
    assert result.returncode == 0, result.stderr
    assert '"algorithm": "FedPAC"' in result.stdout
    assert not (tmp_path / "unused").exists()


@pytest.mark.parametrize("method", ["FedAvg", "FedAvgWeighted", "FedCAP", "FedCCFA", "FedDrift"])
def test_single_method_dry_run_has_no_output_side_effects(method, tmp_path, capsys):
    launcher = launch_module("tools/run_final.py")
    output = tmp_path / "a directory with spaces"
    launcher.main([method, "1", "3", "--device", "cpu", "--dry-run", "--output-root", str(output)])
    printed = capsys.readouterr().out
    assert printed.count("DRY-RUN") == 3 and "--cotton-test-eval false" in printed
    assert not output.exists()
    if method == "FedAvgWeighted":
        assert "FedAvgWeighted_Cotton.yaml" in printed


@pytest.mark.parametrize("extra", [["--rounds", "0"], ["--clients", "5"],
    ["--clients", "3", "--sample-ratio", "0.2"], ["--sample-ratio", "nan"],
    ["--sample-ratio", "inf"], ["--sample-ratio", "1.1"]])
def test_single_method_rejects_invalid_protocol_before_launch(extra, tmp_path):
    launcher = launch_module("tools/run_final.py")
    with pytest.raises(SystemExit):
        launcher.main(["FedAvg", "--output-root", str(tmp_path / "unused"), *extra, "--dry-run"])
    assert not (tmp_path / "unused").exists()


def test_single_method_output_phase_and_utf8_log_replay(tmp_path, monkeypatch, capsys):
    launcher = launch_module("tools/run_final.py")
    calls = []
    monkeypatch.setattr(launcher, "run_logged", lambda cmd, **kwargs:
                        (calls.append((cmd, kwargs)) or (0, tmp_path / "fake.log")))
    launcher.main(["FedAvg", "--device", "cpu"])
    assert calls[-1][1]["output_root"] == ROOT / "CottonExperiments/validation"
    launcher.main(["FedAvg", "--device", "cpu", "--test-eval", "true"])
    assert calls[-1][1]["output_root"] == ROOT / "CottonExperiments/final-test"
    command = [sys.executable, "-c", "print('Unicode: '+chr(0x4e2d)+chr(0x6587))"]
    code1, path1 = run_logged(command, cwd=ROOT, output_root=tmp_path, method="FedAvg", seed=1)
    saved = path1.read_bytes()
    code2, path2 = run_logged(command, cwd=ROOT, output_root=tmp_path, method="FedAvg", seed=1)
    assert code1 == code2 == 0 and path1 != path2 and path1.read_bytes() == saved
    assert "Unicode: \u4e2d\u6587" in path2.read_text(encoding="utf-8")


def test_child_failure_has_an_independent_diagnostic_log(tmp_path):
    code, path = run_logged([sys.executable, "-c", "import sys; print('failure'); sys.exit(7)"],
                            cwd=ROOT, output_root=tmp_path, method="FedAvg", seed=1)
    assert code == 7 and "failure" in path.read_text(encoding="utf-8")


def test_tensorboard_paths_separate_feature_protocols(tmp_path, monkeypatch):
    monkeypatch.setattr(baselines, "distribute_cotton", synthetic_loader)
    paths = []
    for features in ("no-expression", "all23"):
        args = baselines.resolve_config(baselines.parser().parse_args([
            "--algorithm", "FedAvg", "--clients", "6", "--rounds", "1", "--sample-ratio", "1",
            "--device", "cpu", "--epochs", "1", "--rep-epochs", "1", "--clf-epochs", "1",
            "--cotton-feature-set", features, "--output-root", str(tmp_path)]))
        baselines.run(args)
        paths.append(args["tensorboard_dir"])
    assert paths[0] != paths[1]
