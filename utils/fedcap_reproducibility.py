"""Opt-in deterministic execution and RNG-neutral experiment fingerprints.

These controls do not change the algorithm, data budget, or training schedule.
Hashes diagnose matched runs; they do not promise equality across hardware or
PyTorch releases. No test examples are used to select parameters.
"""
import hashlib
import json
import os
from pathlib import Path
import platform
import random

import numpy as np
import sklearn
import torch
import torchvision


POLICY = "strict_runtime_and_run_fingerprints_v1"
SOURCE_FILES = (
    "methods/FedCAP.py",
    "utils/fedcap_experiment.py", "utils/fedcap_reproducibility.py", "utils/fedcap_profile_ablation.py", "utils/fedcap_numerics.py",
    "utils/gen_dataset.py", "utils/drift.py", "utils/models.py", "utils/cotton_dataset.py",
    "utils/metric.py", "entities/base.py", "entities/FedCCFA.py", "entities/FedDrift.py",
    "entities/FedCAP.py", "entities/FedCAPCore.py", "entities/FedCAPStability.py",
    "entities/FedCAPCotton.py", "utils/cotton_collaboration.py",
    "entities/FedCAPMergeGuard.py", "entities/FedCAPProfileShare.py", "entities/FedCAPProfileMerge.py", "entities/FedCAPProfile.py",
    "entities/FedCAPRowTaskGate.py",
    "entities/FedCAPHeadCentering.py",
    "entities/FedCAPClassSupport.py",
    "entities/FedCAPProfileConsensus.py",
    "utils/fedcap_mechanism.py",
    "utils/fedcap_stages.py",
    "utils/fedcap_body_proximal.py",
    "utils/fedcap_head_calibration.py",
    "utils/fedcap_output.py",
    "utils/fedcap_dataset_reports.py",
    "utils/cotton_predictions.py",
    "utils/cotton_paths.py",
    "entities/FedCAPOracleClassShare.py",
)


def configure_determinism(enabled, device):
    """False preserves current backend settings; True is strict, never warn-only."""
    if not isinstance(enabled, bool):
        raise ValueError("deterministic must be a boolean")
    if not enabled:
        return
    if torch.device(device).type == "cuda":
        workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        if workspace not in (None, ":4096:8", ":16:8"):
            raise ValueError("Strict CUDA execution requires CUBLAS_WORKSPACE_CONFIG=:4096:8 or :16:8")
        if workspace is None:
            if torch.cuda.is_initialized():
                raise ValueError("CUDA is already initialized; set CUBLAS_WORKSPACE_CONFIG=:4096:8 before launching Python")
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=False)


def runtime_fingerprint(device):
    gpu = None
    if torch.device(device).type == "cuda":
        props = torch.cuda.get_device_properties(torch.device(device))
        gpu = dict(name=props.name, capability=[props.major, props.minor], total_memory=props.total_memory)
    return dict(
        python=platform.python_version(), platform=platform.platform(), torch=str(torch.__version__),
        torchvision=str(torchvision.__version__), numpy=np.__version__, sklearn=sklearn.__version__,
        cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(), device=device, gpu=gpu,
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        deterministic_warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
        cudnn_deterministic=torch.backends.cudnn.deterministic,
        cudnn_benchmark=torch.backends.cudnn.benchmark,
        cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
        matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        float32_matmul_precision=torch.get_float32_matmul_precision(),
        cpu_threads=torch.get_num_threads(), cpu_interop_threads=torch.get_num_interop_threads(),
        environment={key: os.environ.get(key) for key in (
            "CUBLAS_WORKSPACE_CONFIG", "TORCH_CUDNN_V8_API_DISABLED", "CUDA_VISIBLE_DEVICES",
            "OMP_NUM_THREADS", "MKL_NUM_THREADS", "PYTHONHASHSEED")})


def source_fingerprint(root, files=SOURCE_FILES):
    digests = {name: hashlib.sha256((Path(root) / name).read_bytes()).hexdigest() for name in sorted(files)}
    combined = hashlib.sha256(json.dumps(digests, sort_keys=True).encode()).hexdigest()
    return dict(sha256=combined, files=digests)


def _update_array(digest, value):
    if torch.is_tensor(value):
        value = value.detach().cpu().contiguous().numpy()
    array = np.ascontiguousarray(value)
    if array.dtype.hasobject:
        raise ValueError("Cannot fingerprint an object array reliably")
    digest.update(str(array.dtype).encode())
    digest.update(json.dumps(list(array.shape)).encode())
    if array.size:
        digest.update(memoryview(array).cast("B"))


def training_fingerprint(train):
    """Hash ordered raw client inputs/labels without calling stochastic transforms."""
    digest, inputs_complete = hashlib.sha256(), True
    identities_complete = True
    counts = {}
    for identity, dataset in sorted(train.items()):
        counts[str(identity)] = len(dataset)
        digest.update(json.dumps([identity, len(dataset)]).encode())
        _update_array(digest, np.asarray(dataset.targets, dtype=np.int64))
        for name in ("gene_ids", "family_ids"):
            if not hasattr(dataset, name):
                identities_complete = False
                continue
            identities = np.asarray(getattr(dataset, name))
            if identities.ndim != 1 or len(identities) != len(dataset):
                raise ValueError(f"Invalid ordered {name} in client {identity}")
            digest.update(name.encode())
            digest.update(json.dumps([str(v) for v in identities], ensure_ascii=False).encode("utf-8"))
        if hasattr(dataset, "data"):
            _update_array(digest, dataset.data)
        else:
            inputs_complete = False
            digest.update(b"inputs-unavailable")
    return dict(sha256=digest.hexdigest(), inputs_complete=inputs_complete,
                identities_complete=identities_complete, policy="ordered_inputs_labels_gene_family_ids_v2",
                clients=len(train), samples=sum(counts.values()), client_sample_counts=counts)


def state_fingerprint(state):
    digest = hashlib.sha256()
    for key in sorted(state):
        digest.update(key.encode())
        _update_array(digest, state[key])
    return digest.hexdigest()


def whole_head_fingerprint(server):
    state = {"body." + k: v for k, v in server.body_state.items()}
    for identity, head in sorted(server.states.items()):
        state.update({f"cluster.{identity}.{k}": v for k, v in head.items()})
    for identity, head in sorted(server.retained_heads.items()):
        state.update({f"retained.{identity}.{k}": v for k, v in head.items()})
    digest = hashlib.sha256(state_fingerprint(state).encode())
    digest.update(json.dumps(server.membership, sort_keys=True).encode())
    return digest.hexdigest()


def rng_fingerprint(device):
    """Observe, never advance, the RNGs used by sampling, augmentation and SGD."""
    digest = hashlib.sha256(repr(random.getstate()).encode())
    numpy_state = np.random.get_state()
    digest.update(numpy_state[0].encode())
    _update_array(digest, numpy_state[1])
    digest.update(repr(numpy_state[2:]).encode())
    _update_array(digest, torch.get_rng_state())
    if torch.device(device).type == "cuda":
        _update_array(digest, torch.cuda.get_rng_state(torch.device(device)))
    return digest.hexdigest()
