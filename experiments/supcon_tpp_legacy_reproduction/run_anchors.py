from __future__ import annotations

"""Fail-closed orchestrator for the immutable legacy anchor bundle.

The training entry points are intentionally treated as opaque historical
programs.  This module only validates their inputs, derives one target/seed
config per process, isolates every process in a new ``attempt_NNN`` directory,
and validates/hash-binds completed output before it can ever be skipped.

No command runs training accidentally: callers must choose exactly one of
``--dry-run`` or ``--execute``.
"""

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import platform
import re
import subprocess
import sys
import traceback
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

# Probes and historical entry points must not dirty an otherwise clean replay
# worktree with bytecode caches.
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.dont_write_bytecode = True

import preflight
import protocol


SCHEMA_VERSION = "legacy-anchor-runner/v1"
COMPLETION_SCHEMA_VERSION = "legacy-anchor-attempt-completion/v1"
SEAL_SCHEMA_VERSION = "legacy-anchor-attempt-seal/v1"
ATTEMPT_RE = re.compile(r"^attempt_(\d{3,})$")
ARM_IDS = (
    "anchor_ce",
    "anchor_supcon",
    "anchor_tpp_l124",
    "anchor_tpp_residual025",
)
STAGE_SEEDS = {
    "seed42_sentinel": (42,),
    "full": (52, 62),
}
TRACKED_LEGACY_CODE = (
    "train_dg.py",
    "dg_dataset.py",
    "dg_models.py",
    "dg_losses.py",
    "adaptive_kmargin.py",
)
REQUIRED_RUN_FILES = (
    "resolved_config.json",
    "metrics.json",
    "best_model.pt",
    "confusion_matrix.npy",
    "confusion_matrix.csv",
    "split_manifest.json",
)


class RunnerFailure(RuntimeError):
    pass


@dataclass(frozen=True)
class ArmSpec:
    config_id: str
    source_config_path: Path
    source_config_file_sha256: str
    source_config_payload_sha256: str
    runner: str
    base_config: Mapping[str, Any]
    arm_output_root: Path


@dataclass(frozen=True)
class Job:
    arm: ArmSpec
    target: str
    seed: int

    @property
    def job_id(self) -> str:
        return f"{self.arm.config_id}__target_{self.target}__seed_{self.seed}"

    @property
    def job_root(self) -> Path:
        return self.arm.arm_output_root / "runs" / self.target / f"seed_{self.seed}"


@dataclass(frozen=True)
class AttemptScan:
    action: str
    next_attempt: int
    successful_attempt: Path | None
    prior_attempts: tuple[Mapping[str, Any], ...]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(payload: Any) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def payload_sha256(payload: Any) -> str:
    return hashlib.sha256(canonical_json(payload)).hexdigest()


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RunnerFailure(f"Cannot read valid JSON from {path}: {exc}") from exc


def write_json_exclusive(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise RunnerFailure(f"Refusing to overwrite existing file: {path}") from exc


def assert_equal(actual: Any, expected: Any, context: str) -> None:
    if actual != expected:
        raise RunnerFailure(f"{context}: expected {expected!r}, got {actual!r}")


def assert_close(actual: float, expected: float, context: str, tolerance: float = 1e-12) -> None:
    if not math.isfinite(actual) or not math.isfinite(expected) or abs(actual - expected) > tolerance:
        raise RunnerFailure(
            f"{context}: expected {expected!r} +/- {tolerance}, got {actual!r}"
        )


def path_is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def run_checked(command: Sequence[str], cwd: Path) -> str:
    try:
        completed = subprocess.run(
            list(command),
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except OSError as exc:
        raise RunnerFailure(f"Failed to launch {command!r} in {cwd}: {exc}") from exc
    if completed.returncode != 0:
        raise RunnerFailure(
            f"Command failed with exit code {completed.returncode}: {command!r}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    return completed.stdout.strip()


def sha256_tree_files(root: Path, excluded: Iterable[str] = ()) -> dict[str, dict[str, Any]]:
    excluded_set = set(excluded)
    records: dict[str, dict[str, Any]] = {}
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative in excluded_set:
            continue
        records[relative] = {
            "sha256": preflight.sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
    return records


def validate_finite_json(value: Any, context: str) -> None:
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise RunnerFailure(f"{context} contains a non-finite number: {value!r}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            validate_finite_json(item, f"{context}[{index}]")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            validate_finite_json(item, f"{context}.{key}")
        return
    raise RunnerFailure(f"{context} contains an unsupported JSON value: {type(value).__name__}")


def verify_python_modules(python_executable: Path, root: Path) -> dict[str, Any]:
    root = root.resolve()
    code = (
        "import json;import dg_dataset,dg_losses,dg_models,train_dg;"
        "print(json.dumps([dg_dataset.__file__,dg_losses.__file__,"
        "dg_models.__file__,train_dg.__file__]))"
    )
    output = run_checked([str(python_executable), "-c", code], root)
    try:
        module_paths = [Path(item).resolve() for item in json.loads(output)]
    except (json.JSONDecodeError, TypeError) as exc:
        raise RunnerFailure(f"Module probe returned invalid output: {output!r}") from exc
    escaped = [path for path in module_paths if not path_is_within(path, root)]
    if escaped:
        raise RunnerFailure(f"Execution-root module import escaped {root}: {escaped}")
    return {"root": str(root), "module_paths": [str(path) for path in module_paths]}


def tpp_preload_code(clean_worktree: Path, tpp_snapshot_root: Path) -> str:
    """Return a bootstrap that pins every shared import to the clean worktree.

    ``train_tpp_isolated.py`` inserts its own repository root at ``sys.path[0]``.
    Direct execution would therefore import the modified main-repository
    ``train_dg`` and ``dg_dataset``.  The bootstrap loads the legacy shared
    modules first and installs them in ``sys.modules``; it then loads the two
    hash-locked TPP files explicitly.  The historical runner's later imports
    can only resolve to those already-pinned module objects.
    """

    clean_worktree = clean_worktree.resolve()
    tpp_snapshot_root = tpp_snapshot_root.resolve()
    tpp_dir = tpp_snapshot_root / "experiments" / "tpp_isolated"
    clean_hashes = {
        relative: preflight.sha256_file(clean_worktree / relative)
        for relative in TRACKED_LEGACY_CODE
    }
    snapshot_hashes = {
        "train_tpp_isolated.py": protocol.UNVERSIONED_TPP_SNAPSHOT["train_tpp_isolated.py"]["sha256"],
        "tpp_models.py": protocol.UNVERSIONED_TPP_SNAPSHOT["tpp_models.py"]["sha256"],
    }
    lines = [
        "import hashlib,importlib,importlib.util,json,subprocess,sys",
        "from pathlib import Path",
        "sys.dont_write_bytecode=True",
        f"_clean=Path({str(clean_worktree)!r}).resolve()",
        f"_tpp_dir=Path({str(tpp_dir)!r}).resolve()",
        f"_legacy_commit={protocol.LEGACY_COMMIT!r}",
        f"_clean_hashes={clean_hashes!r}",
        f"_snapshot_hashes={snapshot_hashes!r}",
        "def _sha(path):",
        "    digest=hashlib.sha256()",
        "    with Path(path).open('rb') as handle:",
        "        for chunk in iter(lambda:handle.read(4*1024*1024),b''):",
        "            digest.update(chunk)",
        "    return digest.hexdigest()",
        "def _inside(path,parent):",
        "    try:",
        "        Path(path).resolve().relative_to(Path(parent).resolve())",
        "        return True",
        "    except ValueError:",
        "        return False",
        "_head=subprocess.check_output(['git','-C',str(_clean),'rev-parse','HEAD'],text=True,encoding='utf-8').strip()",
        "_status=subprocess.check_output(['git','-C',str(_clean),'status','--short'],text=True,encoding='utf-8').strip()",
        "if _head != _legacy_commit:",
        "    raise RuntimeError(f'legacy clean-worktree commit mismatch: {_head} != {_legacy_commit}')",
        "if _status:",
        "    raise RuntimeError(f'legacy clean worktree is dirty at launch: {_status}')",
        "for _relative,_expected in _clean_hashes.items():",
        "    _actual=_sha(_clean/_relative)",
        "    if _actual != _expected:",
        "        raise RuntimeError(f'legacy shared-code hash mismatch: {_relative}: {_actual} != {_expected}')",
        "for _name,_expected in _snapshot_hashes.items():",
        "    _actual=_sha(_tpp_dir/_name)",
        "    if _actual != _expected:",
        "        raise RuntimeError(f'TPP snapshot hash mismatch: {_name}: {_actual} != {_expected}')",
        "sys.path.insert(0,str(_clean))",
        "_shared_names=['adaptive_kmargin','dg_dataset','dg_losses','dg_models','train_dg']",
        "_shared={_name:importlib.import_module(_name) for _name in _shared_names}",
        "for _name,_module in _shared.items():",
        "    if not _inside(_module.__file__,_clean):",
        "        raise RuntimeError(f'shared module escaped clean worktree: {_name}={_module.__file__}')",
        "_model_spec=importlib.util.spec_from_file_location('tpp_models',_tpp_dir/'tpp_models.py')",
        "if _model_spec is None or _model_spec.loader is None:",
        "    raise RuntimeError('cannot construct tpp_models snapshot spec')",
        "tpp_models=importlib.util.module_from_spec(_model_spec)",
        "sys.modules['tpp_models']=tpp_models",
        "_model_spec.loader.exec_module(tpp_models)",
        "if Path(tpp_models.__file__).resolve() != (_tpp_dir/'tpp_models.py').resolve():",
        "    raise RuntimeError(f'tpp_models resolved to the wrong file: {tpp_models.__file__}')",
    ]
    return "\n".join(lines)


def tpp_runner_bootstrap_code(
    clean_worktree: Path,
    tpp_snapshot_root: Path,
    probe_only: bool,
) -> str:
    lines = [
        tpp_preload_code(clean_worktree, tpp_snapshot_root),
        "_runner_spec=importlib.util.spec_from_file_location('legacy_tpp_train_snapshot',_tpp_dir/'train_tpp_isolated.py')",
        "if _runner_spec is None or _runner_spec.loader is None:",
        "    raise RuntimeError('cannot construct train_tpp_isolated snapshot spec')",
        "legacy_tpp_train_snapshot=importlib.util.module_from_spec(_runner_spec)",
        "sys.modules['legacy_tpp_train_snapshot']=legacy_tpp_train_snapshot",
        "_runner_spec.loader.exec_module(legacy_tpp_train_snapshot)",
        "if Path(legacy_tpp_train_snapshot.__file__).resolve() != (_tpp_dir/'train_tpp_isolated.py').resolve():",
        "    raise RuntimeError(f'TPP runner resolved to the wrong file: {legacy_tpp_train_snapshot.__file__}')",
        "for _name in _shared_names:",
        "    if not _inside(sys.modules[_name].__file__,_clean):",
        "        raise RuntimeError(f'shared module changed after loading TPP runner: {_name}={sys.modules[_name].__file__}')",
    ]
    if probe_only:
        lines.extend(
            [
                "print(json.dumps({'shared_modules':{name:sys.modules[name].__file__ for name in _shared_names},",
                "'tpp_models':tpp_models.__file__,'tpp_runner':legacy_tpp_train_snapshot.__file__},sort_keys=True))",
            ]
        )
    else:
        lines.append("legacy_tpp_train_snapshot.main()")
    return "\n".join(lines)


def verify_execution_roots(
    clean_worktree: Path,
    tpp_snapshot_root: Path,
    python_executable: Path,
    lock: Mapping[str, Any],
) -> dict[str, Any]:
    clean_worktree = clean_worktree.resolve()
    tpp_snapshot_root = tpp_snapshot_root.resolve()
    clean_record = preflight.verify_clean_worktree(clean_worktree)
    clean_record["selected_python_module_probe"] = verify_python_modules(
        python_executable, clean_worktree
    )

    clean_hashes: dict[str, str] = {}
    for relative in TRACKED_LEGACY_CODE:
        path = clean_worktree / relative
        if not path.is_file():
            raise RunnerFailure(f"Legacy execution file is missing: {path}")
        clean_hashes[relative] = preflight.sha256_file(path)

    tpp_entry = tpp_snapshot_root / "experiments" / "tpp_isolated" / "train_tpp_isolated.py"
    tpp_model = tpp_snapshot_root / "experiments" / "tpp_isolated" / "tpp_models.py"
    if not tpp_entry.is_file() or not tpp_model.is_file():
        raise RunnerFailure(
            "The TPP execution snapshot must contain experiments/tpp_isolated/"
            f"train_tpp_isolated.py and tpp_models.py: {tpp_snapshot_root}"
        )

    snapshot_lock = lock.get("unversioned_tpp_snapshot")
    if not isinstance(snapshot_lock, Mapping):
        raise RunnerFailure("protocol_lock.unversioned_tpp_snapshot is missing")
    assert_equal(
        preflight.sha256_file(tpp_entry),
        snapshot_lock["train_tpp_isolated.py"]["sha256"],
        "TPP snapshot train entry hash",
    )
    assert_equal(
        preflight.sha256_file(tpp_model),
        snapshot_lock["tpp_models.py"]["sha256"],
        "TPP snapshot model hash",
    )

    # Never execute the TPP runner directly.  Its own REPO_ROOT insertion would
    # otherwise give a modified checkout precedence over the immutable legacy
    # worktree.  The probe uses the exact same bootstrap as real execution:
    # shared modules are hash-checked, preloaded from the clean worktree, pinned
    # in sys.modules, and only then are the two hash-locked snapshot files
    # loaded explicitly.
    probe_output = run_checked(
        [
            str(python_executable),
            "-c",
            tpp_runner_bootstrap_code(clean_worktree, tpp_snapshot_root, probe_only=True),
        ],
        clean_worktree,
    )
    try:
        tpp_probe = json.loads(probe_output)
    except (json.JSONDecodeError, TypeError) as exc:
        raise RunnerFailure(f"TPP bootstrap probe returned invalid output: {probe_output!r}") from exc
    if not isinstance(tpp_probe, Mapping) or not isinstance(tpp_probe.get("shared_modules"), Mapping):
        raise RunnerFailure(f"TPP bootstrap probe has an invalid schema: {tpp_probe!r}")
    expected_shared_paths = {
        Path(relative).stem: (clean_worktree / relative).resolve()
        for relative in TRACKED_LEGACY_CODE
    }
    actual_shared_paths = {
        str(name): Path(value).resolve()
        for name, value in tpp_probe["shared_modules"].items()
    }
    assert_equal(set(actual_shared_paths), set(expected_shared_paths), "TPP shared module set")
    for name, expected_path in expected_shared_paths.items():
        assert_equal(actual_shared_paths[name], expected_path, f"TPP shared module path {name}")
    assert_equal(Path(tpp_probe["tpp_models"]).resolve(), tpp_model.resolve(), "TPP model module path")
    assert_equal(Path(tpp_probe["tpp_runner"]).resolve(), tpp_entry.resolve(), "TPP runner module path")

    # Import probes must not have created bytecode or any other replay-tree
    # mutation.  Re-check both identity and cleanliness after all probes.
    post_probe_head = run_checked(["git", "rev-parse", "HEAD"], clean_worktree)
    post_probe_status = run_checked(["git", "status", "--short"], clean_worktree)
    assert_equal(post_probe_head, protocol.LEGACY_COMMIT, "post-probe clean worktree commit")
    assert_equal(post_probe_status, "", "post-probe clean worktree status")
    return {
        "clean_worktree": clean_record,
        "clean_code_sha256": clean_hashes,
        "post_probe_head": post_probe_head,
        "post_probe_status_clean": True,
        "tpp_snapshot": {
            "root": str(tpp_snapshot_root),
            "entrypoint": str(tpp_entry),
            "entrypoint_sha256": preflight.sha256_file(tpp_entry),
            "model_sha256": preflight.sha256_file(tpp_model),
            "bootstrap_probe": tpp_probe,
        },
    }


def runtime_record(python_executable: Path) -> dict[str, Any]:
    code = (
        "import json,platform,sys,numpy,sklearn,torch;"
        "print(json.dumps({'python_executable':sys.executable,'python_version':platform.python_version(),"
        "'platform':platform.platform(),'numpy_version':numpy.__version__,"
        "'sklearn_version':sklearn.__version__,'torch_version':torch.__version__,"
        "'torch_cuda_version':torch.version.cuda,'cudnn_version':torch.backends.cudnn.version(),"
        "'cuda_available':torch.cuda.is_available()}))"
    )
    output = run_checked([str(python_executable), "-c", code], protocol.REPO_ROOT)
    try:
        record = json.loads(output)
    except json.JSONDecodeError as exc:
        raise RunnerFailure(f"Runtime probe returned invalid JSON: {output!r}") from exc
    record["orchestrator_python"] = sys.executable
    record["orchestrator_platform"] = platform.platform()
    return record


def load_arm_specs(bundle: Mapping[str, Any], selected_arms: Sequence[str]) -> list[ArmSpec]:
    selected = set(selected_arms)
    registry_records = bundle["registry"].get("anchor_configs")
    if not isinstance(registry_records, list):
        raise RunnerFailure("anchor_registry.anchor_configs must be a list")
    records_by_id = {str(record.get("config_id")): record for record in registry_records}
    assert_equal(set(records_by_id), set(ARM_IDS), "registry arm ids")

    specs: list[ArmSpec] = []
    for config_id in ARM_IDS:
        if config_id not in selected:
            continue
        record = records_by_id[config_id]
        config_path = Path(str(record["path"])).resolve()
        base_config = read_json(config_path)
        assert_equal(preflight.sha256_file(config_path), record["file_sha256"], f"{config_id} file hash")
        assert_equal(payload_sha256(base_config), record["canonical_payload_sha256"], f"{config_id} payload hash")
        runner = str(record["runner"])
        if runner not in {"train_dg.py", "train_tpp_isolated.py"}:
            raise RunnerFailure(f"Unsupported runner for {config_id}: {runner}")
        expected_runner = (
            "train_dg.py"
            if base_config["model"]["name"] == "pa_csi_dg_lite"
            else "train_tpp_isolated.py"
        )
        assert_equal(runner, expected_runner, f"{config_id} runner dispatch")
        arm_output_root = Path(str(base_config["output_root"])).resolve()
        if not path_is_within(arm_output_root, protocol.OUTPUT_ROOT.resolve()):
            raise RunnerFailure(
                f"Arm output root escapes the protocol output root: {config_id}: {arm_output_root}"
            )
        specs.append(
            ArmSpec(
                config_id=config_id,
                source_config_path=config_path,
                source_config_file_sha256=str(record["file_sha256"]),
                source_config_payload_sha256=str(record["canonical_payload_sha256"]),
                runner=runner,
                base_config=base_config,
                arm_output_root=arm_output_root,
            )
        )
    return specs


def derived_job_config(job: Job, payload_root: Path) -> dict[str, Any]:
    config = copy.deepcopy(dict(job.arm.base_config))
    config["target_env"] = job.target
    config["seed_list"] = [job.seed]
    config["output_root"] = str(payload_root.resolve()).replace("\\", "/")
    return config


def artifact_run_dir(job: Job, attempt_dir: Path) -> Path:
    config = job.arm.base_config
    return (
        attempt_dir
        / "payload"
        / str(config["mode"])
        / str(config["experiment_name"])
        / f"target_{job.target}"
        / f"seed_{job.seed}"
    )


def calculate_confusion_metrics(confusion: np.ndarray) -> dict[str, float]:
    total = float(confusion.sum())
    true_support = confusion.sum(axis=1).astype(np.float64)
    predicted_support = confusion.sum(axis=0).astype(np.float64)
    true_positive = np.diag(confusion).astype(np.float64)
    precision = np.divide(
        true_positive,
        predicted_support,
        out=np.zeros_like(true_positive),
        where=predicted_support != 0,
    )
    recall = np.divide(
        true_positive,
        true_support,
        out=np.zeros_like(true_positive),
        where=true_support != 0,
    )
    f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) != 0,
    )
    return {
        "accuracy": float(true_positive.sum() / total),
        "precision_macro": float(precision.mean()),
        "recall_macro": float(recall.mean()),
        "f1_macro": float(f1.mean()),
    }


def validate_history(metrics: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    history = metrics.get("history")
    expected_epochs = int(config["training"]["epochs"])
    if not isinstance(history, list) or len(history) != expected_epochs:
        raise RunnerFailure(
            f"history must contain exactly {expected_epochs} epochs; got "
            f"{len(history) if isinstance(history, list) else type(history).__name__}"
        )
    expected_sequence = list(range(1, expected_epochs + 1))
    actual_sequence = [record.get("epoch") for record in history if isinstance(record, Mapping)]
    assert_equal(actual_sequence, expected_sequence, "history epoch sequence")

    selection_metric = str(config["training"]["selection_metric"])
    best_score = -math.inf
    best_epoch = 0
    max_val_accuracy = -math.inf
    for record in history:
        if not isinstance(record.get("train"), Mapping) or not isinstance(record.get("val"), Mapping):
            raise RunnerFailure(f"Epoch {record.get('epoch')} lacks train/val metrics")
        validate_finite_json(record, f"history.epoch_{record['epoch']}")
        val = record["val"]
        if selection_metric not in val or "accuracy" not in val:
            raise RunnerFailure(f"Epoch {record['epoch']} lacks required validation metrics")
        score = float(val[selection_metric])
        # Historical runner uses >=, retaining the later exact tie.
        if score >= best_score:
            best_score = score
            best_epoch = int(record["epoch"])
        max_val_accuracy = max(max_val_accuracy, float(val["accuracy"]))

    assert_close(float(metrics["best_val_score"]), best_score, "best validation score")
    assert_close(float(metrics["best_val_accuracy"]), max_val_accuracy, "best validation accuracy")
    return {
        "epochs": len(history),
        "derived_best_epoch_late_tie": best_epoch,
        "derived_best_score": best_score,
    }


def validate_checkpoint(
    job: Job,
    checkpoint: Path,
    config_path: Path,
    clean_worktree: Path,
    tpp_snapshot_root: Path,
    python_executable: Path,
    expected_parameters: Mapping[str, Any],
) -> dict[str, Any]:
    if checkpoint.stat().st_size <= 0:
        raise RunnerFailure(f"Checkpoint is empty: {checkpoint}")
    model_name = str(job.arm.base_config["model"]["name"])
    expected_count = int(expected_parameters[model_name])
    if job.arm.runner == "train_dg.py":
        execution_root = clean_worktree.resolve()
        imports = "from dg_models import PaCsiDGLite as Model"
        bootstrap = ""
    else:
        execution_root = clean_worktree.resolve()
        bootstrap = tpp_preload_code(clean_worktree, tpp_snapshot_root) + "\n"
        imports = "from tpp_models import IsolatedPaCsiDGLiteTPP as Model"
    probe = (
        bootstrap
        + "import json,torch;"
        f"{imports};"
        f"cfg=json.load(open({str(config_path)!r},encoding='utf-8'));"
        "model=Model(input_dim=90,num_classes=6,model_config=cfg['model']);"
        f"state=torch.load({str(checkpoint)!r},map_location='cpu',weights_only=True);"
        "model.load_state_dict(state,strict=True);"
        "assert all(torch.isfinite(t).all().item() for t in state.values() if torch.is_tensor(t));"
        "print(json.dumps({'parameters':sum(p.numel() for p in model.parameters()),"
        "'state_keys':len(state),'state_values':sum(t.numel() for t in state.values() if torch.is_tensor(t))}))"
    )
    output = run_checked([str(python_executable), "-c", probe], execution_root)
    try:
        result = json.loads(output)
    except json.JSONDecodeError as exc:
        raise RunnerFailure(f"Checkpoint probe returned invalid JSON: {output!r}") from exc
    assert_equal(int(result["parameters"]), expected_count, f"{job.job_id} parameter count")
    if int(result["state_keys"]) <= 0 or int(result["state_values"]) <= 0:
        raise RunnerFailure(f"Checkpoint state is empty: {checkpoint}")
    return result


def validate_run_outputs(
    job: Job,
    attempt_dir: Path,
    clean_worktree: Path,
    tpp_snapshot_root: Path,
    python_executable: Path,
    lock: Mapping[str, Any],
) -> dict[str, Any]:
    run_dir = artifact_run_dir(job, attempt_dir)
    if not run_dir.is_dir():
        raise RunnerFailure(f"Expected run artifact directory is missing: {run_dir}")
    for name in REQUIRED_RUN_FILES:
        path = run_dir / name
        if not path.is_file():
            raise RunnerFailure(f"Required run artifact is missing: {path}")

    job_config_path = attempt_dir / "job_config.json"
    job_config = read_json(job_config_path)
    expected_config = derived_job_config(job, attempt_dir / "payload")
    assert_equal(job_config, expected_config, "attempt job config")
    resolved_config = read_json(run_dir / "resolved_config.json")
    resolved_without_runtime = dict(resolved_config)
    device_actual = resolved_without_runtime.pop("device_actual", None)
    assert_equal(resolved_without_runtime, expected_config, "resolved config")
    if not isinstance(device_actual, str) or not device_actual:
        raise RunnerFailure("resolved_config.device_actual is missing")

    metrics = read_json(run_dir / "metrics.json")
    validate_finite_json(metrics, "metrics")
    assert_equal(int(metrics["seed"]), job.seed, "metrics seed")
    assert_equal(str(metrics["target_env"]), job.target, "metrics target")
    assert_equal(str(metrics["selection_metric"]), "accuracy", "selection metric")
    assert_equal(int(metrics["train_size"]), 4800, "train size")
    assert_equal(int(metrics["val_size"]), 1200, "validation size")
    assert_equal(int(metrics["test_size"]), 3000, "test size")
    expected_sources = [env for env in protocol.ENV_ORDER if env != job.target]
    assert_equal(list(metrics["source_envs"]), expected_sources, "source environment order")
    assert_equal(str(metrics["sampler_mode"]), str(job_config["training"]["sampler"]), "sampler")
    history_health = validate_history(metrics, job_config)

    split = read_json(run_dir / "split_manifest.json")
    assert_equal(split["mode"], "dg_loeo", "split mode")
    assert_equal(split["target_env"], job.target, "split target")
    assert_equal(list(split["source_envs"]), expected_sources, "split source order")
    for subset, count in (("train", 4800), ("val", 1200), ("test", 3000)):
        assert_equal(int(split[subset]["count"]), count, f"split {subset} count")
    assert_equal(split["test"]["env_counts"], {job.target: 3000}, "target-only test split")

    confusion = np.load(run_dir / "confusion_matrix.npy", allow_pickle=False)
    if confusion.shape != (6, 6):
        raise RunnerFailure(f"Confusion matrix must be 6x6, got {confusion.shape}")
    if not np.issubdtype(confusion.dtype, np.number):
        raise RunnerFailure(f"Confusion matrix has non-numeric dtype: {confusion.dtype}")
    if not np.all(np.isfinite(confusion)) or not np.all(confusion >= 0) or not np.all(confusion == np.floor(confusion)):
        raise RunnerFailure("Confusion matrix must contain finite non-negative integers")
    confusion = confusion.astype(np.int64, copy=False)
    assert_equal(int(confusion.sum()), 3000, "confusion support")
    with (run_dir / "confusion_matrix.csv").open("r", encoding="utf-8", newline="") as handle:
        csv_rows = [[int(value) for value in row] for row in csv.reader(handle)]
    csv_confusion = np.asarray(csv_rows, dtype=np.int64)
    if not np.array_equal(confusion, csv_confusion):
        raise RunnerFailure("CSV and NPY confusion matrices differ")

    derived_metrics = calculate_confusion_metrics(confusion)
    test_metrics = metrics.get("test_metrics")
    if not isinstance(test_metrics, Mapping):
        raise RunnerFailure("metrics.test_metrics must be an object")
    for name, expected in derived_metrics.items():
        assert_close(float(test_metrics[name]), expected, f"test metric {name}")

    checkpoint_health = validate_checkpoint(
        job=job,
        checkpoint=run_dir / "best_model.pt",
        config_path=job_config_path,
        clean_worktree=clean_worktree,
        tpp_snapshot_root=tpp_snapshot_root,
        python_executable=python_executable,
        expected_parameters=lock["expected_parameters"],
    )
    return {
        "run_dir": str(run_dir),
        "device_actual": device_actual,
        "history": history_health,
        "confusion_sha256": preflight.sha256_file(run_dir / "confusion_matrix.npy"),
        "derived_test_metrics": derived_metrics,
        "checkpoint": checkpoint_health,
    }


def validate_completion_binding(attempt_dir: Path) -> Mapping[str, Any]:
    completion_path = attempt_dir / "attempt_completed.json"
    seal_path = attempt_dir / "attempt_seal.json"
    if completion_path.exists() != seal_path.exists():
        raise RunnerFailure(
            f"Attempt has a partial completion/seal pair and requires diagnosis: {attempt_dir}"
        )
    if not completion_path.is_file():
        raise RunnerFailure(f"Attempt is not marked complete: {attempt_dir}")
    completion = read_json(completion_path)
    seal = read_json(seal_path)
    assert_equal(completion.get("schema_version"), COMPLETION_SCHEMA_VERSION, "completion schema")
    assert_equal(completion.get("status"), "success", "completion status")
    assert_equal(seal.get("schema_version"), SEAL_SCHEMA_VERSION, "seal schema")
    assert_equal(
        preflight.sha256_file(completion_path),
        seal.get("completion_manifest_sha256"),
        "completion manifest seal",
    )
    artifacts = completion.get("artifacts")
    if not isinstance(artifacts, Mapping) or not artifacts:
        raise RunnerFailure(f"Completion manifest has no artifact hashes: {completion_path}")
    actual = sha256_tree_files(
        attempt_dir,
        excluded=("attempt_completed.json", "attempt_seal.json"),
    )
    assert_equal(set(actual), set(artifacts), "completion artifact member set")
    for relative, record in artifacts.items():
        if not isinstance(record, Mapping):
            raise RunnerFailure(f"Malformed completion artifact record: {relative}")
        assert_equal(actual[relative]["sha256"], record.get("sha256"), f"artifact hash {relative}")
        assert_equal(actual[relative]["size_bytes"], record.get("size_bytes"), f"artifact size {relative}")
    return completion


def validate_successful_attempt(
    job: Job,
    attempt_dir: Path,
    clean_worktree: Path,
    tpp_snapshot_root: Path,
    python_executable: Path,
    lock: Mapping[str, Any],
) -> dict[str, Any]:
    completion = validate_completion_binding(attempt_dir)
    assert_equal(completion.get("job_id"), job.job_id, "completion job id")
    assert_equal(completion.get("arm"), job.arm.config_id, "completion arm")
    assert_equal(completion.get("target"), job.target, "completion target")
    assert_equal(int(completion.get("seed")), job.seed, "completion seed")
    assert_equal(int(completion.get("returncode")), 0, "completion return code")
    assert_equal(
        completion.get("source_config_file_sha256"),
        job.arm.source_config_file_sha256,
        "completion source config file hash",
    )
    assert_equal(
        completion.get("source_config_payload_sha256"),
        job.arm.source_config_payload_sha256,
        "completion source config payload hash",
    )
    current_health = validate_run_outputs(
        job,
        attempt_dir,
        clean_worktree,
        tpp_snapshot_root,
        python_executable,
        lock,
    )
    assert_equal(payload_sha256(current_health), completion.get("health_sha256"), "completion health hash")
    return {
        "attempt_dir": str(attempt_dir),
        "completion_manifest_sha256": preflight.sha256_file(attempt_dir / "attempt_completed.json"),
        "health": current_health,
    }


def attempt_directories(job_root: Path) -> list[tuple[int, Path]]:
    if not job_root.exists():
        return []
    if not job_root.is_dir():
        raise RunnerFailure(f"Job root is not a directory: {job_root}")
    attempts: list[tuple[int, Path]] = []
    for child in sorted(job_root.iterdir()):
        if not child.name.startswith("attempt_"):
            raise RunnerFailure(f"Unexpected member in job root: {child}")
        match = ATTEMPT_RE.fullmatch(child.name)
        if match is None or not child.is_dir():
            raise RunnerFailure(f"Malformed attempt entry: {child}")
        attempts.append((int(match.group(1)), child))
    numbers = [number for number, _ in attempts]
    if len(numbers) != len(set(numbers)):
        raise RunnerFailure(f"Duplicate attempt numbers under {job_root}")
    return sorted(attempts)


def scan_attempts(
    job: Job,
    clean_worktree: Path,
    tpp_snapshot_root: Path,
    python_executable: Path,
    lock: Mapping[str, Any],
) -> AttemptScan:
    attempts = attempt_directories(job.job_root)
    prior: list[Mapping[str, Any]] = []
    successful: list[Path] = []
    for number, attempt_dir in attempts:
        completed_exists = (attempt_dir / "attempt_completed.json").exists()
        seal_exists = (attempt_dir / "attempt_seal.json").exists()
        failed_exists = (attempt_dir / "attempt_failed.json").exists()
        started_exists = (attempt_dir / "attempt_started.json").exists()
        if completed_exists or seal_exists:
            # Any attempt claiming completion must be completely healthy or the
            # whole job fails closed; silently replacing a corrupted success is
            # not a scientifically valid recovery.
            health = validate_successful_attempt(
                job,
                attempt_dir,
                clean_worktree,
                tpp_snapshot_root,
                python_executable,
                lock,
            )
            successful.append(attempt_dir)
            prior.append({"attempt": number, "status": "success", **health})
        elif failed_exists:
            if not started_exists:
                raise RunnerFailure(f"Failed attempt has no start record: {attempt_dir}")
            failure = read_json(attempt_dir / "attempt_failed.json")
            assert_equal(failure.get("job_id"), job.job_id, "failed attempt job id")
            prior.append({"attempt": number, "status": "failed", "path": str(attempt_dir)})
        elif started_exists:
            prior.append({"attempt": number, "status": "interrupted", "path": str(attempt_dir)})
        else:
            raise RunnerFailure(f"Attempt has no lifecycle record: {attempt_dir}")

    if len(successful) > 1:
        raise RunnerFailure(f"Multiple successful attempts exist for {job.job_id}: {successful}")
    next_attempt = max((number for number, _ in attempts), default=0) + 1
    return AttemptScan(
        action="skip" if successful else "run",
        next_attempt=next_attempt,
        successful_attempt=successful[0] if successful else None,
        prior_attempts=tuple(prior),
    )


def reserve_attempt(job: Job, starting_number: int) -> tuple[int, Path]:
    job.job_root.mkdir(parents=True, exist_ok=True)
    number = starting_number
    while True:
        attempt_dir = job.job_root / f"attempt_{number:03d}"
        try:
            attempt_dir.mkdir(exist_ok=False)
            return number, attempt_dir
        except FileExistsError:
            number += 1


def command_for_job(
    job: Job,
    job_config_path: Path,
    clean_worktree: Path,
    tpp_snapshot_root: Path,
    python_executable: Path,
) -> tuple[list[str], Path]:
    if job.arm.runner == "train_dg.py":
        cwd = clean_worktree.resolve()
        entrypoint = cwd / "train_dg.py"
        if not entrypoint.is_file():
            raise RunnerFailure(f"Training entrypoint is missing: {entrypoint}")
        command = [str(python_executable), str(entrypoint), "--config", str(job_config_path)]
    else:
        cwd = clean_worktree.resolve()
        entrypoint = (
            tpp_snapshot_root.resolve()
            / "experiments"
            / "tpp_isolated"
            / "train_tpp_isolated.py"
        )
        if not entrypoint.is_file():
            raise RunnerFailure(f"Training entrypoint is missing: {entrypoint}")
        command = [
            str(python_executable),
            "-c",
            tpp_runner_bootstrap_code(clean_worktree, tpp_snapshot_root, probe_only=False),
            "--config",
            str(job_config_path),
        ]
    return command, cwd


def execute_job(
    job: Job,
    scan: AttemptScan,
    clean_worktree: Path,
    tpp_snapshot_root: Path,
    python_executable: Path,
    lock: Mapping[str, Any],
    invocation_id: str,
) -> dict[str, Any]:
    attempt_number, attempt_dir = reserve_attempt(job, scan.next_attempt)
    payload_root = attempt_dir / "payload"
    job_config = derived_job_config(job, payload_root)
    job_config_path = attempt_dir / "job_config.json"
    write_json_exclusive(job_config_path, job_config)
    command, cwd = command_for_job(
        job,
        job_config_path,
        clean_worktree,
        tpp_snapshot_root,
        python_executable,
    )
    start_record = {
        "schema_version": SCHEMA_VERSION,
        "status": "started",
        "invocation_id": invocation_id,
        "job_id": job.job_id,
        "arm": job.arm.config_id,
        "target": job.target,
        "seed": job.seed,
        "attempt": attempt_number,
        "started_at": utc_now(),
        "source_config": str(job.arm.source_config_path),
        "source_config_file_sha256": job.arm.source_config_file_sha256,
        "source_config_payload_sha256": job.arm.source_config_payload_sha256,
        "job_config_file_sha256": preflight.sha256_file(job_config_path),
        "job_config_payload_sha256": payload_sha256(job_config),
        "command": command,
        "cwd": str(cwd),
    }
    write_json_exclusive(attempt_dir / "attempt_started.json", start_record)

    stdout_path = attempt_dir / "runner.stdout.log"
    stderr_path = attempt_dir / "runner.stderr.log"
    returncode: int | None = None
    failure_message: str | None = None
    try:
        with stdout_path.open("x", encoding="utf-8", newline="") as stdout_handle, stderr_path.open(
            "x", encoding="utf-8", newline=""
        ) as stderr_handle:
            completed = subprocess.run(
                command,
                cwd=cwd,
                stdout=stdout_handle,
                stderr=stderr_handle,
                check=False,
                text=True,
            )
            returncode = int(completed.returncode)
        if returncode != 0:
            raise RunnerFailure(f"Training process exited with code {returncode}")

        health = validate_run_outputs(
            job,
            attempt_dir,
            clean_worktree,
            tpp_snapshot_root,
            python_executable,
            lock,
        )
        artifacts = sha256_tree_files(
            attempt_dir,
            excluded=("attempt_completed.json", "attempt_seal.json"),
        )
        completion = {
            "schema_version": COMPLETION_SCHEMA_VERSION,
            "status": "success",
            "invocation_id": invocation_id,
            "job_id": job.job_id,
            "arm": job.arm.config_id,
            "target": job.target,
            "seed": job.seed,
            "attempt": attempt_number,
            "started_at": start_record["started_at"],
            "completed_at": utc_now(),
            "returncode": 0,
            "source_config_file_sha256": job.arm.source_config_file_sha256,
            "source_config_payload_sha256": job.arm.source_config_payload_sha256,
            "job_config_file_sha256": preflight.sha256_file(job_config_path),
            "job_config_payload_sha256": payload_sha256(job_config),
            "run_artifact_dir": str(artifact_run_dir(job, attempt_dir).relative_to(attempt_dir).as_posix()),
            "health": health,
            "health_sha256": payload_sha256(health),
            "artifacts": artifacts,
        }
        completion_path = attempt_dir / "attempt_completed.json"
        write_json_exclusive(completion_path, completion)
        write_json_exclusive(
            attempt_dir / "attempt_seal.json",
            {
                "schema_version": SEAL_SCHEMA_VERSION,
                "completion_manifest_sha256": preflight.sha256_file(completion_path),
            },
        )
        # Validate the exact on-disk form before reporting success.
        validate_successful_attempt(
            job,
            attempt_dir,
            clean_worktree,
            tpp_snapshot_root,
            python_executable,
            lock,
        )
        return {"job_id": job.job_id, "status": "success", "attempt_dir": str(attempt_dir)}
    except Exception as exc:
        failure_message = f"{type(exc).__name__}: {exc}"
        failure = {
            "schema_version": SCHEMA_VERSION,
            "status": "failed",
            "invocation_id": invocation_id,
            "job_id": job.job_id,
            "arm": job.arm.config_id,
            "target": job.target,
            "seed": job.seed,
            "attempt": attempt_number,
            "started_at": start_record["started_at"],
            "failed_at": utc_now(),
            "returncode": returncode,
            "error": failure_message,
            "traceback": traceback.format_exc(),
            "preserved_files": sha256_tree_files(
                attempt_dir,
                excluded=("attempt_failed.json",),
            ),
        }
        write_json_exclusive(attempt_dir / "attempt_failed.json", failure)
        raise RunnerFailure(
            f"Job failed; all artifacts were preserved in {attempt_dir}: {failure_message}"
        ) from exc


@contextmanager
def exclusive_runner_lock(output_root: Path, invocation_id: str):
    output_root.mkdir(parents=True, exist_ok=True)
    lock_path = output_root / ".anchor_runner.lock"
    descriptor: int | None = None
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        payload = canonical_json(
            {
                "schema_version": SCHEMA_VERSION,
                "invocation_id": invocation_id,
                "pid": os.getpid(),
                "created_at": utc_now(),
            }
        )
        os.write(descriptor, payload)
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
    except FileExistsError as exc:
        raise RunnerFailure(
            f"Another runner may be active, or a stale lock needs manual audit: {lock_path}"
        ) from exc
    try:
        yield lock_path
    finally:
        if descriptor is not None:
            os.close(descriptor)
        # This lock belongs to this invocation and is the only transient file
        # removed by the orchestrator. Attempt evidence is never deleted.
        try:
            lock_payload = json.loads(lock_path.read_text(encoding="utf-8"))
            if lock_payload.get("invocation_id") == invocation_id:
                lock_path.unlink()
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            pass


def perform_preflight(
    generated_root: Path,
    clean_worktree: Path,
    tpp_snapshot_root: Path,
    python_executable: Path,
) -> tuple[dict[str, Any], Mapping[str, Any]]:
    try:
        bundle = preflight.verify_bundle(generated_root.resolve())
        snapshot = preflight.verify_snapshot_hashes()
        data = preflight.verify_data(full_hash=True)
        splits = preflight.verify_reconstructed_splits()
        configs = preflight.verify_configs(bundle, clean_worktree.resolve())
        roots = verify_execution_roots(
            clean_worktree,
            tpp_snapshot_root,
            python_executable,
            bundle["lock"],
        )
    except (preflight.AuditFailure, KeyError, OSError, subprocess.SubprocessError) as exc:
        raise RunnerFailure(f"Preflight failed closed: {exc}") from exc
    audit = {
        "schema_version": SCHEMA_VERSION,
        "checked_at": utc_now(),
        "bundle": {key: value for key, value in bundle.items() if key not in {"lock", "registry"}},
        "snapshot_hashes": snapshot,
        "data": data,
        "reconstructed_splits": splits,
        "configs": configs,
        "execution_roots": roots,
        "runtime": runtime_record(python_executable),
    }
    return audit, bundle


def build_jobs(arms: Sequence[ArmSpec], stage: str) -> list[Job]:
    seeds = STAGE_SEEDS[stage]
    return [
        Job(arm=arm, target=target, seed=seed)
        for arm in arms
        for target in protocol.TARGETS
        for seed in seeds
    ]


def require_sentinel_successes(
    arms: Sequence[ArmSpec],
    clean_worktree: Path,
    tpp_snapshot_root: Path,
    python_executable: Path,
    lock: Mapping[str, Any],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    missing: list[str] = []
    for arm in arms:
        for target in protocol.TARGETS:
            job = Job(arm=arm, target=target, seed=42)
            scan = scan_attempts(
                job,
                clean_worktree,
                tpp_snapshot_root,
                python_executable,
                lock,
            )
            if scan.action != "skip":
                missing.append(job.job_id)
            else:
                records.append(
                    {
                        "job_id": job.job_id,
                        "attempt_dir": str(scan.successful_attempt),
                        "status": "healthy",
                    }
                )
    if missing:
        raise RunnerFailure(
            "The full stage is locked until every selected arm has a healthy seed-42 "
            f"sentinel for every target. Missing: {missing}"
        )
    return records


def plan_jobs(
    jobs: Sequence[Job],
    clean_worktree: Path,
    tpp_snapshot_root: Path,
    python_executable: Path,
    lock: Mapping[str, Any],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for job in jobs:
        scan = scan_attempts(
            job,
            clean_worktree,
            tpp_snapshot_root,
            python_executable,
            lock,
        )
        preview_dir = job.job_root / f"attempt_{scan.next_attempt:03d}"
        preview_config = derived_job_config(job, preview_dir / "payload")
        preview_config_path = preview_dir / "job_config.json"
        command, cwd = command_for_job(
            job,
            preview_config_path,
            clean_worktree,
            tpp_snapshot_root,
            python_executable,
        )
        records.append(
            {
                "job_id": job.job_id,
                "arm": job.arm.config_id,
                "target": job.target,
                "seed": job.seed,
                "action": scan.action,
                "successful_attempt": str(scan.successful_attempt) if scan.successful_attempt else None,
                "next_attempt": scan.next_attempt,
                "next_attempt_dir": str(preview_dir),
                "derived_config_payload_sha256": payload_sha256(preview_config),
                "command": command,
                "cwd": str(cwd),
                "prior_attempts": list(scan.prior_attempts),
            }
        )
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fail-closed, attempt-preserving legacy anchor runner."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="Validate and print the plan without writes or training.")
    mode.add_argument("--execute", action="store_true", help="Execute the validated plan sequentially.")
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="For dry-run, emit a compact machine-readable summary instead of commands.",
    )
    parser.add_argument("--stage", required=True, choices=sorted(STAGE_SEEDS))
    parser.add_argument(
        "--arms",
        nargs="+",
        choices=list(ARM_IDS),
        default=list(ARM_IDS),
        help="Anchor arms to validate/run; defaults to all four.",
    )
    parser.add_argument("--generated-root", type=Path, default=protocol.GENERATED_ROOT)
    parser.add_argument("--clean-worktree", type=Path, default=protocol.DEFAULT_CLEAN_WORKTREE)
    parser.add_argument(
        "--tpp-snapshot-root",
        type=Path,
        default=protocol.REPO_ROOT,
        help=(
            "Root containing the hash-locked experiments/tpp_isolated snapshot; shared "
            "modules are always preloaded from --clean-worktree. Defaults to the current "
            "repository, whose two snapshot files must match protocol_lock SHA-256."
        ),
    )
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generated_root = args.generated_root.resolve()
    clean_worktree = args.clean_worktree.resolve()
    tpp_snapshot_root = args.tpp_snapshot_root.resolve()
    python_executable = args.python.resolve()
    if not python_executable.is_file():
        raise SystemExit(f"Python executable does not exist: {python_executable}")

    invocation_id = uuid.uuid4().hex
    try:
        preflight_audit, bundle = perform_preflight(
            generated_root,
            clean_worktree,
            tpp_snapshot_root,
            python_executable,
        )
        arms = load_arm_specs(bundle, args.arms)
        sentinel_gate: list[dict[str, Any]] = []
        if args.stage == "full":
            sentinel_gate = require_sentinel_successes(
                arms,
                clean_worktree,
                tpp_snapshot_root,
                python_executable,
                bundle["lock"],
            )
        jobs = build_jobs(arms, args.stage)
        plan = plan_jobs(
            jobs,
            clean_worktree,
            tpp_snapshot_root,
            python_executable,
            bundle["lock"],
        )
        plan_payload = {
            "schema_version": SCHEMA_VERSION,
            "invocation_id": invocation_id,
            "mode": "dry_run" if args.dry_run else "execute",
            "stage": args.stage,
            "arms": [arm.config_id for arm in arms],
            "preflight_sha256": payload_sha256(preflight_audit),
            "sentinel_gate": sentinel_gate,
            "planned_jobs": len(plan),
            "jobs_to_run": sum(record["action"] == "run" for record in plan),
            "jobs_to_skip": sum(record["action"] == "skip" for record in plan),
            "jobs": plan,
        }
        if args.dry_run:
            if args.summary_only:
                compact = {
                    key: plan_payload[key]
                    for key in (
                        "schema_version",
                        "invocation_id",
                        "mode",
                        "stage",
                        "arms",
                        "preflight_sha256",
                        "planned_jobs",
                        "jobs_to_run",
                        "jobs_to_skip",
                    )
                }
                compact["jobs"] = [
                    {
                        "job_id": record["job_id"],
                        "arm": record["arm"],
                        "target": record["target"],
                        "seed": record["seed"],
                        "action": record["action"],
                        "cwd": record["cwd"],
                        "uses_tpp_bootstrap": record["arm"].startswith("anchor_tpp_"),
                    }
                    for record in plan
                ]
                print(json.dumps(compact, ensure_ascii=False, indent=2, allow_nan=False))
            else:
                print(json.dumps(plan_payload, ensure_ascii=False, indent=2, allow_nan=False))
            return

        output_root = protocol.OUTPUT_ROOT.resolve()
        with exclusive_runner_lock(output_root, invocation_id):
            invocation_dir = output_root / "runner_invocations" / invocation_id
            invocation_dir.mkdir(parents=True, exist_ok=False)
            write_json_exclusive(invocation_dir / "preflight.json", preflight_audit)
            write_json_exclusive(invocation_dir / "plan.json", plan_payload)
            results: list[dict[str, Any]] = []
            for job, record in zip(jobs, plan):
                if record["action"] == "skip":
                    results.append(
                        {
                            "job_id": job.job_id,
                            "status": "skipped_healthy_success",
                            "attempt_dir": record["successful_attempt"],
                        }
                    )
                    continue
                scan = scan_attempts(
                    job,
                    clean_worktree,
                    tpp_snapshot_root,
                    python_executable,
                    bundle["lock"],
                )
                if scan.action == "skip":
                    # Another result appeared after planning.  The process lock
                    # should make this impossible, so fail rather than silently
                    # changing the frozen plan.
                    raise RunnerFailure(f"Job state changed after planning: {job.job_id}")
                results.append(
                    execute_job(
                        job,
                        scan,
                        clean_worktree,
                        tpp_snapshot_root,
                        python_executable,
                        bundle["lock"],
                        invocation_id,
                    )
                )
            write_json_exclusive(
                invocation_dir / "completed.json",
                {
                    "schema_version": SCHEMA_VERSION,
                    "invocation_id": invocation_id,
                    "status": "success",
                    "completed_at": utc_now(),
                    "results": results,
                },
            )
    except RunnerFailure as exc:
        raise SystemExit(f"FAIL-CLOSED: {exc}") from exc


if __name__ == "__main__":
    main()
