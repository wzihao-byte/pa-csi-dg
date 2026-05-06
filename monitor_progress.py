from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parent
ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
TQDM_RE = re.compile(
    r"(?P<percent>\d+)%\|.*?\|\s*(?P<current>\d+)/(?P<total>\d+)\s*\[(?P<timing>[^\]]+)\]"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Monitor PA-CSI DG experiment progress without touching the running training process. "
            "Progress is inferred from output artifacts written by train_dg.py."
        )
    )
    parser.add_argument("--config", required=True, help="Path to the JSON config used by the experiment.")
    parser.add_argument("--output-root", help="Override the output root if the run used --output-root.")
    parser.add_argument("--mode", help="Override the mode if the run used --mode.")
    parser.add_argument("--target-env", dest="target_env", help="Override target env if the run used --target-env.")
    parser.add_argument("--seeds", nargs="+", type=int, help="Override seed list if the run used --seeds.")
    parser.add_argument("--log-file", help="Optional stderr log to read for the live batch bar.")
    parser.add_argument("--no-log", action="store_true", help="Disable live stderr-log parsing.")
    parser.add_argument("--poll-seconds", type=float, default=5.0, help="Seconds between output-folder checks.")
    parser.add_argument("--once", action="store_true", help="Print the current progress once and exit.")
    return parser.parse_args()


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def choose_target_envs(config: Mapping[str, Any], target_env_override: Optional[str]) -> List[str]:
    target_env = target_env_override if target_env_override is not None else config.get("target_env")
    env_files = config.get("data", {}).get("env_files", {})
    env_names = list(env_files.keys())

    if config.get("mode") == "random_split":
        return ["random_split"]
    if target_env in (None, "all"):
        return env_names
    return [str(target_env)]


def build_run_specs(config: Mapping[str, Any], args: argparse.Namespace) -> Tuple[Path, str, str, List[Tuple[str, int]]]:
    mode = str(args.mode or config.get("mode", "dg_loeo"))
    experiment_name = str(config.get("experiment_name", Path(args.config).stem))
    output_root = resolve_path(args.output_root or str(config.get("output_root", "outputs")))
    runtime_config = dict(config)
    runtime_config["mode"] = mode
    target_envs = choose_target_envs(runtime_config, args.target_env)
    seeds = args.seeds or [int(seed) for seed in config.get("seed_list", [42])]
    return output_root, mode, experiment_name, [(target_env, int(seed)) for target_env in target_envs for seed in seeds]


def run_dir(output_root: Path, mode: str, experiment_name: str, target_env: str, seed: int) -> Path:
    return output_root / mode / experiment_name / f"target_{target_env}" / f"seed_{seed}"


def newest_mtime(paths: Iterable[Path]) -> Optional[float]:
    mtimes = [path.stat().st_mtime for path in paths if path.exists()]
    if not mtimes:
        return None
    return max(mtimes)


def inspect_run(path: Path) -> Dict[str, Any]:
    metrics_path = path / "metrics.json"
    if metrics_path.exists():
        try:
            with metrics_path.open("r", encoding="utf-8") as handle:
                metrics = json.load(handle)
            test_metrics = metrics.get("test_metrics", {})
            return {
                "status": "done",
                "accuracy": test_metrics.get("accuracy"),
                "f1_macro": test_metrics.get("f1_macro"),
                "updated_at": metrics_path.stat().st_mtime,
            }
        except (OSError, json.JSONDecodeError):
            return {"status": "writing", "updated_at": metrics_path.stat().st_mtime}

    active_files = [
        path / "split_manifest.json",
        path / "best_model.pt",
        path / "confusion_matrix.npy",
        path / "confusion_matrix.csv",
    ]
    updated_at = newest_mtime(active_files)
    if updated_at is not None:
        status = "active" if (path / "best_model.pt").exists() else "started"
        return {"status": status, "updated_at": updated_at}

    if path.exists():
        return {"status": "created", "updated_at": path.stat().st_mtime}
    return {"status": "pending", "updated_at": None}


def format_active(statuses: Mapping[Tuple[str, int], Mapping[str, Any]]) -> str:
    candidates = [
        (spec, state)
        for spec, state in statuses.items()
        if state["status"] in {"created", "started", "active", "writing"}
    ]
    if not candidates:
        return "waiting"

    def state_mtime(item: Tuple[Tuple[str, int], Mapping[str, Any]]) -> float:
        return float(item[1].get("updated_at") or 0.0)

    (target_env, seed), state = max(candidates, key=state_mtime)
    return f"{state['status']} target={target_env} seed={seed}"


def read_log_tail(path: Path, max_bytes: int = 131_072) -> str:
    with path.open("rb") as handle:
        handle.seek(0, 2)
        size = handle.tell()
        handle.seek(max(0, size - max_bytes), 0)
        return handle.read().decode("utf-8", errors="replace")


def parse_latest_tqdm(path: Optional[Path]) -> Optional[Dict[str, Any]]:
    if path is None or not path.exists():
        return None

    try:
        text = read_log_tail(path)
    except OSError:
        return None

    for raw_line in reversed(text.replace("\r", "\n").splitlines()):
        line = ANSI_RE.sub("", raw_line).strip()
        match = TQDM_RE.search(line)
        if not match:
            continue
        timing = match.group("timing").strip()
        return {
            "current": int(match.group("current")),
            "total": int(match.group("total")),
            "percent": int(match.group("percent")),
            "timing": timing,
            "path": str(path),
        }
    return None


def format_tqdm_snapshot(snapshot: Optional[Mapping[str, Any]]) -> str:
    if snapshot is None:
        return "no live tqdm log yet"
    return (
        f"{int(snapshot['current'])}/{int(snapshot['total'])} "
        f"({int(snapshot['percent'])}%) [{snapshot['timing']}]"
    )


def resolve_log_path(args: argparse.Namespace, output_root: Path, experiment_name: str) -> Optional[Path]:
    if args.no_log:
        return None
    if args.log_file:
        return resolve_path(args.log_file)
    return output_root / "run_logs" / f"{experiment_name}.stderr.log"


def print_once(
    statuses: Mapping[Tuple[str, int], Mapping[str, Any]],
    total: int,
    log_snapshot: Optional[Mapping[str, Any]],
) -> None:
    done = sum(1 for state in statuses.values() if state["status"] == "done")
    active = format_active(statuses)
    print(f"Completed runs: {done}/{total}; {active}")
    print(f"Current loader: {format_tqdm_snapshot(log_snapshot)}")
    for (target_env, seed), state in statuses.items():
        details = state["status"]
        if state.get("accuracy") is not None:
            details += f", accuracy={float(state['accuracy']):.4f}"
        if state.get("f1_macro") is not None:
            details += f", f1_macro={float(state['f1_macro']):.4f}"
        print(f"- target={target_env} seed={seed}: {details}")


def main() -> None:
    args = parse_args()
    config = load_config(resolve_path(args.config))
    output_root, mode, experiment_name, specs = build_run_specs(config, args)
    log_path = resolve_log_path(args, output_root, experiment_name)
    total = len(specs)

    if total == 0:
        raise ValueError("No target-env/seed runs were found to monitor.")

    def read_statuses() -> Dict[Tuple[str, int], Dict[str, Any]]:
        return {
            spec: inspect_run(run_dir(output_root, mode, experiment_name, spec[0], spec[1]))
            for spec in specs
        }

    if args.once:
        print_once(read_statuses(), total, parse_latest_tqdm(log_path))
        return

    live_bar: Optional[tqdm] = None
    with tqdm(total=total, desc="completed runs", unit="run", dynamic_ncols=True, position=0) as bar:
        while True:
            statuses = read_statuses()
            completed = sum(1 for state in statuses.values() if state["status"] == "done")
            bar.n = completed
            bar.set_postfix_str(format_active(statuses))
            bar.refresh()

            snapshot = parse_latest_tqdm(log_path)
            if snapshot is not None:
                current = int(snapshot["current"])
                total_batches = int(snapshot["total"])
                needs_new_live_bar = (
                    live_bar is None
                    or live_bar.total != total_batches
                    or current < live_bar.n
                )
                if needs_new_live_bar:
                    if live_bar is not None:
                        live_bar.close()
                    live_bar = tqdm(
                        total=total_batches,
                        desc="current loader",
                        unit="batch",
                        dynamic_ncols=True,
                        position=1,
                        leave=False,
                    )
                live_bar.n = current
                live_bar.total = total_batches
                live_bar.set_postfix_str(f"{int(snapshot['percent'])}% [{snapshot['timing']}]")
                live_bar.refresh()

            if completed >= total:
                break
            time.sleep(max(0.5, float(args.poll_seconds)))

    if live_bar is not None:
        live_bar.close()


if __name__ == "__main__":
    main()
