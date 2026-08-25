from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from train_tpp_isolated import hard_pair_report


TARGET_ENVS = ("E1", "E2", "E3")
TPP_SEEDS = (42, 52, 62)
BASELINE_SEEDS = (52, 62)
TPP_CONFIGS = (
    ("tpp_l12", REPO_ROOT / "experiments" / "tpp_isolated" / "configs" / "tpp_l12.json"),
    ("tpp_l124", REPO_ROOT / "experiments" / "tpp_isolated" / "configs" / "tpp_l124.json"),
)
BASELINE_CONFIG = REPO_ROOT / "experiments" / "tpp_isolated" / "configs" / "anchor_ce_baseline_ref.json"


@dataclass(frozen=True)
class Job:
    kind: str
    experiment_name: str
    config_path: Path
    entrypoint: Path
    target_env: str
    seed: int
    run_dir: Path

    @property
    def name(self) -> str:
        return f"{self.kind}:{self.experiment_name}:target_{self.target_env}:seed_{self.seed}"

    @property
    def log_name(self) -> str:
        return f"{self.kind}_{self.experiment_name}_target_{self.target_env}_seed_{self.seed}.log"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the isolated TPP experiment matrix sequentially.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned jobs without running them.")
    parser.add_argument("--force", action="store_true", help="Run jobs even when completed artifacts exist.")
    parser.add_argument("--python", default=sys.executable, help="Python executable to use for child jobs.")
    parser.add_argument("--targets", nargs="+", default=list(TARGET_ENVS), choices=list(TARGET_ENVS))
    parser.add_argument("--tpp-seeds", nargs="+", type=int, default=list(TPP_SEEDS))
    parser.add_argument("--baseline-seeds", nargs="+", type=int, default=list(BASELINE_SEEDS))
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def run_dir_for_config(config_path: Path, target_env: str, seed: int) -> Path:
    config = load_json(config_path)
    output_root = REPO_ROOT / str(config.get("output_root", "outputs"))
    return (
        output_root
        / str(config.get("mode", "dg_loeo"))
        / str(config.get("experiment_name", config_path.stem))
        / f"target_{target_env}"
        / f"seed_{seed}"
    )


def planned_jobs(targets: Sequence[str], tpp_seeds: Sequence[int], baseline_seeds: Sequence[int]) -> List[Job]:
    jobs: List[Job] = []
    for target_env in targets:
        for seed in baseline_seeds:
            jobs.append(
                Job(
                    kind="baseline",
                    experiment_name="anchor_ce_baseline_ref",
                    config_path=BASELINE_CONFIG,
                    entrypoint=REPO_ROOT / "train_dg.py",
                    target_env=target_env,
                    seed=int(seed),
                    run_dir=run_dir_for_config(BASELINE_CONFIG, target_env, int(seed)),
                )
            )

    for experiment_name, config_path in TPP_CONFIGS:
        for target_env in targets:
            for seed in tpp_seeds:
                jobs.append(
                    Job(
                        kind="tpp",
                        experiment_name=experiment_name,
                        config_path=config_path,
                        entrypoint=REPO_ROOT / "experiments" / "tpp_isolated" / "train_tpp_isolated.py",
                        target_env=target_env,
                        seed=int(seed),
                        run_dir=run_dir_for_config(config_path, target_env, int(seed)),
                    )
                )
    return jobs


def has_confusion(run_dir: Path) -> bool:
    return (run_dir / "confusion_matrix.npy").exists() or (run_dir / "confusion_matrix.csv").exists()


def is_complete(job: Job) -> bool:
    return (job.run_dir / "metrics.json").exists() and has_confusion(job.run_dir)


def command_for_job(job: Job, python_executable: str) -> List[str]:
    return [
        python_executable,
        str(job.entrypoint),
        "--config",
        str(job.config_path),
        "--target-env",
        job.target_env,
        "--seeds",
        str(job.seed),
    ]


def read_confusion(run_dir: Path) -> np.ndarray:
    npy_path = run_dir / "confusion_matrix.npy"
    if npy_path.exists():
        return np.load(npy_path)
    csv_path = run_dir / "confusion_matrix.csv"
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        return np.array([[int(value) for value in row] for row in csv.reader(handle)], dtype=np.int64)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def ensure_hard_pair_report(job: Job) -> None:
    if not is_complete(job):
        return
    report_path = job.run_dir / "hard_pair_report.json"
    if report_path.exists():
        return
    write_json(report_path, hard_pair_report(read_confusion(job.run_dir)))


def run_job(job: Job, python_executable: str, log_dir: Path) -> None:
    command = command_for_job(job, python_executable)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / job.log_name
    start = time.time()
    print(f"[RUN] {job.name}", flush=True)
    print(f"      log: {log_path}", flush=True)
    with log_path.open("w", encoding="utf-8", newline="") as log_file:
        log_file.write("$ " + " ".join(command) + "\n\n")
        log_file.flush()
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=False,
        )
    elapsed = time.time() - start
    if completed.returncode != 0:
        raise RuntimeError(f"Job failed after {elapsed:.1f}s: {job.name}. See {log_path}")
    ensure_hard_pair_report(job)
    print(f"[OK]  {job.name} ({elapsed / 60.0:.1f} min)", flush=True)


def print_plan(jobs: Iterable[Job], python_executable: str, force: bool) -> None:
    for index, job in enumerate(jobs, start=1):
        status = "run"
        if is_complete(job) and not force:
            status = "skip-complete"
        print(f"{index:02d}. [{status}] {job.name}", flush=True)
        print("    " + " ".join(command_for_job(job, python_executable)), flush=True)


def main() -> None:
    args = parse_args()
    jobs = planned_jobs(args.targets, args.tpp_seeds, args.baseline_seeds)
    log_dir = REPO_ROOT / "outputs_tpp_isolated" / "run_logs"

    if args.dry_run:
        print_plan(jobs, args.python, args.force)
        return

    total = len(jobs)
    skipped = 0
    for index, job in enumerate(jobs, start=1):
        print(f"\n[{index}/{total}] {job.name}", flush=True)
        if is_complete(job) and not args.force:
            ensure_hard_pair_report(job)
            skipped += 1
            print("[SKIP] completed artifacts already exist", flush=True)
            continue
        run_job(job, args.python, log_dir)

    print(f"\nCompleted matrix: {total - skipped} run, {skipped} skipped, {total} planned.", flush=True)


if __name__ == "__main__":
    main()
