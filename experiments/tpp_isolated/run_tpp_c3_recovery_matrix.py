from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
TRAIN_ENTRYPOINT = REPO_ROOT / "experiments" / "tpp_isolated" / "train_tpp_isolated.py"
CONFIG_ROOT = REPO_ROOT / "experiments" / "tpp_isolated" / "configs"
TARGET_ENVS = ("E1", "E2", "E3")
SCREENING_SEEDS = (52, 62)
CONFIRMATION_SEEDS = (42, 52, 62)
EXPERIMENT_CONFIGS = {
    "residual025": CONFIG_ROOT / "tpp_c3_residual025_l124.json",
    "residual050": CONFIG_ROOT / "tpp_c3_residual050_l124.json",
    "mixed050": CONFIG_ROOT / "tpp_c3_mixed050_l124.json",
}


@dataclass(frozen=True)
class Job:
    experiment_key: str
    experiment_name: str
    config_path: Path
    target_env: str
    seed: int
    run_dir: Path

    @property
    def name(self) -> str:
        return f"{self.experiment_key}:target_{self.target_env}:seed_{self.seed}"

    @property
    def log_name(self) -> str:
        return f"{self.experiment_name}_target_{self.target_env}_seed_{self.seed}.log"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the staged C3-recovery TPP experiment matrix sequentially."
    )
    parser.add_argument(
        "--stage",
        choices=["screening", "confirmation"],
        default="screening",
        help="Screen with seeds 52/62, or confirm with the complete 42/52/62 matrix.",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=sorted(EXPERIMENT_CONFIGS),
        default=sorted(EXPERIMENT_CONFIGS),
    )
    parser.add_argument("--targets", nargs="+", choices=list(TARGET_ENVS), default=list(TARGET_ENVS))
    parser.add_argument("--seeds", nargs="+", type=int, default=None, help="Override the stage seed list.")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def run_dir_for_config(config_path: Path, target_env: str, seed: int) -> Path:
    config = load_json(config_path)
    return (
        REPO_ROOT
        / str(config.get("output_root", "outputs_tpp_c3_recovery"))
        / str(config.get("mode", "dg_loeo"))
        / str(config.get("experiment_name", config_path.stem))
        / f"target_{target_env}"
        / f"seed_{seed}"
    )


def planned_jobs(
    experiment_keys: Sequence[str], targets: Sequence[str], seeds: Sequence[int]
) -> List[Job]:
    jobs: List[Job] = []
    for experiment_key in experiment_keys:
        config_path = EXPERIMENT_CONFIGS[experiment_key]
        experiment_name = str(load_json(config_path)["experiment_name"])
        for target_env in targets:
            for seed in seeds:
                jobs.append(
                    Job(
                        experiment_key=experiment_key,
                        experiment_name=experiment_name,
                        config_path=config_path,
                        target_env=target_env,
                        seed=int(seed),
                        run_dir=run_dir_for_config(config_path, target_env, int(seed)),
                    )
                )
    return jobs


def is_complete(job: Job) -> bool:
    return (
        (job.run_dir / "metrics.json").exists()
        and (job.run_dir / "confusion_matrix.npy").exists()
    )


def command_for_job(job: Job, python_executable: str) -> List[str]:
    return [
        python_executable,
        str(TRAIN_ENTRYPOINT),
        "--config",
        str(job.config_path),
        "--target-env",
        job.target_env,
        "--seeds",
        str(job.seed),
    ]


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
    print(f"[OK]  {job.name} ({elapsed / 60.0:.1f} min)", flush=True)


def main() -> None:
    args = parse_args()
    default_seeds = SCREENING_SEEDS if args.stage == "screening" else CONFIRMATION_SEEDS
    seeds = tuple(args.seeds) if args.seeds is not None else default_seeds
    jobs = planned_jobs(args.experiments, args.targets, seeds)

    print(
        f"C3 recovery stage={args.stage}; experiments={args.experiments}; "
        f"targets={args.targets}; seeds={list(seeds)}",
        flush=True,
    )
    for index, job in enumerate(jobs, start=1):
        status = "run" if args.force or not is_complete(job) else "skip-complete"
        print(f"{index:02d}. [{status}] {job.name}", flush=True)
        if args.dry_run:
            print("    " + " ".join(command_for_job(job, args.python)), flush=True)

    if args.dry_run:
        return

    log_dir = REPO_ROOT / "outputs_tpp_c3_recovery" / "run_logs"
    run_count = 0
    skipped_count = 0
    for index, job in enumerate(jobs, start=1):
        print(f"\n[{index}/{len(jobs)}] {job.name}", flush=True)
        if is_complete(job) and not args.force:
            skipped_count += 1
            print("[SKIP] completed artifacts already exist", flush=True)
            continue
        run_job(job, args.python, log_dir)
        run_count += 1

    print(
        f"\nCompleted C3 matrix: {run_count} run, {skipped_count} skipped, "
        f"{len(jobs)} planned.",
        flush=True,
    )


if __name__ == "__main__":
    main()
