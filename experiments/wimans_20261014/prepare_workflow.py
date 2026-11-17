"""Durable CPU-only acquisition then candidate preparation, with no training."""
from __future__ import annotations

import os
import subprocess
import sys

from experiments.wimans_20261014.acquisition import DEFAULT_ROOT, REPO, now, write_json


def main():
    status_path = DEFAULT_ROOT / "workflow_status.json"
    record = {"pid": os.getpid(), "started_at_utc": now(), "stages": [],
              "training_started": False, "target_model_evaluation_permitted": False,
              "formal_input_freeze_allowed": False}
    for name, module in (("acquisition", "acquisition"), ("candidate_preparation", "prepare_inputs")):
        record.update(status="running", stage=name, updated_at_utc=now())
        write_json(status_path, record)
        env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
        result = subprocess.run([sys.executable, "-m", "experiments.wimans_20261014." + module], cwd=REPO, env=env)
        record["stages"].append({"stage": name, "exit_code": result.returncode, "finished_at_utc": now()})
        if result.returncode:
            record.update(status="failed", updated_at_utc=now())
            write_json(status_path, record)
            raise SystemExit(result.returncode)
    record.update(status="storage_order_inputs_prepared_source_qa_pending", updated_at_utc=now())
    write_json(status_path, record)


if __name__ == "__main__":
    main()
