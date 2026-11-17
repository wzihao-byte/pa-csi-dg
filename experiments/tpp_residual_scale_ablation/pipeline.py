"""Run the locked residual-scale matrix, then its prewritten analysis."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
OUTPUT_ROOT = REPO_ROOT / "outputs_tpp_residual_scale_ablation" / "h11_l1248_v1"


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_ROOT / "pipeline.stdout.log").open("ab") as stdout, (
        OUTPUT_ROOT / "pipeline.stderr.log"
    ).open("ab") as stderr:
        subprocess.run(
            [sys.executable, str(HERE / "run_scale.py"), "--execute"],
            cwd=HERE,
            stdout=stdout,
            stderr=stderr,
            check=True,
        )
        subprocess.run(
            [sys.executable, str(HERE / "analyze_scale.py")],
            cwd=HERE,
            stdout=stdout,
            stderr=stderr,
            check=True,
        )


if __name__ == "__main__":
    main()
