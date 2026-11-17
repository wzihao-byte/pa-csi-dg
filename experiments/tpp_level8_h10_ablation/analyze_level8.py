"""Run the sealed Level-8 paired analysis for the H10 output root."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
SOURCE = REPO_ROOT / "experiments" / "tpp_level8_ablation" / "analyze_level8.py"


def main() -> None:
    spec = importlib.util.spec_from_file_location("level8_h10_analysis_base", SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load analysis: {SOURCE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.BUNDLE = HERE / "generated_l1248_h10_v1"
    module.OUTPUT = REPO_ROOT / "outputs_tpp_level8_ablation" / "l1248_h10_v1"
    module.RESULTS = module.OUTPUT / "level8_results.json"
    module.ANALYSIS_DIR = module.OUTPUT / "analysis"
    module.main()


if __name__ == "__main__":
    main()

