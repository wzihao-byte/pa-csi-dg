"""Full amplitude-only development training with source-only checkpoint selection.

Example: python -m experiments.revision_validation.train_source --target E1
         --arm R00 --seed 401 --output outputs_revision/R00_E1_401/attempt_001

The target environment is only a fold name. This program has no target loader.
Each attempt gets a new directory; no historical H/Y checkpoint is overwritten.
"""
from __future__ import annotations

import argparse
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from dg_losses import DomainAwareSupConLoss
from experiments.revision_validation.models import ARMS, MODEL_CONFIG, build_model
from experiments.revision_validation.prepare_verified_protocol import ROOT, OUT, sha256, write
from experiments.revision_validation.source_data import VerifiedSources, epoch_schedule, schedule_hash


def score_source(model, dataset, device):
    model.eval()
    confusion = np.zeros((6, 6), dtype=np.int64)
    predictions = []
    with torch.no_grad():
        for start in range(0, len(dataset.manifest["val"]), 32):
            amplitude, labels, rows = dataset.batch("val", range(start, min(start + 32, len(dataset.manifest["val"]))), device)
            logits = model(amplitude, project=False)["logits"]
            if not torch.isfinite(logits).all():
                raise FloatingPointError("Nonfinite source-validation logits")
            predicted = logits.argmax(dim=1)
            actual, predicted = labels.cpu().numpy(), predicted.cpu().numpy()
            np.add.at(confusion, (actual, predicted), 1)
            predictions.extend({"sample_id": row["sample_id"], "subject_id": row["subject_id"],
                                "acquisition_id": row["acquisition_id"], "true": int(a), "predicted": int(p)}
                               for row, a, p in zip(rows, actual, predicted))
    tp = np.diag(confusion)
    denominator = confusion.sum(axis=0) + confusion.sum(axis=1)
    f1 = np.divide(2 * tp, denominator, out=np.zeros(6, dtype=float), where=denominator > 0)
    return {"macro_f1": float(f1.mean()), "class_f1": f1.tolist(),
            "accuracy": float(tp.sum() / confusion.sum()), "confusion_matrix": confusion.tolist()}, predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True, choices=["E1", "E2", "E3"])
    parser.add_argument("--arm", required=True, choices=list(ARMS))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--lambda-supcon", type=float, default=0.025)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.epochs < 1 or args.lambda_supcon not in (0.01, 0.025):
        raise ValueError("Use positive epochs and one of the declared source-search lambda candidates")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Use a fresh attempt directory; completed/failed attempts are retained")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    manifest_path = OUT / f"source_{args.target}.json"
    dataset = VerifiedSources(manifest_path)
    spec = ARMS[args.arm]
    lam = args.lambda_supcon if spec["supcon"] else 0.0
    config = {"stage": "full_source_development" if args.epochs == 200 else "short_source_pilot",
              "arm": args.arm, "seed": args.seed, "target_name_only": args.target,
              "source_manifest_path": str(manifest_path), "source_manifest_sha256": sha256(manifest_path),
              "opened_data_paths": dataset.opened_paths, "target_data_loaded": False,
              "model": MODEL_CONFIG, "modality": "amplitude_only", "phase_loaded": False,
              "input_processing": "Hash-bound historical 850-frame amplitude cast to float32",
              "epochs": args.epochs, "updates_per_epoch": 300, "batch_size": 16,
              "optimizer": "Adam", "learning_rate": 0.001, "weight_decay": 0.0,
              "scheduler": "Continuous exponential decay: initial_lr * 0.9 ** (completed_updates / 10000)",
              "lambda_supcon": lam, "temperature": 0.2, "warmup_steps": 6000,
              "checkpoint_metric": "source_validation_macro_f1", "tie_rule": "earliest",
              "validation_sample_count": 1200, "development_environments": True,
              "torch_version": torch.__version__, "device": args.device,
              "code_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in (
                  Path(__file__), Path(__file__).with_name("models.py"), Path(__file__).with_name("source_data.py"),
                  ROOT / "dg_models.py", ROOT / "dg_losses.py", ROOT / "experiments/tpp_isolated/tpp_models.py")}}
    write(args.output / "config.json", config)
    model = build_model(args.arm, args.seed).to(args.device)
    torch.manual_seed(args.seed + 17)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0)
    contrastive = DomainAwareSupConLoss(temperature=0.2, positive_mode="all_views")
    history, best, selected_epoch, step = [], -1.0, None, 0
    started = time.perf_counter()
    try:
        for epoch in range(args.epochs):
            model.train()
            schedule = epoch_schedule(dataset.manifest["train"], args.seed, epoch)
            losses = []
            for batch in schedule:
                step += 1
                amplitude, labels, _ = dataset.batch("train", batch, args.device)
                optimizer.zero_grad(set_to_none=True)
                output = model(amplitude, project=spec["supcon"])
                ce = F.cross_entropy(output["logits"], labels)
                sc = contrastive(output["projection"], labels) if spec["supcon"] else ce.new_zeros(())
                weight = lam * min(1.0, step / 6000)
                loss = ce + weight * sc
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite loss at step {step}")
                loss.backward()
                if not torch.stack([torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None]).all():
                    raise FloatingPointError(f"Nonfinite gradient at step {step}")
                optimizer.step()
                for group in optimizer.param_groups:
                    group["lr"] = 0.001 * 0.9 ** (step / 10000)
                losses.append([float(ce.detach()), float(sc.detach()), float(loss.detach())])
            metrics, predictions = score_source(model, dataset, args.device)
            row = {"epoch": epoch + 1, "optimizer_steps": step, "source_validation": metrics,
                   "mean_ce_supcon_total": np.mean(losses, axis=0).tolist(),
                   "schedule_sha256": schedule_hash(schedule), "lambda_at_epoch_end": weight,
                   "learning_rate_after_epoch": optimizer.param_groups[0]["lr"]}
            history.append(row)
            if metrics["macro_f1"] > best:
                best, selected_epoch = metrics["macro_f1"], epoch + 1
                checkpoint = args.output / "best_model.pt"
                temporary = args.output / "best_model.pt.tmp"
                torch.save(model.state_dict(), temporary)
                temporary.replace(checkpoint)
                write(args.output / "selected_source_predictions.json", predictions)
            write(args.output / "history.json", history)
            write(args.output / "status.json", {"status": "running", "epoch": epoch + 1,
                  "best_epoch": selected_epoch, "best_source_macro_f1": best, "target_data_loaded": False})
            print(f"{args.target} {args.arm} seed {args.seed}: epoch {epoch+1}/{args.epochs}, "
                  f"source macro-F1 {metrics['macro_f1']:.4f}, best {best:.4f}", flush=True)
        write(args.output / "status.json", {"status": "completed", "epochs": args.epochs,
              "optimizer_updates": step, "best_epoch": selected_epoch, "best_source_macro_f1": best,
              "elapsed_seconds": time.perf_counter() - started, "target_data_loaded": False,
              "best_model_sha256": sha256(args.output / "best_model.pt"),
              "history_sha256": sha256(args.output / "history.json"),
              "predictions_sha256": sha256(args.output / "selected_source_predictions.json"),
              "config_sha256": sha256(args.output / "config.json")})
    except Exception:
        write(args.output / "status.json", {"status": "failed", "optimizer_updates": step,
              "elapsed_seconds": time.perf_counter() - started, "error": traceback.format_exc(),
              "target_data_loaded": False})
        raise


if __name__ == "__main__":
    main()
