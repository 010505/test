from __future__ import annotations

import argparse
import json
import math
import platform
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .progressive import (
    DEFAULT_OBSERVATION_RATIOS,
    ClassDiffusionModel,
    PrefixSequenceDataset,
    load_raw_shrec17_npz,
    stratified_raw_split,
)
from .progressive_benchmark import (
    calibrate_early_exit,
    choose_device,
    decision_metrics,
    error_recovery_metrics,
    evaluate_prefixes,
    measure_online_latency,
    set_reproducible,
)


BASE_BETAS = (0.25, 0.45, 0.65, 0.90)
SUPPORTED_DEPTHS = (1, 2, 4, 8)


def resample_betas(depth: int, base_betas=BASE_BETAS) -> tuple[float, ...]:
    """Resample the four-step schedule while preserving its terminal noise.

    One/two steps merge adjacent transitions. Eight steps split every original
    transition into two equal-retention transitions. This keeps
    prod(1 - beta) identical for every depth.
    """
    if depth == 4:
        return tuple(float(beta) for beta in base_betas)
    if depth == 1:
        retention = math.prod(1.0 - float(beta) for beta in base_betas)
        return (1.0 - retention,)
    if depth == 2:
        return tuple(
            1.0 - math.prod(1.0 - float(beta) for beta in pair)
            for pair in (base_betas[:2], base_betas[2:])
        )
    if depth == 8:
        result = []
        for beta in base_betas:
            split_beta = 1.0 - math.sqrt(1.0 - float(beta))
            result.extend((split_beta, split_beta))
        return tuple(result)
    raise ValueError(f"depth must be one of {SUPPORTED_DEPTHS}, got {depth}")


def stats(values):
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
    }


def make_loaders(args, seed, labels, official_train, official_test):
    train_samples, validation_samples = stratified_raw_split(
        official_train, args.val_ratio, seed
    )
    ratios = tuple(args.observation_ratios)
    train_loader = DataLoader(
        PrefixSequenceDataset(
            train_samples, labels, args.frames, ratios, augment=True
        ),
        batch_size=args.sequence_batch_size,
        shuffle=True,
        drop_last=True,
    )
    validation_loader = DataLoader(
        PrefixSequenceDataset(validation_samples, labels, args.frames, ratios),
        batch_size=args.eval_batch_size,
    )
    test_loader = DataLoader(
        PrefixSequenceDataset(official_test, labels, args.frames, ratios),
        batch_size=args.eval_batch_size,
    )
    return train_loader, validation_loader, test_loader


def evaluate_and_save(
    args, model, output, seed, depth, betas, labels, validation_loader, test_loader, device,
    history, best_epoch, reused_checkpoint=None,
):
    ratios = tuple(args.observation_ratios)
    validation = evaluate_prefixes(
        model, validation_loader, device, ratios, collect=True
    )
    calibration = calibrate_early_exit(validation, ratios)
    test = evaluate_prefixes(model, test_loader, device, ratios, collect=True)
    decision = decision_metrics(
        test["probabilities"], test["targets"], ratios,
        calibration["confidence_threshold"], calibration["margin_threshold"],
    )
    recovery = error_recovery_metrics(
        test["probabilities"], test["targets"], ratios
    )
    latency = measure_online_latency(
        model, test_loader, device, iterations=args.latency_iterations
    )
    validation.pop("probabilities")
    validation.pop("targets")
    test.pop("probabilities")
    test.pop("targets")
    result = {
        "experiment": f"class_diffusion_depth_{depth}",
        "seed": seed,
        "depth": depth,
        "betas": list(betas),
        "terminal_clean_retention": float(math.prod(1.0 - beta for beta in betas)),
        "terminal_uniform_weight": float(1.0 - math.prod(1.0 - beta for beta in betas)),
        "epochs": args.epochs,
        "best_epoch": best_epoch,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "selection_metric": "validation_prefix_auc",
        "validation": validation,
        "official_test": test,
        "early_exit_calibration": calibration,
        "official_test_early_exit": decision,
        "official_test_error_recovery": recovery,
        "online_latency": latency,
        "device": str(device),
        "python_version": platform.python_version(),
        "reused_checkpoint": str(reused_checkpoint) if reused_checkpoint else None,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    (output / "model.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def train_depth(
    args, seed, depth, labels, official_train, official_test, device,
):
    output = Path(args.output) / f"depth_{depth}" / f"seed_{seed}"
    result_path = output / "model.json"
    if result_path.exists() and not args.force:
        print(f"SKIP completed depth={depth} seed={seed}: {result_path}", flush=True)
        return json.loads(result_path.read_text(encoding="utf-8"))

    set_reproducible(seed)
    train_loader, validation_loader, test_loader = make_loaders(
        args, seed, labels, official_train, official_test
    )
    betas = resample_betas(depth)
    model = ClassDiffusionModel(
        num_classes=len(labels), dropout=args.dropout, betas=betas
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    accumulation = max(1, math.ceil(args.batch_size / args.sequence_batch_size))
    history = []
    start_epoch = 1
    best_score = -1.0
    best_epoch = 0
    last_path = output / "last.pt"
    output.mkdir(parents=True, exist_ok=True)
    if last_path.exists() and not args.force:
        checkpoint = torch.load(last_path, map_location=device, weights_only=True)
        if checkpoint.get("betas") == list(betas) and checkpoint.get("seed") == seed:
            model.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            history = checkpoint["history"]
            start_epoch = int(checkpoint["epoch"]) + 1
            best_score = float(checkpoint["best_score"])
            best_epoch = int(checkpoint["best_epoch"])
            print(
                f"RESUME depth={depth} seed={seed} at epoch {start_epoch}", flush=True
            )

    print(
        f"START depth={depth} seed={seed} betas={tuple(round(x, 6) for x in betas)} "
        f"params={sum(p.numel() for p in model.parameters()):,}", flush=True
    )
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        seen = 0
        for batch_index, (views, lengths, progress, targets) in enumerate(train_loader):
            views = views.to(device)
            lengths = lengths.to(device)
            progress = progress.to(device)
            targets = targets.to(device)
            log_probabilities, conditions = model(
                views, lengths, progress, return_auxiliary=True
            )
            primary = F.nll_loss(
                log_probabilities.reshape(-1, len(labels)),
                targets[:, None].expand(-1, log_probabilities.shape[1]).reshape(-1),
            )
            denoising = model.denoising_loss(conditions, targets)
            loss = primary + 0.5 * denoising
            (loss / accumulation).backward()
            if (batch_index + 1) % accumulation == 0 or batch_index + 1 == len(train_loader):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            loss_sum += float(loss.detach()) * len(targets)
            seen += len(targets)
        scheduler.step()
        validation = evaluate_prefixes(
            model, validation_loader, device, tuple(args.observation_ratios)
        )
        row = {
            "epoch": epoch,
            "train_loss": loss_sum / max(seen, 1),
            "val_prefix_auc": validation["prefix_auc"],
            "val_full_accuracy": validation["full_accuracy"],
            "val_ratio_accuracy": validation["ratio_accuracy"],
        }
        history.append(row)
        if row["val_prefix_auc"] >= best_score:
            best_score = row["val_prefix_auc"]
            best_epoch = epoch
            torch.save({
                "model_state": model.state_dict(),
                "experiment": f"class_diffusion_depth_{depth}",
                "labels": labels,
                "frames": args.frames,
                "observation_ratios": tuple(args.observation_ratios),
                "betas": list(betas),
                "depth": depth,
                "seed": seed,
                "epochs": args.epochs,
                "best_epoch": best_epoch,
                "best_validation_score": best_score,
            }, output / "best.pt")
        torch.save({
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "history": history,
            "epoch": epoch,
            "best_score": best_score,
            "best_epoch": best_epoch,
            "betas": list(betas),
            "seed": seed,
        }, last_path)
        print(
            f"depth={depth} seed={seed} epoch={epoch:02d}/{args.epochs} "
            f"loss={row['train_loss']:.4f} valAUC={row['val_prefix_auc']:.2%} "
            f"full={row['val_full_accuracy']:.2%} best={best_score:.2%}@{best_epoch}",
            flush=True,
        )

    checkpoint = torch.load(output / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    return evaluate_and_save(
        args, model, output, seed, depth, betas, labels,
        validation_loader, test_loader, device, history, best_epoch,
    )


def reuse_four_step(args, seed, labels, official_train, official_test, device):
    source = Path(args.four_step_runs) / f"progressive_seed{seed}" / "04_class_diffusion" / "best.pt"
    if not source.exists():
        return None
    output = Path(args.output) / "depth_4" / f"seed_{seed}"
    result_path = output / "model.json"
    if result_path.exists() and not args.force:
        return json.loads(result_path.read_text(encoding="utf-8"))
    _, validation_loader, test_loader = make_loaders(
        args, seed, labels, official_train, official_test
    )
    checkpoint = torch.load(source, map_location=device, weights_only=True)
    model = ClassDiffusionModel(
        num_classes=len(labels), dropout=args.dropout, betas=BASE_BETAS
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    history_path = source.parent / "history.json"
    history = json.loads(history_path.read_text(encoding="utf-8"))
    best_epoch = max(history, key=lambda row: row["val_prefix_auc"])["epoch"]
    output.mkdir(parents=True, exist_ok=True)
    torch.save({**checkpoint, "betas": list(BASE_BETAS), "depth": 4}, output / "best.pt")
    print(f"REUSE four-step checkpoint seed={seed}: {source}", flush=True)
    return evaluate_and_save(
        args, model, output, seed, 4, BASE_BETAS, labels,
        validation_loader, test_loader, device, history, best_epoch, source,
    )


def aggregate_results(args):
    rows = {}
    for depth in args.depths:
        experiments = []
        for seed in args.seeds:
            path = Path(args.output) / f"depth_{depth}" / f"seed_{seed}" / "model.json"
            if path.exists():
                experiments.append(json.loads(path.read_text(encoding="utf-8")))
        if not experiments:
            continue
        ratios = list(experiments[0]["official_test"]["ratio_accuracy"])
        rows[str(depth)] = {
            "depth": depth,
            "seeds": [row["seed"] for row in experiments],
            "betas": experiments[0]["betas"],
            "terminal_uniform_weight": experiments[0]["terminal_uniform_weight"],
            "parameters": experiments[0]["parameters"],
            "ratio_accuracy": {
                ratio: stats([row["official_test"]["ratio_accuracy"][ratio] for row in experiments])
                for ratio in ratios
            },
            "prefix_auc": stats([row["official_test"]["prefix_auc"] for row in experiments]),
            "full_accuracy": stats([row["official_test"]["full_accuracy"] for row in experiments]),
            "decision_accuracy": stats([row["official_test_early_exit"]["accuracy"] for row in experiments]),
            "average_decision_ratio": stats([row["official_test_early_exit"]["average_decision_ratio"] for row in experiments]),
            "final_recovery_rate": stats([row["official_test_error_recovery"]["final_recovery_rate"] for row in experiments]),
            "online_mean_update_ms": stats([row["online_latency"]["mean_update_ms"] for row in experiments]),
        }
    output = Path(args.output)
    (output / "aggregate.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    lines = [
        "# Independently trained class-diffusion depth ablation",
        "",
        "All depths use the same AGCRN encoder, splits, 40-epoch protocol, losses, and "
        "98.55625% terminal uniform weight. Only the number of categorical diffusion steps changes.",
        "",
        "| Steps | Seeds | 25% | 50% | 65% | 80% | 100% | Prefix AUC | Decision acc. | ADR | Recovery | Online/update |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for depth in args.depths:
        row = rows.get(str(depth))
        if not row:
            continue
        ratio_values = [row["ratio_accuracy"][key]["mean"] for key in row["ratio_accuracy"]]
        lines.append(
            f"| {depth} | {','.join(map(str, row['seeds']))} | "
            + " | ".join(f"{value:.2%}" for value in ratio_values)
            + f" | {row['prefix_auc']['mean']:.2%} +/- {row['prefix_auc']['std']:.2%}"
            + f" | {row['decision_accuracy']['mean']:.2%}"
            + f" | {row['average_decision_ratio']['mean']:.3f}"
            + f" | {row['final_recovery_rate']['mean']:.2%}"
            + f" | {row['online_mean_update_ms']['mean']:.3f} ms |"
        )
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


def main():
    parser = argparse.ArgumentParser(description="Independent 1/2/4/8-step class-diffusion ablation")
    parser.add_argument("--data", default="data/shrec17_ddnet_npz")
    parser.add_argument("--output", default="runs/diffusion_depth_ablation")
    parser.add_argument("--depths", type=int, nargs="+", default=list(SUPPORTED_DEPTHS), choices=SUPPORTED_DEPTHS)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--sequence-batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--frames", type=int, default=64)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--latency-iterations", type=int, default=100)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--observation-ratios", type=float, nargs="+", default=list(DEFAULT_OBSERVATION_RATIOS))
    parser.add_argument("--four-step-runs", default="runs")
    parser.add_argument("--retrain-four-step", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    ratios = tuple(args.observation_ratios)
    if sorted(ratios) != list(ratios) or ratios[-1] != 1.0:
        raise ValueError("observation ratios must be sorted and end at 1.0")
    device = choose_device(args.device)
    official_train = load_raw_shrec17_npz(args.data, "train")
    official_test = load_raw_shrec17_npz(args.data, "test")
    labels = sorted({sample.label for sample in official_train + official_test})
    print(
        f"DATA train={len(official_train)} official_test={len(official_test)} "
        f"classes={len(labels)} device={device}", flush=True
    )
    for depth in args.depths:
        for seed in args.seeds:
            if depth == 4 and not args.retrain_four_step:
                reused = reuse_four_step(
                    args, seed, labels, official_train, official_test, device
                )
                if reused is not None:
                    aggregate_results(args)
                    continue
            train_depth(
                args, seed, depth, labels, official_train, official_test, device
            )
            aggregate_results(args)
    aggregate_results(args)


if __name__ == "__main__":
    main()
