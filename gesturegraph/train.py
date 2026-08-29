from __future__ import annotations

import argparse
import json
import platform
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .backbones import EXPERIMENTAL_MODEL_NAMES
from .data import GestureDataset, discover_samples, stratified_split
from .model import build_model
from .shrec import load_shrec17, load_shrec17_npz


def set_reproducible(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def evaluate(model, loader, criterion, device, num_classes):
    model.eval(); total_loss = correct = total = 0
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs); total_loss += criterion(logits, targets).item() * len(targets)
            predictions = logits.argmax(dim=1)
            correct += (predictions == targets).sum().item(); total += len(targets)
            for truth, prediction in zip(targets.cpu().tolist(), predictions.cpu().tolist()):
                confusion[truth, prediction] += 1
    return total_loss / max(total, 1), correct / max(total, 1), confusion


def train(args):
    set_reproducible(args.seed)
    if args.dataset in {"shrec17", "shrec17_npz"}:
        loader = load_shrec17 if args.dataset == "shrec17" else load_shrec17_npz
        official_train = loader(args.data, "train", args.frames, args.classes)
        test_samples = loader(args.data, "test", args.frames, args.classes)
        train_samples, val_samples = stratified_split(official_train, args.val_ratio, args.seed)
        samples = official_train + test_samples
    else:
        samples = discover_samples(args.data, args.frames)
        train_samples, val_samples = stratified_split(samples, args.val_ratio, args.seed)
        test_samples = []
    labels = sorted({sample.label for sample in samples})
    if len(labels) < 2:
        raise ValueError("training requires at least two gesture labels")
    if not val_samples:
        raise ValueError("validation split is empty; record at least two samples per label")
    print("Dataset:", dict(sorted(Counter(sample.label for sample in samples).items())))
    print(f"Split: {len(train_samples)} train / {len(val_samples)} validation / {len(test_samples)} official test")

    train_loader = DataLoader(GestureDataset(train_samples, labels, augment=True), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(GestureDataset(val_samples, labels), batch_size=args.batch_size)
    test_loader = DataLoader(GestureDataset(test_samples, labels), batch_size=args.batch_size) if test_samples else None
    device = choose_device(args.device); print("Device:", device)
    model_config = {
        "pe_dim": args.pe_dim,
        "attention_heads": args.attention_heads,
        "adaptive_dim": args.adaptive_dim,
        "spectral_weighting": "laplacian_eigenvalue",
    }
    model = build_model(args.model, len(labels), args.frames, args.dropout, args.ablation, model_config).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"Model: {args.model} | parameters: {parameter_count:,} | config: {model_config}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    criterion = nn.CrossEntropyLoss(label_smoothing=.05)

    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    best_accuracy = -1.0; history = []
    for epoch in range(1, args.epochs + 1):
        model.train(); loss_sum = correct = total = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs); loss = criterion(logits, targets)
            loss.backward(); optimizer.step()
            loss_sum += loss.item() * len(targets); correct += (logits.argmax(1) == targets).sum().item(); total += len(targets)
        scheduler.step()
        val_loss, val_accuracy, confusion = evaluate(model, val_loader, criterion, device, len(labels))
        row = {"epoch": epoch, "train_loss": loss_sum/max(total,1), "train_accuracy": correct/max(total,1), "val_loss": val_loss, "val_accuracy": val_accuracy}
        history.append(row)
        print(f"Epoch {epoch:03d} | train {row['train_accuracy']:.1%} | val {val_accuracy:.1%} | loss {val_loss:.4f}")
        if val_accuracy >= best_accuracy:
            best_accuracy = val_accuracy
            checkpoint = {
                "model_state": model.state_dict(),
                "labels": labels,
                "frames": args.frames,
                "best_accuracy": best_accuracy,
                "model_name": args.model,
                "ablation": args.ablation,
                "dropout": args.dropout,
                "model_config": model_config,
            }
            torch.save(checkpoint, output / "best.pt")
            (output / "confusion.json").write_text(json.dumps({"labels": labels, "matrix": confusion.tolist()}, indent=2), encoding="utf-8")

    best = torch.load(output / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(best["model_state"])
    test_accuracy = None
    if test_loader:
        test_loss, test_accuracy, test_confusion = evaluate(model, test_loader, criterion, device, len(labels))
        (output / "test_confusion.json").write_text(json.dumps({"labels": labels, "matrix": test_confusion.tolist()}, indent=2), encoding="utf-8")
        print(f"Official test | accuracy {test_accuracy:.1%} | loss {test_loss:.4f}")
    if hasattr(model, "adjacency_components"):
        components = model.adjacency_components()
        np.savez(
            output / "adjacency_matrices.npz",
            **{name: value.numpy() for name, value in components.items()},
        )
    (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    metadata = {
        "architecture": args.model,
        "ablation": args.ablation,
        "dataset": args.dataset,
        "labels": labels,
        "frames": args.frames,
        "best_validation_accuracy": best_accuracy,
        "official_test_accuracy": test_accuracy,
        "samples": len(samples),
        "parameters": parameter_count,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "seed": args.seed,
        "device": str(device),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "deterministic_cudnn": bool(device.type == "cuda" and torch.backends.cudnn.deterministic),
        "model_config": model_config,
    }
    (output / "model.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Best model: {output/'best.pt'} ({best_accuracy:.1%})")


def build_parser():
    parser = argparse.ArgumentParser(description="Train a 22-node hand ST-GCN")
    parser.add_argument("--data", default="data/recordings")
    parser.add_argument("--dataset", default="recordings", choices=["recordings", "shrec17", "shrec17_npz"])
    parser.add_argument("--classes", type=int, default=14, choices=[14, 28])
    parser.add_argument("--model", default="stgcn", choices=["stgcn", "mlp", *EXPERIMENTAL_MODEL_NAMES])
    parser.add_argument("--ablation", default="none", choices=["none", "no_graph", "single_frame"])
    parser.add_argument("--output", default="runs/stgcn")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--frames", type=int, default=64)
    parser.add_argument("--val-ratio", type=float, default=.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=.15)
    parser.add_argument("--pe-dim", type=int, default=8)
    parser.add_argument("--attention-heads", "--gat-heads", dest="attention_heads", type=int, default=4)
    parser.add_argument("--adaptive-dim", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    return parser


if __name__ == "__main__":
    train(build_parser().parse_args())
