import json
import os
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import (accuracy_score, classification_report, f1_score,
                             precision_score, recall_score)
from tqdm import tqdm


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_logits(model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    output = model(images)
    # model của HuggingFace trả về object có .logits
    return output.logits if hasattr(output, "logits") else output


def train_one_epoch(model: nn.Module, dataloader: DataLoader, loss_fn: nn.Module,
                    optimizer: torch.optim.Optimizer, device: str, epoch: int,
                    scaler: torch.amp.GradScaler | None = None) -> float:
    model.train()
    losses = []
    use_amp = scaler is not None
    with tqdm(dataloader, desc=f"Epoch {epoch} - Training") as pb:
        for item in pb:
            images = item["image"].to(device, non_blocking=True)
            labels = item["label"].to(device, non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                logits = get_logits(model, images)
                loss = loss_fn(logits, labels)
            losses.append(loss.item())

            optimizer.zero_grad()
            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            pb.set_postfix({"loss": float(np.mean(losses))})
    return float(np.mean(losses))


@torch.no_grad()
def predict(model: nn.Module, dataloader: DataLoader, device: str, desc: str = "Evaluating"):
    model.eval()
    predictions, trues = [], []
    for item in tqdm(dataloader, desc=desc):
        images = item["image"].to(device, non_blocking=True)
        logits = get_logits(model, images)
        predictions.extend(logits.argmax(dim=-1).cpu().tolist())
        trues.extend(item["label"].tolist())
    return trues, predictions


def compute_metrics(trues: list[int], predictions: list[int]) -> dict:
    return {
        "accuracy": accuracy_score(trues, predictions),
        "precision": precision_score(trues, predictions, average="macro", zero_division=0),
        "recall": recall_score(trues, predictions, average="macro", zero_division=0),
        "f1": f1_score(trues, predictions, average="macro", zero_division=0),
    }


def evaluate(model: nn.Module, dataloader: DataLoader, device: str, desc: str = "Evaluating") -> dict:
    trues, predictions = predict(model, dataloader, device, desc)
    return compute_metrics(trues, predictions)


def print_scores(scores: dict, prefix: str = ""):
    for name, value in scores.items():
        print(f"\t- {prefix}{name}: {value:.4f}")


def fit(model: nn.Module, train_dataloader: DataLoader, dev_dataloader: DataLoader,
        optimizer: torch.optim.Optimizer, device: str, epochs: int, patience: int,
        save_fn, best_score_name: str = "f1", use_amp: bool = True) -> list[dict]:
    loss_fn = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda") if (use_amp and device == "cuda") else None

    history = []
    best_score, bad_epochs = -1.0, 0
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_dataloader, loss_fn, optimizer, device, epoch, scaler)
        dev_scores = evaluate(model, dev_dataloader, device, desc=f"Epoch {epoch} - Dev")
        print(f"Epoch {epoch}: train loss = {train_loss:.4f}")
        print_scores(dev_scores, prefix="dev ")
        history.append({"epoch": epoch, "train_loss": train_loss, **{f"dev_{k}": v for k, v in dev_scores.items()}})

        # lưu model tốt nhất theo F1 trên dev, dừng sớm nếu không cải thiện sau `patience` epoch
        if dev_scores[best_score_name] > best_score:
            best_score, bad_epochs = dev_scores[best_score_name], 0
            save_fn()
            print(f"\t=> Lưu checkpoint mới (dev {best_score_name} = {best_score:.4f})")
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"Dừng sớm sau {epoch} epoch (dev {best_score_name} không cải thiện {patience} epoch).")
                break
    return history


def report_test(model: nn.Module, test_dataloader: DataLoader, device: str,
                class_names: list[str], output_dir: str, history: list[dict]) -> dict:
    trues, predictions = predict(model, test_dataloader, device, desc="Testing")
    scores = compute_metrics(trues, predictions)
    print("Kết quả trên tập test (checkpoint tốt nhất theo dev):")
    print_scores(scores)
    report = classification_report(
        trues, predictions, labels=list(range(len(class_names))),
        target_names=class_names, digits=4, zero_division=0,
    )
    print(report)

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump({"test": scores, "history": history}, f, ensure_ascii=False, indent=2)
    with open(os.path.join(output_dir, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(report)
    return scores
