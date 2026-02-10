import argparse
import logging
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.benchmark_dataset import (
    EmbeddingEncoder,
    BenchmarkDataset,
    split_train_val_with_cdrh3_constraint,
)
from model.model import AffinityPredictor


def save_checkpoint(state, is_best, checkpoint_dir, filename="checkpoint.pt"):
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, filename)
    torch.save(state, path)
    if is_best:
        best_path = os.path.join(checkpoint_dir, "model_best.pt")
        torch.save(state, best_path)


def _safe_auc_metrics(all_labels, all_probs):
    unique_labels = set(all_labels)
    if len(unique_labels) < 2:
        return float("nan"), float("nan")

    roc_auc = roc_auc_score(all_labels, all_probs)
    pr_auc = average_precision_score(all_labels, all_probs)
    return roc_auc, pr_auc


def _binary_classification_metrics(all_labels, all_preds):
    tp = sum(1 for y, p in zip(all_labels, all_preds) if y == 1 and p == 1)
    tn = sum(1 for y, p in zip(all_labels, all_preds) if y == 0 and p == 0)
    fp = sum(1 for y, p in zip(all_labels, all_preds) if y == 0 and p == 1)
    fn = sum(1 for y, p in zip(all_labels, all_preds) if y == 1 and p == 0)
    total = tp + tn + fp + fn

    accuracy = 100.0 * (tp + tn) / max(total, 1)
    precision = 100.0 * tp / max(tp + fp, 1)
    recall = 100.0 * tp / max(tp + fn, 1)
    specificity = 100.0 * tn / max(tn + fp, 1)
    f1 = 0.0
    if precision + recall > 0:
        f1 = 2.0 * precision * recall / (precision + recall)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "num_samples": total,
    }


def validate(model, dataloader, criterion, device, step):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    all_labels = []
    all_probs = []
    all_preds = []

    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f"[Val @ step {step}]")
        for batch in pbar:
            heavy = batch["heavy"].to(device)
            light = batch["light"].to(device)
            antigen = batch["antigen"].to(device)
            label = batch["label"].to(device)

            logits = model(heavy, light, antigen)
            loss = criterion(logits, label)
            running_loss += loss.item()

            probs = F.softmax(logits, dim=1)[:, 1]
            _, predicted = logits.max(1)

            all_labels.extend(label.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())
            all_preds.extend(predicted.cpu().numpy().tolist())

            total += label.size(0)
            correct += predicted.eq(label).sum().item()
            pbar.set_postfix({"loss": running_loss / (pbar.n + 1), "acc": 100.0 * correct / max(total, 1)})

    roc_auc, pr_auc = _safe_auc_metrics(all_labels, all_probs)
    cls_metrics = _binary_classification_metrics(all_labels, all_preds)
    cls_metrics["roc_auc"] = roc_auc
    cls_metrics["pr_auc"] = pr_auc
    cls_metrics["loss"] = running_loss / len(dataloader)
    return cls_metrics


def setup_logger(log_path):
    log_dir = os.path.dirname(log_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger("benchmark_train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def build_column_map(args):
    column_map = {}
    if args.heavy_col:
        column_map["heavy"] = args.heavy_col
    if args.light_col:
        column_map["light"] = args.light_col
    if args.antigen_col:
        column_map["antigen"] = args.antigen_col
    column_map["label"] = args.label_col
    return column_map or None


def infinite_loader(dataloader):
    while True:
        for batch in dataloader:
            yield batch


def train_with_fixed_steps(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    train_steps,
    eval_every_steps,
    log_every_steps,
    logger,
    checkpoint_dir,
    model_name,
):
    if train_steps < 1:
        raise ValueError("train_steps must be >= 1")
    if eval_every_steps < 1:
        raise ValueError("eval_every_steps must be >= 1")
    if log_every_steps < 1:
        raise ValueError("log_every_steps must be >= 1")

    step_iter = infinite_loader(train_loader)
    best_val_acc = -1.0
    best_metrics = None
    running_loss = 0.0
    running_correct = 0
    running_total = 0

    model.train()
    for step in range(1, train_steps + 1):
        batch = next(step_iter)
        heavy = batch["heavy"].to(device)
        light = batch["light"].to(device)
        antigen = batch["antigen"].to(device)
        label = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(heavy, light, antigen)
        loss = criterion(logits, label)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = logits.max(1)
        running_total += label.size(0)
        running_correct += predicted.eq(label).sum().item()

        if step % log_every_steps == 0 or step == 1:
            logger.info(
                "Step=%d/%d TrainLoss=%.4f TrainAcc=%.2f%%",
                step,
                train_steps,
                running_loss / max(step, 1),
                100.0 * running_correct / max(running_total, 1),
            )

        should_eval = (step % eval_every_steps == 0) or (step == train_steps)
        if should_eval:
            val_metrics = validate(model, val_loader, criterion, device, step)
            val_acc = val_metrics["accuracy"]
            is_best = val_acc > best_val_acc
            if is_best:
                best_val_acc = val_acc
                best_metrics = dict(val_metrics)
                best_metrics["best_step"] = step

            save_checkpoint(
                {
                    "model_name": model_name,
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_acc": best_val_acc,
                    "val_metrics": val_metrics,
                },
                is_best,
                checkpoint_dir,
                filename=f"checkpoint_step_{step}.pt",
            )

            logger.info(
                "Eval@Step=%d ValLoss=%.4f ValAcc=%.2f%% ValPrecision=%.2f%% ValRecall=%.2f%% "
                "ValF1=%.2f%% ValSpecificity=%.2f%% ROC-AUC=%.4f PR-AUC=%.4f N=%d",
                step,
                val_metrics["loss"],
                val_metrics["accuracy"],
                val_metrics["precision"],
                val_metrics["recall"],
                val_metrics["f1"],
                val_metrics["specificity"],
                val_metrics["roc_auc"],
                val_metrics["pr_auc"],
                val_metrics["num_samples"],
            )
            model.train()

    return best_metrics


def main(args):
    logger = setup_logger(args.log_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)
    logger.info("Model name: %s", args.model_name)
    logger.info("Split method: %s", args.split_method)
    logger.info("Train steps: %d", args.train_steps)

    encoder = EmbeddingEncoder(
        model_path=args.model_path,
        cache_dir=args.cache_dir,
        device=device,
        max_length=args.max_length,
        cache_prefix=args.model_name,
    )
    column_map = build_column_map(args)

    train_df, val_df, split_stats = split_train_val_with_cdrh3_constraint(
        data_source=args.data_path,
        method=args.split_method,
        val_ratio=args.val_ratio,
        cdrh3_col=args.cdrh3_col,
        random_state=args.random_state,
        similarity_threshold=args.similarity_threshold,
        min_diff_k=args.min_diff_k,
    )
    logger.info("Split stats: %s", split_stats)

    train_dataset = BenchmarkDataset(train_df, encoder, column_map=column_map)
    val_dataset = BenchmarkDataset(val_df, encoder, column_map=column_map)
    if len(train_dataset) == 0:
        raise ValueError(
            "Train set is empty after split filtering. Lower min_diff_k or use a less strict split configuration."
        )
    if len(val_dataset) == 0:
        raise ValueError("Validation set is empty after splitting. Increase data size or val_ratio.")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

    model = AffinityPredictor(hidden_size=args.hidden_size, seq_len=args.max_length).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    checkpoint_dir = os.path.join(args.checkpoint_dir, args.model_name)
    best_metrics = train_with_fixed_steps(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        train_steps=args.train_steps,
        eval_every_steps=args.eval_every_steps,
        log_every_steps=args.log_every_steps,
        logger=logger,
        checkpoint_dir=checkpoint_dir,
        model_name=args.model_name,
    )

    if best_metrics is None:
        logger.warning("No evaluation metrics were recorded.")
        return

    logger.info(
        "Best Val Metrics @ step=%d | Acc=%.2f%% Precision=%.2f%% Recall=%.2f%% F1=%.2f%% "
        "Specificity=%.2f%% ROC-AUC=%.4f PR-AUC=%.4f N=%d",
        best_metrics["best_step"],
        best_metrics["accuracy"],
        best_metrics["precision"],
        best_metrics["recall"],
        best_metrics["f1"],
        best_metrics["specificity"],
        best_metrics["roc_auc"],
        best_metrics["pr_auc"],
        best_metrics["num_samples"],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark training with CDRH3-constrained train/val splitting")
    parser.add_argument("--data_path", type=str, default="./data/virAbBench.csv", help="CSV path for training/validation")
    parser.add_argument("--model_path", type=str, default="/mnt/data/home/majiahao/LucaBCRTasks/huggingface/lucabcr_hf_model", help="Path to HF model")
    parser.add_argument("--cache_dir", type=str, default="./cache", help="Directory for embedding cache")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--log_path", type=str, default="./logs/train.log", help="Path to training log file")
    parser.add_argument("--model_name", type=str, required=True, help="Model name used for logs/checkpoints/cache prefix")

    parser.add_argument("--max_length", type=int, default=1500, help="Max sequence length")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--train_steps", type=int, default=4000, help="Fixed number of optimization steps")
    parser.add_argument("--eval_every_steps", type=int, default=200, help="Validation interval in steps")
    parser.add_argument("--log_every_steps", type=int, default=20, help="Training log interval in steps")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--hidden_size", type=int, default=1536, help="Hidden size for predictor")

    parser.add_argument("--split_method", type=str, default="min_diff_k", choices=["similarity", "min_diff_k"])
    parser.add_argument("--val_ratio", type=float, default=0.05, help="Validation split ratio")
    parser.add_argument("--cdrh3_col", type=str, default=None, help="CDRH3 column name")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed for split")
    parser.add_argument("--similarity_threshold", type=float, default=0.8, help="CDRH3 similarity threshold")
    parser.add_argument("--min_diff_k", type=int, default=10, help="Minimum CDRH3 position differences")

    parser.add_argument("--heavy_col", type=str, default=None, help="Heavy chain column name")
    parser.add_argument("--light_col", type=str, default=None, help="Light chain column name")
    parser.add_argument("--antigen_col", type=str, default=None, help="Antigen column name")
    parser.add_argument("--label_col", type=str, default="bind", help="Label column name")

    args = parser.parse_args()
    main(args)
