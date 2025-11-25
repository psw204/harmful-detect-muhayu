import os
import yaml
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import MultiModalDataset
from model import MultimodalClassifier


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def extract_threshold(cfg):
    """
    cfg["threshold"]가 dict이든 숫자든 안정적으로 float로 변환.
    """
    th_cfg = cfg.get("threshold", 0.5)

    # 1) threshold: dict (default, optimize 등)
    if isinstance(th_cfg, dict):
        return float(th_cfg.get("default", 0.5))

    # 2) threshold: float or str
    return float(th_cfg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)

    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    print(f"[Device] Using: {device}")

    # ===== Dataset =====
    train_set = MultiModalDataset(cfg["train_manifest"], cfg, split="train")
    val_set = MultiModalDataset(cfg["val_manifest"], cfg, split="val")

    train_loader = DataLoader(
        train_set,
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
        shuffle=False,
    )

    # ===== Model =====
    model = MultimodalClassifier(cfg).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"]
    )

    criterion = nn.BCEWithLogitsLoss()

    os.makedirs(cfg["save_dir"], exist_ok=True)

    # ===== Extract Threshold Safely =====
    threshold = extract_threshold(cfg)
    print(f"[Threshold] Using threshold = {threshold}")

    # ===== TRAIN =====
    for epoch in range(cfg["epochs"]):
        print(f"\n===== Epoch {epoch+1}/{cfg['epochs']} =====")

        model.train()
        train_losses = []

        pbar = tqdm(train_loader)
        for batch in pbar:

            # Move to GPU
            batch["video"] = batch["video"].to(device)
            batch["audio"] = batch["audio"].to(device)
            batch["label"] = batch["label"].to(device)

            for tk, tv in batch["text"].items():
                batch["text"][tk] = tv.to(device)

            # Forward
            logits = model(batch)
            labels = batch["label"]

            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            pbar.set_description(f"loss={loss.item():.4f}")

        print(f"[Train] loss={np.mean(train_losses):.4f}")

        # ===== VALIDATION =====
        model.eval()
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(val_loader):

                batch["video"] = batch["video"].to(device)
                batch["audio"] = batch["audio"].to(device)
                batch["label"] = batch["label"].to(device)

                for tk, tv in batch["text"].items():
                    batch["text"][tk] = tv.to(device)

                logits = model(batch)
                probs = torch.sigmoid(logits).cpu().numpy()
                labels = batch["label"].cpu().numpy()

                all_probs.extend(probs.tolist())
                all_labels.extend(labels.tolist())

        all_probs = np.array(all_probs, dtype=float)
        all_labels = np.array(all_labels, dtype=int)

        pred_bin = (all_probs >= threshold).astype(int)

        # ===== ACC =====
        acc = (pred_bin == all_labels).mean()

        # ===== Precision, Recall, F1 =====
        tp = np.sum((pred_bin == 1) & (all_labels == 1))
        fp = np.sum((pred_bin == 1) & (all_labels == 0))
        fn = np.sum((pred_bin == 0) & (all_labels == 1))

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        print(f"[Val] ACC={acc:.4f}  P={precision:.4f}  R={recall:.4f}  F1={f1:.4f}")

        # ===== SAVE CHECKPOINT =====
        ckpt = {
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }

        save_path = os.path.join(cfg["save_dir"], f"epoch{epoch+1}.pt")
        torch.save(ckpt, save_path)
        print(f"Saved → {save_path}")


if __name__ == "__main__":
    main()
