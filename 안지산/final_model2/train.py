# -*- coding: utf-8 -*-
import argparse
import os
import yaml

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModel

from dataset_avt import AVTDataset
from model_avt import AVTModel
from losses import bce_logits, info_nce

from sklearn.metrics import f1_score, accuracy_score, roc_auc_score


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="config.yaml")
    return ap.parse_args()


@torch.no_grad()
def extract_text_cls(txt_model, input_ids, attention_mask, device):
    """
    KoBERT from transformers:
    반환: CLS 임베딩 (B,768)
    """
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    outputs = txt_model(input_ids=input_ids, attention_mask=attention_mask)
    # last_hidden_state: (B,L,H)
    cls_emb = outputs.last_hidden_state[:, 0, :]  # (B,H)
    return cls_emb


@torch.no_grad()
def evaluate(model, loader, txt_model, threshold, device):
    model.eval()
    txt_model.eval()

    all_labels = []
    all_probs = []

    for vid, wav, ids, mask, y in loader:
        vid = vid.to(device)           # (B,T,C,H,W)
        wav = wav.to(device)           # (B,1,N)
        y = y.to(device)               # (B,)

        cls_emb = extract_text_cls(txt_model, ids, mask, device)  # (B,768)

        logits, _, _ = model(vid, wav, cls_emb)
        probs = torch.sigmoid(logits)

        all_labels.extend(y.detach().cpu().tolist())
        all_probs.extend(probs.detach().cpu().tolist())

    all_labels = torch.tensor(all_labels)
    all_probs = torch.tensor(all_probs)

    preds = (all_probs >= threshold).long()

    acc = accuracy_score(all_labels, preds)
    f1 = f1_score(all_labels, preds, zero_division=0)

    try:
        auroc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auroc = 0.0

    return acc, f1, auroc


def main():
    args = get_args()
    cfg = yaml.safe_load(open(args.cfg, "r", encoding="utf-8"))
    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(cfg["save_dir"], exist_ok=True)

    # ---------------------------
    # Text tokenizer / model (KoBERT)
    # ---------------------------
    text_model_name = cfg["text"].get("model_name", "skt/kobert-base-v1")
    tokenizer = AutoTokenizer.from_pretrained(text_model_name)
    txt_model = AutoModel.from_pretrained(text_model_name).to(device)
    txt_model.eval()  # 학습 중에는 freeze
    for param in txt_model.parameters():
        param.requires_grad = False

    # ---------------------------
    # Dataset / DataLoader
    # ---------------------------
    train_ds = AVTDataset(cfg["train_manifest"], cfg, tokenizer)
    val_ds = AVTDataset(cfg["val_manifest"], cfg, tokenizer)

    train_ld = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        pin_memory=True,
        drop_last=True,
    )

    val_ld = DataLoader(
        val_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=True,
        drop_last=False,
    )

    # ---------------------------
    # Model / Optimizer
    # ---------------------------
    model = AVTModel(text_hidden_dim=768, dim=256).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
    )

    epochs = cfg["epochs"]
    w_cls = cfg["loss_weights"]["cls"]
    w_nce = cfg["loss_weights"]["nce"]
    best_f1 = 0.0

    # ---------------------------
    # Training loop
    # ---------------------------
    for epoch in range(1, epochs + 1):
        model.train()
        txt_model.eval()

        pbar = tqdm(train_ld, desc=f"[Epoch {epoch}/{epochs}]")
        running_loss = 0.0

        for vid, wav, ids, mask, y in pbar:
            vid = vid.to(device)
            wav = wav.to(device)
            y = y.to(device)

            with torch.no_grad():
                cls_emb = extract_text_cls(txt_model, ids, mask, device)

            logits, z_av, z_t = model(vid, wav, cls_emb)

            cls_loss = bce_logits(logits, y)
            nce_loss = torch.tensor(0.0, device=device)

            loss = cls_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix(
                loss=f"{running_loss / (pbar.n or 1):.4f}",
                cls=f"{cls_loss.item():.4f}",
                nce=f"{nce_loss.item():.4f}",
            )

        # -----------------------
        # Validation
        # -----------------------
        acc, f1, auroc = evaluate(
            model,
            val_ld,
            txt_model,
            threshold=cfg["threshold"],
            device=device,
        )

        print(
            f"[Val] acc={acc:.4f} f1={f1:.4f} auroc={auroc:.4f} "
            f"(epoch {epoch})"
        )

        # best checkpoint 저장 (F1 기준)
        if f1 > best_f1:
            best_f1 = f1
            ckpt_path = os.path.join(cfg["save_dir"], "best_f1_final_model2.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"✓ checkpoint updated: {ckpt_path} (best_f1={best_f1:.4f})")

    print(f"[Done] best F1 = {best_f1:.4f}")


if __name__ == "__main__":
    main()
