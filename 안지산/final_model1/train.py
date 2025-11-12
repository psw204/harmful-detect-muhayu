# -*- coding: utf-8 -*-
import argparse, yaml, os, torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from dataset_avt import AVTDataset
from model_avt import AVTModel
from losses import bce_logits, info_nce
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score

def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', default='config.yaml')
    return ap.parse_args()

@torch.no_grad()
def evaluate_with_text(model, loader, txt_model, threshold=0.5, device='cuda'):
    model.eval()
    ys, ps = [], []
    for v, a, ids, mask, y in loader:
        v, a, ids, mask, y = v.to(device), a.to(device), ids.to(device), mask.to(device), y.to(device)
        out = txt_model(input_ids=ids, attention_mask=mask)
        t_cls = out.last_hidden_state[:, 0, :]  # [CLS]
        logits, _, _ = model(v, a, t_cls)
        prob = torch.sigmoid(logits)
        ys.extend(y.cpu().tolist())
        ps.extend(prob.cpu().tolist())

    preds = [1 if p >= threshold else 0 for p in ps]
    acc = accuracy_score(ys, preds)
    f1 = f1_score(ys, preds)
    try:
        auroc = roc_auc_score(ys, ps)
    except Exception:
        auroc = 0.0
    return acc, f1, auroc

def main():
    args = get_args()
    cfg = yaml.safe_load(open(args.cfg, 'r', encoding='utf-8'))
    device = cfg.get('device', 'cuda')

    # 토크나이저/텍스트 모델(KoBERT)
    tok = AutoTokenizer.from_pretrained('skt/kobert-base-v1')
    txt_model = AutoModel.from_pretrained('skt/kobert-base-v1').to(device)
    txt_model.eval()

    # 데이터
    train_ds = AVTDataset(cfg['train_manifest'], cfg, tok)
    val_ds   = AVTDataset(cfg['val_manifest'], cfg, tok)
    train_ld = DataLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True,
                          num_workers=cfg['num_workers'], pin_memory=True, drop_last=True)
    val_ld   = DataLoader(val_ds, batch_size=cfg['batch_size'], shuffle=False,
                          num_workers=cfg['num_workers'], pin_memory=True)

    # 모델
    model = AVTModel().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])

    best_f1 = -1.0
    for epoch in range(1, cfg['epochs'] + 1):
        model.train()
        pbar = tqdm(train_ld, desc=f"Epoch {epoch}")
        for v, a, ids, mask, y in pbar:
            v, a, ids, mask, y = v.to(device), a.to(device), ids.to(device), mask.to(device), y.to(device)
            # KoBERT CLS
            with torch.no_grad():
                out = txt_model(input_ids=ids, attention_mask=mask)
                t_cls = out.last_hidden_state[:, 0, :]  # [CLS]

            logits, z_av, z_t = model(v, a, t_cls)
            loss_cls = bce_logits(logits, y)
            loss_nce = info_nce(z_av, z_t) * cfg['loss_weights']['nce']
            loss = cfg['loss_weights']['cls'] * loss_cls + loss_nce

            opt.zero_grad()
            loss.backward()
            opt.step()
            pbar.set_postfix(loss=float(loss.item()))

        acc, f1, auroc = evaluate_with_text(
            model, val_ld, txt_model, threshold=cfg['threshold'], device=device
        )
        print(f"[Val] acc={acc:.4f} f1={f1:.4f} auroc={auroc:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            os.makedirs(cfg['save_dir'], exist_ok=True)
            torch.save(model.state_dict(), os.path.join(cfg['save_dir'], 'best_f1.pt'))
            print(f"✓ checkpoint updated (best_f1={best_f1:.4f})")

if __name__ == '__main__':
    main()
