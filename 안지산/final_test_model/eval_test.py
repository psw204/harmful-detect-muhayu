import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import MultiModalDataset
from model import MultimodalClassifier
from utils import compute_metrics


def search_best_threshold(labels, probs):
    best_thr = 0.0
    best_f1 = -1
    best_metrics = None

    for thr in [i / 100 for i in range(0, 101)]:
        preds = (probs >= thr).astype(int)
        m = compute_metrics(labels, preds)

        # compute_metrics() 기준: "F1-Score"
        f1 = m.get("F1-Score")

        if f1 is None:
            raise ValueError(f"F1 key not found in metrics dict: {m}")

        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr
            best_metrics = m

    return best_thr, best_metrics



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()

    # -----------------------------------------
    # Load Config
    # -----------------------------------------
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = cfg.get("device", "cuda")
    print("[Device]", device)

    # -----------------------------------------
    # Dataset & Loader
    # -----------------------------------------
    test_dataset = MultiModalDataset(
        manifest_path=args.manifest,
        config=cfg,
        split="test"
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.get("batch_size", 4),
        num_workers=0,    # 🔥 worker 모두 비활성화
        shuffle=False,
        pin_memory=False
    )


    # -----------------------------------------
    # Load Model
    # -----------------------------------------
    model = MultimodalClassifier(cfg).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)

    # final_model3 checkpoint 구조 대응
    if "model" in ckpt:
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt

    model.load_state_dict(state_dict, strict=True)
    print(f"✔ Loaded checkpoint: {args.checkpoint}")

    model.eval()

    # -----------------------------------------
    # Inference
    # -----------------------------------------
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Eval"):
            video = batch["video"].to(device)
            audio = batch["audio"].to(device)
            text = {k: v.to(device) for k, v in batch["text"].items()}
            labels = batch["label"].cpu().numpy()

            logits = model({
                "video": video,
                "audio": audio,
                "text": text
            })

            probs = torch.sigmoid(logits).detach().cpu().numpy()

            all_labels.extend(labels)
            all_probs.extend(probs)

    all_labels = torch.tensor(all_labels).numpy()
    all_probs = torch.tensor(all_probs).numpy()

    # -----------------------------------------
    # Default threshold 평가
    # -----------------------------------------
    thr_cfg = cfg.get("threshold", {})
    default_thr = thr_cfg.get("default", 0.5)

    preds_default = (all_probs >= default_thr).astype(int)
    metrics_default = compute_metrics(all_labels, preds_default)

    print("\n=== 📌 Default Threshold Evaluation ===")
    print(f"Threshold = {default_thr}")
    for k, v in metrics_default.items():
        print(f"{k.capitalize()}: {v:.4f}")

    # -----------------------------------------
    # Auto Threshold Search
    # -----------------------------------------
    if thr_cfg.get("optimize", True):
        print("\n=== 🔍 Auto Threshold Search (0.00 ~ 1.00) ===")
        best_thr, best_metrics = search_best_threshold(all_labels, all_probs)

        print(f"\n🎯 Best Threshold: {best_thr:.2f}")
        print("=== Metrics at Best Threshold ===")
        for k, v in best_metrics.items():
            print(f"{k.capitalize()}: {v:.4f}")

    print("\n🎉 Evaluation Completed!")


if __name__ == "__main__":
    main()
