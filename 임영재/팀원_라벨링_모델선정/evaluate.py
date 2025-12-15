#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json

# -------------------------------
# 경로 설정
# -------------------------------
BASE = "/home/jovyan/kau-muhayu-multimodal-harmful-content-detect/임영재/팀원_라벨링_모델선정"
RESULT_DIR = os.path.join(BASE, "결과_데이터_32")
LABEL_DIR = os.path.join(BASE, "팀원_라벨링")

PEOPLE = ["박상원", "안지산", "임영재"]

#12
# -------------------------------
# 유틸 함수들
# -------------------------------
def load_json(path):
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            print(f"⚠️ JSON LOAD ERROR: {path}")
            return {}
    return {}


def find_gt_key_for_pred(pred_name, gt, used):
    """
    pred 파일 이름 안에 ground truth key 문자열이 '포함'되어 있으면 매칭.
    (여러 개 매칭되면 가장 긴 문자열 사용)
    """
    best = None
    best_len = -1
    for k in gt.keys():
        if k in used:
            continue
        if k in pred_name:
            if len(k) > best_len:
                best = k
                best_len = len(k)
    return best


def eval_one(pred_name, pred_label, gt, used_gt_keys):
    """
    pred 하나(파일 하나)에 대해 TP/TN/FP/FN 계산
    """
    matched = find_gt_key_for_pred(pred_name, gt, used_gt_keys)
    if matched is None:
        # 매칭 실패
        return 0, 0, 0, 0, False, None

    real = int(gt.get(matched, {}).get("is_harmful", 0))
    pred = int(pred_label)

    if real == 1 and pred == 1:
        return 1, 0, 0, 0, True, matched  # TP
    elif real == 1 and pred == 0:
        return 0, 0, 0, 1, True, matched  # FN
    elif real == 0 and pred == 0:
        return 0, 1, 0, 0, True, matched  # TN
    elif real == 0 and pred == 1:
        return 0, 0, 1, 0, True, matched  # FP
    return 0, 0, 0, 0, False, None


def evaluate_group(pred_dict, gt, used_gt_keys):
    """
    verified / safe 그룹 하나에 대해 TP/TN/FP/FN 합산
    """
    TP = TN = FP = FN = 0
    matched_pred = 0
    unmatched_pred = 0

    for fname, info in pred_dict.items():
        pred = int(info.get("final_label", info.get("pred_label", 0)))
        tp, tn, fp, fn, used, gkey = eval_one(fname, pred, gt, used_gt_keys)

        TP += tp
        TN += tn
        FP += fp
        FN += fn
        if used:
            matched_pred += 1
            used_gt_keys.add(gkey)
        else:
            unmatched_pred += 1

    return TP, TN, FP, FN, matched_pred, unmatched_pred


def calc_metrics(TP, TN, FP, FN):
    total = TP + TN + FP + FN
    acc = (TP + TN) / total if total > 0 else 0.0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return acc, precision, recall, f1, total


# -------------------------------
# 사람별 평가
# -------------------------------
def evaluate_person(person):
    print("\n==============================")
    print(f"📊 평가 시작: {person}")
    print("==============================")

    # GT 로드
    gt_path = os.path.join(LABEL_DIR, f"{person}_labels_categorized.json")
    gt = load_json(gt_path)

    # 예측 결과 로드
    pred_base = os.path.join(RESULT_DIR, person, "라벨_결과")
    verified_img = load_json(os.path.join(pred_base, "verified_labels.json"))
    safe_img     = load_json(os.path.join(pred_base, "safe_labels.json"))
    verified_vid = load_json(os.path.join(pred_base, "verified_video_labels.json"))
    safe_vid     = load_json(os.path.join(pred_base, "safe_video_labels.json"))

    used_gt_img = set()
    used_gt_vid = set()

    # -------- IMAGE --------
    TP_i1, TN_i1, FP_i1, FN_i1, M_i1, UM_i1 = evaluate_group(verified_img, gt, used_gt_img)
    TP_i2, TN_i2, FP_i2, FN_i2, M_i2, UM_i2 = evaluate_group(safe_img, gt, used_gt_img)

    TP_img = TP_i1 + TP_i2
    TN_img = TN_i1 + TN_i2
    FP_img = FP_i1 + FP_i2
    FN_img = FN_i1 + FN_i2

    acc_img, prec_img, rec_img, f1_img, total_img = calc_metrics(TP_img, TN_img, FP_img, FN_img)

    # -------- VIDEO --------
    TP_v1, TN_v1, FP_v1, FN_v1, M_v1, UM_v1 = evaluate_group(verified_vid, gt, used_gt_vid)
    TP_v2, TN_v2, FP_v2, FN_v2, M_v2, UM_v2 = evaluate_group(safe_vid, gt, used_gt_vid)

    TP_vid = TP_v1 + TP_v2
    TN_vid = TN_v1 + TN_v2
    FP_vid = FP_v1 + FP_v2
    FN_vid = FN_v1 + FN_v2

    acc_vid, prec_vid, rec_vid, f1_vid, total_vid = calc_metrics(TP_vid, TN_vid, FP_vid, FN_vid)

    # -------- 출력 --------
    print("\n📸 IMAGE Metrics ====================")
    print(f"TP={TP_img} | TN={TN_img} | FP={FP_img} | FN={FN_img} | Total={total_img}")
    print(f"🎯 Accuracy : {acc_img*100:.2f}%")
    print(f"🎯 Precision: {prec_img*100:.2f}%")
    print(f"🎯 Recall   : {rec_img*100:.2f}%")
    print(f"🎯 F1-score : {f1_img*100:.2f}%")

    print("\n🎬 VIDEO Metrics ====================")
    print(f"TP={TP_vid} | TN={TN_vid} | FP={FP_vid} | FN={FN_vid} | Total={total_vid}")
    print(f"🎯 Accuracy : {acc_vid*100:.2f}%")
    print(f"🎯 Precision: {prec_vid*100:.2f}%")
    print(f"🎯 Recall   : {rec_vid*100:.2f}%")
    print(f"🎯 F1-score : {f1_vid*100:.2f}%\n")

    return {
        "person": person,

        "TP_img": TP_img, "TN_img": TN_img, "FP_img": FP_img, "FN_img": FN_img,
        "TP_vid": TP_vid, "TN_vid": TN_vid, "FP_vid": FP_vid, "FN_vid": FN_vid,

        "img_acc": acc_img, "img_prec": prec_img, "img_rec": rec_img, "img_f1": f1_img,
        "vid_acc": acc_vid, "vid_prec": prec_vid, "vid_rec": rec_vid, "vid_f1": f1_vid,

        "img_total": total_img,
        "vid_total": total_vid,
    }


# -------------------------------
# 메인: 전체 요약 + 전체 Metrics
# -------------------------------
def main():
    results = [evaluate_person(p) for p in PEOPLE]

    print("\n=========== 개인별 요약 ===========")
    for r in results:
        print(
            f"{r['person']} → "
            f"IMG Acc:{r['img_acc']*100:.2f}% / P:{r['img_prec']*100:.2f}% / "
            f"R:{r['img_rec']*100:.2f}% / F1:{r['img_f1']*100:.2f}% | "
            f"VID Acc:{r['vid_acc']*100:.2f}% / P:{r['vid_prec']*100:.2f}% / "
            f"R:{r['vid_rec']*100:.2f}% / F1:{r['vid_f1']*100:.2f}%"
        )

    # -------- 전체(이미지) 합산 --------
    TP_img_all = sum(r["TP_img"] for r in results)
    TN_img_all = sum(r["TN_img"] for r in results)
    FP_img_all = sum(r["FP_img"] for r in results)
    FN_img_all = sum(r["FN_img"] for r in results)

    acc_img_all, prec_img_all, rec_img_all, f1_img_all, total_img_all = calc_metrics(
        TP_img_all, TN_img_all, FP_img_all, FN_img_all
    )

    # -------- 전체(비디오) 합산 --------
    TP_vid_all = sum(r["TP_vid"] for r in results)
    TN_vid_all = sum(r["TN_vid"] for r in results)
    FP_vid_all = sum(r["FP_vid"] for r in results)
    FN_vid_all = sum(r["FN_vid"] for r in results)

    acc_vid_all, prec_vid_all, rec_vid_all, f1_vid_all, total_vid_all = calc_metrics(
        TP_vid_all, TN_vid_all, FP_vid_all, FN_vid_all
    )

    print("\n=========== 전체 Metrics (ALL) ===========")
    print("📸 IMAGE (ALL)")
    print(f"TP={TP_img_all} | TN={TN_img_all} | FP={FP_img_all} | FN={FN_img_all} | Total={total_img_all}")
    print(f"🎯 Accuracy : {acc_img_all*100:.2f}%")
    print(f"🎯 Precision: {prec_img_all*100:.2f}%")
    print(f"🎯 Recall   : {rec_img_all*100:.2f}%")
    print(f"🎯 F1-score : {f1_img_all*100:.2f}%\n")

    print("🎬 VIDEO (ALL)")
    print(f"TP={TP_vid_all} | TN={TN_vid_all} | FP={FP_vid_all} | FN={FN_vid_all} | Total={total_vid_all}")
    print(f"🎯 Accuracy : {acc_vid_all*100:.2f}%")
    print(f"🎯 Precision: {prec_vid_all*100:.2f}%")
    print(f"🎯 Recall   : {rec_vid_all*100:.2f}%")
    print(f"🎯 F1-score : {f1_vid_all*100:.2f}%\n")


if __name__ == "__main__":
    main()
