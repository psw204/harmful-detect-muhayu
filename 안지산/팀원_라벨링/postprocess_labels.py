# postprocess_labels.py
import os, json, argparse

def read_jsonl(path):
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line=line.strip()
            if not line: continue
            yield json.loads(line)

def main(mani_path, out_dir, verified_img=None, verified_vid=None):
    os.makedirs(out_dir, exist_ok=True)

    # 1) 매니페스트에서 이미지/비디오 경로 수집
    img_all, vid_all = [], []
    for item in read_jsonl(mani_path):
        m = item.get("modality") or item.get("type")
        p = item.get("path") or item.get("filepath") or item.get("url")
        if not p or not m: continue
        if m.lower().startswith("img") or m.lower()=="image":
            img_all.append(p)
        elif m.lower().startswith("vid") or m.lower()=="video":
            vid_all.append(p)

    # 2) 이미 생성된 verified 라벨 읽기(없으면 빈 dict)
    vimg = {}
    if verified_img and os.path.isfile(verified_img):
        vimg = json.load(open(verified_img, "r", encoding="utf-8"))

    vvid = {}
    if verified_vid and os.path.isfile(verified_vid):
        vvid = json.load(open(verified_vid, "r", encoding="utf-8"))

    # 3) safe = 전체 - verified (경로 기준)
    simg = {p: 0 for p in img_all if p not in vimg}
    svid = {p: 0 for p in vid_all if p not in vvid}

    # 4) 누락된 파일만 생성
    out_files = []
    if vimg:
        out_files.append(("verified_labels.json", vimg))
    if vvid:
        out_files.append(("verified_video_labels.json", vvid))
    if simg:  # 이미지 safe
        out_files.append(("safe_labels.json", simg))
    if svid:  # 비디오 safe
        out_files.append(("safe_video_labels.json", svid))

    for name, data in out_files:
        out_path = os.path.join(out_dir, name)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[저장] {name}: {len(data)}개")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--verified-img", default=None)
    ap.add_argument("--verified-vid", default=None)
    args = ap.parse_args()

    main(args.manifest, args.out_dir, args.verified_img, args.verified_vid)
