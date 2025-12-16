import os
import json
from glob import glob

DATA_ROOT = "/home/jovyan/kau-muhayu-multimodal-harmful-content-detect/무하유_유해콘텐츠_데이터/2_실제_수집_데이터(개인)"

OUT_PATH = "test_manifest.jsonl"
results = []

print(f"📌 DATA_ROOT = {DATA_ROOT}")

# 개인 폴더 탐색
persons = [p for p in os.listdir(DATA_ROOT) if os.path.isdir(os.path.join(DATA_ROOT, p))]
print(f"📌 탐색된 개인: {persons}")

for person in persons:
    print(f"\n=== {person} 처리 중 ===")
    person_root = os.path.join(DATA_ROOT, person)

    # ==================================================================
    # 1) 비디오 폴더 내 manifests/*.jsonl 탐색
    # ==================================================================
    video_root = os.path.join(person_root, "비디오")
    if os.path.isdir(video_root):
        print("  ▶ 비디오 검색 중...")

        # manifests 디렉터리만 타겟으로 하는 개선된 패턴
        video_jsonl = glob(os.path.join(video_root, "**", "manifests", "*.jsonl"), recursive=True)

        print(f"    📌 비디오 manifest 개수: {len(video_jsonl)}")

        for jsonl_path in video_jsonl:
            print(f"    📌 발견(비디오): {jsonl_path}")
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        results.append(json.loads(line))
                    except:
                        continue
    else:
        print(f"  ⚠️ 비디오 폴더 없음: {video_root}")

    # ==================================================================
    # 2) 안전_비디오 폴더 내 manifests/*.jsonl 탐색
    # ==================================================================
    safe_root = os.path.join(person_root, "안전_비디오")
    if os.path.isdir(safe_root):
        print("  ▶ 안전_비디오 검색 중...")

        safe_jsonl = glob(os.path.join(safe_root, "**", "manifests", "*.jsonl"), recursive=True)

        print(f"    📌 안전_비디오 manifest 개수: {len(safe_jsonl)}")

        for jsonl_path in safe_jsonl:
            print(f"    📌 발견(안전): {jsonl_path}")
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        results.append(json.loads(line))
                    except:
                        continue
    else:
        print(f"  ⚠️ 안전_비디오 폴더 없음: {safe_root}")

print("\n🎉 test_manifest.jsonl 생성 시작...")

with open(OUT_PATH, "w", encoding="utf-8") as f:
    for item in results:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"🎉 test_manifest.jsonl 생성 완료!")
print(f"📌 최종 총 개수: {len(results)}")
