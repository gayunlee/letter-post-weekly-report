"""채널톡 3분류 파인튜닝 모델 벤치마크

Golden set에서 기타를 제외하고 3분류(결제·구독/콘텐츠·수강/기술·오류) 테스트.
파인튜닝된 KcELECTRA 모델 평가.

사용법:
    python3 scripts/benchmark_channel_4class_ft.py --model-dir models/channel_3class/kcelectra-base-v2022/final_model
"""
import json
import argparse
from pathlib import Path
from collections import defaultdict

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

GOLDEN_PATH = Path("data/channel_io/golden/golden_multilabel_270.json")
BENCHMARKS_DIR = Path("benchmarks")

TOPICS_3 = ["결제·구독", "콘텐츠·수강", "기술·오류"]

MERGE_MAP = {
    "결제·환불": "결제·구독",
    "구독·멤버십": "결제·구독",
    "콘텐츠·수강": "콘텐츠·수강",
    "기술·오류": "기술·오류",
    "기타": "기타",
}

WORKFLOW_NOISE = [
    "그 외 기타 문의(오류/구독해지/환불)",
    "그 외 기타 문의",
    "💬1:1 상담 문의하기",
    "💬 1:1 고객센터 문의하기",
    "💬상담매니저에게 직접 문의",
    "어스 구독신청/결제하기",
    "어스 이용방법",
    "사이트 및 동영상 오류",
    "수강 및 상품문의",
    "라이브 콘텐츠 참여 방법",
    "수강방법",
    "↩ 이전으로",
    "✅ 1:1 문의하기",
    "구독 상품변경/결제정보 확인",
    "결제실패 후 카드변경 방법",
    "💬 구독 결제/변경/정보 확인 직접 문의하기",
    "구독상품 변경",
    "구독 결제/변경/정보 직접 문의하기",
]


def strip_workflow_buttons(text: str) -> str:
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        stripped = line.strip()
        if stripped in WORKFLOW_NOISE:
            continue
        if stripped.startswith("👆🏻"):
            continue
        cleaned.append(line)
    return "\n".join(cleaned).strip()


def merge_topic(topics_list):
    merged = set()
    for t in topics_list:
        merged.add(MERGE_MAP.get(t, t))
    for priority in TOPICS_3:
        if priority in merged:
            return priority
    return "기타"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="models/channel_3class/kcelectra-base-v2022/final_model")
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    print(f"모델: {model_dir}")

    # 모델 로드
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    model.eval()

    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    model = model.to(device)
    print(f"Device: {device}")

    # 카테고리 매핑
    config_path = model_dir / "category_config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        id_to_category = {int(k): v for k, v in config["id_to_category"].items()}
    else:
        id_to_category = {0: "결제·구독", 1: "콘텐츠·수강", 2: "기술·오류"}

    # Golden set 로드 — 기타 제외
    with open(GOLDEN_PATH) as f:
        golden_all = json.load(f)

    golden = []
    gold_topics = []
    skipped_기타 = 0
    for item in golden_all:
        topic = merge_topic(item["topics"])
        if topic == "기타":
            skipped_기타 += 1
            continue
        golden.append(item)
        gold_topics.append(topic)

    dist = defaultdict(int)
    for t in gold_topics:
        dist[t] += 1
    print(f"테스트: {len(golden)}건 (기타 {skipped_기타}건 제외)")
    print(f"Golden 분포: {dict(dist)}")
    print()

    # 분류
    correct = 0
    predictions = []
    for i, (item, gold_t) in enumerate(zip(golden, gold_topics)):
        text = strip_workflow_buttons(item["text"])
        if len(text.strip()) < 3:
            text = item["text"]

        inputs = tokenizer(text, truncation=True, padding="max_length",
                          max_length=args.max_length, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            pred_id = outputs.logits.argmax(-1).item()
            confidence = torch.softmax(outputs.logits, dim=-1).max().item()

        pred_topic = id_to_category[pred_id]
        predictions.append({"topic": pred_topic, "confidence": confidence})

        if gold_t == pred_topic:
            correct += 1
        else:
            print(f"  [{i:3d}] ❌ gold={gold_t:8s} pred={pred_topic:8s} conf={confidence:.2f}")

    # 결과
    n = len(golden)
    accuracy = correct / n * 100

    # 혼동 행렬
    confusion = defaultdict(lambda: defaultdict(int))
    for gold_t, pred in zip(gold_topics, predictions):
        confusion[gold_t][pred["topic"]] += 1

    # Per-class metrics
    per_class = {}
    for t in TOPICS_3:
        tp = confusion[t][t]
        fp = sum(confusion[other][t] for other in TOPICS_3 if other != t)
        fn = sum(confusion[t][other] for other in TOPICS_3 if other != t)
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        per_class[t] = {"precision": round(p, 3), "recall": round(r, 3), "f1": round(f1, 3),
                        "tp": tp, "fp": fp, "fn": fn}

    print(f"\n{'='*60}")
    print(f"정확도: {correct}/{n} ({accuracy:.1f}%)")
    print(f"\n토픽별:")
    for t in TOPICS_3:
        m = per_class[t]
        print(f"  {t}: P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} (TP={m['tp']} FP={m['fp']} FN={m['fn']})")

    print(f"\n혼동 행렬:")
    header = "".join(f"{t:>10s}" for t in TOPICS_3)
    print(f"{'':>12s}{header}")
    for gt in TOPICS_3:
        row = "".join(f"{confusion[gt][pt]:>10d}" for pt in TOPICS_3)
        print(f"{gt:>12s}{row}")

    # 저장
    BENCHMARKS_DIR.mkdir(exist_ok=True)
    model_name = model_dir.parent.name
    out_path = BENCHMARKS_DIR / f"channel_3class_ft_{model_name}_{n}items.json"
    result = {
        "model": str(model_dir),
        "model_name": model_name,
        "n": n,
        "n_excluded_기타": skipped_기타,
        "accuracy": round(accuracy, 1),
        "correct": correct,
        "per_class": per_class,
        "confusion": {gt: dict(confusion[gt]) for gt in TOPICS_3},
    }
    with open(out_path, "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n저장: {out_path}")

    # 비교
    print(f"\n{'='*60}")
    print(f"  비교:")
    print(f"  KcELECTRA v4 (4분류, 270건): 87.0%")
    print(f"  KcELECTRA v5 (3분류, {n}건): {accuracy:.1f}%")
    print(f"  Haiku API (단일 분류): 99.3%")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
