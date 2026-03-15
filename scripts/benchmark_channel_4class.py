"""채널톡 4분류 단일 분류 벤치마크 — 결제·구독 합침

Golden set(270건)에서 결제·환불+구독·멤버십 → 결제·구독으로 합쳐서 4분류 테스트.
단일 분류 (topic 1개만 선택).

사용법:
  python3 scripts/benchmark_channel_4class.py --model exaone3.5:32b --sample 20
  python3 scripts/benchmark_channel_4class.py --model exaone3.5:32b --all
"""
import json
import time
import random
import argparse
import subprocess
from pathlib import Path
from collections import defaultdict

GOLDEN_PATH = Path("data/channel_io/golden/golden_multilabel_270.json")
BENCHMARKS_DIR = Path("benchmarks")

TOPICS_4 = ["결제·구독", "콘텐츠·수강", "기술·오류", "기타"]

# 5분류 → 4분류 매핑
MERGE_MAP = {
    "결제·환불": "결제·구독",
    "구독·멤버십": "결제·구독",
    "콘텐츠·수강": "콘텐츠·수강",
    "기술·오류": "기술·오류",
    "기타": "기타",
}

SYSTEM_PROMPT = """당신은 금융 교육 플랫폼의 CS 문의 분류기입니다.
채널톡으로 접수된 고객 상담을 아래 4개 중 **하나만** 선택하여 분류합니다.

### 결제·구독
결제, 환불, 구독 신청, 해지, 등급 변경, 갱신, 멤버십, 계좌이체, 카드 변경, 영수증 요청.
- "환불해주세요", "구독 해지", "상품 변경", "멤버십 취소"
- 아직 이용하지 않는 강의에 대한 신청 의사도 여기

### 콘텐츠·수강
이미 이용 중인 강의 접근, 수강 방법, 자료/교재 요청, 이용 방법. 오프라인 행사/세미나/특강도 여기.
- "강의 어디서 보나요", "수강 방법", "교재 배송", "세미나 참석"

### 기술·오류
확실히 시스템 문제인 경우. 로그인 안 됨, 앱 오류, 접속 문제. 기술 기능 건의도 여기.
- 로그인 안 됨, 앱 크래시, "멤버인데 안 됨" (시스템 원인)

### 기타
위 3개에 해당하지 않는 문의. 단순 인사, 투자 질문, 번호 변경, 비기술 기능 건의.

## 응답: JSON만 출력 (다른 텍스트 없이)

{"topic": "결제·구독", "summary": "요약 1문장"}"""


def load_golden():
    with open(GOLDEN_PATH) as f:
        return json.load(f)


def merge_topic(topics_list):
    """5분류 topics 배열 → 4분류 단일 topic."""
    merged = set()
    for t in topics_list:
        merged.add(MERGE_MAP.get(t, t))
    # 우선순위: 결제·구독 > 콘텐츠·수강 > 기술·오류 > 기타
    for priority in TOPICS_4:
        if priority in merged:
            return priority
    return "기타"


def sample_balanced(golden, n_per_topic=5):
    """4분류 기준 토픽당 n건씩 균등 샘플링."""
    by_topic = defaultdict(list)
    for i, item in enumerate(golden):
        t4 = merge_topic(item["topics"])
        by_topic[t4].append(i)

    sampled_indices = set()
    for topic in TOPICS_4:
        candidates = by_topic[topic]
        random.shuffle(candidates)
        sampled_indices.update(candidates[:n_per_topic])

    return sorted(sampled_indices)


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
]


def strip_workflow_buttons(text: str) -> str:
    """워크플로우 버튼 텍스트 제거."""
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        stripped = line.strip()
        if stripped in WORKFLOW_NOISE:
            continue
        # 👆🏻 으로 시작하는 워크플로우 라인도 제거
        if stripped.startswith("👆🏻"):
            continue
        cleaned.append(line)
    return "\n".join(cleaned).strip()


def classify_with_ollama(model: str, text: str, timeout: int = 120) -> dict:
    """Ollama API로 분류."""
    prompt = strip_workflow_buttons(text)[:500]
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 150},
    }

    start = time.time()
    result = subprocess.run(
        ["curl", "-s", "http://localhost:11434/api/chat", "-d", json.dumps(payload)],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    elapsed = time.time() - start

    try:
        resp = json.loads(result.stdout)
        raw = resp.get("message", {}).get("content", "")
    except json.JSONDecodeError:
        return {"topic": "기타", "raw": result.stdout[:200], "time": elapsed, "error": "json_decode"}

    try:
        parsed = parse_json(raw)
        topic = parsed.get("topic", "기타")
        if topic not in TOPICS_4:
            # 5분류 이름으로 답한 경우 매핑
            topic = MERGE_MAP.get(topic, "기타")
        if topic not in TOPICS_4:
            topic = "기타"
        return {"topic": topic, "raw": raw[:300], "time": elapsed}
    except Exception:
        return {"topic": "기타", "raw": raw[:300], "time": elapsed, "error": "parse"}


def parse_json(raw: str) -> dict:
    """LLM 응답에서 JSON 추출."""
    if "```" in raw:
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        raw = raw[start : end + 1]
    return json.loads(raw)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="exaone3.5:32b")
    parser.add_argument("--sample", type=int, default=20, help="샘플 수 (토픽당 N/4건)")
    parser.add_argument("--all", action="store_true", help="전체 270건")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    golden = load_golden()

    if args.all:
        indices = list(range(len(golden)))
    else:
        indices = sample_balanced(golden, n_per_topic=args.sample // 4)

    items = [golden[i] for i in indices]
    gold_topics = [merge_topic(item["topics"]) for item in items]

    print(f"모델: {args.model}")
    print(f"테스트: {len(items)}건")
    dist = defaultdict(int)
    for t in gold_topics:
        dist[t] += 1
    print(f"Golden 분포: {dict(dist)}")
    print()

    predictions = []
    correct = 0
    for i, (item, gold_t) in enumerate(zip(items, gold_topics)):
        pred = classify_with_ollama(args.model, item["text"])
        predictions.append(pred)

        match = "✅" if gold_t == pred["topic"] else "❌"
        if gold_t == pred["topic"]:
            correct += 1
        print(f"  [{indices[i]:3d}] {match} gold={gold_t:8s} pred={pred['topic']:8s} ({pred['time']:.1f}s)")

    # 결과
    n = len(items)
    accuracy = correct / n * 100

    # 혼동 행렬
    confusion = defaultdict(lambda: defaultdict(int))
    for gold_t, pred in zip(gold_topics, predictions):
        confusion[gold_t][pred["topic"]] += 1

    # Per-class metrics
    per_class = {}
    for t in TOPICS_4:
        tp = confusion[t][t]
        fp = sum(confusion[other][t] for other in TOPICS_4 if other != t)
        fn = sum(confusion[t][other] for other in TOPICS_4 if other != t)
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        per_class[t] = {"precision": round(p, 3), "recall": round(r, 3), "f1": round(f1, 3),
                        "tp": tp, "fp": fp, "fn": fn}

    print(f"\n{'='*60}")
    print(f"정확도: {correct}/{n} ({accuracy:.1f}%)")
    print(f"\n토픽별:")
    for t in TOPICS_4:
        m = per_class[t]
        print(f"  {t}: P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} (TP={m['tp']} FP={m['fp']} FN={m['fn']})")

    print(f"\n혼동 행렬:")
    header = "".join(f"{t:>10s}" for t in TOPICS_4)
    print(f"{'':>12s}{header}")
    for gt in TOPICS_4:
        row = "".join(f"{confusion[gt][pt]:>10d}" for pt in TOPICS_4)
        print(f"{gt:>12s}{row}")

    # 저장
    BENCHMARKS_DIR.mkdir(exist_ok=True)
    model_name = args.model.replace(":", "_").replace("/", "_")
    out_path = BENCHMARKS_DIR / f"channel_4class_{model_name}_{n}items.json"
    result = {
        "model": args.model,
        "n": n,
        "accuracy": round(accuracy, 1),
        "correct": correct,
        "per_class": per_class,
        "confusion": {gt: dict(confusion[gt]) for gt in TOPICS_4},
        "indices": indices,
    }
    with open(out_path, "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n저장: {out_path}")


if __name__ == "__main__":
    main()
