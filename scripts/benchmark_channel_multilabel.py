"""채널톡 다중 태그 분류 벤치마크 — EXAONE / KcELECTRA 비교용

Golden set(270건)에서 토픽당 4건씩 20건 샘플링하여 Ollama 모델 테스트.
결과는 benchmarks/ 에 JSON으로 저장.

사용법:
  python3 scripts/benchmark_channel_multilabel.py --model exaone3.5:32b --sample 20
  python3 scripts/benchmark_channel_multilabel.py --model exaone3.5:32b --all
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

TOPICS = ["결제·환불", "구독·멤버십", "콘텐츠·수강", "기술·오류", "기타"]

SYSTEM_PROMPT = """당신은 금융 교육 플랫폼의 CS 문의 분류기입니다.
채널톡으로 접수된 고객 상담을 분류합니다.

## 분류 규칙

아래 5개 주제 중 **관련된 주제를 모두** topics에 포함하세요.
하나만 해당되면 하나만, 여러 개 해당되면 여러 개 넣으세요.

### 결제·환불
결제, 환불, 계좌이체, 카드 결제/변경, 영수증 요청, 의도치 않은 자동결제.

### 구독·멤버십
구독 신청, 해지, 등급 변경, 갱신, 멤버십 관련. 아직 이용하지 않는 강의에 대한 신청 의사도 여기.

### 콘텐츠·수강
이미 이용 중인 강의 접근, 수강 방법, 자료/교재 요청, 이용 방법. 오프라인 행사/세미나/특강도 여기.

### 기술·오류
확실히 시스템 문제인 경우. 로그인 안 됨, 앱 오류, 접속 문제. 기술 기능 건의도 여기.

### 기타
위 4개에 해당하지 않는 문의. 단순 인사, 투자 질문, 기능 개선 건의(비기술).

## 다중 주제 처리

여러 주제가 섞여 있으면 관련된 주제를 모두 topics에 넣으세요.
- "구독 해지 + 환불해주세요" → ["결제·환불", "구독·멤버십"]
- "로그인 안 됨 + 환불" → ["기술·오류", "결제·환불"]
하나만 해당되면 하나만 넣으면 됩니다.

## 응답: JSON만 출력 (다른 텍스트 없이)

{"topics": ["결제·환불"], "summary": "요약 1문장"}"""


def load_golden():
    with open(GOLDEN_PATH) as f:
        return json.load(f)


def sample_balanced(golden, n_per_topic=4):
    """토픽당 n건씩 균등 샘플링 (단일 태그 항목 우선)."""
    by_topic = defaultdict(list)
    for i, item in enumerate(golden):
        if len(item["topics"]) == 1:
            by_topic[item["topics"][0]].append(i)

    sampled_indices = set()
    for topic in TOPICS:
        candidates = by_topic[topic]
        random.shuffle(candidates)
        sampled_indices.update(candidates[:n_per_topic])

    return sorted(sampled_indices)


def classify_with_ollama(model: str, text: str, timeout: int = 120) -> dict:
    """Ollama API로 분류."""
    prompt = text[:500]
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 200},
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
        return {"topics": ["기타"], "raw": result.stdout[:200], "time": elapsed, "error": "json_decode"}

    # JSON 파싱
    try:
        parsed = parse_json(raw)
        topics = parsed.get("topics", parsed.get("topic", ["기타"]))
        if isinstance(topics, str):
            topics = [topics]
        topics = [t for t in topics if t in TOPICS]
        if not topics:
            topics = ["기타"]
        return {"topics": topics, "raw": raw[:300], "time": elapsed}
    except Exception:
        return {"topics": ["기타"], "raw": raw[:300], "time": elapsed, "error": "parse"}


def parse_json(raw: str) -> dict:
    """LLM 응답에서 JSON 추출."""
    if "```" in raw:
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()
    # 첫 { 부터 마지막 } 까지
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        raw = raw[start : end + 1]
    return json.loads(raw)


def evaluate(golden_items, predictions):
    """Multi-label 평가 메트릭 계산."""
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
    exact_match = 0

    for item, pred in zip(golden_items, predictions):
        gold_set = set(item["topics"])
        pred_set = set(pred["topics"])

        if gold_set == pred_set:
            exact_match += 1

        for t in TOPICS:
            if t in gold_set and t in pred_set:
                tp[t] += 1
            elif t not in gold_set and t in pred_set:
                fp[t] += 1
            elif t in gold_set and t not in pred_set:
                fn[t] += 1

    # Per-topic metrics
    per_topic = {}
    for t in TOPICS:
        p = tp[t] / (tp[t] + fp[t]) if (tp[t] + fp[t]) > 0 else 0
        r = tp[t] / (tp[t] + fn[t]) if (tp[t] + fn[t]) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        per_topic[t] = {"precision": round(p, 3), "recall": round(r, 3), "f1": round(f1, 3),
                        "tp": tp[t], "fp": fp[t], "fn": fn[t]}

    # Micro average
    total_tp = sum(tp.values())
    total_fp = sum(fp.values())
    total_fn = sum(fn.values())
    micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0

    n = len(golden_items)
    return {
        "n": n,
        "exact_match": exact_match,
        "exact_match_pct": round(exact_match / n * 100, 1),
        "micro_precision": round(micro_p, 3),
        "micro_recall": round(micro_r, 3),
        "micro_f1": round(micro_f1, 3),
        "per_topic": per_topic,
        "avg_pred_topics": round(sum(len(p["topics"]) for p in predictions) / n, 2),
        "avg_gold_topics": round(sum(len(g["topics"]) for g in golden_items) / n, 2),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="exaone3.5:32b")
    parser.add_argument("--sample", type=int, default=20, help="샘플 수 (토픽당 N/5건)")
    parser.add_argument("--all", action="store_true", help="전체 270건")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    golden = load_golden()

    if args.all:
        indices = list(range(len(golden)))
    else:
        indices = sample_balanced(golden, n_per_topic=args.sample // 5)

    items = [golden[i] for i in indices]
    print(f"모델: {args.model}")
    print(f"테스트: {len(items)}건 (indices: {indices[:10]}...)")
    print(f"Golden 토픽 분포: { {t: sum(1 for it in items if t in it['topics']) for t in TOPICS} }")
    print()

    predictions = []
    errors = 0
    for i, item in enumerate(items):
        pred = classify_with_ollama(args.model, item["text"])
        predictions.append(pred)

        gold = item["topics"]
        match = "✅" if set(gold) == set(pred["topics"]) else "❌"
        print(f"  [{indices[i]:3d}] {match} gold={gold} pred={pred['topics']} ({pred['time']:.1f}s)")

        if "error" in pred:
            errors += 1

    # 평가
    metrics = evaluate(items, predictions)
    metrics["model"] = args.model
    metrics["errors"] = errors
    metrics["indices"] = indices

    print(f"\n{'='*60}")
    print(f"Exact Match: {metrics['exact_match']}/{metrics['n']} ({metrics['exact_match_pct']}%)")
    print(f"Micro P={metrics['micro_precision']:.3f} R={metrics['micro_recall']:.3f} F1={metrics['micro_f1']:.3f}")
    print(f"평균 예측 topics: {metrics['avg_pred_topics']} (golden: {metrics['avg_gold_topics']})")
    print(f"\n토픽별:")
    for t in TOPICS:
        m = metrics["per_topic"][t]
        print(f"  {t}: P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} (TP={m['tp']} FP={m['fp']} FN={m['fn']})")

    # 저장
    BENCHMARKS_DIR.mkdir(exist_ok=True)
    model_name = args.model.replace(":", "_").replace("/", "_")
    out_path = BENCHMARKS_DIR / f"channel_multilabel_{model_name}_{len(items)}items.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"\n저장: {out_path}")


if __name__ == "__main__":
    main()
