"""v3 분류 체계 검증 스크립트

기존 분류된 데이터를 새 기능/대응 주체 기반 체계로 LLM 재분류하여
분류 일관성과 명확성을 검증합니다.

검증 항목:
1. 분류 일치율: 동일 건을 2회 분류했을 때 동일 결과 비율
2. v2 → v3 매핑 자연스러움: 기존 분류 결과와의 관계
3. 경계 사례 분석: 신뢰도 낮은 건의 패턴

사용법:
    python3 scripts/validate_v3_taxonomy.py
    python3 scripts/validate_v3_taxonomy.py --sample-size 100 --rounds 2
"""
import sys
import os
import json
import argparse
import random
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

# v3 분류 체계 정의 (5분류)
V3_TOPICS = {
    "운영 피드백": "운영팀/사업팀이 사람 대 사람으로 처리할 요청·이슈 (세미나, 환불, 멤버십, 배송, 가격 정책)",
    "서비스 피드백": "개발팀이 시스템을 수정해야 하는 기술적 이슈·요청 (앱 버그, 결제 오류, 로그인 실패, 기능 요청, 사칭 제보)",
    "콘텐츠 반응": "마스터/콘텐츠가 주된 대상인 감상·반응·피드백 (강의 후기, 마스터 칭찬/불만, 콘텐츠 품질)",
    "투자 담론": "시장/종목/투자전략이 주된 대상인 의견·질문·공유 (투자 질문, 수익 공유, 시장 분석, 종목 토론)",
    "기타": "분석 가치가 낮은 인사·잡담·테스트 (새해 인사, 안부, 축하)",
}

CLASSIFY_PROMPT = """당신은 금융 교육 플랫폼의 VOC 데이터 분류기입니다.

## 분류 기준

| 카테고리 | 기준 | 예시 |
|---------|------|------|
| 운영 피드백 | 운영팀/사업팀이 사람 대 사람으로 처리 | 세미나 문의, 환불 요청, 멤버십 문의, 배송, 가격 정책 질문 |
| 서비스 피드백 | 개발팀이 시스템 수정 필요 | 앱 크래시, 결제 오류(시스템), 로그인 실패, 기능 요청, 사칭 제보 |
| 콘텐츠 반응 | 마스터/콘텐츠가 주된 대상 | 강의 후기, 마스터 칭찬/불만, 콘텐츠 품질 의견, 리포트 반응 |
| 투자 담론 | 시장/종목/투자전략이 주된 대상 | 투자 질문, 수익/손실 공유, 시장 분석, 종목 토론, 매매 타이밍 |
| 기타 | 분류 불필요 | 새해 인사, 안부, 축하, 테스트 글 |

## 경계 판별 규칙
- "결제가 안 돼요" → 서비스 피드백 (시스템 장애)
- "환불해주세요" → 운영 피드백 (정책/프로세스)
- "쌤 강의 너무 좋아요 덕분에 수익 났어요" → 콘텐츠 반응 (주된 대상이 마스터/강의)
- "요즘 장이 불안한데 현금 비중 어떻게 가져가세요?" → 투자 담론 (주된 대상이 시장/투자)
- "쌤 리포트대로 매수했는데 수익 30% 났어요" → 콘텐츠 반응 (마스터 리포트에 대한 반응)
- "삼성전자 지금 들어가도 될까요?" → 투자 담론 (종목/매매 질문)
- "앱에서 강의가 안 보여요" → 서비스 피드백 (기술 이슈)
- "새해 복 많이 받으세요" → 기타 (인사)

## 마스터란?
투자 교육 커뮤니티를 운영하는 금융 콘텐츠 크리에이터입니다.

## 응답 형식
반드시 아래 JSON만 출력하세요:
{"topic": "콘텐츠 반응", "confidence": 0.92, "reason": "마스터 강의에 대한 긍정적 반응"}"""


def load_sample_data(data_dir: str, sample_size: int) -> list:
    """기존 분류된 데이터에서 샘플 추출"""
    all_items = []

    for filename in sorted(os.listdir(data_dir)):
        if not filename.endswith(".json"):
            continue
        filepath = os.path.join(data_dir, filename)
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)

        for item in data.get("letters", []):
            item["_source_type"] = "letter"
            item["_content_field"] = "message"
            all_items.append(item)

        for item in data.get("posts", []):
            item["_source_type"] = "post"
            item["_content_field"] = "textBody"
            all_items.append(item)

    # 기존 Topic별 균형 샘플링
    by_topic = {}
    for item in all_items:
        topic = item.get("classification", {}).get("topic", "미분류")
        by_topic.setdefault(topic, []).append(item)

    per_topic = max(1, sample_size // len(by_topic)) if by_topic else sample_size
    sample = []
    for topic, items in by_topic.items():
        n = min(per_topic, len(items))
        sample.extend(random.sample(items, n))

    # 부족하면 전체에서 추가
    if len(sample) < sample_size:
        remaining = [i for i in all_items if i not in sample]
        extra = min(sample_size - len(sample), len(remaining))
        sample.extend(random.sample(remaining, extra))

    random.shuffle(sample)
    return sample[:sample_size]


def classify_item(client: Anthropic, model: str, text: str) -> dict:
    """단일 건 v3 분류"""
    if not text or len(text.strip()) < 10:
        return {"topic": "기타", "confidence": 1.0, "reason": "빈 텍스트"}

    try:
        response = client.messages.create(
            model=model,
            max_tokens=200,
            system=CLASSIFY_PROMPT,
            messages=[{"role": "user", "content": text[:500]}],
        )
        raw = response.content[0].text.strip()
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()
        if raw.startswith("{{") and raw.endswith("}}"):
            raw = raw[1:-1]
        result = json.loads(raw)

        topic = result.get("topic", "기타")
        if topic not in V3_TOPICS:
            topic = "기타"

        return {
            "topic": topic,
            "confidence": max(0.0, min(1.0, float(result.get("confidence", 0.5)))),
            "reason": result.get("reason", ""),
        }
    except Exception as e:
        return {"topic": "기타", "confidence": 0.0, "reason": f"오류: {e}"}


def run_classification_round(
    client: Anthropic, model: str, items: list, max_workers: int = 5
) -> list:
    """전체 샘플에 대해 1회차 분류"""
    results = [None] * len(items)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {}
        for i, item in enumerate(items):
            content_field = item.get("_content_field", "message")
            text = item.get(content_field, "") or item.get("textBody", "") or item.get("body", "")
            future = executor.submit(classify_item, client, model, text)
            future_map[future] = i

        done = 0
        for future in as_completed(future_map):
            idx = future_map[future]
            results[idx] = future.result()
            done += 1
            if done % 20 == 0:
                print(f"    {done}/{len(items)} 완료")

    return results


def analyze_results(items: list, round1: list, round2: list = None):
    """분류 결과 분석"""
    print("\n" + "=" * 70)
    print("  v3 분류 체계 검증 결과")
    print("=" * 70)

    # 1. 분포
    topic_dist = Counter(r["topic"] for r in round1)
    print(f"\n  [1] v3 Topic 분포 (Round 1)")
    for topic, count in topic_dist.most_common():
        pct = count / len(round1) * 100
        print(f"    {topic}: {count}건 ({pct:.1f}%)")

    # 2. 신뢰도 분포
    confidences = [r["confidence"] for r in round1]
    avg_conf = sum(confidences) / len(confidences) if confidences else 0
    low_conf = sum(1 for c in confidences if c < 0.7)
    print(f"\n  [2] 신뢰도")
    print(f"    평균: {avg_conf:.3f}")
    print(f"    0.7 미만 (경계 사례): {low_conf}건 ({low_conf/len(round1)*100:.1f}%)")

    # 3. v2 → v3 매핑 분석
    v2_to_v3 = Counter()
    for item, r1 in zip(items, round1):
        v2_topic = item.get("classification", {}).get("topic", "미분류")
        v3_topic = r1["topic"]
        v2_to_v3[(v2_topic, v3_topic)] += 1

    print(f"\n  [3] v2 → v3 매핑 패턴")
    print(f"    {'v2 Topic':<16} → {'v3 Topic':<12} : 건수")
    print(f"    {'-'*50}")
    for (v2, v3), count in sorted(v2_to_v3.items(), key=lambda x: -x[1]):
        print(f"    {v2:<16} → {v3:<12} : {count}")

    # 4. 일관성 (2회차가 있으면)
    if round2:
        match = sum(1 for r1, r2 in zip(round1, round2) if r1["topic"] == r2["topic"])
        consistency = match / len(round1) * 100
        print(f"\n  [4] 일관성 (2회 분류 일치율)")
        print(f"    일치: {match}/{len(round1)} ({consistency:.1f}%)")

        # 불일치 건 분석
        mismatches = []
        for i, (r1, r2) in enumerate(zip(round1, round2)):
            if r1["topic"] != r2["topic"]:
                content_field = items[i].get("_content_field", "message")
                text = items[i].get(content_field, "")[:80]
                mismatches.append({
                    "text": text,
                    "round1": r1["topic"],
                    "round2": r2["topic"],
                    "r1_conf": r1["confidence"],
                    "r2_conf": r2["confidence"],
                })

        if mismatches:
            print(f"\n    불일치 건 ({len(mismatches)}건):")
            for m in mismatches[:10]:
                print(f"      R1={m['round1']}({m['r1_conf']:.2f}) vs R2={m['round2']}({m['r2_conf']:.2f})")
                print(f"        \"{m['text']}...\"")

    # 5. 경계 사례 (낮은 신뢰도)
    low_conf_items = []
    for item, r1 in zip(items, round1):
        if r1["confidence"] < 0.7:
            content_field = item.get("_content_field", "message")
            text = item.get(content_field, "")[:80]
            low_conf_items.append({
                "text": text,
                "v3_topic": r1["topic"],
                "confidence": r1["confidence"],
                "reason": r1["reason"],
                "v2_topic": item.get("classification", {}).get("topic", "미분류"),
            })

    if low_conf_items:
        print(f"\n  [5] 경계 사례 (신뢰도 < 0.7, 상위 10건)")
        for lc in sorted(low_conf_items, key=lambda x: x["confidence"])[:10]:
            print(f"    [{lc['v3_topic']}] conf={lc['confidence']:.2f} (v2: {lc['v2_topic']})")
            print(f"      \"{lc['text']}...\"")
            print(f"      사유: {lc['reason']}")

    return {
        "topic_distribution": dict(topic_dist),
        "avg_confidence": round(avg_conf, 4),
        "low_confidence_count": low_conf,
        "low_confidence_rate": round(low_conf / len(round1) * 100, 2),
        "v2_to_v3_mapping": {f"{v2}->{v3}": c for (v2, v3), c in v2_to_v3.items()},
        "consistency": (
            round(sum(1 for r1, r2 in zip(round1, round2) if r1["topic"] == r2["topic"]) / len(round1) * 100, 2)
            if round2 else None
        ),
        "sample_size": len(items),
    }


def main():
    parser = argparse.ArgumentParser(description="v3 분류 체계 검증")
    parser.add_argument("--sample-size", type=int, default=50, help="검증 샘플 크기")
    parser.add_argument("--rounds", type=int, default=2, help="분류 반복 횟수 (일관성 검증)")
    parser.add_argument("--model", default="claude-haiku-4-5-20251001", help="분류 모델")
    parser.add_argument("--max-workers", type=int, default=5, help="병렬 호출 수")
    parser.add_argument("--seed", type=int, default=42, help="랜덤 시드")
    args = parser.parse_args()

    random.seed(args.seed)
    data_dir = "./data/classified_data_two_axis"

    if not os.path.exists(data_dir):
        print(f"  데이터 디렉토리 없음: {data_dir}")
        return

    print("=" * 70)
    print("  v3 분류 체계 검증")
    print(f"  샘플: {args.sample_size}건, 반복: {args.rounds}회")
    print(f"  모델: {args.model}")
    print("=" * 70)

    # 데이터 로드
    print("\n[1] 샘플 데이터 로드")
    items = load_sample_data(data_dir, args.sample_size)
    print(f"  {len(items)}건 샘플 추출")

    v2_dist = Counter(i.get("classification", {}).get("topic", "미분류") for i in items)
    print(f"  v2 분포: {dict(v2_dist)}")

    client = Anthropic()

    # Round 1
    print(f"\n[2] Round 1 분류")
    round1 = run_classification_round(client, args.model, items, args.max_workers)

    # Round 2 (일관성)
    round2 = None
    if args.rounds >= 2:
        print(f"\n[3] Round 2 분류 (일관성 검증)")
        round2 = run_classification_round(client, args.model, items, args.max_workers)

    # 분석
    results = analyze_results(items, round1, round2)

    # 결과 저장
    output_dir = "./reports"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "v3_taxonomy_validation.json")

    # 상세 결과 (개별 건)
    detailed = []
    for i, item in enumerate(items):
        content_field = item.get("_content_field", "message")
        detailed.append({
            "text": (item.get(content_field, "") or "")[:200],
            "v2_topic": item.get("classification", {}).get("topic", "미분류"),
            "v2_sentiment": item.get("classification", {}).get("sentiment", "미분류"),
            "v3_topic_r1": round1[i]["topic"],
            "v3_confidence_r1": round1[i]["confidence"],
            "v3_reason_r1": round1[i]["reason"],
            "v3_topic_r2": round2[i]["topic"] if round2 else None,
            "v3_confidence_r2": round2[i]["confidence"] if round2 else None,
        })

    save_data = {
        "summary": results,
        "detailed": detailed,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2)
    print(f"\n  결과 저장: {output_path}")


if __name__ == "__main__":
    main()
