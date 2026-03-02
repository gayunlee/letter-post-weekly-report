"""v3 분류 체계 LLM 라벨링 스크립트

기존 데이터를 LLM으로 새 v3 5분류 체계(운영 피드백/서비스 피드백/콘텐츠 반응/투자 담론/기타)로 분류.
결과를 data/training_data/v3/ 에 저장.

사용법:
    python3 scripts/label_v3_taxonomy.py
    python3 scripts/label_v3_taxonomy.py --max-workers 10
    python3 scripts/label_v3_taxonomy.py --dry-run          # 비용만 추정
"""
import sys
import os
import json
import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

V3_TOPICS = ["운영 피드백", "서비스 피드백", "콘텐츠 반응", "투자 담론", "기타"]

CLASSIFY_PROMPT = """당신은 금융 교육 플랫폼의 VOC 데이터 분류기입니다.

## 분류 기준

| 카테고리 | 기준 | 예시 |
|---------|------|------|
| 운영 피드백 | 운영팀/사업팀이 사람 대 사람으로 처리 | 세미나 문의, 환불 요청, 멤버십 문의, 배송, 가격 정책 질문 |
| 서비스 피드백 | 개발팀이 시스템 수정 필요 | 앱 크래시, 결제 오류(시스템), 로그인 실패, 기능 요청, 사칭 제보 |
| 콘텐츠 반응 | 마스터/콘텐츠가 주된 대상 | 강의 후기, 마스터 칭찬/불만, 콘텐츠 품질 의견, 리포트 반응 |
| 투자 담론 | 시장/종목/투자전략이 주된 대상 | 투자 질문, 수익/손실 공유, 시장 분석, 종목 토론, 매매 타이밍 |
| 기타 | 분류 불필요 | 새해 인사, 안부, 축하, 테스트 글 |

## 경계 판별
- 시스템 장애/버그 → 서비스 피드백
- 정책/프로세스/인력 대응 → 운영 피드백
- 마스터/콘텐츠가 주된 대상 → 콘텐츠 반응
- 시장/종목/투자가 주된 대상 → 투자 담론
- 인사/잡담 → 기타

## 마스터란?
투자 교육 커뮤니티를 운영하는 금융 콘텐츠 크리에이터입니다.

## 응답: JSON만 출력
{"topic": "콘텐츠 반응", "confidence": 0.92}"""

OUTPUT_DIR = "./data/training_data/v3"


def load_existing_data(data_dir: str) -> list:
    """기존 분류 데이터 전체 로드"""
    all_items = []
    for filename in sorted(os.listdir(data_dir)):
        if not filename.endswith(".json"):
            continue
        filepath = os.path.join(data_dir, filename)
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
        week = filename.replace(".json", "")
        for item in data.get("letters", []):
            item["_source"] = "letter"
            item["_week"] = week
            item["_text"] = item.get("message", "")
            all_items.append(item)
        for item in data.get("posts", []):
            item["_source"] = "post"
            item["_week"] = week
            item["_text"] = item.get("textBody") or item.get("body", "")
            all_items.append(item)
    return all_items


def classify_single(client: Anthropic, model: str, text: str) -> dict:
    """단일 건 v3 분류"""
    if not text or len(text.strip()) < 10:
        return {"topic": "기타", "confidence": 1.0}
    try:
        response = client.messages.create(
            model=model,
            max_tokens=100,
            system=CLASSIFY_PROMPT,
            messages=[{"role": "user", "content": text[:500]}],
            timeout=30.0,
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
        conf = max(0.0, min(1.0, float(result.get("confidence", 0.5))))

        return {"topic": topic, "confidence": conf}
    except Exception as e:
        return {"topic": "기타", "confidence": 0.0, "error": str(e)[:80]}


def load_progress(output_path: str) -> dict:
    """이미 라벨링된 결과 로드 (재시작 지원)"""
    if os.path.exists(output_path):
        with open(output_path, encoding="utf-8") as f:
            data = json.load(f)
        return {item["_id"]: item for item in data if "_id" in item}
    return {}


def main():
    parser = argparse.ArgumentParser(description="v3 분류 체계 LLM 라벨링")
    parser.add_argument("--model", default="claude-haiku-4-5-20251001")
    parser.add_argument("--max-workers", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true", help="비용만 추정")
    args = parser.parse_args()

    data_dir = "./data/classified_data_two_axis"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "labeled_all.json")

    # 데이터 로드
    print("=" * 70)
    print("  v3 분류 체계 LLM 라벨링")
    print("=" * 70)

    items = load_existing_data(data_dir)
    print(f"\n  전체 데이터: {len(items)}건")

    if args.dry_run:
        # 비용 추정: Haiku input ~500 tokens, output ~30 tokens per item
        est_input = len(items) * 500 / 1000 * 0.001
        est_output = len(items) * 30 / 1000 * 0.005
        print(f"\n  비용 추정:")
        print(f"    Input:  ~${est_input:.2f}")
        print(f"    Output: ~${est_output:.2f}")
        print(f"    합계:   ~${est_input + est_output:.2f}")
        return

    # 진행 상황 로드 (중단 후 재시작 지원)
    done_map = load_progress(output_path)
    remaining = [item for item in items if item.get("_id") not in done_map]
    print(f"  이미 완료: {len(done_map)}건, 남은 작업: {len(remaining)}건")

    if not remaining:
        print("  모든 항목 라벨링 완료!")
        return

    client = Anthropic()
    results = list(done_map.values())
    start_time = time.time()
    batch_size = 50

    for batch_start in range(0, len(remaining), batch_size):
        batch_end = min(batch_start + batch_size, len(remaining))
        batch = remaining[batch_start:batch_end]

        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            future_map = {}
            for i, item in enumerate(batch):
                future = executor.submit(classify_single, client, args.model, item["_text"])
                future_map[future] = i

            batch_results = [None] * len(batch)
            for future in as_completed(future_map):
                idx = future_map[future]
                batch_results[idx] = future.result()

        # 결과 병합
        for i, item in enumerate(batch):
            v3_result = batch_results[i]
            labeled = {
                "_id": item.get("_id", ""),
                "_source": item["_source"],
                "_week": item["_week"],
                "text": item["_text"][:500],
                "v2_topic": item.get("classification", {}).get("topic", ""),
                "v2_sentiment": item.get("classification", {}).get("sentiment", ""),
                "v3_topic": v3_result["topic"],
                "v3_confidence": v3_result["confidence"],
                "masterName": item.get("masterName", ""),
            }
            results.append(labeled)

        # 중간 저장 (50건마다)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        elapsed = time.time() - start_time
        total_done = len(done_map) + batch_end
        rate = (batch_end) / elapsed if elapsed > 0 else 0
        print(f"  {total_done}/{len(items)} 완료 ({rate:.1f}건/초)")

    # 최종 통계
    from collections import Counter
    v3_dist = Counter(r["v3_topic"] for r in results)
    print(f"\n{'='*70}")
    print(f"  라벨링 완료: {len(results)}건")
    print(f"  v3 분포:")
    for topic, count in v3_dist.most_common():
        pct = count / len(results) * 100
        print(f"    {topic}: {count}건 ({pct:.1f}%)")

    avg_conf = sum(r["v3_confidence"] for r in results) / len(results)
    print(f"  평균 신뢰도: {avg_conf:.3f}")
    print(f"  저장: {output_path}")


if __name__ == "__main__":
    main()
