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

V3_TOPICS = ["운영 피드백", "서비스 피드백", "콘텐츠·투자", "일상·감사", "기타"]

CLASSIFY_PROMPT = """당신은 금융 교육 플랫폼의 VOC 데이터 분류기입니다.

## 분류 우선순위 (위에서 아래로 판별, 먼저 해당되면 확정)

### 1순위: 운영 피드백
운영팀/사업팀이 사람 대 사람으로 처리할 요청·이슈.
예: 세미나 문의, 환불 요청, 멤버십 가입/해지 문의, 배송, 가격 정책 질문, 구독 관련 민원

### 2순위: 서비스 피드백
개발팀이 시스템을 수정해야 하는 기술적 이슈·요청.
예: 앱 크래시, 결제 오류(시스템), 로그인 실패, 기능 요청, 사칭 제보, 링크 오류

### 3순위: 콘텐츠·투자 ⭐ 가장 넓은 범위
투자/콘텐츠/마스터/시장/종목과 **조금이라도** 관련된 모든 글.
아래 신호가 **하나라도** 있으면 무조건 콘텐츠·투자:
- 종목명, 섹터, ETF, 지수, 코인 언급
- 수익/손실/매수/매도/포트폴리오/리밸런싱
- 마스터의 분석/강의/뷰/관점에 대한 반응 (칭찬이든 비판이든)
- 시장 상황, 거시경제, 금리, 환율 언급
- "강의 잘 들었습니다", "덕분에 공부했습니다" 등 학습 언급
- 멤버십 불만이지만 콘텐츠 품질이 이유인 경우

### 4순위: 일상·감사
위 1~3순위에 **전혀** 해당하지 않는 순수 인사·감사·안부·응원·격려·잡담.
투자/콘텐츠 신호가 **0개**일 때만 해당.
예: "감사합니다 명절 잘 보내세요", "힘내세요", 날씨 이야기, MBTI, 자기소개

### 5순위: 기타
무의미 노이즈, 분류 불가. 예: ".", "1", "?", 자음만, 테스트 글

## 핵심 규칙
- 감사/응원 + 투자 신호 → **콘텐츠·투자** (3순위 우선)
- 감사/응원만, 투자 신호 0개 → **일상·감사** (4순위)
- 애매하면 → **콘텐츠·투자** (투자 교육 커뮤니티이므로 콘텐츠·투자가 기본값)

## 마스터란?
투자 교육 커뮤니티를 운영하는 금융 콘텐츠 크리에이터입니다.

## 응답: JSON만 출력
{"topic": "콘텐츠·투자", "confidence": 0.92}"""

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
    if not text or len(text.strip()) < 3:
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
    parser.add_argument("--model", default="claude-opus-4-6")
    parser.add_argument("--max-workers", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true", help="비용만 추정")
    args = parser.parse_args()

    data_dir = "./data/classified_data_two_axis"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "labeled_all_v3b.json")

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
