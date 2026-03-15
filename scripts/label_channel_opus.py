"""채널톡 3분류 Opus 라벨링 스크립트

기존 분류 데이터에서 미사용 건을 추출하여 Opus로 3분류 라벨링.
golden set + 기존 학습 데이터 chatId 제외.

사용법:
    python3 scripts/label_channel_opus.py
    python3 scripts/label_channel_opus.py --limit 100  # 테스트용
"""
import json
import os
import sys
import time
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).parent.parent

SYSTEM_PROMPT = """당신은 금융 교육 플랫폼의 CS 문의 분류기입니다.
채널톡으로 접수된 고객 상담을 **3개 카테고리 중 하나**로 분류합니다.

## 분류 카테고리

### 결제·구독
결제, 환불, 카드 변경, 영수증, 구독 신청/해지/변경, 멤버십, 자동결제.
- 돈이 오가는 모든 것은 여기
- "환불해주세요", "구독 해지", "결제 안 됨", "상품 변경"
- "강의 결제", "수강권 환불" → 돈 관련이면 무조건 여기

### 콘텐츠·수강
이미 이용 중인 강의 접근, 수강 방법, 자료/교재, 오프라인 행사/세미나.
- "강의 어디서 보나요", "교재 배송", "세미나 참석", "동반자 가능한가요"
- 투자 종목 질문, 마스터 콘텐츠 관련
- 구독 범위/포함 콘텐츠 문의 (결제 의도 없으면)

### 기술·오류
시스템 문제, 로그인 불가, 앱 오류, 기능 건의.
- "로그인 안 됨", "앱이 안 열려요", "링크가 안 돼요"
- "프린트 기능 추가해주세요" (기술적 기능 건의)

## 우선순위 규칙
1. **결제·구독 우선**: 돈/결제/환불/구독이 언급되면 무조건 결제·구독
2. **기술·오류는 확실할 때만**: 시스템 문제가 명확한 경우만
3. **애매하면 콘텐츠·수강**: 결제도 기술도 아니면 콘텐츠·수강

## 워크플로우 버튼
메시지에 워크플로우 버튼 텍스트가 포함될 수 있습니다.
버튼은 참고만 하고, **실제 텍스트 내용 기준**으로 분류하세요.

## 응답
JSON만 출력: {"topic": "결제·구독", "confidence": 0.95}
topic은 반드시 "결제·구독", "콘텐츠·수강", "기술·오류" 중 하나."""

VALID_TOPICS = {"결제·구독", "콘텐츠·수강", "기술·오류"}

WORKFLOW_NOISE = [
    "그 외 기타 문의(오류/구독해지/환불)", "그 외 기타 문의",
    "💬1:1 상담 문의하기", "💬 1:1 고객센터 문의하기",
    "💬상담매니저에게 직접 문의", "어스 구독신청/결제하기",
    "어스 이용방법", "사이트 및 동영상 오류", "수강 및 상품문의",
    "라이브 콘텐츠 참여 방법", "수강방법", "↩ 이전으로",
    "✅ 1:1 문의하기", "구독 상품변경/결제정보 확인",
    "결제실패 후 카드변경 방법",
    "💬 구독 결제/변경/정보 확인 직접 문의하기",
    "구독상품 변경", "구독 결제/변경/정보 직접 문의하기",
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


def classify_single(client, text: str, workflow_buttons=None):
    """단일 건 Opus 분류."""
    cleaned = strip_workflow_buttons(text)
    if len(cleaned.strip()) < 5:
        cleaned = text

    prompt_text = cleaned[:700]
    if workflow_buttons:
        button_str = ", ".join(workflow_buttons[:3])
        prompt_text = f"[워크플로우: {button_str}]\n{prompt_text}"

    try:
        response = client.messages.create(
            model="claude-opus-4-20250514",
            max_tokens=100,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt_text}],
            timeout=60.0,
        )

        raw = response.content[0].text.strip()
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        result = json.loads(raw)
        topic = result.get("topic", "")
        confidence = float(result.get("confidence", 0.5))

        if topic not in VALID_TOPICS:
            topic = "콘텐츠·수강"  # fallback

        return {
            "topic": topic,
            "confidence": confidence,
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        }
    except Exception as e:
        return {
            "topic": "콘텐츠·수강",
            "confidence": 0.0,
            "error": str(e)[:100],
            "input_tokens": 0,
            "output_tokens": 0,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="라벨링 건수 제한 (0=전체)")
    parser.add_argument("--workers", type=int, default=5, help="병렬 워커 수")
    parser.add_argument("--output", default="data/channel_io/training_data/labeled_opus_8800.json")
    args = parser.parse_args()

    # 1. 제외할 chatId 수집
    golden_path = PROJECT_ROOT / "data/channel_io/golden/golden_multilabel_270.json"
    labeled_path = PROJECT_ROOT / "data/channel_io/training_data/labeled_1500_v5.json"

    with open(golden_path) as f:
        golden = json.load(f)
    with open(labeled_path) as f:
        labeled = json.load(f)

    exclude_ids = set(item["chatId"] for item in golden) | set(item["chatId"] for item in labeled)

    # 2. 미사용 데이터 추출
    classified_path = PROJECT_ROOT / "data/channel_io/classified_2025-08-01_2025-12-01.json"
    with open(classified_path) as f:
        data = json.load(f)

    all_items = data["items"]
    unlabeled = [item for item in all_items if item["chatId"] not in exclude_ids]

    if args.limit > 0:
        unlabeled = unlabeled[:args.limit]

    print("=" * 60)
    print("  채널톡 3분류 Opus 라벨링")
    print(f"  전체 데이터: {len(all_items)}")
    print(f"  제외 (golden+학습): {len(exclude_ids)}")
    print(f"  라벨링 대상: {len(unlabeled)}")
    print(f"  병렬 워커: {args.workers}")
    print("=" * 60)

    # 3. Opus 라벨링
    client = Anthropic()
    results = []
    total_input = 0
    total_output = 0
    errors = 0
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_map = {}
        for i, item in enumerate(unlabeled):
            future = executor.submit(
                classify_single, client,
                item.get("text", ""),
                item.get("workflow_buttons", []),
            )
            future_map[future] = i

        done = 0
        for future in as_completed(future_map):
            idx = future_map[future]
            item = unlabeled[idx]
            result = future.result()

            results.append({
                "chatId": item["chatId"],
                "text": item.get("text", ""),
                "topic": result["topic"],
            })

            total_input += result.get("input_tokens", 0)
            total_output += result.get("output_tokens", 0)
            if "error" in result:
                errors += 1

            done += 1
            if done % 100 == 0 or done == len(unlabeled):
                elapsed = time.time() - start_time
                rate = done / elapsed if elapsed > 0 else 0
                input_cost = total_input * 15 / 1_000_000
                output_cost = total_output * 75 / 1_000_000
                cost = input_cost + output_cost
                print(f"  {done}/{len(unlabeled)} ({rate:.1f}건/초) | 비용: ${cost:.2f} | 에러: {errors}")

    # 4. 저장
    output_path = PROJECT_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # 통계
    from collections import Counter
    dist = Counter(item["topic"] for item in results)
    elapsed = time.time() - start_time
    input_cost = total_input * 15 / 1_000_000
    output_cost = total_output * 75 / 1_000_000

    print(f"\n{'='*60}")
    print(f"  완료: {len(results)}건 라벨링")
    print(f"  분포:")
    for topic, count in dist.most_common():
        print(f"    {topic}: {count}건 ({count/len(results)*100:.1f}%)")
    print(f"  소요: {elapsed:.0f}초 ({elapsed/60:.1f}분)")
    print(f"  비용: ${input_cost + output_cost:.2f} (input: ${input_cost:.2f}, output: ${output_cost:.2f})")
    print(f"  에러: {errors}건")
    print(f"  저장: {output_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
