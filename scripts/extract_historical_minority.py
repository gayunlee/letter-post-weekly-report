"""과거 데이터에서 소수 카테고리 후보 추출

1단계: BigQuery에서 기존 학습 데이터 이전 기간의 편지/게시글 추출
2단계: 로컬 v3c 모델로 추론 → 서비스 피드백 / 운영 피드백 예측 건만 필터
3단계: 필터된 후보를 Opus로 재라벨링 (별도 실행)

사용법:
    # 1+2단계: 추출 + 로컬 모델 필터링
    python3 scripts/extract_historical_minority.py

    # 3단계: Opus 라벨링 (별도)
    python3 scripts/extract_historical_minority.py --label-with-opus
"""
import sys
import os
import json
import argparse
import time
from collections import Counter
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

OUTPUT_DIR = "./data/historical_minority"
MODEL_DIR = "models/v3/topic_v3c_full_clean/final_model"

# 기존 학습 데이터 시작일 이전까지 추출
CUTOFF_DATE = "2025-12-08T00:00:00Z"


def extract_from_bigquery():
    """BigQuery에서 과거 편지/게시글 추출"""
    from src.bigquery.client import BigQueryClient
    client = BigQueryClient()

    print("  [1/2] 편지 추출 중...")
    letters_query = f"""
    SELECT _id, message, masterId, type, createdAt
    FROM us_plus.usermastermessages
    WHERE type = 'LETTER'
      AND createdAt < '{CUTOFF_DATE}'
    """
    letters = client.execute_query(letters_query)
    print(f"    편지: {len(letters)}건")

    print("  [2/2] 게시글 추출 중...")
    posts_query = f"""
    SELECT _id, title, textBody, body, createdAt
    FROM us_plus.posts
    WHERE deleted = 'false'
      AND createdAt < '{CUTOFF_DATE}'
    """
    posts = client.execute_query(posts_query)
    print(f"    게시글: {len(posts)}건")

    # 통합 포맷으로 변환
    items = []
    for row in letters:
        text = row.get('message', '') or ''
        if len(text.strip()) < 20:
            continue
        items.append({
            '_id': row['_id'],
            'text': text[:500],
            '_source': 'letter',
            'createdAt': row['createdAt'],
        })

    for row in posts:
        text = row.get('textBody') or row.get('body', '') or ''
        if len(text.strip()) < 20:
            continue
        items.append({
            '_id': row['_id'],
            'text': text[:500],
            '_source': 'post',
            'createdAt': row['createdAt'],
        })

    print(f"    유효 항목: {len(items)}건 (20자 이상)")
    return items


def run_local_model(items, batch_size=64):
    """로컬 v3c 모델로 추론하여 소수 카테고리 필터링"""
    print(f"\n  로컬 모델 로드: {MODEL_DIR}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()

    with open(f'{MODEL_DIR}/category_config.json') as f:
        config = json.load(f)
    id_to_cat = {int(k): v for k, v in config['id_to_category'].items()}

    minority_targets = {'서비스 피드백', '운영 피드백'}
    all_preds = []
    minority_items = []

    total = len(items)
    start_time = time.time()

    for i in range(0, total, batch_size):
        batch = items[i:i + batch_size]
        texts = [item['text'] for item in batch]

        inputs = tokenizer(
            texts,
            truncation=True,
            padding='max_length',
            max_length=256,
            return_tensors='pt'
        )

        with torch.no_grad():
            outputs = model(**inputs)

        probs = torch.softmax(outputs.logits, dim=-1)
        pred_ids = probs.argmax(dim=-1)

        for j, item in enumerate(batch):
            pred_cat = id_to_cat[pred_ids[j].item()]
            conf = probs[j][pred_ids[j]].item()
            all_preds.append(pred_cat)

            if pred_cat in minority_targets:
                item['model_pred'] = pred_cat
                item['model_conf'] = round(conf, 4)

                # top-2 정보 추가
                topk = torch.topk(probs[j], k=2)
                item['model_top2'] = id_to_cat[topk.indices[1].item()]
                item['model_top2_conf'] = round(topk.values[1].item(), 4)
                minority_items.append(item)

        if (i + batch_size) % (batch_size * 10) == 0 or i + batch_size >= total:
            elapsed = time.time() - start_time
            rate = (i + len(batch)) / elapsed
            print(f"    {min(i + batch_size, total)}/{total} 추론 완료 ({rate:.0f}건/초)")

    # 분포 출력
    pred_dist = Counter(all_preds)
    print(f"\n  로컬 모델 예측 분포 (전체 {total}건):")
    for cat, cnt in pred_dist.most_common():
        pct = cnt / total * 100
        marker = " ← TARGET" if cat in minority_targets else ""
        print(f"    {cat}: {cnt}건 ({pct:.1f}%){marker}")

    return minority_items


def label_with_opus(items, max_workers=5, model_name="claude-opus-4-6"):
    """필터된 후보를 Opus로 재라벨링"""
    from anthropic import Anthropic
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from dotenv import load_dotenv
    load_dotenv()

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

    V3_TOPICS = ["운영 피드백", "서비스 피드백", "콘텐츠·투자", "일상·감사", "기타"]

    client = Anthropic()
    output_path = os.path.join(OUTPUT_DIR, "opus_labeled_minority.json")

    # 이미 라벨링된 항목 로드 (재시작 지원)
    done_map = {}
    if os.path.exists(output_path):
        with open(output_path, encoding='utf-8') as f:
            done_map = {item['_id']: item for item in json.load(f)}
    remaining = [item for item in items if item['_id'] not in done_map]

    print(f"\n  Opus 라벨링: 전체 {len(items)}건, 완료 {len(done_map)}건, 남은 {len(remaining)}건")

    if not remaining:
        print("  모든 항목 라벨링 완료!")
        return list(done_map.values())

    def classify_single(text):
        if not text or len(text.strip()) < 3:
            return {"topic": "기타", "confidence": 1.0}
        try:
            response = client.messages.create(
                model=model_name,
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

    results = list(done_map.values())
    start_time = time.time()
    batch_size = 50

    for batch_start in range(0, len(remaining), batch_size):
        batch = remaining[batch_start:batch_start + batch_size]

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {}
            for i, item in enumerate(batch):
                future = executor.submit(classify_single, item['text'])
                future_map[future] = i

            batch_results = [None] * len(batch)
            for future in as_completed(future_map):
                idx = future_map[future]
                batch_results[idx] = future.result()

        for i, item in enumerate(batch):
            opus_result = batch_results[i]
            labeled = {
                '_id': item['_id'],
                'text': item['text'],
                '_source': item['_source'],
                'createdAt': item.get('createdAt', ''),
                'model_pred': item.get('model_pred', ''),
                'model_conf': item.get('model_conf', 0),
                'v3c_topic': opus_result['topic'],
                'v3_confidence': opus_result['confidence'],
            }
            results.append(labeled)

        # 중간 저장
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        elapsed = time.time() - start_time
        total_done = len(done_map) + batch_start + len(batch)
        rate = (batch_start + len(batch)) / elapsed if elapsed > 0 else 0
        print(f"    {total_done}/{len(items)} 완료 ({rate:.1f}건/초)")

    # 최종 통계
    opus_dist = Counter(r['v3c_topic'] for r in results)
    print(f"\n  Opus 라벨링 결과 ({len(results)}건):")
    for topic, cnt in opus_dist.most_common():
        pct = cnt / len(results) * 100
        print(f"    {topic}: {cnt}건 ({pct:.1f}%)")

    # 실제 서비스/운영으로 확정된 건수
    confirmed = sum(1 for r in results if r['v3c_topic'] in {'서비스 피드백', '운영 피드백'})
    print(f"\n  소수 카테고리 확정: {confirmed}건 / {len(results)}건")
    print(f"  저장: {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="과거 데이터 소수 카테고리 후보 추출")
    parser.add_argument("--label-with-opus", action="store_true",
                        help="필터된 후보를 Opus로 라벨링")
    parser.add_argument("--max-workers", type=int, default=5)
    parser.add_argument("--model", default="claude-opus-4-6")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("  과거 데이터 소수 카테고리 후보 추출")
    print("=" * 70)

    filtered_path = os.path.join(OUTPUT_DIR, "model_filtered_minority.json")

    if args.label_with_opus:
        # 3단계: Opus 라벨링
        if not os.path.exists(filtered_path):
            print(f"  ERROR: {filtered_path} 없음. 먼저 필터링을 실행하세요.")
            return
        with open(filtered_path, encoding='utf-8') as f:
            filtered = json.load(f)
        print(f"\n  필터된 후보: {len(filtered)}건")
        label_with_opus(filtered, max_workers=args.max_workers, model_name=args.model)
        return

    # 1단계: BigQuery 추출
    raw_path = os.path.join(OUTPUT_DIR, "raw_historical.json")
    if os.path.exists(raw_path):
        print(f"\n  캐시 사용: {raw_path}")
        with open(raw_path, encoding='utf-8') as f:
            items = json.load(f)
        print(f"  로드: {len(items)}건")
    else:
        items = extract_from_bigquery()
        with open(raw_path, 'w', encoding='utf-8') as f:
            json.dump(items, f, ensure_ascii=False, indent=2)
        print(f"  저장: {raw_path} ({len(items)}건)")

    # 2단계: 로컬 모델 필터링
    print(f"\n{'='*70}")
    print(f"  로컬 v3c 모델 추론 + 소수 카테고리 필터링")
    print(f"{'='*70}")

    minority_items = run_local_model(items)

    with open(filtered_path, 'w', encoding='utf-8') as f:
        json.dump(minority_items, f, ensure_ascii=False, indent=2)

    print(f"\n  필터된 소수 카테고리 후보: {len(minority_items)}건")
    print(f"  저장: {filtered_path}")
    print(f"\n  다음 단계:")
    print(f"    python3 scripts/extract_historical_minority.py --label-with-opus")


if __name__ == "__main__":
    main()
