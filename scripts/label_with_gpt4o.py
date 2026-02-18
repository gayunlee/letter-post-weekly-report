#!/usr/bin/env python3
"""GPT-4o-mini로 VOC 라벨링 (5개 카테고리)

사용법:
    python scripts/label_with_gpt4o.py
"""
import json
import os
import sys
from pathlib import Path
from typing import List, Dict
from collections import Counter
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))

from openai import OpenAI
from src.bigquery.client import BigQueryClient

# 5개 카테고리 정의
CATEGORIES = {
    "서비스 이슈": "서비스/운영/CS에 대한 불만, 문의, 개선요청 (콘텐츠 업로드 지연, 링크 오류, 결제 문제 등)",
    "서비스 칭찬": "콘텐츠/마스터/플랫폼에 대한 구체적인 감사, 칭찬 (강의 좋았다, 덕분에 배웠다 등)",
    "투자 질문": "종목, 비중, 매매 타이밍, 포트폴리오 전략 등 투자 관련 질문",
    "정보/의견": "뉴스, 분석, 시장 전망 공유 + 수익/손실 후기, 투자 심리 표현",
    "일상 소통": "인사, 안부, 축하, 조문, 가입인사 등 일상적 소통",
}

SYSTEM_PROMPT = """너는 금융 콘텐츠 플랫폼의 VOC(고객의 소리) 분류 전문가야.

## 분류 카테고리
1. 서비스 이슈: 서비스/운영에 대한 불만, 문의, 개선요청 (대응 필요)
   - 콘텐츠 업로드 지연, 링크 오류, 결제 문제, 답변 없음 불만 등

2. 서비스 칭찬: 콘텐츠/마스터/플랫폼에 대한 구체적 감사
   - "강의 덕분에 많이 배웠어요", "콘텐츠 퀄리티가 좋아요"

3. 투자 질문: 투자 관련 질문 (종목, 비중, 전략)
   - "삼성전자 지금 사도 될까요?", "비중 어떻게 가져가야 하나요?"

4. 정보/의견: 뉴스, 분석 공유 + 수익/손실 후기 + 시장 심리
   - "오늘 시장 분석입니다", "10% 수익 봤어요", "요즘 장이 불안하네요"

5. 일상 소통: 인사, 안부, 잡담
   - "새해 복 많이 받으세요", "가입인사드립니다", "고인의 명복을 빕니다"

## 중요 구분 기준
- 서비스 문제(링크, 결제, 답변 없음) → 서비스 이슈
- 마스터/콘텐츠 칭찬 → 서비스 칭찬
- 시장/증시/본인 수익 이야기 → 정보/의견 (서비스 이슈 아님!)
- 형식적 감사("감사합니다" 한마디) → 일상 소통
- 구체적 감사("강의 덕분에...") → 서비스 칭찬
"""


def load_data_from_bigquery(start_date: str, end_date: str) -> List[Dict]:
    """BigQuery에서 데이터 로드"""
    client = BigQueryClient()
    project_id = client.project_id

    query = f"""
    SELECT
        _id as id,
        message as text,
        'letter' as source,
        createdAt
    FROM `{project_id}.us_plus.usermastermessages`
    WHERE TIMESTAMP(createdAt) >= '{start_date}'
        AND TIMESTAMP(createdAt) < '{end_date}'
        AND type = 'LETTER' AND isBlock = 'false'

    UNION ALL

    SELECT
        _id as id,
        COALESCE(textBody, body) as text,
        'post' as source,
        createdAt
    FROM `{project_id}.us_plus.posts`
    WHERE TIMESTAMP(createdAt) >= '{start_date}'
        AND TIMESTAMP(createdAt) < '{end_date}'
        AND isBlock = 'false' AND deleted = 'false'

    ORDER BY createdAt DESC
    """

    results = client.execute_query(query)
    # 빈 텍스트 제거
    return [r for r in results if r.get('text') and len(r['text'].strip()) > 5]


def classify_batch(client: OpenAI, items: List[Dict], batch_size: int = 20) -> List[Dict]:
    """배치 단위로 GPT-4o-mini 분류"""

    results = []
    categories_list = list(CATEGORIES.keys())

    for i in range(0, len(items), batch_size):
        batch = items[i:i+batch_size]

        # 배치 텍스트 구성
        items_text = ""
        for j, item in enumerate(batch):
            text = item["text"][:300].replace("\n", " ").strip()
            items_text += f"{j+1}. {text}\n"

        user_prompt = f"""다음 {len(batch)}개의 글을 분류해줘.

{items_text}

각 글의 카테고리를 JSON 배열로 답변해:
[{{"id": 1, "category": "카테고리명"}}, ...]

카테고리는 반드시 다음 중 하나: {categories_list}"""

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0,
                response_format={"type": "json_object"}
            )

            content = response.choices[0].message.content
            parsed = json.loads(content)

            # 결과가 리스트인지 확인
            if isinstance(parsed, dict) and "results" in parsed:
                classifications = parsed["results"]
            elif isinstance(parsed, list):
                classifications = parsed
            else:
                classifications = list(parsed.values())[0] if parsed else []

            for item, cls in zip(batch, classifications):
                cat = cls.get("category", "") if isinstance(cls, dict) else ""
                if cat in CATEGORIES:
                    item["category"] = cat
                else:
                    item["category"] = "일상 소통"  # fallback
                results.append(item)

        except Exception as e:
            print(f"  오류: {e}")
            for item in batch:
                item["category"] = "일상 소통"
                results.append(item)

        print(f"  진행: {min(i+batch_size, len(items))}/{len(items)}")

    return results


def main():
    project_root = Path(__file__).parent.parent
    output_path = project_root / "data" / "labeling" / "gpt4o_labeled.json"

    # 날짜 범위 (2주간)
    start_date = "2026-01-05"
    end_date = "2026-01-19"

    print("=" * 60)
    print("GPT-4o-mini VOC 라벨링 (5개 카테고리)")
    print("=" * 60)
    print(f"기간: {start_date} ~ {end_date}")

    # BigQuery에서 데이터 로드
    print("\n1. BigQuery 데이터 로드")
    items = load_data_from_bigquery(start_date, end_date)
    print(f"  {len(items)}건 로드")

    if not items:
        print("데이터가 없습니다.")
        return

    # GPT-4o-mini 분류
    print("\n2. GPT-4o-mini 분류 중...")
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    labeled = classify_batch(client, items, batch_size=20)

    # 통계
    print("\n3. 분류 결과")
    dist = Counter(item["category"] for item in labeled)
    for cat, count in dist.most_common():
        pct = count / len(labeled) * 100
        print(f"  {cat}: {count}건 ({pct:.1f}%)")

    # 카테고리별 샘플
    print("\n4. 카테고리별 샘플")
    for cat in CATEGORIES:
        cat_items = [d for d in labeled if d["category"] == cat][:3]
        print(f"\n  [{cat}] ({len([d for d in labeled if d['category'] == cat])}건)")
        for item in cat_items:
            text = item["text"][:60].replace("\n", " ")
            print(f"    - {text}...")

    # 저장
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(labeled, f, ensure_ascii=False, indent=2)
    print(f"\n저장: {output_path}")
    print(f"총 {len(labeled)}건")


if __name__ == "__main__":
    main()
