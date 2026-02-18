#!/usr/bin/env python3
"""라벨 정제 스크립트

자동 라벨링된 데이터를 LLM으로 검증/수정합니다.

사용법:
    python scripts/refine_labels.py
"""
import json
import os
import sys
from pathlib import Path
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()

from anthropic import Anthropic

NEW_CATEGORIES = {
    "긍정 피드백": "감사, 칭찬, 만족 표현, 수익 후기",
    "부정 피드백": "불만, 개선요청, 실망, 답답함 표현",
    "질문/문의": "투자 질문, 서비스 문의, 정보 요청",
    "정보 공유": "뉴스, 분석, 의견 공유, 정보 전달",
    "일상 소통": "인사, 안부, 축하, 조문, 잡담",
}

FEW_SHOT = """
예제:
- "감사합니다. 덕분에 수익 봤어요" → 긍정 피드백 (구체적 감사)
- "삼가 고인의 명복을 빕니다" → 일상 소통 (조문/안부)
- "새해 복 많이 받으세요" → 일상 소통 (인사)
- "왜 답변이 없나요?" → 부정 피드백 (불만)
- "링크가 안 열려요" → 질문/문의 (서비스 문의)
- "제 분석 공유합니다" → 정보 공유
- "어떻게 해야 할까요?" → 질문/문의 (질문)
"""


def refine_batch(client: Anthropic, items: List[Dict], batch_size: int = 20) -> List[Dict]:
    """배치 단위로 LLM 검증"""

    category_desc = "\n".join([f"- {k}: {v}" for k, v in NEW_CATEGORIES.items()])

    results = []

    for i in range(0, len(items), batch_size):
        batch = items[i:i+batch_size]

        # 배치 프롬프트 구성
        items_text = ""
        for j, item in enumerate(batch):
            text = item["text"][:200].replace("\n", " ")
            current = item.get("new_label", "")
            items_text += f"{j+1}. [{current}] {text}\n"

        prompt = f"""다음 글들의 분류가 올바른지 검증하고, 틀린 것만 수정해주세요.

[카테고리]
{category_desc}

[분류 규칙]
- 조문(고인의 명복), 안부는 "일상 소통"
- 형식적 감사(감사합니다 한마디)는 "일상 소통"
- 구체적 감사(덕분에, 수익 후기)는 "긍정 피드백"
- 물음표(?)가 있고 답변을 기대하면 "질문/문의"
- 정보를 전달하는 게 목적이면 "정보 공유"

[예제]
{FEW_SHOT}

[검증할 항목들]
{items_text}

각 항목의 올바른 카테고리를 JSON 배열로 답변하세요. 형식:
[{{"id": 1, "category": "카테고리명"}}, ...]

현재 분류가 맞으면 그대로, 틀리면 수정된 카테고리를 넣으세요."""

        try:
            message = client.messages.create(
                model="claude-sonnet-4-5-20250929",
                max_tokens=1000,
                temperature=0,
                messages=[{"role": "user", "content": prompt}]
            )

            response = message.content[0].text.strip()

            # JSON 파싱
            import re
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                corrections = json.loads(json_match.group())

                # 결과 적용
                for item, correction in zip(batch, corrections):
                    new_cat = correction.get("category", item.get("new_label"))
                    if new_cat in NEW_CATEGORIES:
                        old_cat = item.get("new_label")
                        item["new_label"] = new_cat
                        if old_cat != new_cat:
                            item["refined"] = True
                            item["old_label"] = old_cat
                    results.append(item)
            else:
                results.extend(batch)

        except Exception as e:
            print(f"  오류: {e}")
            results.extend(batch)

        print(f"  진행: {min(i+batch_size, len(items))}/{len(items)}")

    return results


def main():
    project_root = Path(__file__).parent.parent
    input_path = project_root / "data" / "labeling" / "auto_labeled.json"
    output_path = project_root / "data" / "labeling" / "refined_labeled.json"

    print("=" * 60)
    print("라벨 정제 (LLM 검증)")
    print("=" * 60)

    # 로드
    print(f"\n로드: {input_path}")
    with open(input_path, encoding="utf-8") as f:
        items = json.load(f)
    print(f"  {len(items)}건")

    # 정제
    print("\nLLM 검증 중...")
    client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    refined = refine_batch(client, items, batch_size=20)

    # 통계
    changed = sum(1 for item in refined if item.get("refined"))
    print(f"\n수정된 항목: {changed}건")

    # 수정된 샘플 출력
    if changed > 0:
        print("\n수정 샘플:")
        for item in refined:
            if item.get("refined"):
                text = item["text"][:50].replace("\n", " ")
                print(f"  [{item.get('old_label')}] → [{item['new_label']}]: {text}...")
                if sum(1 for i in refined if i.get("refined")) >= 10:
                    break

    # 저장
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(refined, f, ensure_ascii=False, indent=2)
    print(f"\n저장: {output_path}")

    # 분포
    from collections import Counter
    dist = Counter(item["new_label"] for item in refined)
    print("\n최종 분포:")
    for cat, count in dist.most_common():
        print(f"  {cat}: {count}건")

    print("\n다음 단계:")
    print("  python -m src.classifier_v2.prepare_data --labeling-file data/labeling/refined_labeled.json")


if __name__ == "__main__":
    main()
