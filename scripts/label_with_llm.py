#!/usr/bin/env python3
"""LLM 직접 분류 스크립트 (6개 카테고리)

사용법:
    python scripts/label_with_llm.py
"""
import json
import os
from pathlib import Path
from typing import List, Dict
from collections import Counter
from dotenv import load_dotenv

load_dotenv()

from anthropic import Anthropic

# 새로운 6개 카테고리
CATEGORIES = {
    "서비스 피드백": "콘텐츠, 마스터, 플랫폼에 대한 긍정/부정 반응 (칭찬, 불만, 개선요청)",
    "서비스 문의": "결제, 링크, 배송, 기능 등 서비스 이용 관련 문의",
    "투자 질문": "종목, 비중, 매매 타이밍, 포트폴리오 전략 등 투자 관련 질문",
    "정보 공유": "뉴스, 분석, 시장 정보, 종목 분석 공유 (질문 없이 정보 전달)",
    "투자 심리": "수익/손실 후기, 시장 불안, 기대감 등 투자 관련 감정/심리 표현",
    "일상 소통": "인사, 안부, 축하, 조문, 가입인사 등 일상적 소통",
}

FEW_SHOT = """
[서비스 피드백]
- "콘텐츠 정말 유익해요. 감사합니다" (긍정)
- "업로드가 너무 늦어요. 개선 부탁드립니다" (부정)
- "답변이 없어서 답답합니다" (부정)

[서비스 문의]
- "강의 링크가 안 열려요"
- "결제 취소는 어떻게 하나요?"
- "자료 언제 올라오나요?"

[투자 질문]
- "삼성전자 지금 매수해도 될까요?"
- "포트폴리오 비중을 어떻게 가져가면 좋을까요?"
- "이 종목 손절해야 할까요?"

[정보 공유]
- "오늘 FOMC 결과 공유합니다"
- "제가 분석한 내용입니다"
- "이런 뉴스가 있네요"

[투자 심리]
- "오늘 10% 수익 봤어요!"
- "손실이 커서 불안합니다"
- "시장이 불안해서 걱정이에요"
- "믿고 기다리겠습니다"

[일상 소통]
- "새해 복 많이 받으세요"
- "삼가 고인의 명복을 빕니다"
- "가입인사드립니다"
- "건강하세요"
"""


def classify_batch(client: Anthropic, items: List[Dict], batch_size: int = 15) -> List[Dict]:
    """배치 단위로 LLM 분류"""

    category_desc = "\n".join([f"- {k}: {v}" for k, v in CATEGORIES.items()])
    results = []

    for i in range(0, len(items), batch_size):
        batch = items[i:i+batch_size]

        # 배치 텍스트 구성
        items_text = ""
        for j, item in enumerate(batch):
            text = item["text"][:200].replace("\n", " ").strip()
            items_text += f"{j+1}. {text}\n"

        prompt = f"""다음 글들을 6개 카테고리 중 하나로 분류하세요.

[카테고리]
{category_desc}

[분류 예시]
{FEW_SHOT}

[분류 규칙]
1. 서비스(콘텐츠/마스터/플랫폼)에 대한 반응 → 서비스 피드백
2. 서비스 이용 문의 → 서비스 문의
3. 투자 방법/종목 질문 → 투자 질문
4. 정보 전달이 목적 → 정보 공유
5. 수익/손실/불안/기대 등 심리 표현 → 투자 심리
6. 인사/안부/잡담 → 일상 소통

[분류할 글]
{items_text}

JSON 배열로 답변하세요:
[{{"id": 1, "category": "카테고리명"}}, ...]"""

        try:
            message = client.messages.create(
                model="claude-sonnet-4-5-20250929",
                max_tokens=800,
                temperature=0,
                messages=[{"role": "user", "content": prompt}]
            )

            response = message.content[0].text.strip()

            # JSON 파싱
            import re
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                classifications = json.loads(json_match.group())

                for item, cls in zip(batch, classifications):
                    cat = cls.get("category", "")
                    if cat in CATEGORIES:
                        item["new_label"] = cat
                    else:
                        item["new_label"] = "일상 소통"  # fallback
                    results.append(item)
            else:
                # 파싱 실패시 fallback
                for item in batch:
                    item["new_label"] = "일상 소통"
                    results.append(item)

        except Exception as e:
            print(f"  오류: {e}")
            for item in batch:
                item["new_label"] = "일상 소통"
                results.append(item)

        print(f"  진행: {min(i+batch_size, len(items))}/{len(items)}")

    return results


def load_samples(data_dir: Path, sample_size: int = 900) -> List[Dict]:
    """기존 데이터에서 샘플 추출 (6개 카테고리 균형)"""
    import random
    random.seed(42)

    all_items = []

    for json_file in sorted(data_dir.glob("*.json")):
        with open(json_file, encoding="utf-8") as f:
            data = json.load(f)

        for item in data.get("letters", []):
            text = item.get("message", "").strip()
            if text and len(text) > 10:
                all_items.append({
                    "id": item.get("_id", ""),
                    "text": text,
                    "source": "letter",
                })

        for item in data.get("posts", []):
            text = (item.get("textBody") or item.get("body") or "").strip()
            if text and len(text) > 10:
                all_items.append({
                    "id": item.get("_id", ""),
                    "text": text,
                    "source": "post",
                })

    random.shuffle(all_items)
    return all_items[:sample_size]


def main():
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data" / "classified_data"
    output_path = project_root / "data" / "labeling" / "labeled_6cat.json"

    print("=" * 60)
    print("LLM 직접 분류 (6개 카테고리)")
    print("=" * 60)

    # 샘플 추출
    print("\n1. 샘플 추출")
    samples = load_samples(data_dir, sample_size=900)
    print(f"  {len(samples)}건 추출")

    # LLM 분류
    print("\n2. LLM 분류 중...")
    client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    labeled = classify_batch(client, samples, batch_size=15)

    # 통계
    print("\n3. 결과")
    dist = Counter(item["new_label"] for item in labeled)
    for cat, count in dist.most_common():
        pct = count / len(labeled) * 100
        print(f"  {cat}: {count}건 ({pct:.1f}%)")

    # 샘플 출력
    print("\n4. 카테고리별 샘플")
    for cat in CATEGORIES:
        items = [d for d in labeled if d["new_label"] == cat][:3]
        print(f"\n  [{cat}]")
        for item in items:
            text = item["text"][:60].replace("\n", " ")
            print(f"    - {text}...")

    # 저장
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(labeled, f, ensure_ascii=False, indent=2)
    print(f"\n저장: {output_path}")


if __name__ == "__main__":
    main()
