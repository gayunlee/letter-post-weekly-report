#!/usr/bin/env python3
"""자동 라벨링 스크립트

1. 기존 분류 데이터 → 새 카테고리 매핑
2. 낮은 신뢰도 항목 → LLM 재분류
3. 결과 저장 후 검토

사용법:
    python scripts/auto_label.py
"""
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any
from collections import Counter
from dotenv import load_dotenv

load_dotenv()

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from anthropic import Anthropic

# 새로운 5개 카테고리
NEW_CATEGORIES = {
    "긍정 피드백": "감사, 칭찬, 만족 표현, 수익 후기",
    "부정 피드백": "불만, 개선요청, 실망, 답답함 표현",
    "질문/문의": "투자 질문, 서비스 문의, 정보 요청 (물음표로 끝나는 경우 많음)",
    "정보 공유": "뉴스, 분석, 의견 공유, 정보 전달이 목적",
    "일상 소통": "인사, 안부, 축하, 잡담, 소통 자체가 목적",
}

# 기존 → 새 카테고리 매핑
OLD_TO_NEW_MAPPING = {
    "감사·후기": "긍정 피드백",
    "질문·토론": "질문/문의",
    "정보성 글": "정보 공유",
    "서비스 피드백": "질문/문의",
    "불편사항": "부정 피드백",
    "일상·공감": "일상 소통",
}

# Few-shot 예제 (새 카테고리 기준)
FEW_SHOT_EXAMPLES = """
예제 1:
내용: "쌤 덕분에 투자의 눈을 떠가는 1인입니다. 정말 감사합니다."
분류: 긍정 피드백
이유: 마스터에 대한 구체적인 감사와 만족을 표현

예제 2:
내용: "일주일에 한번도 안 올라오는 컨텐츠에 무엇을 기대할까요..."
분류: 부정 피드백
이유: 서비스에 대한 불만과 실망을 표현

예제 3:
내용: "삼성전자를 스터디 목록에 편입하지 않으시는 이유가 궁금합니다."
분류: 질문/문의
이유: 투자 전략에 대한 질문, 답변을 기대

예제 4:
내용: "강의 자료 링크가 연결되지 않습니다. 확인 부탁드립니다."
분류: 질문/문의
이유: 서비스 문의, 답변/조치를 기대

예제 5:
내용: "AI 데이터센터 수요가 폭발적으로 증가하고 있습니다. 제 분석을 공유합니다."
분류: 정보 공유
이유: 정보 전달이 주된 목적

예제 6:
내용: "새해 복 많이 받으세요! 건강하고 행복한 한 해 되세요."
분류: 일상 소통
이유: 인사, 소통 자체가 목적

예제 7:
내용: "감사합니다. 그런데 포트폴리오 비중을 어떻게 가져가면 좋을까요?"
분류: 질문/문의
이유: 감사+질문 복합이지만 질문이 핵심 내용

예제 8:
내용: "안녕하세요~ 감사합니다!"
분류: 일상 소통
이유: 형식적 인사, 구체적 감사 대상 없음
"""


class AutoLabeler:
    """자동 라벨링 클래스"""

    def __init__(self, confidence_threshold: float = 0.2, max_llm_calls: int = 100):
        self.confidence_threshold = confidence_threshold
        self.max_llm_calls = max_llm_calls
        self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.llm_call_count = 0

    def map_old_to_new(self, old_category: str) -> str:
        """기존 카테고리를 새 카테고리로 매핑"""
        return OLD_TO_NEW_MAPPING.get(old_category, "")

    def classify_with_llm(self, content: str) -> Dict[str, Any]:
        """LLM으로 새 카테고리 분류"""
        category_desc = "\n".join([
            f"- {cat}: {desc}"
            for cat, desc in NEW_CATEGORIES.items()
        ])

        prompt = f"""다음은 금융 콘텐츠 크리에이터 플랫폼의 사용자가 작성한 글입니다.
이 글을 아래 5개 카테고리 중 하나로 분류해주세요.

[분류 카테고리]
{category_desc}

[분류 규칙]
- 감사 + 질문 복합 → 질문이 실질적 내용이면 "질문/문의"
- 인사 + 감사 → 감사가 구체적이면 "긍정 피드백", 형식적이면 "일상 소통"
- 불만 형태의 질문 → 불만 해소가 목적이면 "부정 피드백"

[Few-shot 예제]
{FEW_SHOT_EXAMPLES}

[분류할 내용]
{content[:500]}

위 내용을 가장 적합한 카테고리 하나로 분류하고, 다음 JSON 형식으로만 답변해주세요:
{{"category": "카테고리명", "confidence": "높음/중간/낮음", "reason": "분류 이유"}}"""

        try:
            message = self.client.messages.create(
                model="claude-sonnet-4-5-20250929",
                max_tokens=200,
                temperature=0,
                messages=[{"role": "user", "content": prompt}]
            )

            self.llm_call_count += 1
            response_text = message.content[0].text.strip()

            # JSON 파싱
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                return {
                    "category": result.get("category", ""),
                    "confidence": 0.9 if result.get("confidence") == "높음" else 0.7 if result.get("confidence") == "중간" else 0.5,
                    "reason": result.get("reason", ""),
                    "method": "llm"
                }
        except Exception as e:
            print(f"  LLM 오류: {e}")

        return {"category": "", "confidence": 0, "reason": "LLM 오류", "method": "error"}

    def process_item(self, item: Dict) -> Dict:
        """단일 항목 처리"""
        text = item.get("text", "")
        old_category = item.get("old_category", "")
        old_confidence = item.get("confidence", 0)

        # 1. 기존 카테고리 매핑
        mapped_category = self.map_old_to_new(old_category)

        # 2. 신뢰도 체크
        needs_llm = (
            old_confidence < self.confidence_threshold or
            not mapped_category or
            old_category in ["서비스 피드백"]  # 애매한 카테고리
        )

        if needs_llm and text and self.llm_call_count < self.max_llm_calls:
            # LLM 재분류
            llm_result = self.classify_with_llm(text)
            if llm_result["category"] in NEW_CATEGORIES:
                return {
                    **item,
                    "new_label": llm_result["category"],
                    "label_confidence": llm_result["confidence"],
                    "label_method": "llm",
                    "label_reason": llm_result.get("reason", "")
                }

        # 매핑 결과 사용
        return {
            **item,
            "new_label": mapped_category if mapped_category else None,
            "label_confidence": old_confidence if mapped_category else 0,
            "label_method": "mapping" if mapped_category else "none",
            "label_reason": f"기존: {old_category}" if mapped_category else ""
        }


def load_existing_data(data_dir: Path) -> List[Dict]:
    """기존 분류 데이터 로드"""
    all_items = []

    for json_file in sorted(data_dir.glob("*.json")):
        print(f"  로딩: {json_file.name}")
        with open(json_file, encoding="utf-8") as f:
            data = json.load(f)

        # 편지
        for item in data.get("letters", []):
            text = item.get("message", "").strip()
            classification = item.get("classification", {})
            if text:
                all_items.append({
                    "id": item.get("_id", ""),
                    "text": text,
                    "source": "letter",
                    "old_category": classification.get("category", ""),
                    "confidence": classification.get("confidence", 0),
                    "file": json_file.name,
                })

        # 게시글
        for item in data.get("posts", []):
            text = (item.get("textBody") or item.get("body") or "").strip()
            classification = item.get("classification", {})
            if text:
                all_items.append({
                    "id": item.get("_id", ""),
                    "text": text,
                    "source": "post",
                    "old_category": classification.get("category", ""),
                    "confidence": classification.get("confidence", 0),
                    "file": json_file.name,
                })

    return all_items


def sample_for_training(items: List[Dict], sample_size: int = 750) -> List[Dict]:
    """훈련용 샘플 추출 (새 카테고리 기준 균형)"""
    import random
    random.seed(42)

    # 새 카테고리별 그룹화
    by_new_category = {cat: [] for cat in NEW_CATEGORIES}

    for item in items:
        new_cat = item.get("new_label")
        if new_cat in by_new_category:
            by_new_category[new_cat].append(item)

    # 균등 샘플링
    per_category = sample_size // len(NEW_CATEGORIES)
    sampled = []

    print(f"\n카테고리별 샘플링 (목표: 카테고리당 {per_category}건)")
    for cat in NEW_CATEGORIES:
        available = by_new_category[cat]
        random.shuffle(available)
        take = min(per_category, len(available))
        sampled.extend(available[:take])
        print(f"  {cat}: {len(available)}건 중 {take}건")

    random.shuffle(sampled)
    return sampled[:sample_size]


def print_stats(items: List[Dict], title: str = "통계"):
    """통계 출력"""
    print(f"\n{title}:")
    print(f"  총 항목: {len(items)}건")

    # 새 카테고리 분포
    by_new = Counter(item.get("new_label", "(미분류)") for item in items)
    print("\n  새 카테고리 분포:")
    for cat, count in by_new.most_common():
        pct = count / len(items) * 100
        print(f"    {cat}: {count}건 ({pct:.1f}%)")

    # 라벨링 방법 분포
    by_method = Counter(item.get("label_method", "none") for item in items)
    print("\n  라벨링 방법:")
    for method, count in by_method.most_common():
        print(f"    {method}: {count}건")


def show_samples(items: List[Dict], n: int = 5):
    """샘플 출력"""
    print(f"\n샘플 {n}건:")
    print("-" * 80)
    for item in items[:n]:
        text = item["text"][:100].replace("\n", " ")
        old = item.get("old_category", "")
        new = item.get("new_label", "")
        method = item.get("label_method", "")
        print(f"[{old}] → [{new}] ({method})")
        print(f"  {text}...")
        print()


def main():
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data" / "classified_data"
    output_path = project_root / "data" / "labeling" / "auto_labeled.json"

    print("=" * 60)
    print("자동 라벨링")
    print("=" * 60)

    # 데이터 로드
    print("\n1. 기존 데이터 로드")
    all_items = load_existing_data(data_dir)
    print(f"  총 {len(all_items)}건")

    # 자동 라벨링
    print("\n2. 자동 라벨링 시작")
    print(f"  신뢰도 임계값: 0.4 (미만은 LLM 재분류)")

    labeler = AutoLabeler(confidence_threshold=0.4)
    labeled_items = []

    for i, item in enumerate(all_items):
        result = labeler.process_item(item)
        labeled_items.append(result)

        if (i + 1) % 100 == 0:
            print(f"  진행: {i + 1}/{len(all_items)} (LLM: {labeler.llm_call_count}회)")

    print(f"\n  완료! LLM 호출: {labeler.llm_call_count}회")

    # 통계
    print_stats(labeled_items, "전체 라벨링 결과")

    # 훈련용 샘플 추출
    print("\n3. 훈련용 샘플 추출 (750건)")
    training_samples = sample_for_training(labeled_items, 750)
    print_stats(training_samples, "훈련 샘플")

    # 샘플 출력
    show_samples(training_samples, 10)

    # 저장
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(training_samples, f, ensure_ascii=False, indent=2)
    print(f"\n저장 완료: {output_path}")

    # 검토 안내
    print("\n" + "=" * 60)
    print("다음 단계:")
    print("  1. 위 샘플을 검토하세요")
    print("  2. 문제 없으면 재훈련:")
    print("     python -m src.classifier_v2.prepare_data --labeling-file data/labeling/auto_labeled.json")
    print("=" * 60)


if __name__ == "__main__":
    main()
