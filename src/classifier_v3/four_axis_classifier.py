"""3축 분류기 래퍼 (ThreeAxisClassifier)

기존 TwoAxisClassifier(Topic + Sentiment)를 래핑하고,
DetailTagExtractor에서 추출한 Intent를 추가합니다.

분류 결과 스키마:
  classification: {topic, sentiment, intent + 각 confidence}
  detail_tags: {category_tags, free_tags, summary}

부서별 라우팅이나 긴급도는 분류 축에 포함하지 않습니다.
각 부서가 3축 + detail_tags를 자유롭게 조합하여 쿼리합니다.
"""
from typing import List, Dict, Any, Optional

from src.classifier_v2.two_axis_classifier import TwoAxisClassifier
from src.classifier_v2.detail_tag_extractor import DetailTagExtractor, aggregate_category_tags


class ThreeAxisClassifier:
    """3축(Topic × Sentiment × Intent) 분류기"""

    def __init__(
        self,
        two_axis_classifier: Optional[TwoAxisClassifier] = None,
        detail_tag_model: str = "claude-haiku-4-5-20251001",
        max_workers: int = 5,
    ):
        self.classifier = two_axis_classifier or TwoAxisClassifier()
        self.tag_extractor = DetailTagExtractor(
            model=detail_tag_model,
            max_workers=max_workers,
        )

    def classify_and_enrich(
        self,
        items: List[Dict[str, Any]],
        content_field: str = "message",
        batch_size: int = 32,
        skip_detail_tags: bool = False,
    ) -> List[Dict[str, Any]]:
        """3축 분류 전체 파이프라인

        1. 2축 분류 (Topic + Sentiment) — 파인튜닝 모델
        2. detail_tags + Intent 추출 — Haiku 1회 호출

        Args:
            items: BigQuery에서 가져온 원본 데이터
            content_field: 텍스트 필드명
            batch_size: 2축 분류 배치 크기
            skip_detail_tags: True이면 detail_tags/intent 추출 건너뜀

        Returns:
            각 item에 classification + detail_tags가 부착된 리스트
        """
        needs_two_axis = []
        needs_two_axis_idx = []
        results = [None] * len(items)

        for i, item in enumerate(items):
            if item.get("classification", {}).get("topic"):
                results[i] = item.copy()
            else:
                needs_two_axis.append(item)
                needs_two_axis_idx.append(i)

        # Step 1: 2축 분류
        if needs_two_axis:
            print(f"  2축 분류: {len(needs_two_axis)}건")
            classified = self.classifier.classify_batch(
                needs_two_axis, content_field, batch_size
            )
            for j, idx in enumerate(needs_two_axis_idx):
                results[idx] = classified[j]

        # Step 2: detail_tags + Intent 추출
        if not skip_detail_tags:
            needs_tags = []
            needs_tags_idx = []
            for i, item in enumerate(results):
                if not item.get("detail_tags"):
                    needs_tags.append(item)
                    needs_tags_idx.append(i)

            if needs_tags:
                print(f"  detail_tags + intent 추출: {len(needs_tags)}건")
                enriched = self.tag_extractor.extract_tags_batch(
                    needs_tags, content_field
                )
                for j, idx in enumerate(needs_tags_idx):
                    results[idx] = enriched[j]

        return results

    def get_cost_report(self) -> Dict[str, Any]:
        return self.tag_extractor.get_cost_report()

    def get_tag_aggregation(self, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        return aggregate_category_tags(items)
