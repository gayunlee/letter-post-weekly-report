"""분류 데이터 저장 및 로드 (2-Tier Storage)"""
import json
import os
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from decimal import Decimal
from pathlib import Path


class DecimalEncoder(json.JSONEncoder):
    """Decimal을 JSON 직렬화 가능하도록 변환하는 인코더"""
    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        return super(DecimalEncoder, self).default(obj)


class ClassifiedDataStore:
    """주차별 분류 결과 저장/로드 관리자 (2-Tier)"""

    def __init__(
        self,
        classified_data_dir: str = "./data/classified_data",
        stats_dir: str = "./data/stats"
    ):
        """
        ClassifiedDataStore 초기화

        Args:
            classified_data_dir: 전체 라벨링 데이터 저장 경로
            stats_dir: 통계 요약 저장 경로
        """
        self.classified_data_dir = Path(classified_data_dir)
        self.stats_dir = Path(stats_dir)

        # 디렉토리 생성
        self.classified_data_dir.mkdir(parents=True, exist_ok=True)
        self.stats_dir.mkdir(parents=True, exist_ok=True)

    def _get_classified_data_path(self, week_start: str) -> Path:
        """전체 분류 데이터 파일 경로 반환"""
        return self.classified_data_dir / f"{week_start}.json"

    def _get_stats_path(self, week_start: str) -> Path:
        """통계 요약 파일 경로 반환"""
        return self.stats_dir / f"{week_start}.json"

    def exists(self, week_start: str) -> bool:
        """
        주차별 분류 결과 존재 여부 확인

        Args:
            week_start: 주차 시작일 (YYYY-MM-DD)

        Returns:
            존재 여부
        """
        return (
            self._get_classified_data_path(week_start).exists() and
            self._get_stats_path(week_start).exists()
        )

    def save_weekly_data(
        self,
        week_start: str,
        week_end: str,
        classified_letters: List[Dict[str, Any]],
        classified_posts: List[Dict[str, Any]],
        stats: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        주차별 분류 결과 저장 (2-Tier)

        Args:
            week_start: 주차 시작일 (YYYY-MM-DD)
            week_end: 주차 종료일 (YYYY-MM-DD)
            classified_letters: 분류된 편지글 리스트
            classified_posts: 분류된 게시글 리스트
            stats: 통계 데이터 (선택, 없으면 자동 생성)
        """
        # 1. 전체 라벨링 데이터 저장
        classified_data = {
            "week_start": week_start,
            "week_end": week_end,
            "letters": classified_letters,
            "posts": classified_posts
        }

        classified_data_path = self._get_classified_data_path(week_start)
        with open(classified_data_path, 'w', encoding='utf-8') as f:
            json.dump(classified_data, f, ensure_ascii=False, indent=2, cls=DecimalEncoder)

        # 2. 통계 요약 저장
        if stats is None:
            stats = self._compute_stats(
                week_start, week_end,
                classified_letters, classified_posts
            )

        stats_path = self._get_stats_path(week_start)
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2, cls=DecimalEncoder)

    def load_weekly_data(self, week_start: str) -> Dict[str, Any]:
        """
        전체 라벨링 데이터 로드

        Args:
            week_start: 주차 시작일 (YYYY-MM-DD)

        Returns:
            {
                "week_start": str,
                "week_end": str,
                "letters": List[Dict],
                "posts": List[Dict]
            }
        """
        classified_data_path = self._get_classified_data_path(week_start)

        if not classified_data_path.exists():
            raise FileNotFoundError(
                f"분류 데이터를 찾을 수 없습니다: {week_start}"
            )

        with open(classified_data_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def load_weekly_stats(self, week_start: str) -> Dict[str, Any]:
        """
        통계 요약만 로드 (빠름)

        Args:
            week_start: 주차 시작일 (YYYY-MM-DD)

        Returns:
            {
                "week_start": str,
                "week_end": str,
                "total": {...},
                "categories": {...},
                "masters": {...},
                "service_feedback_count": int
            }
        """
        stats_path = self._get_stats_path(week_start)

        if not stats_path.exists():
            raise FileNotFoundError(
                f"통계 데이터를 찾을 수 없습니다: {week_start}"
            )

        with open(stats_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def get_latest_stats(self) -> Optional[Dict[str, Any]]:
        """
        가장 최근 주차 통계 반환

        Returns:
            최신 통계 데이터 또는 None
        """
        stats_files = sorted(self.stats_dir.glob("*.json"), reverse=True)

        if not stats_files:
            return None

        latest_file = stats_files[0]
        with open(latest_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def list_available_weeks(self) -> List[str]:
        """
        저장된 주차 목록 반환 (날짜 순서)

        Returns:
            주차 시작일 리스트 (YYYY-MM-DD)
        """
        stats_files = sorted(self.stats_dir.glob("*.json"))
        return [f.stem for f in stats_files]

    def _compute_stats(
        self,
        week_start: str,
        week_end: str,
        classified_letters: List[Dict[str, Any]],
        classified_posts: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        통계 요약 계산

        Args:
            week_start: 주차 시작일
            week_end: 주차 종료일
            classified_letters: 분류된 편지글
            classified_posts: 분류된 게시글

        Returns:
            통계 요약 딕셔너리
        """
        # 전체 통계
        total = {
            "letters": len(classified_letters),
            "posts": len(classified_posts),
            "total": len(classified_letters) + len(classified_posts)
        }

        # 카테고리별 통계
        categories = {}

        for item in classified_letters + classified_posts:
            classification = item.get("classification", {})
            category = classification.get("category", "미분류")
            categories[category] = categories.get(category, 0) + 1

        # 마스터별 통계
        masters = {}

        for item in classified_letters + classified_posts:
            master_id = item.get("masterId", "unknown")

            if master_id not in masters:
                masters[master_id] = {
                    "letters": 0,
                    "posts": 0,
                    "total": 0,
                    "categories": {}
                }

            # 편지 vs 게시글 구분
            is_letter = "message" in item and "textBody" not in item
            if is_letter:
                masters[master_id]["letters"] += 1
            else:
                masters[master_id]["posts"] += 1

            masters[master_id]["total"] += 1

            # 카테고리별 통계
            classification = item.get("classification", {})
            category = classification.get("category", "미분류")
            master_categories = masters[master_id]["categories"]
            master_categories[category] = master_categories.get(category, 0) + 1

        # 서비스 피드백 개수
        service_feedback_count = categories.get("서비스 피드백", 0)

        return {
            "week_start": week_start,
            "week_end": week_end,
            "total": total,
            "categories": categories,
            "masters": masters,
            "service_feedback_count": service_feedback_count
        }
