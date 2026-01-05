"""주간 리포트를 위한 통계 분석 모듈"""
from typing import List, Dict, Any
from collections import defaultdict, Counter


class WeeklyAnalytics:
    """주간 데이터 통계 분석"""

    def __init__(self):
        """WeeklyAnalytics 초기화"""
        pass

    def analyze_weekly_data(
        self,
        letters: List[Dict[str, Any]],
        posts: List[Dict[str, Any]],
        previous_letters: List[Dict[str, Any]] = None,
        previous_posts: List[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        주간 데이터 통계 분석

        Args:
            letters: 이번 주 편지글
            posts: 이번 주 게시글
            previous_letters: 전주 편지글 (선택)
            previous_posts: 전주 게시글 (선택)

        Returns:
            통계 분석 결과
        """
        # 전체 통계
        total_stats = self._calculate_total_stats(
            letters, posts,
            previous_letters, previous_posts
        )

        # 마스터별 통계
        master_stats = self._calculate_master_stats(
            letters, posts,
            previous_letters, previous_posts
        )

        # 카테고리별 통계
        category_stats = self._calculate_category_stats(letters, posts)

        # 서비스 피드백 추출
        service_feedbacks = self._extract_service_feedbacks(letters, posts)

        return {
            "total_stats": total_stats,
            "master_stats": master_stats,
            "category_stats": category_stats,
            "service_feedbacks": service_feedbacks
        }

    def _calculate_total_stats(
        self,
        letters: List[Dict[str, Any]],
        posts: List[Dict[str, Any]],
        previous_letters: List[Dict[str, Any]] = None,
        previous_posts: List[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        전체 통계 계산

        Returns:
            {
                "this_week": {"letters": int, "posts": int, "total": int},
                "last_week": {"letters": int, "posts": int, "total": int},
                "change": {"letters": int, "posts": int, "total": int}
            }
        """
        this_week = {
            "letters": len(letters),
            "posts": len(posts),
            "total": len(letters) + len(posts)
        }

        last_week = {
            "letters": len(previous_letters) if previous_letters else 0,
            "posts": len(previous_posts) if previous_posts else 0,
            "total": (len(previous_letters) if previous_letters else 0) +
                     (len(previous_posts) if previous_posts else 0)
        }

        change = {
            "letters": this_week["letters"] - last_week["letters"],
            "posts": this_week["posts"] - last_week["posts"],
            "total": this_week["total"] - last_week["total"]
        }

        return {
            "this_week": this_week,
            "last_week": last_week,
            "change": change
        }

    def _calculate_master_stats(
        self,
        letters: List[Dict[str, Any]],
        posts: List[Dict[str, Any]],
        previous_letters: List[Dict[str, Any]] = None,
        previous_posts: List[Dict[str, Any]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        마스터별 통계 계산

        Returns:
            {
                "master_id": {
                    "this_week": {"letters": int, "posts": int, "total": int},
                    "last_week": {"letters": int, "posts": int, "total": int},
                    "change": {"letters": int, "posts": int, "total": int},
                    "categories": {"감사·후기": int, ...},
                    "contents": [...]
                }
            }
        """
        master_stats = defaultdict(lambda: {
            "this_week": {"letters": 0, "posts": 0, "total": 0},
            "last_week": {"letters": 0, "posts": 0, "total": 0},
            "categories": defaultdict(int),
            "contents": []
        })

        # 이번 주 데이터 집계
        for letter in letters:
            # actualMasterId 사용 (없으면 masterId)
            master_id = letter.get("actualMasterId") or letter.get("masterId", "unknown")
            master_stats[master_id]["this_week"]["letters"] += 1
            master_stats[master_id]["this_week"]["total"] += 1

            # 카테고리 집계
            category = letter.get("classification", {}).get("category", "미분류")
            master_stats[master_id]["categories"][category] += 1

            # 콘텐츠 저장
            master_stats[master_id]["contents"].append({
                "type": "letter",
                "content": letter.get("message", "")[:100],
                "category": category,
                "createdAt": letter.get("createdAt", ""),
                "masterName": letter.get("masterName", ""),
                "masterClubName": letter.get("masterClubName", "")
            })

        for post in posts:
            # actualMasterId 사용 (게시판이 아닌 마스터로 그룹핑)
            master_id = post.get("actualMasterId") or post.get("postBoardId", "unknown")
            master_stats[master_id]["this_week"]["posts"] += 1
            master_stats[master_id]["this_week"]["total"] += 1

            # 카테고리 집계
            category = post.get("classification", {}).get("category", "미분류")
            master_stats[master_id]["categories"][category] += 1

            # 콘텐츠 저장
            content = post.get("textBody") or post.get("body", "")
            master_stats[master_id]["contents"].append({
                "type": "post",
                "content": content[:100],
                "category": category,
                "title": post.get("title", ""),
                "createdAt": post.get("createdAt", ""),
                "masterName": post.get("masterName", ""),
                "masterClubName": post.get("masterClubName", "")
            })

        # 전주 데이터 집계
        if previous_letters:
            for letter in previous_letters:
                master_id = letter.get("actualMasterId") or letter.get("masterId", "unknown")
                master_stats[master_id]["last_week"]["letters"] += 1
                master_stats[master_id]["last_week"]["total"] += 1

        if previous_posts:
            for post in previous_posts:
                master_id = post.get("actualMasterId") or post.get("postBoardId", "unknown")
                master_stats[master_id]["last_week"]["posts"] += 1
                master_stats[master_id]["last_week"]["total"] += 1

        # 증감 계산
        for master_id in master_stats:
            master_stats[master_id]["change"] = {
                "letters": (master_stats[master_id]["this_week"]["letters"] -
                           master_stats[master_id]["last_week"]["letters"]),
                "posts": (master_stats[master_id]["this_week"]["posts"] -
                         master_stats[master_id]["last_week"]["posts"]),
                "total": (master_stats[master_id]["this_week"]["total"] -
                         master_stats[master_id]["last_week"]["total"])
            }

        return dict(master_stats)

    def _calculate_category_stats(
        self,
        letters: List[Dict[str, Any]],
        posts: List[Dict[str, Any]]
    ) -> Dict[str, int]:
        """
        카테고리별 통계 계산

        Returns:
            {"감사·후기": int, "질문·토론": int, ...}
        """
        category_counts = Counter()

        for letter in letters:
            category = letter.get("classification", {}).get("category", "미분류")
            category_counts[category] += 1

        for post in posts:
            category = post.get("classification", {}).get("category", "미분류")
            category_counts[category] += 1

        return dict(category_counts)

    def _extract_service_feedbacks(
        self,
        letters: List[Dict[str, Any]],
        posts: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        서비스 피드백 추출

        Returns:
            [{"content": str, "reason": str, "masterId": str}, ...]
        """
        feedbacks = []

        for letter in letters:
            classification = letter.get("classification", {})
            if classification.get("category") == "서비스 피드백":
                feedbacks.append({
                    "type": "letter",
                    "content": letter.get("message", "")[:200],
                    "reason": classification.get("reason", ""),
                    "masterId": letter.get("masterId", "unknown"),
                    "createdAt": letter.get("createdAt", "")
                })

        for post in posts:
            classification = post.get("classification", {})
            if classification.get("category") == "서비스 피드백":
                content = post.get("textBody") or post.get("body", "")
                feedbacks.append({
                    "type": "post",
                    "title": post.get("title", ""),
                    "content": content[:200],
                    "reason": classification.get("reason", ""),
                    "masterId": post.get("postBoardId", "unknown"),
                    "createdAt": post.get("createdAt", "")
                })

        return feedbacks

    def get_top_contents_by_category(
        self,
        letters: List[Dict[str, Any]],
        posts: List[Dict[str, Any]],
        category: str,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        특정 카테고리의 상위 콘텐츠 추출

        Args:
            letters: 편지글 리스트
            posts: 게시글 리스트
            category: 카테고리명
            limit: 반환할 최대 개수

        Returns:
            해당 카테고리의 콘텐츠 리스트
        """
        contents = []

        for letter in letters:
            if letter.get("classification", {}).get("category") == category:
                contents.append({
                    "type": "letter",
                    "content": letter.get("message", "")[:200],
                    "masterId": letter.get("masterId", ""),
                    "createdAt": letter.get("createdAt", "")
                })

        for post in posts:
            if post.get("classification", {}).get("category") == category:
                content_text = post.get("textBody") or post.get("body", "")
                contents.append({
                    "type": "post",
                    "title": post.get("title", ""),
                    "content": content_text[:200],
                    "masterId": post.get("postBoardId", ""),
                    "createdAt": post.get("createdAt", ""),
                    "likeCount": post.get("likeCount", 0),
                    "replyCount": post.get("replyCount", 0)
                })

        # 게시글의 경우 좋아요+댓글 수로 정렬
        contents_sorted = sorted(
            contents,
            key=lambda x: (x.get("likeCount", 0) + x.get("replyCount", 0)),
            reverse=True
        )

        return contents_sorted[:limit]
