"""2축 분류 체계 기반 주간 통계 분석 모듈

1축 analytics.py와 독립적으로 동작합니다.
핵심 가치: 마스터별 감성 변화 추이 → 부정 급증 마스터 조기 감지
"""
import re
from typing import List, Dict, Any, Tuple
from collections import defaultdict
from src.utils.text_utils import clean_text


# 2축 카테고리 정의
TOPICS = ["콘텐츠 반응", "투자 이야기", "서비스 이슈", "커뮤니티 소통"]
SENTIMENTS = ["긍정", "부정", "중립"]

# 부정 감성 증감 임계값: 이 비율(pp) 이상 증가하면 하이라이트
NEGATIVE_SPIKE_THRESHOLD_PP = 10  # 10%p 이상 증가


class TwoAxisAnalytics:
    """2축(Topic × Sentiment) 주간 데이터 통계 분석"""

    def _get_master_group_name(self, master_name: str) -> str:
        """마스터 이름 끝 숫자 제거 (서재형2 → 서재형)"""
        if not master_name:
            return "Unknown"
        return re.sub(r'\d+$', '', master_name).strip()

    def analyze_weekly_data(
        self,
        letters: List[Dict[str, Any]],
        posts: List[Dict[str, Any]],
        previous_letters: List[Dict[str, Any]] = None,
        previous_posts: List[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        2축 기반 주간 통계 분석

        Returns:
            {
                "total_stats": {...},
                "topic_sentiment_matrix": {...},
                "master_stats": {...},
                "negative_spike_masters": [...],
                "service_issues": [...],
            }
        """
        all_items = self._tag_items(letters, posts)
        prev_items = self._tag_items(
            previous_letters or [], previous_posts or []
        )

        total_stats = self._calc_total_stats(all_items, prev_items)
        matrix = self._calc_topic_sentiment_matrix(all_items)
        master_stats = self._calc_master_stats(all_items, prev_items)
        negative_spikes = self._detect_negative_spikes(master_stats)
        negative_drops = self._detect_negative_drops(master_stats)
        service_issues = self._extract_service_issues(all_items)

        return {
            "total_stats": total_stats,
            "topic_sentiment_matrix": matrix,
            "master_stats": master_stats,
            "negative_spike_masters": negative_spikes,
            "negative_drop_masters": negative_drops,
            "service_issues": service_issues,
        }

    def _tag_items(
        self,
        letters: List[Dict[str, Any]],
        posts: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """편지+게시글을 통합하고 필요한 필드를 정규화"""
        items = []
        for letter in letters:
            cls = letter.get("classification", {})
            items.append({
                "type": "letter",
                "master_group": self._get_master_group_name(letter.get("masterName", "Unknown")),
                "master_name": letter.get("masterName", "Unknown"),
                "club_name": letter.get("masterClubName", ""),
                "topic": cls.get("topic", "미분류"),
                "sentiment": cls.get("sentiment", "미분류"),
                "content": clean_text(letter.get("message", ""), 200),
                "raw_content": letter.get("message", ""),
                "created_at": letter.get("createdAt", ""),
            })

        for post in posts:
            cls = post.get("classification", {})
            content = post.get("textBody") or post.get("body", "")
            items.append({
                "type": "post",
                "master_group": self._get_master_group_name(post.get("masterName", "Unknown")),
                "master_name": post.get("masterName", "Unknown"),
                "club_name": post.get("masterClubName", ""),
                "topic": cls.get("topic", "미분류"),
                "sentiment": cls.get("sentiment", "미분류"),
                "title": post.get("title", ""),
                "content": clean_text(content, 200),
                "raw_content": content,
                "created_at": post.get("createdAt", ""),
            })

        return items

    def _calc_total_stats(
        self,
        items: List[Dict[str, Any]],
        prev_items: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """전체 통계 + 전주 대비 증감"""
        this_letters = sum(1 for i in items if i["type"] == "letter")
        this_posts = sum(1 for i in items if i["type"] == "post")
        prev_letters = sum(1 for i in prev_items if i["type"] == "letter")
        prev_posts = sum(1 for i in prev_items if i["type"] == "post")

        this_week = {"letters": this_letters, "posts": this_posts, "total": this_letters + this_posts}
        last_week = {"letters": prev_letters, "posts": prev_posts, "total": prev_letters + prev_posts}

        # 전체 감성 분포
        sent_dist = {s: 0 for s in SENTIMENTS}
        for item in items:
            s = item["sentiment"]
            if s in sent_dist:
                sent_dist[s] += 1

        prev_sent_dist = {s: 0 for s in SENTIMENTS}
        for item in prev_items:
            s = item["sentiment"]
            if s in prev_sent_dist:
                prev_sent_dist[s] += 1

        return {
            "this_week": this_week,
            "last_week": last_week,
            "change": {
                "letters": this_week["letters"] - last_week["letters"],
                "posts": this_week["posts"] - last_week["posts"],
                "total": this_week["total"] - last_week["total"],
            },
            "sentiment_distribution": sent_dist,
            "prev_sentiment_distribution": prev_sent_dist,
        }

    def _calc_topic_sentiment_matrix(
        self, items: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, int]]:
        """Topic × Sentiment 교차 집계 매트릭스"""
        matrix = {t: {s: 0 for s in SENTIMENTS} for t in TOPICS}
        for item in items:
            t, s = item["topic"], item["sentiment"]
            if t in matrix and s in matrix.get(t, {}):
                matrix[t][s] += 1
        return matrix

    def _calc_master_stats(
        self,
        items: List[Dict[str, Any]],
        prev_items: List[Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        """
        마스터별 2축 통계

        각 마스터:
          this_week: {letters, posts, total}
          last_week: {letters, posts, total}
          change: {letters, posts, total}
          sentiment: {긍정: n, 부정: n, 중립: n}
          prev_sentiment: {긍정: n, 부정: n, 중립: n}
          negative_ratio: float  (이번 주 부정 비율)
          prev_negative_ratio: float  (전주 부정 비율)
          negative_change_pp: float  (부정 비율 증감, %p)
          topics: {콘텐츠 반응: n, ...}
          contents: [...]
          club_names: set
        """
        stats = defaultdict(lambda: {
            "this_week": {"letters": 0, "posts": 0, "total": 0},
            "last_week": {"letters": 0, "posts": 0, "total": 0},
            "sentiment": {s: 0 for s in SENTIMENTS},
            "prev_sentiment": {s: 0 for s in SENTIMENTS},
            "topics": {t: 0 for t in TOPICS},
            "contents": [],
            "club_names": set(),
        })

        # 이번 주
        for item in items:
            mg = item["master_group"]
            key = "letters" if item["type"] == "letter" else "posts"
            stats[mg]["this_week"][key] += 1
            stats[mg]["this_week"]["total"] += 1

            s = item["sentiment"]
            if s in stats[mg]["sentiment"]:
                stats[mg]["sentiment"][s] += 1

            t = item["topic"]
            if t in stats[mg]["topics"]:
                stats[mg]["topics"][t] += 1

            if item["club_name"]:
                stats[mg]["club_names"].add(item["club_name"])

            stats[mg]["contents"].append(item)

        # 전주
        for item in prev_items:
            mg = item["master_group"]
            key = "letters" if item["type"] == "letter" else "posts"
            stats[mg]["last_week"][key] += 1
            stats[mg]["last_week"]["total"] += 1

            s = item["sentiment"]
            if s in stats[mg]["prev_sentiment"]:
                stats[mg]["prev_sentiment"][s] += 1

            if item["club_name"]:
                stats[mg]["club_names"].add(item["club_name"])

        # 증감 + 부정 비율 계산
        for mg in stats:
            tw = stats[mg]["this_week"]
            lw = stats[mg]["last_week"]
            stats[mg]["change"] = {
                "letters": tw["letters"] - lw["letters"],
                "posts": tw["posts"] - lw["posts"],
                "total": tw["total"] - lw["total"],
            }

            # 부정 비율 (%)
            total_this = tw["total"]
            total_prev = lw["total"]
            neg_this = stats[mg]["sentiment"].get("부정", 0)
            neg_prev = stats[mg]["prev_sentiment"].get("부정", 0)

            stats[mg]["negative_ratio"] = (neg_this / total_this * 100) if total_this > 0 else 0.0
            stats[mg]["prev_negative_ratio"] = (neg_prev / total_prev * 100) if total_prev > 0 else 0.0
            stats[mg]["negative_change_pp"] = stats[mg]["negative_ratio"] - stats[mg]["prev_negative_ratio"]

        return dict(stats)

    def _detect_negative_spikes(
        self, master_stats: Dict[str, Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """부정 감성 급증 마스터 감지

        조건: 부정 비율이 전주 대비 NEGATIVE_SPIKE_THRESHOLD_PP 이상 증가
        + 이번 주 총 건수 5건 이상 (소량 데이터 노이즈 제외)
        """
        spikes = []
        for master, data in master_stats.items():
            if master == "Unknown":
                continue
            if data["this_week"]["total"] < 5:
                continue
            if data["negative_change_pp"] >= NEGATIVE_SPIKE_THRESHOLD_PP:
                # 부정 콘텐츠 샘플 수집
                neg_samples = [
                    c for c in data["contents"]
                    if c["sentiment"] == "부정"
                ][:5]

                spikes.append({
                    "master": master,
                    "negative_ratio": round(data["negative_ratio"], 1),
                    "prev_negative_ratio": round(data["prev_negative_ratio"], 1),
                    "change_pp": round(data["negative_change_pp"], 1),
                    "negative_count": data["sentiment"].get("부정", 0),
                    "total_count": data["this_week"]["total"],
                    "samples": neg_samples,
                })

        # 증감폭 내림차순 정렬
        spikes.sort(key=lambda x: x["change_pp"], reverse=True)
        return spikes

    def _detect_negative_drops(
        self, master_stats: Dict[str, Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """부정 감성 급감 마스터 감지 (개선 시그널)

        조건: 부정 비율이 전주 대비 NEGATIVE_SPIKE_THRESHOLD_PP 이상 감소
        + 전주 총 건수 5건 이상 (전주에 충분한 데이터가 있어야 의미 있음)
        """
        drops = []
        for master, data in master_stats.items():
            if master == "Unknown":
                continue
            if data["last_week"]["total"] < 5:
                continue
            if data["negative_change_pp"] <= -NEGATIVE_SPIKE_THRESHOLD_PP:
                drops.append({
                    "master": master,
                    "negative_ratio": round(data["negative_ratio"], 1),
                    "prev_negative_ratio": round(data["prev_negative_ratio"], 1),
                    "change_pp": round(data["negative_change_pp"], 1),
                    "negative_count": data["sentiment"].get("부정", 0),
                    "total_count": data["this_week"]["total"],
                })

        # 감소폭 큰 순서 (절댓값)
        drops.sort(key=lambda x: x["change_pp"])
        return drops

    def _extract_service_issues(
        self, items: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """서비스 이슈 (Topic=서비스 이슈) 중 부정 감성 추출"""
        issues = []
        for item in items:
            if item["topic"] == "서비스 이슈" and item["sentiment"] == "부정":
                issues.append({
                    "type": item["type"],
                    "master": item["master_group"],
                    "content": item["content"],
                    "title": item.get("title", ""),
                    "created_at": item["created_at"],
                })
        return issues
