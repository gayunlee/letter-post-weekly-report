"""서비스 이슈 클러스터링 + 특이 패턴 LLM 요약 모듈

서비스 이슈: 임베딩 클러스터링으로 세부 이슈 자동 태깅 (환불, 접속장애, 멤버십 등)
나머지 토픽: 부정 글이 일정 수 이상이면 LLM으로 공통 테마 요약 (1회 호출)

비용: 임베딩 로컬(무료) + LLM 클러스터 라벨링 ~10회 + 패턴 요약 3-5회 = 주당 $0.01 이하
"""
import os
import json
import numpy as np
from typing import List, Dict, Any, Optional
from collections import Counter

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

TOPICS = ["콘텐츠 반응", "투자 이야기", "서비스 이슈", "커뮤니티 소통"]

# 클러스터링 최소 건수
MIN_ITEMS_FOR_CLUSTERING = 10
# 특이 패턴 감지: 부정 건수 이상이면 LLM 요약 트리거
NOTABLE_NEGATIVE_THRESHOLD = 5


class SubThemeAnalyzer:
    """대분류 내 하위 테마 분석"""

    def __init__(self, model: str = "gpt-4o-mini"):
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("CALLME_OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key) if api_key else None
        self.model = model
        self._embedder = None

    @property
    def embedder(self):
        """lazy loading — 임베딩 모델은 필요할 때만 로드"""
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer("jhgan/ko-sroberta-multitask")
        return self._embedder

    def analyze(
        self,
        letters: List[Dict[str, Any]],
        posts: List[Dict[str, Any]],
        stats: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """서비스 이슈 클러스터링 + 특이 패턴 요약"""
        items = self._prepare_items(letters, posts)

        # 1. 서비스 이슈 클러스터링
        service_items = [i for i in items if i["topic"] == "서비스 이슈"]
        service_clusters = {}
        if len(service_items) >= MIN_ITEMS_FOR_CLUSTERING:
            print(f"  서비스 이슈 {len(service_items)}건 클러스터링...")
            service_clusters = self._cluster_service_issues(service_items)

        # 2. 특이 패턴 감지 + LLM 요약
        notable_patterns = self._detect_notable_patterns(items, stats)

        return {
            "service_clusters": service_clusters,
            "notable_patterns": notable_patterns,
        }

    def _prepare_items(
        self,
        letters: List[Dict[str, Any]],
        posts: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """원본 데이터를 분석용으로 정규화"""
        import re
        items = []
        for letter in letters:
            cls = letter.get("classification", {})
            master = letter.get("masterName", "Unknown")
            master_group = re.sub(r'\d+$', '', master).strip() if master else "Unknown"
            text = letter.get("message", "")
            items.append({
                "topic": cls.get("topic", "미분류"),
                "sentiment": cls.get("sentiment", "미분류"),
                "master": master_group,
                "text": text,
                "text_short": text[:200] if text else "",
                "type": "편지",
            })

        for post in posts:
            cls = post.get("classification", {})
            master = post.get("masterName", "Unknown")
            master_group = re.sub(r'\d+$', '', master).strip() if master else "Unknown"
            text = post.get("textBody") or post.get("body", "")
            items.append({
                "topic": cls.get("topic", "미분류"),
                "sentiment": cls.get("sentiment", "미분류"),
                "master": master_group,
                "text": text,
                "text_short": text[:200] if text else "",
                "type": "게시글",
            })
        return items

    # ── 서비스 이슈 클러스터링 ──────────────────────────────

    def _cluster_service_issues(
        self, items: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """서비스 이슈 임베딩 클러스터링 + LLM 라벨링"""
        from sklearn.decomposition import PCA
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score

        texts = [i["text"][:500] for i in items]
        embeddings = self.embedder.encode(texts, show_progress_bar=False, batch_size=64)

        # PCA 차원 축소
        n_comp = min(50, len(embeddings) - 1)
        reduced = PCA(n_components=n_comp).fit_transform(embeddings)

        # 최적 k 탐색 (실루엣 스코어)
        best_k, best_score = 3, -1
        max_k = min(15, len(items) // 3)
        for k in range(3, max_k + 1):
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(reduced)
            score = silhouette_score(reduced, labels, sample_size=min(500, len(reduced)))
            if score > best_score:
                best_k, best_score = k, score

        print(f"  최적 k={best_k} (silhouette={best_score:.3f})")

        # 최종 클러스터링
        labels = KMeans(n_clusters=best_k, random_state=42, n_init=10).fit_predict(reduced)

        # 클러스터별 정보 수집
        clusters = {}
        for cid in sorted(set(labels)):
            cluster_items = [items[i] for i, l in enumerate(labels) if l == cid]
            sentiments = dict(Counter(it["sentiment"] for it in cluster_items))
            masters = Counter(it["master"] for it in cluster_items).most_common(3)
            samples = [it["text_short"] for it in cluster_items[:5]]

            # LLM으로 클러스터 라벨링
            label = self._label_cluster(samples)

            clusters[str(cid)] = {
                "label": label,
                "count": len(cluster_items),
                "sentiment_dist": sentiments,
                "top_masters": [[m, c] for m, c in masters],
                "samples": samples,
            }

        return clusters

    def _label_cluster(self, samples: List[str]) -> str:
        """LLM으로 클러스터 이름 생성"""
        if not self.client:
            return "미분류"

        joined = "\n---\n".join(s[:200] for s in samples[:5])
        prompt = f"""다음은 금융 콘텐츠 플랫폼의 서비스 관련 VOC(고객 의견) 그룹입니다.
이 그룹의 공통 이슈를 2-4단어의 한국어 명사구로 명명해주세요.
예시: "환불 요청", "강의 접속 장애", "멤버십 해지 문의", "콘텐츠 업로드 지연"

답변은 이름만 출력하세요 (따옴표 없이).

[VOC 샘플]
{joined}"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=30,
                temperature=0.1,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content.strip().strip('"\'')
        except Exception:
            return "서비스 이슈"

    # ── 특이 패턴 감지 + LLM 요약 ──────────────────────────

    def _detect_notable_patterns(
        self,
        items: List[Dict[str, Any]],
        stats: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """각 토픽(서비스 이슈 제외)에서 부정 글이 NOTABLE_NEGATIVE_THRESHOLD 이상이면 요약"""
        patterns = []

        for topic in TOPICS:
            if topic == "서비스 이슈":
                continue

            topic_items = [i for i in items if i["topic"] == topic]
            neg_items = [i for i in topic_items if i["sentiment"] == "부정"]

            if len(neg_items) < NOTABLE_NEGATIVE_THRESHOLD:
                continue

            print(f"  [{topic}] 부정 {len(neg_items)}건 — LLM 요약 중...")
            summary = self._summarize_negative_pattern(neg_items, topic)

            patterns.append({
                "topic": topic,
                "negative_count": len(neg_items),
                "total_in_topic": len(topic_items),
                "negative_ratio": round(len(neg_items) / len(topic_items) * 100, 1) if topic_items else 0,
                "top_masters": [
                    [m, c] for m, c in
                    Counter(i["master"] for i in neg_items).most_common(3)
                ],
                "summary": summary,
            })

        return patterns

    def _summarize_negative_pattern(
        self, items: List[Dict[str, Any]], topic: str
    ) -> str:
        """부정 글들의 공통 테마를 LLM으로 요약"""
        if not self.client:
            return "요약 불가 (API 키 없음)"

        samples = [i["text_short"] for i in items[:20]]
        joined = "\n---\n".join(samples)

        prompt = f"""다음은 금융 콘텐츠 플랫폼의 '{topic}' 주제에서 부정적 감성으로 분류된 이용자 글 {len(items)}건 중 샘플입니다.

공통으로 나타나는 테마/불만 2-3개를 간결하게 정리해주세요.
각 테마는 한 줄로, 불릿 포인트(- ) 형식으로 작성하세요.

[부정 VOC 샘플]
{joined}"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=300,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content.strip()
        except Exception:
            return "- 요약 생성 실패"


def save_sub_themes(result: Dict[str, Any], date: str, output_dir: str = "./data/sub_themes"):
    """서브 테마 분석 결과 저장"""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{date}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"date": date, **result}, f, ensure_ascii=False, indent=2)
    return path


def load_sub_themes(date: str, data_dir: str = "./data/sub_themes") -> Optional[Dict[str, Any]]:
    """저장된 서브 테마 분석 결과 로드"""
    path = os.path.join(data_dir, f"{date}.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None
