"""클러스터링 PoC — 대분류 내 하위 테마 자동 발견

기존 2축 분류 데이터에서 특정 대분류(예: 콘텐츠 반응)의 글들을
임베딩 클러스터링으로 하위 그룹핑하여 어떤 세부 테마가 존재하는지 확인.

사용법:
    python scripts/poc_clustering.py
    python scripts/poc_clustering.py --topic "투자 이야기"
    python scripts/poc_clustering.py --all-topics
"""
import sys
import os
import json
import argparse
import numpy as np
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def load_data(data_path: str, topic_filter: str = None):
    """2축 분류 데이터 로드 및 필터링"""
    with open(data_path, encoding="utf-8") as f:
        data = json.load(f)

    items = []
    for item in data.get("letters", []) + data.get("posts", []):
        cls = item.get("classification", {})
        topic = cls.get("topic", "")
        if topic_filter and topic != topic_filter:
            continue
        text = item.get("message") or item.get("textBody") or item.get("body", "")
        if text and len(text.strip()) > 10:
            items.append({
                "text": text.strip()[:500],  # 임베딩용 최대 500자
                "full_text": text.strip(),
                "topic": topic,
                "sentiment": cls.get("sentiment", ""),
                "master": item.get("masterName", "Unknown"),
                "type": "편지" if "message" in item else "게시글",
            })
    return items


def embed_texts(texts: list[str], model_name: str = "jhgan/ko-sroberta-multitask"):
    """sentence-transformers로 임베딩"""
    from sentence_transformers import SentenceTransformer
    print(f"  임베딩 모델 로드: {model_name}")
    model = SentenceTransformer(model_name)
    print(f"  {len(texts)}건 임베딩 중...")
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=64)
    return embeddings


def cluster_hdbscan(embeddings, min_cluster_size: int = 10):
    """HDBSCAN 클러스터링 — 클러스터 수를 자동 결정"""
    try:
        import hdbscan
        print(f"  HDBSCAN 클러스터링 (min_cluster_size={min_cluster_size})")
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=5,
            metric='euclidean',
        )
        labels = clusterer.fit_predict(embeddings)
        return labels
    except ImportError:
        print("  hdbscan 미설치 — k-means로 대체")
        return None


def cluster_kmeans(embeddings, n_clusters: int = 8):
    """K-means 클러스터링"""
    from sklearn.cluster import KMeans
    print(f"  K-means 클러스터링 (k={n_clusters})")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)
    return labels


def find_optimal_k(embeddings, max_k: int = 15):
    """실루엣 스코어로 최적 k 탐색"""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    scores = {}
    for k in range(3, min(max_k + 1, len(embeddings) // 5)):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        score = silhouette_score(embeddings, labels, sample_size=min(1000, len(embeddings)))
        scores[k] = score

    best_k = max(scores, key=scores.get)
    print(f"  실루엣 스코어: {', '.join(f'k={k}: {s:.3f}' for k, s in sorted(scores.items()))}")
    print(f"  최적 k = {best_k} (score={scores[best_k]:.3f})")
    return best_k


def print_cluster_summary(items, labels, topic_name: str):
    """클러스터별 요약 출력"""
    cluster_ids = sorted(set(labels))
    n_clusters = len([c for c in cluster_ids if c >= 0])
    noise = sum(1 for l in labels if l == -1)

    print(f"\n{'='*70}")
    print(f"  [{topic_name}] 클러스터링 결과: {n_clusters}개 하위 그룹")
    if noise > 0:
        print(f"  (노이즈: {noise}건)")
    print(f"{'='*70}")

    for cid in cluster_ids:
        if cid == -1:
            continue
        cluster_items = [items[i] for i, l in enumerate(labels) if l == cid]
        sentiments = Counter(it["sentiment"] for it in cluster_items)
        masters = Counter(it["master"] for it in cluster_items)

        print(f"\n--- 클러스터 {cid} ({len(cluster_items)}건) ---")
        print(f"  감성: {dict(sentiments)}")
        print(f"  주요 마스터: {', '.join(f'{m}({c})' for m, c in masters.most_common(3))}")
        print(f"  대표 샘플:")
        # 가장 짧은 것 + 가장 긴 것 + 랜덤 포함하여 다양하게
        samples = sorted(cluster_items, key=lambda x: len(x["full_text"]))
        indices = [0, len(samples)//3, len(samples)*2//3, min(len(samples)-1, 4)]
        shown = set()
        for idx in indices:
            if idx in shown or idx >= len(samples):
                continue
            shown.add(idx)
            text = samples[idx]["full_text"][:150].replace("\n", " ")
            print(f"    [{samples[idx]['sentiment']}] {text}...")
            if len(shown) >= 4:
                break


def run_poc(data_path: str, topic: str, method: str = "both"):
    """단일 토픽에 대해 PoC 실행"""
    items = load_data(data_path, topic_filter=topic)
    if len(items) < 20:
        print(f"  [{topic}] 데이터 부족: {len(items)}건 (최소 20건 필요)")
        return

    print(f"\n[{topic}] {len(items)}건 분석 시작")

    texts = [it["text"] for it in items]
    embeddings = embed_texts(texts)

    # PCA 차원 축소
    from sklearn.decomposition import PCA
    n_comp = min(50, len(embeddings) - 1)
    print(f"  PCA 차원 축소 (768 → {n_comp})")
    pca = PCA(n_components=n_comp)
    reduced = pca.fit_transform(embeddings)

    # 방법 1: K-means (실루엣 스코어로 최적 k)
    if method in ("both", "kmeans"):
        print("\n  === K-means 클러스터링 ===")
        optimal_k = find_optimal_k(reduced)
        kmeans_labels = cluster_kmeans(reduced, n_clusters=optimal_k)
        print_cluster_summary(items, kmeans_labels, f"{topic} (K-means, k={optimal_k})")

    # 방법 2: HDBSCAN (더 작은 min_cluster_size)
    if method in ("both", "hdbscan"):
        print("\n  === HDBSCAN 클러스터링 ===")
        hdb_labels = cluster_hdbscan(reduced, min_cluster_size=5)
        if hdb_labels is not None:
            print_cluster_summary(items, hdb_labels, f"{topic} (HDBSCAN)")

    return items, kmeans_labels if method != "hdbscan" else hdb_labels


def main():
    parser = argparse.ArgumentParser(description="클러스터링 PoC")
    parser.add_argument("--data", default="./data/classified_data_two_axis/2026-02-09.json",
                        help="2축 분류 데이터 파일")
    parser.add_argument("--topic", default="콘텐츠 반응", help="분석할 토픽")
    parser.add_argument("--all-topics", action="store_true", help="전체 토픽 분석")
    args = parser.parse_args()

    if not os.path.exists(args.data):
        print(f"데이터 파일 없음: {args.data}")
        return

    if args.all_topics:
        for topic in ["콘텐츠 반응", "투자 이야기", "커뮤니티 소통", "서비스 이슈"]:
            run_poc(args.data, topic, method="kmeans")
    else:
        run_poc(args.data, args.topic)


if __name__ == "__main__":
    main()
