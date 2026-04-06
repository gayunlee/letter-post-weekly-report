"""주간 리포트용 피드백 클러스터링 — 인메모리 버전

피드백 아이템 리스트를 받아서 클러스터 ID와 클러스터 라벨을 주입한다.
cluster_tech_issues.py의 파이프라인을 주간 리포트 흐름에 맞게 간소화.
"""
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_distances

logger = logging.getLogger(__name__)

DESCRIBE_MODEL = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
EMBED_MODEL = "amazon.titan-embed-text-v2:0"
DISTANCE_THRESHOLD = 0.50  # 주간 리포트는 약간 넓게
KEYWORD_WEIGHT = 0.4


def _load_domain_rules() -> str:
    rules_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "data", "config", "domain_rules.md"
    )
    if os.path.exists(rules_path):
        with open(rules_path) as f:
            return f.read()
    return ""


DESCRIBE_PROMPT = f"""이 텍스트에서 사용자가 말하고 있는 내용을 한 문장으로 요약하세요.

## 핵심 규칙
- **형식: "[서비스명] — 구체적 증상"** (서비스명 맨 앞)
- 원문에 보이는 사실만 적을 것
- 평가/판단/인과추정 금지
- 특정 부서/담당자 귀책 표현 금지 ("미흡", "부재", "부족", "지연" 등)
- 100자 이내, 한 문장만 출력

## 도메인 규칙
{_load_domain_rules()}

응답: 한 문장만 (JSON 아님)"""


def _describe_single(bedrock, text: str) -> str:
    if not text or len(text.strip()) < 5:
        return f"짧은 텍스트: {text[:30]}"
    for attempt in range(3):
        try:
            resp = bedrock.invoke_model(
                modelId=DESCRIBE_MODEL,
                body=json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 100,
                    "system": DESCRIBE_PROMPT,
                    "messages": [{"role": "user", "content": text[:500]}],
                }),
            )
            result = json.loads(resp["body"].read())
            return result["content"][0]["text"].strip()
        except Exception as e:
            if "Throttl" in str(e) and attempt < 2:
                time.sleep(3 * (attempt + 1))
                continue
            return f"서술 실패: {text[:50]}"


def _embed_single(bedrock, text: str):
    for attempt in range(3):
        try:
            resp = bedrock.invoke_model(
                modelId=EMBED_MODEL,
                body=json.dumps({"inputText": text[:2000], "dimensions": 256}),
            )
            result = json.loads(resp["body"].read())
            return result["embedding"]
        except Exception as e:
            if "Throttl" in str(e) and attempt < 2:
                time.sleep(3 * (attempt + 1))
                continue
            return None


def _keyword_overlap_matrix(feedback_items):
    n = len(feedback_items)
    overlap = np.zeros((n, n))
    tag_sets = []
    for item in feedback_items:
        tags = item.get("tags", [])
        if isinstance(tags, str):
            tags = [t.strip() for t in tags.split(",")]
        words = set()
        for tag in tags or []:
            for part in str(tag).replace("·", " ").replace("_", " ").split():
                if len(part) >= 2:
                    words.add(part)
        tag_sets.append(words)

    for i in range(n):
        for j in range(i + 1, n):
            if not tag_sets[i] or not tag_sets[j]:
                continue
            intersection = len(tag_sets[i] & tag_sets[j])
            union = len(tag_sets[i] | tag_sets[j])
            if union > 0:
                jac = intersection / union
                overlap[i][j] = jac
                overlap[j][i] = jac
    return overlap


def cluster_feedbacks(feedback_items: list) -> tuple[list, dict]:
    """피드백 아이템에 클러스터 ID + 라벨 주입

    Args:
        feedback_items: service_feedbacks 리스트. 각 item은 dict (content, tags 등 포함).

    Returns:
        (enriched_items, cluster_labels)
        enriched_items: item에 'cluster_id', 'cluster_label' 추가
        cluster_labels: {cluster_id: {'label': str, 'size': int, 'items': [idx]}}
    """
    if not feedback_items:
        return [], {}

    # 이용문의 제외 (주간 리포트는 기존 피드백 정의를 따르되, 단순 문의 제거)
    items = [
        (i, item)
        for i, item in enumerate(feedback_items)
        if item.get("subtag") != "이용문의"
    ]

    if not items:
        return feedback_items, {}

    indices = [i for i, _ in items]
    targets = [item for _, item in items]

    logger.info(f"  클러스터링 대상: {len(targets)}건")

    bedrock = boto3.client("bedrock-runtime", region_name="us-west-2")

    # Step 1: 서술 생성
    descriptions = [None] * len(targets)
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_map = {}
        for i, item in enumerate(targets):
            text = item.get("content", "") or item.get("reason", "")
            future = executor.submit(_describe_single, bedrock, text)
            future_map[future] = i
        for future in as_completed(future_map):
            idx = future_map[future]
            descriptions[idx] = future.result()

    # Step 2: 임베딩 (서술 + 원문 concat)
    embeddings = []
    valid = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_map = {}
        for i, (item, desc) in enumerate(zip(targets, descriptions)):
            text = item.get("content", "")[:200]
            embed_text = f"{desc} | {text}"
            future = executor.submit(_embed_single, bedrock, embed_text)
            future_map[future] = i
        for future in as_completed(future_map):
            idx = future_map[future]
            emb = future.result()
            if emb is not None:
                embeddings.append((idx, emb))

    if len(embeddings) < 2:
        # 클러스터링 불가
        for i, desc in zip(indices, descriptions):
            feedback_items[i]["cluster_id"] = None
            feedback_items[i]["cluster_label"] = desc or ""
        return feedback_items, {}

    embeddings.sort(key=lambda x: x[0])
    valid_indices = [e[0] for e in embeddings]
    emb_matrix = np.array([e[1] for e in embeddings])

    # Step 3: 클러스터링
    cosine_dist = cosine_distances(emb_matrix)
    valid_targets = [targets[i] for i in valid_indices]
    kw_overlap = _keyword_overlap_matrix(valid_targets)
    dist_matrix = cosine_dist * (1 - KEYWORD_WEIGHT * kw_overlap)

    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=DISTANCE_THRESHOLD,
        metric="precomputed",
        linkage="average",
    )
    labels = clustering.fit_predict(dist_matrix)

    # 클러스터별 대표 라벨 (가장 긴 서술)
    cluster_info = {}
    for local_idx, cid in enumerate(labels):
        cid = int(cid)
        global_idx = indices[valid_indices[local_idx]]
        desc = descriptions[valid_indices[local_idx]]
        if cid not in cluster_info:
            cluster_info[cid] = {"label": desc, "size": 0, "items": []}
        cluster_info[cid]["size"] += 1
        cluster_info[cid]["items"].append(global_idx)
        # 더 긴 서술이 있으면 업데이트 (구체성 우선)
        if desc and len(desc) > len(cluster_info[cid]["label"]):
            cluster_info[cid]["label"] = desc

    # 피드백 아이템에 주입
    for local_idx, cid in enumerate(labels):
        global_idx = indices[valid_indices[local_idx]]
        feedback_items[global_idx]["cluster_id"] = int(cid)
        feedback_items[global_idx]["cluster_label"] = cluster_info[int(cid)]["label"]

    # 임베딩 실패한 건
    for i, idx in enumerate(indices):
        if i not in valid_indices:
            feedback_items[idx]["cluster_id"] = None
            feedback_items[idx]["cluster_label"] = descriptions[i] if i < len(descriptions) else ""

    n_clusters = len(cluster_info)
    multi = sum(1 for c in cluster_info.values() if c["size"] >= 2)
    logger.info(f"  클러스터링 완료: {n_clusters}개 (2건+: {multi}개)")

    return feedback_items, cluster_info


def enrich_master_stats_with_clusters(stats: dict) -> None:
    """master_stats의 피드백 콘텐츠에 클러스터 라벨을 주입 (in-place).

    service_feedbacks에 이미 cluster_label이 있으면,
    각 마스터의 contents 중 피드백 건에 매칭해서 라벨 주입.
    """
    feedbacks = stats.get("service_feedbacks", [])
    master_stats = stats.get("master_stats", {})

    if not feedbacks or not master_stats:
        return

    # content 앞 60자로 매칭 키 생성
    def _key(text: str) -> str:
        return (text or "").strip()[:60]

    content_to_cluster = {}
    for fb in feedbacks:
        key = _key(fb.get("content", ""))
        if key:
            content_to_cluster[key] = fb.get("cluster_label", "")

    # 각 마스터의 contents에서 피드백 건에 cluster_label 주입
    for master_name, data in master_stats.items():
        for c in data.get("contents", []):
            topic = c.get("topic", "")
            if topic not in ("피드백", "대응 필요"):
                continue
            key = _key(c.get("content", ""))
            if key in content_to_cluster:
                c["cluster_label"] = content_to_cluster[key]
