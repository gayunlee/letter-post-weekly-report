"""채널톡 golden set 추출 스크립트

분류된 결과에서 route × topic 층화 샘플링으로 수동 검수용 데이터 추출.

원칙 (MEMORY.md):
  - Golden set은 절대 대충 넣지 않는다
  - LLM 라벨은 "후보 생성"까지만. 최종 라벨은 반드시 1건씩 검수
  - 파이프라인: LLM 후보 → 1건씩 검수 → 판단 근거 주석 → 확정

사용법:
    # 분류 결과 파일에서 추출
    python3 scripts/extract_channel_golden.py data/channel_io/classified_2025-11-24_2025-12-01.json

    # 샘플 수 조정 (기본 100건)
    python3 scripts/extract_channel_golden.py data/channel_io/classified_*.json --n 150

    # 여러 파일 합산 후 추출
    python3 scripts/extract_channel_golden.py data/channel_io/classified_*.json --n 200

출력:
    data/channel_io/golden/channel_golden_candidates.json
"""
import sys
import os
import json
import argparse
import random
import glob
from datetime import datetime
from collections import Counter

OUTPUT_DIR = "./data/channel_io/golden"


def load_classified_items(paths):
    """분류 결과 파일(들)에서 아이템 로드"""
    all_items = []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        items = data.get("items", [])
        # 분류 결과가 있는 것만
        all_items.extend(items)
    return all_items


def stratified_sample(items, n, seed=42):
    """route × topic 층화 샘플링

    각 (route, topic) 셀에서 비례 배분 + 최소 보장.
    abandoned은 topic이 없으므로 route만으로 그룹.
    """
    random.seed(seed)

    # 그룹핑
    groups = {}
    for item in items:
        route = item.get("route", "unknown")
        cls = item.get("classification")
        topic = cls.get("topic", "미분류") if cls else "미분류"
        key = (route, topic)
        groups.setdefault(key, []).append(item)

    # 비례 배분 (최소 2건, 최대 전체)
    total = len(items)
    sampled = []
    remaining_n = n

    # 1차: 각 그룹에서 최소 2건
    min_per_group = 2
    for key, group_items in sorted(groups.items()):
        take = min(min_per_group, len(group_items))
        chosen = random.sample(group_items, take)
        sampled.extend(chosen)
        remaining_n -= take

    # 2차: 남은 건수를 비례 배분
    already_ids = {item["chatId"] for item in sampled}
    remaining_items = {
        key: [i for i in group if i["chatId"] not in already_ids]
        for key, group in groups.items()
    }

    if remaining_n > 0:
        remaining_total = sum(len(v) for v in remaining_items.values())
        for key, group_items in sorted(remaining_items.items()):
            if not group_items or remaining_total == 0:
                continue
            proportion = len(group_items) / remaining_total
            take = max(0, min(int(remaining_n * proportion), len(group_items)))
            if take > 0:
                chosen = random.sample(group_items, take)
                sampled.extend(chosen)

    # 부족분 랜덤 채우기
    if len(sampled) < n:
        already_ids = {item["chatId"] for item in sampled}
        pool = [i for i in items if i["chatId"] not in already_ids]
        extra = min(n - len(sampled), len(pool))
        if extra > 0:
            sampled.extend(random.sample(pool, extra))

    return sampled[:n]


def main():
    parser = argparse.ArgumentParser(description="채널톡 golden set 후보 추출")
    parser.add_argument("files", nargs="+", help="분류 결과 JSON 파일 (glob 지원)")
    parser.add_argument("--n", type=int, default=100, help="추출 건수 (기본 100)")
    parser.add_argument("--seed", type=int, default=42, help="랜덤 시드")
    parser.add_argument("--output", default=None, help="출력 파일 경로")
    args = parser.parse_args()

    # glob 확장
    paths = []
    for pattern in args.files:
        matched = glob.glob(pattern)
        paths.extend(matched)
    paths = sorted(set(paths))

    if not paths:
        print("  파일을 찾을 수 없습니다.")
        return

    print("=" * 60)
    print("  채널톡 Golden Set 후보 추출")
    print(f"  입력: {len(paths)}개 파일")
    print(f"  목표: {args.n}건")
    print("=" * 60)

    # 로드
    print("\n[1] 분류 결과 로드...")
    items = load_classified_items(paths)
    print(f"  총 {len(items):,}건 로드")

    # 분포 확인
    route_topic = Counter()
    for item in items:
        route = item.get("route", "unknown")
        cls = item.get("classification")
        topic = cls.get("topic", "미분류") if cls else "미분류"
        route_topic[(route, topic)] += 1

    print(f"\n  전체 route × topic 분포:")
    for (route, topic), count in sorted(route_topic.items()):
        print(f"    {route:<20} × {topic:<12} = {count}건")

    # 층화 샘플링
    print(f"\n[2] 층화 샘플링 ({args.n}건)...")
    sampled = stratified_sample(items, args.n, seed=args.seed)
    print(f"  {len(sampled)}건 추출 완료")

    # 샘플 분포
    sampled_dist = Counter()
    for item in sampled:
        route = item.get("route", "unknown")
        cls = item.get("classification")
        topic = cls.get("topic", "미분류") if cls else "미분류"
        sampled_dist[(route, topic)] += 1

    print(f"\n  추출 분포:")
    for (route, topic), count in sorted(sampled_dist.items()):
        print(f"    {route:<20} × {topic:<12} = {count}건")

    # golden set 포맷으로 변환
    golden_candidates = []
    for item in sampled:
        cls = item.get("classification") or {}
        golden_candidates.append({
            "chatId": item["chatId"],
            "text": item["text"],
            "route": item.get("route", "unknown"),
            # LLM 후보 라벨 (검수 전)
            "llm_topic": cls.get("topic"),
            "llm_summary": cls.get("summary"),
            "llm_tags": cls.get("tags", []),
            "llm_confidence": cls.get("confidence"),
            # 수동 검수 필드 (빈칸 → 검수자가 채움)
            "verified_topic": None,
            "verified_route": None,
            "review_note": None,
            "reviewer": None,
            # 메타데이터
            "message_count": item.get("message_count"),
            "user_message_count": item.get("user_message_count"),
            "manager_message_count": item.get("manager_message_count"),
            "workflow_buttons": item.get("workflow_buttons", []),
            "has_free_text": item.get("has_free_text"),
        })

    # 저장
    output_path = args.output or os.path.join(OUTPUT_DIR, "channel_golden_candidates.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    export = {
        "metadata": {
            "purpose": "채널톡 CS golden set 후보 — 수동 검수 필요",
            "source_files": paths,
            "total_source_items": len(items),
            "sampled": len(golden_candidates),
            "sampling_method": "stratified (route × topic)",
            "seed": args.seed,
            "created_at": datetime.now().isoformat(),
            "instructions": (
                "1. 각 항목의 text를 읽고 verified_topic/verified_route를 직접 판정\n"
                "2. llm_topic과 다르면 review_note에 근거 기록\n"
                "3. 검수 완료 후 reviewer 필드에 이름 기록\n"
                "4. 확정된 파일을 data/channel_io/golden/channel_golden_v1.json으로 저장"
            ),
            "topic_options": ["결제·환불", "구독·멤버십", "콘텐츠·수강", "기술·오류", "기타"],
            "route_options": ["manager_resolved", "bot_resolved", "abandoned"],
        },
        "items": golden_candidates,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(export, f, ensure_ascii=False, indent=2)

    file_size = os.path.getsize(output_path) / 1024
    print(f"\n[3] 저장 완료: {output_path} ({file_size:.1f}KB)")

    # 검수 가이드
    print(f"\n{'='*60}")
    print("  다음 단계:")
    print("  1. 위 파일을 열어 1건씩 text를 읽고 verified_topic 채우기")
    print("  2. LLM 라벨과 다르면 review_note에 판단 근거 기록")
    print("  3. 완료 후 → data/channel_io/golden/channel_golden_v1.json")
    print("  4. python3 scripts/benchmark_channel_golden.py 로 벤치마크")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
