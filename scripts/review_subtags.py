"""subtag 리뷰 큐 — 새 subtag 후보를 사람이 검수하는 스크립트

사용법:
    python3 scripts/review_subtags.py                 # 리뷰 큐 확인 + 승인/거부
    python3 scripts/review_subtags.py --show           # 리뷰 큐만 확인
    python3 scripts/review_subtags.py --add-to-queue   # 수동으로 후보 추가 (테스트용)

워크플로우:
    1. reclassify_others()가 new_candidates 발견 → append_to_queue()로 큐에 추가
    2. 사람이 이 스크립트 실행 → 후보 하나씩 보고 승인/거부/스킵
    3. 승인된 subtag → v5_subtags.json에 자동 추가
"""
import json
import os
import sys
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
QUEUE_PATH = BASE_DIR / "data" / "config" / "subtag_review_queue.json"
SUBTAG_CONFIG = BASE_DIR / "data" / "config" / "v5_subtags.json"
REVIEW_LOG = BASE_DIR / "data" / "config" / "subtag_review_log.json"


def load_queue() -> list:
    if QUEUE_PATH.exists():
        with open(QUEUE_PATH, encoding="utf-8") as f:
            return json.load(f)
    return []


def save_queue(queue: list):
    os.makedirs(QUEUE_PATH.parent, exist_ok=True)
    with open(QUEUE_PATH, "w", encoding="utf-8") as f:
        json.dump(queue, f, ensure_ascii=False, indent=2)


def load_subtags() -> dict:
    with open(SUBTAG_CONFIG, encoding="utf-8") as f:
        return json.load(f)


def save_subtags(subtags: dict):
    with open(SUBTAG_CONFIG, "w", encoding="utf-8") as f:
        json.dump(subtags, f, ensure_ascii=False, indent=2)


def append_review_log(entry: dict):
    """승인/거부 이력 기록."""
    log = []
    if REVIEW_LOG.exists():
        with open(REVIEW_LOG, encoding="utf-8") as f:
            log = json.load(f)
    log.append(entry)
    with open(REVIEW_LOG, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)


def append_to_queue(candidates: list):
    """reclassify_others()의 new_candidates를 큐에 추가."""
    queue = load_queue()
    for c in candidates:
        c["queued_at"] = datetime.now().isoformat()
        c["status"] = "pending"
        queue.append(c)
    save_queue(queue)
    print(f"  {len(candidates)}건 큐에 추가 (총 {len(queue)}건)")


def show_queue():
    queue = load_queue()
    pending = [q for q in queue if q.get("status") == "pending"]
    print(f"\n{'='*60}")
    print(f"  subtag 리뷰 큐 — {len(pending)}건 대기 중 (전체 {len(queue)}건)")
    print(f"{'='*60}")

    if not pending:
        print("  대기 중인 항목 없음")
        return

    for i, item in enumerate(pending):
        print(f"\n  [{i+1}] topic: {item['topic']}")
        print(f"      제안: {item['suggested_subtag']}")
        print(f"      이유: {item.get('reason', '')[:80]}")
        print(f"      예시: {item.get('text_preview', '')[:80]}")
        print(f"      등록: {item.get('queued_at', '?')[:10]}")


def review_queue():
    queue = load_queue()
    pending_indices = [i for i, q in enumerate(queue) if q.get("status") == "pending"]

    if not pending_indices:
        print("  대기 중인 항목 없음")
        return

    subtags = load_subtags()
    approved_count = 0
    rejected_count = 0

    for qi in pending_indices:
        item = queue[qi]
        topic = item["topic"]
        suggested = item["suggested_subtag"]
        current_list = subtags.get(topic, [])

        print(f"\n{'─'*50}")
        print(f"  topic: {topic}")
        print(f"  제안 subtag: {suggested}")
        print(f"  이유: {item.get('reason', '')}")
        print(f"  예시: {item.get('text_preview', '')}")
        print(f"  현재 목록: {', '.join(current_list)}")
        print()

        while True:
            choice = input("  [a]승인 / [r]거부 / [m]기존으로 매핑 / [s]스킵 / [q]종료: ").strip().lower()
            if choice in ("a", "r", "m", "s", "q"):
                break
            print("  a/r/m/s/q 중 선택하세요")

        if choice == "q":
            break
        elif choice == "a":
            # 기타 앞에 삽입
            if "기타" in current_list:
                idx = current_list.index("기타")
                current_list.insert(idx, suggested)
            else:
                current_list.append(suggested)
            subtags[topic] = current_list
            queue[qi]["status"] = "approved"
            approved_count += 1
            append_review_log({
                "action": "approved",
                "topic": topic,
                "subtag": suggested,
                "date": datetime.now().isoformat(),
            })
            print(f"  ✓ '{suggested}' → {topic} 목록에 추가됨")

        elif choice == "r":
            queue[qi]["status"] = "rejected"
            rejected_count += 1
            append_review_log({
                "action": "rejected",
                "topic": topic,
                "subtag": suggested,
                "reason": item.get("reason", ""),
                "date": datetime.now().isoformat(),
            })
            print(f"  ✗ 거부됨")

        elif choice == "m":
            print(f"  매핑할 기존 subtag 선택:")
            valid = [s for s in current_list if s != "기타"]
            for j, s in enumerate(valid):
                print(f"    [{j+1}] {s}")
            try:
                mi = int(input("  번호: ").strip()) - 1
                mapped = valid[mi]
                queue[qi]["status"] = "mapped"
                queue[qi]["mapped_to"] = mapped
                append_review_log({
                    "action": "mapped",
                    "topic": topic,
                    "subtag": suggested,
                    "mapped_to": mapped,
                    "date": datetime.now().isoformat(),
                })
                print(f"  → '{suggested}'는 '{mapped}'에 매핑됨")
            except (ValueError, IndexError):
                print("  잘못된 입력, 스킵합니다")

        elif choice == "s":
            pass  # 다음으로

    # 저장
    save_queue(queue)
    if approved_count > 0:
        save_subtags(subtags)
        print(f"\n  승인 {approved_count}건 → v5_subtags.json 업데이트 완료")
    if rejected_count > 0:
        print(f"  거부 {rejected_count}건")


def main():
    if "--show" in sys.argv:
        show_queue()
    elif "--add-to-queue" in sys.argv:
        # 테스트용 수동 추가
        topic = input("topic: ").strip()
        subtag = input("subtag: ").strip()
        reason = input("reason: ").strip()
        append_to_queue([{
            "topic": topic,
            "suggested_subtag": subtag,
            "reason": reason,
            "text_preview": "(수동 추가)",
        }])
    else:
        show_queue()
        print()
        review_queue()


if __name__ == "__main__":
    main()
