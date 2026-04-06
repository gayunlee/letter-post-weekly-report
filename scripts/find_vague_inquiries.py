"""추가 응답 유도가 필요한 문의 패턴 수집

수집 대상 (유저 첫 메시지만으로 처리 불가):
A. 워크플로우 버튼만 누르고 이탈 (자유텍스트 0)
B. 짧은 메시지로 추가 정보 필요 (의도 명확/불명확 무관)
   - 이탈 케이스 + 매니저 추가 질문 케이스 모두

버튼 감지: WORKFLOW_BUTTONS + 이모지 prefix (👆🏻, 💬) + 알려진 패턴
"""
import sys, os, json, re
from datetime import datetime
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.bigquery.client import BigQueryClient
from src.bigquery.channel_queries import ChannelQueryService
from src.bigquery.channel_preprocessor import (
    dedup_messages, group_by_chat, WORKFLOW_BUTTONS
)

# 확장된 버튼 패턴 (기존 WORKFLOW_BUTTONS + 이모지 prefix + 알려진 패턴)
BUTTON_PREFIXES = ["👆🏻", "👆", "💬", "📩"]
EXTRA_BUTTON_PATTERNS = {
    "어스 구독신청/결제하기",
    "💬상담매니저에게 직접 문의",
    "💬1:1 상담 문의하기",
    "그 외 기타 문의(오류/구독해지/환불)",
}


def is_button_text(text: str) -> bool:
    """워크플로우 버튼인지 판별"""
    text = text.strip()
    if text in WORKFLOW_BUTTONS or text in EXTRA_BUTTON_PATTERNS:
        return True
    for prefix in BUTTON_PREFIXES:
        if text.startswith(prefix):
            return True
    return False


def classify_pattern(chat_messages, chat_state):
    """패턴 분류"""
    user_msgs = [m for m in chat_messages if m.get("personType") == "user"]
    manager_msgs = [m for m in chat_messages if m.get("personType") == "manager"]

    if not user_msgs:
        return None

    user_texts = [(m.get("plainText") or "").strip() for m in user_msgs]
    user_texts = [t for t in user_texts if t]  # 빈 문자열 제거

    buttons = [t for t in user_texts if is_button_text(t)]
    free_texts = [t for t in user_texts if not is_button_text(t)]
    has_manager = len(manager_msgs) > 0
    suffix = "_followup" if has_manager else "_abandoned"

    # 패턴 A: 버튼만 (자유텍스트 0)
    if buttons and not free_texts:
        return "A_button_only" + suffix

    # 패턴 B: 짧은 자유텍스트 (전체 합산 60자 이하, 메시지 1-2개)
    if free_texts and len(free_texts) <= 2:
        total_chars = sum(len(t) for t in free_texts)
        if total_chars <= 60:
            return "B_short_msg" + suffix

    # 패턴 C: 버튼 + 짧은 자유텍스트 1개
    if buttons and free_texts and len(free_texts) == 1 and len(free_texts[0]) <= 60:
        return "C_button_plus_short" + suffix

    return None


def format_conversation(chat_id, chat_messages, chat_state, pattern):
    """대화를 보기 좋게 포맷"""
    lines = []
    lines.append(f"{'='*70}")
    lines.append(f"Chat ID: {chat_id}  |  패턴: {pattern}  |  상태: {chat_state or '?'}")

    user_count = sum(1 for m in chat_messages if m.get("personType") == "user")
    manager_count = sum(1 for m in chat_messages if m.get("personType") == "manager")
    bot_count = sum(1 for m in chat_messages if m.get("personType") == "bot")
    lines.append(f"메시지: user={user_count}, manager={manager_count}, bot={bot_count}")

    first_ts = chat_messages[0].get("createdAt", 0)
    last_ts = chat_messages[-1].get("createdAt", 0)
    if isinstance(first_ts, (int, float)) and first_ts > 0:
        first_dt = datetime.fromtimestamp(first_ts / 1000)
        last_dt = datetime.fromtimestamp(last_ts / 1000)
        lines.append(f"기간: {first_dt.strftime('%m/%d %H:%M')} ~ {last_dt.strftime('%m/%d %H:%M')}")

    lines.append("-" * 50)

    for msg in chat_messages:
        person = msg.get("personType", "?")
        text = (msg.get("plainText") or "").strip()
        if not text:
            continue

        # 봇 메시지 축약
        if person == "bot" and len(text) > 80:
            text = text[:80] + "..."

        prefix = {"user": "👤", "manager": "💼", "bot": "🤖"}.get(person, "?")
        btn_mark = " [BTN]" if person == "user" and is_button_text(text) else ""
        lines.append(f"  {prefix} {text}{btn_mark}")

    lines.append("")
    return "\n".join(lines)


def main():
    client = BigQueryClient()
    cq = ChannelQueryService(client)

    print("데이터 조회 중 (2026-01-01 ~ 2026-04-01)...")
    messages, chat_states = cq.get_weekly_conversations("2026-01-01", "2026-04-01")
    print(f"메시지 {len(messages)}건 조회")

    messages = dedup_messages(messages)
    print(f"중복 제거 후 {len(messages)}건")

    groups = group_by_chat(messages)
    print(f"채팅 {len(groups)}건\n")

    # 패턴별 분류
    patterns = {}
    for chat_id, chat_msgs in groups.items():
        state = chat_states.get(chat_id, "")
        pattern = classify_pattern(chat_msgs, state)
        if pattern:
            patterns.setdefault(pattern, []).append((chat_id, chat_msgs, state))

    # ===== 요약 =====
    total = sum(len(v) for v in patterns.values())
    print(f"{'='*70}")
    print(f"전체 {len(groups)}건 중 추가 응답 유도 필요 {total}건 ({total/len(groups)*100:.1f}%)")
    print(f"{'='*70}\n")

    for p in sorted(patterns.keys()):
        print(f"  {p}: {len(patterns[p])}건")

    # ===== 유저 메시지 빈도 분석 (짧은 메시지 패턴) =====
    print(f"\n{'='*70}")
    print("유저 자유텍스트 빈도 TOP 30 (B_short_msg 패턴)")
    print(f"{'='*70}\n")

    free_text_counter = Counter()
    for p, items in patterns.items():
        if not p.startswith("B_"):
            continue
        for chat_id, chat_msgs, state in items:
            for m in chat_msgs:
                if m.get("personType") == "user":
                    t = (m.get("plainText") or "").strip()
                    if t and not is_button_text(t):
                        free_text_counter[t] += 1

    for text, count in free_text_counter.most_common(30):
        print(f"  {count:3d}회  {text}")

    # ===== 상세 원문 =====
    print(f"\n{'='*70}")
    print("상세 원문 (패턴별 최대 15건)")
    print(f"{'='*70}")

    for pattern_name in sorted(patterns.keys()):
        items = patterns[pattern_name]
        print(f"\n### {pattern_name} ({len(items)}건)\n")
        for chat_id, chat_msgs, state in items[:15]:
            print(format_conversation(chat_id, chat_msgs, state, pattern_name))

    # ===== JSON 저장 =====
    export = []
    for pattern_name, items in patterns.items():
        for chat_id, chat_msgs, state in items:
            user_texts = []
            manager_texts = []
            for m in chat_msgs:
                t = (m.get("plainText") or "").strip()
                if not t:
                    continue
                if m.get("personType") == "user":
                    user_texts.append({"text": t, "is_button": is_button_text(t)})
                elif m.get("personType") == "manager":
                    manager_texts.append(t)

            export.append({
                "chatId": chat_id,
                "pattern": pattern_name,
                "state": state,
                "user_messages": user_texts,
                "manager_messages": manager_texts,
                "total_messages": len(chat_msgs),
            })

    os.makedirs("exports", exist_ok=True)
    out_path = "exports/vague_inquiries.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(export, f, ensure_ascii=False, indent=2)
    print(f"\n→ {out_path} 저장 ({len(export)}건)")


if __name__ == "__main__":
    main()
