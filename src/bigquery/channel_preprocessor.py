"""채널톡 대화 전처리 모듈

chatId별로 메시지를 그룹핑하고, 사용자 메시지만 연결하여
분류 단위(1대화 = 1분류)로 변환합니다.

전처리 규칙 (spec.md):
- 분류 단위: chatId 기준 1대화 = 1분류
- 사용자 메시지만 연결
- 앞 500자 + 마지막 200자
"""
from typing import List, Dict, Any
from collections import defaultdict


def group_by_chat(messages: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """chatId별로 메시지 그룹핑 (시간순 정렬 유지)"""
    groups = defaultdict(list)
    for msg in messages:
        chat_id = msg.get("chatId")
        if chat_id:
            groups[chat_id].append(msg)

    # 각 그룹 내 시간순 정렬
    for chat_id in groups:
        groups[chat_id].sort(key=lambda m: m.get("createdAt", ""))

    return dict(groups)


def extract_user_text(messages: List[Dict[str, Any]], max_front: int = 500, max_tail: int = 200) -> str:
    """사용자 메시지만 연결 → 앞 500자 + 마지막 200자

    Args:
        messages: 한 chatId의 메시지 리스트 (시간순)
        max_front: 앞부분 최대 글자 수
        max_tail: 뒷부분 최대 글자 수

    Returns:
        연결된 사용자 텍스트
    """
    user_texts = []
    for msg in messages:
        if msg.get("personType") == "user":
            text = (msg.get("plainText") or "").strip()
            if text:
                user_texts.append(text)

    if not user_texts:
        return ""

    full_text = "\n".join(user_texts)

    if len(full_text) <= max_front + max_tail:
        return full_text

    front = full_text[:max_front]
    tail = full_text[-max_tail:]
    return f"{front}\n...\n{tail}"


def build_chat_items(
    messages: List[Dict[str, Any]],
    min_user_chars: int = 10,
) -> List[Dict[str, Any]]:
    """메시지 리스트 → 분류용 아이템 리스트 변환

    Args:
        messages: 전체 메시지 리스트
        min_user_chars: 최소 사용자 텍스트 길이 (이하 제외)

    Returns:
        chatId별 분류용 아이템 리스트
        [{chatId, text, message_count, user_message_count, first_message_at, ...}]
    """
    groups = group_by_chat(messages)
    items = []

    for chat_id, chat_messages in groups.items():
        text = extract_user_text(chat_messages)

        if len(text) < min_user_chars:
            continue

        user_count = sum(1 for m in chat_messages if m.get("personType") == "user")
        bot_count = sum(1 for m in chat_messages if m.get("personType") == "bot")
        manager_count = sum(1 for m in chat_messages if m.get("personType") == "manager")

        items.append({
            "chatId": chat_id,
            "text": text,
            "message_count": len(chat_messages),
            "user_message_count": user_count,
            "bot_message_count": bot_count,
            "manager_message_count": manager_count,
            "first_message_at": chat_messages[0].get("createdAt", ""),
            "last_message_at": chat_messages[-1].get("createdAt", ""),
            "source": "channel_io",
        })

    return items
