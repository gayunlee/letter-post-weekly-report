"""채널톡 리포트용 집계 헬퍼.

자연어 VOC 분류와 워크플로우 버튼 선택 신호를 분리해서 집계한다.
"""
from collections import Counter
from typing import Any, Dict, Iterable


def build_channel_talk_report_context(rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    total_chats = 0
    voc_chats = 0
    workflow_only_chats = 0
    topic_counts = Counter()
    subtag_counts = Counter()
    workflow_button_counts = Counter()
    interaction_type_counts = Counter()

    for row in rows:
        total_chats += 1
        interaction_type = row.get("interaction_type") or (
            "free_text" if row.get("has_free_text") else "workflow_only"
        )
        interaction_type_counts[interaction_type] += 1

        buttons = row.get("workflow_buttons") or []
        if isinstance(buttons, str):
            buttons = [buttons]
        for button in buttons:
            if button:
                workflow_button_counts[button] += 1

        if not row.get("has_free_text"):
            workflow_only_chats += 1
            continue

        voc_chats += 1
        topic = row.get("topic") or ""
        subtag = row.get("subtag") or ""
        if topic:
            topic_counts[topic] += 1
        if subtag:
            subtag_counts[subtag] += 1

    return {
        "total_chats": total_chats,
        "voc_chats": voc_chats,
        "workflow_only_chats": workflow_only_chats,
        "topic_counts": topic_counts,
        "subtag_counts": subtag_counts,
        "workflow_button_counts": workflow_button_counts,
        "interaction_type_counts": interaction_type_counts,
    }


def _pct(value: int, total: int) -> str:
    if total <= 0:
        return "0.0%"
    return f"{value / total * 100:.1f}%"


def render_workflow_intent_section(context: Dict[str, Any], top_n: int = 8) -> str:
    total = context.get("total_chats", 0)
    workflow_only = context.get("workflow_only_chats", 0)
    button_counts = context.get("workflow_button_counts", Counter())

    lines = [
        "## 문의 진입 의도",
        "",
        (
            f"워크플로우 버튼만 선택한 대화는 **{workflow_only}건"
            f"({_pct(workflow_only, total)})**입니다. 이 건들은 고객이 직접 작성한"
            " 자연어 문의가 없어 VOC 주제 카운트에는 포함하지 않습니다."
        ),
        "",
    ]

    if not button_counts:
        lines.append("이번 기간에 집계된 워크플로우 버튼 선택 신호는 없습니다.")
        return "\n".join(lines)

    lines.extend([
        "| 순위 | 선택 버튼 | 건수 | 비중 |",
        "|---:|---|---:|---:|",
    ])
    for rank, (button, count) in enumerate(button_counts.most_common(top_n), start=1):
        lines.append(f"| {rank} | {button} | {count}건 | {_pct(count, total)} |")

    lines.extend([
        "",
        "이 수치는 고객이 어떤 문의 경로로 진입했는지를 보여주는 신호입니다. 실제 불편 내용은 자연어 문의가 있는 상담만 별도로 해석합니다.",
    ])
    return "\n".join(lines)
