"""채널톡 서브태그 LLM 응답 정규화."""
from typing import Any, Dict

from .subtag_prompt import SUBTAGS


def empty_subtag_detail(subtag: str = "기타") -> Dict[str, Any]:
    return {
        "subtag": subtag,
        "is_compound": False,
        "compound_reason": None,
        "summary": "",
        "tags": [],
    }


def normalize_subtag_detail(parsed: Dict[str, Any], topic: str) -> Dict[str, Any]:
    """서브태그 LLM 응답을 저장 가능한 안정된 계약으로 정규화."""
    subtag = parsed.get("subtag", "기타")
    valid = SUBTAGS.get(topic, [])
    if subtag not in valid:
        subtag = "기타"

    is_compound = bool(parsed.get("is_compound", False))
    compound_reason = parsed.get("compound_reason")
    if not is_compound:
        compound_reason = None
    elif compound_reason is None:
        compound_reason = ""
    else:
        compound_reason = str(compound_reason).strip()[:120]

    tags = parsed.get("tags", [])
    if not isinstance(tags, list):
        tags = []
    tags = [str(tag).strip()[:30] for tag in tags if str(tag).strip()][:4]

    return {
        "subtag": subtag,
        "is_compound": is_compound,
        "compound_reason": compound_reason,
        "summary": str(parsed.get("summary", "")).strip()[:200],
        "tags": tags,
    }
