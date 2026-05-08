"""KST 기준 날짜 유틸리티.

Cloud Run / 로컬 KST 어디서 돌든 동일한 결과를 보장한다.
naive datetime + .timestamp() 또는 datetime.now() 같은 local-TZ 의존
패턴을 피하고, KST 명시 timezone을 사용한다.
"""
from datetime import datetime, timedelta, timezone

KST = timezone(timedelta(hours=9))


def yesterday_kst(now: datetime | None = None) -> str:
    """KST 기준 '어제' 날짜를 YYYY-MM-DD 로 반환한다.

    Args:
        now: 기준 시각 (테스트용). 생략 시 현재 UTC 시각을 사용.

    Returns:
        KST 기준 yesterday in YYYY-MM-DD 형식.
    """
    if now is None:
        now = datetime.now(timezone.utc)
    elif now.tzinfo is None:
        raise ValueError("now must be timezone-aware")

    kst_now = now.astimezone(KST)
    return (kst_now - timedelta(days=1)).strftime("%Y-%m-%d")


def next_day(date_str: str) -> str:
    """YYYY-MM-DD 문자열을 받아 다음날 YYYY-MM-DD 를 반환한다."""
    d = datetime.strptime(date_str, "%Y-%m-%d")
    return (d + timedelta(days=1)).strftime("%Y-%m-%d")
