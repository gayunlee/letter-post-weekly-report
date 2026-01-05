"""텍스트 처리 유틸리티"""
import re


def clean_text(text: str, max_length: int = 150) -> str:
    """
    텍스트 정리 및 적절한 길이로 자르기

    Args:
        text: 원본 텍스트
        max_length: 최대 길이

    Returns:
        정리된 텍스트
    """
    if not text:
        return ""

    # 1. 줄바꿈을 공백으로 치환
    cleaned = text.replace('\n', ' ').replace('\r', ' ')

    # 2. 연속 공백을 하나로 정리
    cleaned = re.sub(r'\s+', ' ', cleaned)

    # 3. 앞뒤 공백 제거
    cleaned = cleaned.strip()

    # 4. 최대 길이로 자르기 (문장 단위로 시도)
    if len(cleaned) <= max_length:
        return cleaned

    # 문장 끝(. ! ?)에서 자르기 시도
    truncated = cleaned[:max_length]

    # 마지막 문장 종결 부호 찾기
    last_period = max(
        truncated.rfind('.'),
        truncated.rfind('!'),
        truncated.rfind('?'),
        truncated.rfind('~')
    )

    if last_period > max_length * 0.5:  # 최소 50% 이상이면 문장 단위로 자름
        return truncated[:last_period + 1]

    # 문장 종결 부호가 없으면 단어 단위로 자름
    last_space = truncated.rfind(' ')
    if last_space > max_length * 0.7:  # 최소 70% 이상이면 단어 단위로 자름
        return truncated[:last_space] + "…"

    # 그냥 자르고 말줄임표 추가
    return truncated + "…"


def extract_quote(text: str, max_length: int = 120) -> str:
    """
    인용문으로 사용할 텍스트 추출

    Args:
        text: 원본 텍스트
        max_length: 최대 길이

    Returns:
        인용문 형식의 텍스트
    """
    cleaned = clean_text(text, max_length)
    return cleaned
