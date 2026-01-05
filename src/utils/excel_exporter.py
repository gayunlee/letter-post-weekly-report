"""엑셀 파일 내보내기 유틸리티"""
import os
from typing import List, Dict, Any
from datetime import datetime
import pandas as pd


def export_to_excel(
    letters: List[Dict[str, Any]],
    posts: List[Dict[str, Any]],
    output_path: str
) -> str:
    """
    분류된 데이터를 엑셀 파일로 내보내기

    Args:
        letters: 분류된 편지글 리스트
        posts: 분류된 게시글 리스트
        output_path: 출력 파일 경로

    Returns:
        생성된 엑셀 파일 경로
    """
    # 편지글 데이터 변환
    letter_rows = []
    for letter in letters:
        letter_rows.append({
            "마스터": letter.get("masterName", "Unknown"),
            "클럽": letter.get("masterClubName", ""),
            "내용": letter.get("message", ""),
            "라벨": letter.get("classification", {}).get("category", "미분류"),
            "날짜": _format_date(letter.get("createdAt", ""))
        })

    # 게시글 데이터 변환
    post_rows = []
    for post in posts:
        content = post.get("textBody") or post.get("body", "")
        post_rows.append({
            "마스터": post.get("masterName", "Unknown"),
            "클럽": post.get("masterClubName", ""),
            "제목": post.get("title", ""),
            "내용": content,
            "라벨": post.get("classification", {}).get("category", "미분류"),
            "날짜": _format_date(post.get("createdAt", ""))
        })

    # DataFrame 생성
    df_letters = pd.DataFrame(letter_rows)
    df_posts = pd.DataFrame(post_rows)

    # 디렉토리 생성
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 엑셀 파일로 저장 (시트 분리)
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        if not df_letters.empty:
            df_letters.to_excel(writer, sheet_name='편지글', index=False)
            _set_column_widths(writer.sheets['편지글'], df_letters)
        if not df_posts.empty:
            df_posts.to_excel(writer, sheet_name='게시글', index=False)
            _set_column_widths(writer.sheets['게시글'], df_posts)

    return output_path


def _set_column_widths(worksheet, df: pd.DataFrame):
    """열 너비 설정"""
    # 기본 열 너비 설정
    column_widths = {
        '마스터': 15,
        '클럽': 20,
        '제목': 40,
        '내용': 80,  # 내용 열은 넓게
        '라벨': 15,
        '날짜': 18
    }

    for i, col in enumerate(df.columns):
        col_letter = chr(65 + i)  # A, B, C, ...
        width = column_widths.get(col, 15)
        worksheet.column_dimensions[col_letter].width = width


def _format_date(date_str: str) -> str:
    """날짜 포맷 변환"""
    if not date_str:
        return ""
    try:
        # ISO 형식 파싱 시도
        if 'T' in date_str:
            dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            return dt.strftime('%Y-%m-%d %H:%M')
        return date_str
    except Exception:
        return date_str
