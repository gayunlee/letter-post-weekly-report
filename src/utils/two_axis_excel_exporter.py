"""2축 분류 데이터 엑셀 내보내기

1축 excel_exporter.py와 독립. 컬럼: 주제(Topic) + 감성(Sentiment) 분리.
피벗 분석이 편하도록 '주제×감성' 조합 컬럼도 포함.
"""
import os
from typing import List, Dict, Any
from datetime import datetime
import pandas as pd


def export_two_axis_to_excel(
    letters: List[Dict[str, Any]],
    posts: List[Dict[str, Any]],
    output_path: str,
) -> str:
    """2축 분류된 데이터를 엑셀 파일로 내보내기"""

    letter_rows = []
    for letter in letters:
        cls = letter.get("classification", {})
        detail = letter.get("detail_tags", {})
        letter_rows.append({
            "유형": "편지",
            "마스터": letter.get("masterName", "Unknown"),
            "클럽": letter.get("masterClubName", ""),
            "내용": letter.get("message", ""),
            "주제": cls.get("topic", "미분류"),
            "감성": cls.get("sentiment", "미분류"),
            "주제×감성": f"{cls.get('topic', '미분류')} · {cls.get('sentiment', '미분류')}",
            "카테고리 태그": ", ".join(detail.get("category_tags", [])),
            "자유 태그": ", ".join(detail.get("free_tags", [])),
            "요약": detail.get("summary", ""),
            "주제 신뢰도": round(cls.get("topic_confidence", 0), 3),
            "감성 신뢰도": round(cls.get("sentiment_confidence", 0), 3),
            "날짜": _format_date(letter.get("createdAt", "")),
            "차단": "Y" if letter.get("isBlock") == "true" else "N",
        })

    post_rows = []
    for post in posts:
        cls = post.get("classification", {})
        detail = post.get("detail_tags", {})
        content = post.get("textBody") or post.get("body", "")
        post_rows.append({
            "유형": "게시글",
            "마스터": post.get("masterName", "Unknown"),
            "클럽": post.get("masterClubName", ""),
            "제목": post.get("title", ""),
            "내용": content,
            "주제": cls.get("topic", "미분류"),
            "감성": cls.get("sentiment", "미분류"),
            "주제×감성": f"{cls.get('topic', '미분류')} · {cls.get('sentiment', '미분류')}",
            "카테고리 태그": ", ".join(detail.get("category_tags", [])),
            "자유 태그": ", ".join(detail.get("free_tags", [])),
            "요약": detail.get("summary", ""),
            "주제 신뢰도": round(cls.get("topic_confidence", 0), 3),
            "감성 신뢰도": round(cls.get("sentiment_confidence", 0), 3),
            "날짜": _format_date(post.get("createdAt", "")),
            "차단": "Y" if post.get("isBlock") == "true" else "N",
        })

    df_letters = pd.DataFrame(letter_rows)
    df_posts = pd.DataFrame(post_rows)

    # 전체 통합 시트 (편지+게시글 합친 것 — 피벗 분석용)
    all_rows = letter_rows + post_rows
    df_all = pd.DataFrame(all_rows)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        if not df_all.empty:
            df_all.to_excel(writer, sheet_name="전체", index=False)
            _set_column_widths(writer.sheets["전체"], df_all)

        if not df_letters.empty:
            df_letters.to_excel(writer, sheet_name="편지글", index=False)
            _set_column_widths(writer.sheets["편지글"], df_letters)

        if not df_posts.empty:
            df_posts.to_excel(writer, sheet_name="게시글", index=False)
            _set_column_widths(writer.sheets["게시글"], df_posts)

        # 피벗 요약 시트
        if not df_all.empty:
            _write_pivot_sheet(writer, df_all)

    return output_path


def _write_pivot_sheet(writer, df: pd.DataFrame):
    """피벗 요약 시트 생성 — 주제×감성, 마스터별 감성 분포"""
    # 주제×감성 교차표
    pivot_topic = pd.crosstab(df["주제"], df["감성"], margins=True, margins_name="합계")

    # 마스터별 감성 분포
    pivot_master = pd.crosstab(df["마스터"], df["감성"], margins=True, margins_name="합계")

    # 마스터별 주제 분포
    pivot_master_topic = pd.crosstab(df["마스터"], df["주제"], margins=True, margins_name="합계")

    row = 0
    pivot_topic.to_excel(writer, sheet_name="피벗 요약", startrow=row)
    row += len(pivot_topic) + 3

    pivot_master.to_excel(writer, sheet_name="피벗 요약", startrow=row)
    row += len(pivot_master) + 3

    pivot_master_topic.to_excel(writer, sheet_name="피벗 요약", startrow=row)

    ws = writer.sheets["피벗 요약"]
    ws.column_dimensions["A"].width = 18


def _set_column_widths(worksheet, df: pd.DataFrame):
    """열 너비 설정"""
    widths = {
        "유형": 8,
        "마스터": 15,
        "클럽": 20,
        "제목": 40,
        "내용": 80,
        "주제": 15,
        "감성": 10,
        "주제×감성": 25,
        "카테고리 태그": 25,
        "자유 태그": 35,
        "요약": 40,
        "주제 신뢰도": 12,
        "감성 신뢰도": 12,
        "날짜": 18,
        "차단": 8,
    }
    for i, col in enumerate(df.columns):
        col_letter = chr(65 + i) if i < 26 else chr(65 + i // 26 - 1) + chr(65 + i % 26)
        worksheet.column_dimensions[col_letter].width = widths.get(col, 15)


def _format_date(date_str: str) -> str:
    if not date_str:
        return ""
    try:
        if "T" in date_str:
            dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
            return dt.strftime("%Y-%m-%d %H:%M")
        return date_str
    except Exception:
        return date_str
