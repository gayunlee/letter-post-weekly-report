"""옵시디언 마크다운 노트를 Notion 페이지로 변환
- Obsidian callout ([!example]-, [!info]-) → Notion toggle 블록
- 테이블, 코드블록, 체크리스트 지원
"""
from __future__ import annotations
import os
import re
import requests
from dotenv import load_dotenv

load_dotenv()

NOTION_API_KEY = os.getenv("NOTION_API_KEY")
NOTION_DATABASE_ID = os.getenv("NOTION_DATABASE_ID")
HEADERS = {
    "Authorization": f"Bearer {NOTION_API_KEY}",
    "Notion-Version": "2022-06-28",
    "Content-Type": "application/json",
}

# callout 아이콘 매핑
CALLOUT_ICONS = {
    "example": "📋",
    "info": "💡",
    "tip": "💡",
    "warning": "⚠️",
    "note": "📝",
    "danger": "🔴",
    "abstract": "📄",
    "question": "❓",
}


def _rich_text(text: str) -> list[dict]:
    """마크다운 인라인 → Notion rich_text. 볼드/이탤릭/코드 지원."""
    if not text:
        return [{"type": "text", "text": {"content": ""}}]

    segments = []
    # 볼드 + 이탤릭 + 인라인코드 파싱
    pattern = r"(\*\*[^*]+\*\*|`[^`]+`|\*[^*]+\*|_[^_]+_)"
    parts = re.split(pattern, text)
    for part in parts:
        if not part:
            continue
        if part.startswith("**") and part.endswith("**"):
            segments.append({
                "type": "text",
                "text": {"content": part[2:-2]},
                "annotations": {"bold": True},
            })
        elif part.startswith("`") and part.endswith("`"):
            segments.append({
                "type": "text",
                "text": {"content": part[1:-1]},
                "annotations": {"code": True},
            })
        elif (part.startswith("*") and part.endswith("*")) or (part.startswith("_") and part.endswith("_")):
            segments.append({
                "type": "text",
                "text": {"content": part[1:-1]},
                "annotations": {"italic": True},
            })
        else:
            # 2000자 제한
            for chunk_start in range(0, len(part), 2000):
                segments.append({"type": "text", "text": {"content": part[chunk_start:chunk_start+2000]}})
    return segments or [{"type": "text", "text": {"content": ""}}]


def _heading(text: str, level: int) -> dict:
    htype = f"heading_{level}"
    return {"object": "block", "type": htype, htype: {"rich_text": _rich_text(text)}}


def _paragraph(text: str) -> dict:
    return {"object": "block", "type": "paragraph", "paragraph": {"rich_text": _rich_text(text)}}


def _bullet(text: str) -> dict:
    return {"object": "block", "type": "bulleted_list_item", "bulleted_list_item": {"rich_text": _rich_text(text)}}


def _numbered(text: str) -> dict:
    return {"object": "block", "type": "numbered_list_item", "numbered_list_item": {"rich_text": _rich_text(text)}}


def _code_block(code: str, lang: str = "plain text") -> dict:
    lang_map = {"json": "json", "bash": "bash", "python": "python", "": "plain text"}
    return {
        "object": "block",
        "type": "code",
        "code": {
            "rich_text": [{"type": "text", "text": {"content": code[:2000]}}],
            "language": lang_map.get(lang, "plain text"),
        },
    }


def _divider() -> dict:
    return {"object": "block", "type": "divider", "divider": {}}


def _quote(text: str) -> dict:
    return {"object": "block", "type": "quote", "quote": {"rich_text": _rich_text(text)}}


def _toggle(title: str, children: list[dict], icon: str = "") -> dict:
    """Notion toggle 블록 (= Obsidian callout)"""
    title_text = f"{icon} {title}" if icon else title
    return {
        "object": "block",
        "type": "toggle",
        "toggle": {
            "rich_text": _rich_text(title_text),
            "children": children[:100],  # Notion 제한
        },
    }


def _table(lines: list[str]) -> list[dict]:
    """마크다운 테이블 → Notion table 블록"""
    rows = []
    for line in lines:
        if "---" in line and all(c in "-|: " for c in line):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        rows.append(cells)

    if not rows:
        return []

    n_cols = max(len(r) for r in rows)
    table_rows = []
    for row in rows:
        while len(row) < n_cols:
            row.append("")
        table_rows.append({
            "type": "table_row",
            "table_row": {
                "cells": [[{"type": "text", "text": {"content": cell[:2000]}}] for cell in row[:n_cols]],
            },
        })

    return [{
        "object": "block",
        "type": "table",
        "table": {
            "table_width": n_cols,
            "has_column_header": True,
            "has_row_header": False,
            "children": table_rows,
        },
    }]


def _parse_callout_body(lines: list[str]) -> list[dict]:
    """callout 내부 텍스트 → Notion 블록들"""
    blocks = []
    i = 0
    in_code = False
    code_lines = []
    code_lang = ""

    while i < len(lines):
        line = lines[i]

        # 코드블록
        if line.strip().startswith("```"):
            if in_code:
                blocks.append(_code_block("\n".join(code_lines), code_lang))
                code_lines = []
                in_code = False
            else:
                in_code = True
                code_lang = line.strip().lstrip("`").strip()
            i += 1
            continue

        if in_code:
            code_lines.append(line)
            i += 1
            continue

        stripped = line.strip()

        if not stripped:
            i += 1
            continue

        if stripped == "---":
            blocks.append(_divider())
            i += 1
            continue

        if stripped.startswith("- "):
            blocks.append(_bullet(stripped[2:]))
            i += 1
            continue

        # 볼드 라벨 라인 (예: **패턴**: ...)
        blocks.append(_paragraph(stripped))
        i += 1

    return blocks


def md_to_notion_blocks(md_text: str) -> list[dict]:
    """마크다운 → Notion 블록 변환 (callout → toggle 지원)"""
    blocks = []
    lines = md_text.split("\n")
    i = 0
    in_code = False
    code_lines = []
    code_lang = ""

    while i < len(lines):
        line = lines[i]

        # 코드 블록
        if line.strip().startswith("```") and not _is_in_callout(lines, i):
            if in_code:
                blocks.append(_code_block("\n".join(code_lines), code_lang))
                code_lines = []
                in_code = False
            else:
                in_code = True
                code_lang = line.strip().lstrip("`").strip()
            i += 1
            continue

        if in_code:
            code_lines.append(line)
            i += 1
            continue

        stripped = line.strip()

        # Obsidian callout 감지: > [!type]- Title
        callout_match = re.match(r"^>\s*\[!(\w+)\][- ]\s*(.+)", stripped)
        if callout_match:
            callout_type = callout_match.group(1).lower()
            callout_title = callout_match.group(2)
            icon = CALLOUT_ICONS.get(callout_type, "📌")

            # callout 본문 수집
            i += 1
            body_lines = []
            while i < len(lines):
                cline = lines[i]
                if cline.startswith(">"):
                    # > 제거
                    content = cline[1:].lstrip(" ") if len(cline) > 1 else ""
                    body_lines.append(content)
                    i += 1
                else:
                    break

            children = _parse_callout_body(body_lines)
            blocks.append(_toggle(callout_title, children, icon))
            continue

        # 빈 줄
        if not stripped:
            i += 1
            continue

        # 구분선
        if stripped in ("---", "***", "___"):
            blocks.append(_divider())
            i += 1
            continue

        # 헤딩
        if stripped.startswith("# "):
            blocks.append(_heading(stripped[2:], 1))
            i += 1
            continue
        if stripped.startswith("## "):
            blocks.append(_heading(stripped[3:], 2))
            i += 1
            continue
        if stripped.startswith("### "):
            blocks.append(_heading(stripped[4:], 3))
            i += 1
            continue

        # 테이블
        if "|" in stripped and i + 1 < len(lines) and "---" in lines[i + 1]:
            table_lines = []
            while i < len(lines) and "|" in lines[i].strip():
                table_lines.append(lines[i].strip())
                i += 1
            blocks.extend(_table(table_lines))
            continue

        # 일반 인용문 (callout이 아닌 >)
        if stripped.startswith("> ") and not re.match(r"^>\s*\[!", stripped):
            quote_lines = []
            while i < len(lines) and lines[i].strip().startswith(">"):
                quote_lines.append(lines[i].strip().lstrip("> ").lstrip(">"))
                i += 1
            blocks.append(_quote("\n".join(quote_lines)[:2000]))
            continue

        # 불릿
        if stripped.startswith("- [ ] ") or stripped.startswith("- [x] "):
            checked = stripped.startswith("- [x]")
            blocks.append({
                "object": "block",
                "type": "to_do",
                "to_do": {"rich_text": _rich_text(stripped[6:]), "checked": checked},
            })
            i += 1
            continue

        if stripped.startswith("- ") or stripped.startswith("* "):
            blocks.append(_bullet(stripped[2:]))
            i += 1
            continue

        # 번호 목록
        num_match = re.match(r"^(\d+)\.\s+(.+)", stripped)
        if num_match:
            blocks.append(_numbered(num_match.group(2)))
            i += 1
            continue

        # 일반 텍스트
        blocks.append(_paragraph(stripped))
        i += 1

    return blocks


def _is_in_callout(lines, idx):
    """현재 라인이 callout 블록 안에 있는지 (> 로 시작하는 구간)"""
    if idx > 0 and lines[idx - 1].strip().startswith(">"):
        return True
    return False


def create_notion_page(title: str, blocks: list[dict]) -> str:
    """Notion DB에 페이지 생성 + 블록 추가"""
    payload = {
        "parent": {"database_id": NOTION_DATABASE_ID},
        "properties": {
            "이름": {"title": [{"text": {"content": title}}]},
        },
        "children": blocks[:100],
    }
    resp = requests.post("https://api.notion.com/v1/pages", headers=HEADERS, json=payload)
    if resp.status_code != 200:
        print(f"  오류 상세: {resp.text[:500]}")
    resp.raise_for_status()
    page_id = resp.json()["id"]
    url = resp.json()["url"]

    remaining = blocks[100:]
    while remaining:
        batch = remaining[:100]
        remaining = remaining[100:]
        resp = requests.patch(
            f"https://api.notion.com/v1/blocks/{page_id}/children",
            headers=HEADERS,
            json={"children": batch},
        )
        if resp.status_code != 200:
            print(f"  블록 추가 오류: {resp.text[:300]}")
        resp.raise_for_status()

    return url


def delete_page(page_id: str):
    """Notion 페이지 아카이브 (삭제)"""
    requests.patch(
        f"https://api.notion.com/v1/pages/{page_id}",
        headers=HEADERS,
        json={"archived": True},
    )


def find_existing_pages(title: str) -> list[str]:
    """DB에서 같은 제목의 기존 페이지 찾기"""
    resp = requests.post(
        f"https://api.notion.com/v1/databases/{NOTION_DATABASE_ID}/query",
        headers=HEADERS,
        json={"filter": {"property": "이름", "title": {"equals": title}}},
    )
    if resp.status_code != 200:
        return []
    return [r["id"] for r in resp.json().get("results", [])]


def process_file(filepath: str, title: str):
    """파일 읽고 Notion 페이지 생성 (기존 페이지 있으면 삭제)"""
    # 기존 페이지 삭제
    existing = find_existing_pages(title)
    for pid in existing:
        print(f"  기존 페이지 삭제: {pid}")
        delete_page(pid)

    with open(filepath, encoding="utf-8") as f:
        content = f.read()

    # Obsidian wikilink 정리
    content = re.sub(r"\[\[([^\]]+)\]\]", r"\1", content)

    blocks = md_to_notion_blocks(content)
    print(f"  변환된 블록: {len(blocks)}개")

    url = create_notion_page(title, blocks)
    print(f"  생성 완료: {url}")
    return url


if __name__ == "__main__":
    vault = "/Users/gygygygy/Documents/Obsidian Vault/US/REAL TASK/DEVCRAFT"

    pages = [
        (f"{vault}/채널톡 답변 유형 분석.md", "채널톡 CS 답변 유형 분석"),
        (f"{vault}/CS AI 에이전트 데모 설계.md", "CS AI 에이전트 데모 설계"),
    ]

    for filepath, title in pages:
        print(f"\n[{title}]")
        try:
            process_file(filepath, title)
        except Exception as e:
            print(f"  오류: {e}")
