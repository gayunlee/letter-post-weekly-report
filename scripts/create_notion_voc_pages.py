"""Obsidian VOC 노트를 Notion 페이지로 변환"""
import os
import re
import requests
from dotenv import load_dotenv

load_dotenv()

NOTION_KEY = os.getenv("NOTION_API_KEY")
NOTION_DB_ID = os.getenv("NOTION_DATABASE_ID")
HEADERS = {
    "Authorization": f"Bearer {NOTION_KEY}",
    "Notion-Version": "2022-06-28",
    "Content-Type": "application/json",
}

def rich_text(text, bold=False, italic=False, code=False):
    """Notion rich text object"""
    ann = {}
    if bold: ann["bold"] = True
    if italic: ann["italic"] = True
    if code: ann["code"] = True
    obj = {"type": "text", "text": {"content": text}}
    if ann:
        obj["annotations"] = ann
    return obj

def parse_inline(text):
    """Parse **bold**, `code` from text into rich_text array"""
    parts = []
    i = 0
    while i < len(text):
        # Bold
        if text[i:i+2] == "**":
            end = text.find("**", i+2)
            if end != -1:
                parts.append(rich_text(text[i+2:end], bold=True))
                i = end + 2
                continue
        # Code
        if text[i] == "`" and text[i:i+3] != "```":
            end = text.find("`", i+1)
            if end != -1:
                parts.append(rich_text(text[i+1:end], code=True))
                i = end + 1
                continue
        # Find next special char
        next_bold = text.find("**", i)
        next_code = text.find("`", i)
        nexts = [x for x in [next_bold, next_code] if x > i]
        end = min(nexts) if nexts else len(text)
        if end > i:
            parts.append(rich_text(text[i:end]))
        i = end
    if not parts:
        parts.append(rich_text(text))
    return parts

def heading(level, text):
    h = f"heading_{level}"
    return {"object": "block", "type": h, h: {"rich_text": parse_inline(text)}}

def paragraph(text):
    if not text.strip():
        return {"object": "block", "type": "paragraph", "paragraph": {"rich_text": []}}
    return {"object": "block", "type": "paragraph", "paragraph": {"rich_text": parse_inline(text)}}

def bullet(text):
    return {"object": "block", "type": "bulleted_list_item",
            "bulleted_list_item": {"rich_text": parse_inline(text)}}

def code_block(code, lang="plain text"):
    return {"object": "block", "type": "code",
            "code": {"rich_text": [rich_text(code)], "language": lang}}

def divider():
    return {"object": "block", "type": "divider", "divider": {}}

def toggle_heading(level, title, children):
    """Notion toggle heading with children"""
    h = f"heading_{level}"
    return {
        "object": "block", "type": h,
        h: {"rich_text": parse_inline(title), "is_toggleable": True, "children": children}
    }

def callout(text_lines, icon="💡"):
    """Notion callout block"""
    combined = "\n".join(text_lines)
    return {
        "object": "block", "type": "callout",
        "callout": {"rich_text": parse_inline(combined), "icon": {"type": "emoji", "emoji": icon}}
    }

def table_block(headers, rows):
    """Notion table block"""
    width = len(headers)
    table_rows = []
    # header row
    table_rows.append({
        "type": "table_row",
        "table_row": {"cells": [[rich_text(h, bold=True)] for h in headers]}
    })
    for row in rows:
        cells = [[rich_text(str(c))] for c in row]
        # pad if needed
        while len(cells) < width:
            cells.append([rich_text("")])
        table_rows.append({"type": "table_row", "table_row": {"cells": cells}})

    return {
        "object": "block", "type": "table",
        "table": {
            "table_width": width,
            "has_column_header": True,
            "has_row_header": False,
            "children": table_rows
        }
    }

def parse_md_table(lines):
    """Parse markdown table lines into headers and rows"""
    headers = []
    rows = []
    for i, line in enumerate(lines):
        cells = [c.strip().replace("**", "").replace("*", "") for c in line.strip().strip("|").split("|")]
        if i == 0:
            headers = cells
        elif set(line.strip().replace("|", "").replace("-", "").replace(":", "").strip()) == set():
            continue  # separator line
        else:
            rows.append(cells)
    return headers, rows


def obsidian_to_notion_blocks(md_text):
    """Convert Obsidian markdown to Notion blocks (simplified)"""
    blocks = []
    lines = md_text.split("\n")
    i = 0

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Skip frontmatter
        if stripped == "---" and i == 0:
            i += 1
            while i < len(lines) and lines[i].strip() != "---":
                i += 1
            i += 1
            continue

        # Divider
        if stripped == "---":
            blocks.append(divider())
            i += 1
            continue

        # Obsidian callout (toggle) - collect content
        if stripped.startswith("> [!"):
            # Parse callout title
            m = re.match(r'> \[!(\w+)\]-?\s*(.*)', stripped)
            if m:
                callout_type = m.group(1)
                title = m.group(2).strip()
                # Collect callout content
                toggle_children = []
                i += 1
                content_lines = []
                while i < len(lines) and lines[i].startswith(">"):
                    content = lines[i][1:].strip()
                    if content.startswith("> "):
                        content = content[2:]  # nested callout
                    content_lines.append(content)
                    i += 1

                # Parse content_lines into blocks
                j = 0
                while j < len(content_lines):
                    cl = content_lines[j]
                    if cl.startswith("|"):
                        # table
                        table_lines = []
                        while j < len(content_lines) and content_lines[j].startswith("|"):
                            table_lines.append(content_lines[j])
                            j += 1
                        if table_lines:
                            h, r = parse_md_table(table_lines)
                            if h and r:
                                toggle_children.append(table_block(h, r))
                        continue
                    elif cl.startswith("```"):
                        code_lines = []
                        j += 1
                        while j < len(content_lines) and not content_lines[j].startswith("```"):
                            code_lines.append(content_lines[j])
                            j += 1
                        j += 1
                        toggle_children.append(code_block("\n".join(code_lines)))
                        continue
                    elif cl.startswith("- "):
                        toggle_children.append(bullet(cl[2:]))
                    elif cl.strip():
                        toggle_children.append(paragraph(cl))
                    j += 1

                if not toggle_children:
                    toggle_children = [paragraph("(내용 없음)")]
                blocks.append(toggle_heading(3, f"{'📌' if callout_type == 'note' else 'ℹ️'} {title}", toggle_children))
                continue
            i += 1
            continue

        # Blockquote (not callout)
        if stripped.startswith("> ") and not stripped.startswith("> [!"):
            quote_text = stripped[2:]
            blocks.append({
                "object": "block", "type": "quote",
                "quote": {"rich_text": parse_inline(quote_text)}
            })
            i += 1
            continue

        # Heading
        hm = re.match(r'^(#{1,3})\s+(.+)$', stripped)
        if hm and not stripped.startswith("# Feedback"):
            level = len(hm.group(1))
            text = hm.group(2).strip()
            blocks.append(heading(level, text))
            i += 1
            continue

        # Code block
        if stripped.startswith("```"):
            lang = stripped[3:].strip() or "plain text"
            code_lines = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code_lines.append(lines[i])
                i += 1
            i += 1
            blocks.append(code_block("\n".join(code_lines), lang))
            continue

        # Table
        if stripped.startswith("|"):
            table_lines = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                table_lines.append(lines[i])
                i += 1
            h, r = parse_md_table(table_lines)
            if h and r:
                blocks.append(table_block(h, r))
            continue

        # Bullet
        if stripped.startswith("- "):
            blocks.append(bullet(stripped[2:]))
            i += 1
            continue

        # Obsidian link - skip
        if stripped.startswith("[["):
            i += 1
            continue

        # Feedback section - skip
        if stripped == "# Feedback":
            break

        # Regular paragraph
        if stripped:
            blocks.append(paragraph(stripped))

        i += 1

    return blocks


def create_notion_page(title, blocks, parent_page_id=None):
    """Create a Notion page"""
    # Notion API limits 100 blocks per request
    # Create page first, then append blocks in batches

    if parent_page_id:
        parent = {"type": "page_id", "page_id": parent_page_id}
    else:
        parent = {"type": "database_id", "database_id": NOTION_DB_ID}

    # First batch (up to 100)
    first_batch = blocks[:100]

    data = {
        "parent": parent,
        "properties": {
            "title": {"title": [{"text": {"content": title}}]}
        } if parent_page_id else {
            "이름": {"title": [{"text": {"content": title}}]}
        },
        "children": first_batch,
    }

    r = requests.post("https://api.notion.com/v1/pages", headers=HEADERS, json=data)
    if not r.ok:
        print(f"Error creating page: {r.status_code}")
        print(r.text[:500])
        return None

    page_id = r.json()["id"]
    print(f"Created page: {title} ({page_id})")

    # Append remaining blocks in batches
    remaining = blocks[100:]
    while remaining:
        batch = remaining[:100]
        remaining = remaining[100:]
        r = requests.patch(
            f"https://api.notion.com/v1/blocks/{page_id}/children",
            headers=HEADERS,
            json={"children": batch}
        )
        if not r.ok:
            print(f"Error appending blocks: {r.status_code}")
            print(r.text[:300])

    return page_id


def main():
    base_path = "/Users/gygygygy/Documents/Obsidian Vault/Forge/notes/voc 데이터 라벨링"

    # 1. Create main page
    print("=== 메인 페이지 생성 ===")
    with open(f"{base_path}/Project - VOC 데이터 라벨링.md", encoding="utf-8") as f:
        main_md = f.read()

    main_blocks = obsidian_to_notion_blocks(main_md)
    main_page_id = create_notion_page("VOC 분류 시스템 브리핑", main_blocks)

    if not main_page_id:
        print("메인 페이지 생성 실패")
        return

    # 2. Create detail page as child
    print("\n=== 상세 자료 하위 페이지 생성 ===")
    with open(f"{base_path}/상세 자료 — VOC 분류기 실험 로그.md", encoding="utf-8") as f:
        detail_md = f.read()

    detail_blocks = obsidian_to_notion_blocks(detail_md)
    detail_page_id = create_notion_page("상세 자료 — VOC 분류기 실험 로그", detail_blocks, parent_page_id=main_page_id)

    if detail_page_id:
        print(f"\n완료!")
        print(f"메인: https://notion.so/{main_page_id.replace('-', '')}")
        print(f"상세: https://notion.so/{detail_page_id.replace('-', '')}")


if __name__ == "__main__":
    main()
