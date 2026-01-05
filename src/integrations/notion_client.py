"""Notion API 연동 모듈"""
import os
import re
from typing import Dict, Any, List
from datetime import datetime
from notion_client import Client
from dotenv import load_dotenv

load_dotenv()


class NotionReportClient:
    """Notion 데이터베이스에 리포트를 작성하는 클라이언트"""

    def __init__(
        self,
        api_key: str = None,
        database_id: str = None
    ):
        """
        NotionReportClient 초기화

        Args:
            api_key: Notion API 키
            database_id: 리포트를 저장할 데이터베이스 ID
        """
        self.api_key = api_key or os.getenv("NOTION_API_KEY")
        self.database_id = database_id or os.getenv("NOTION_DATABASE_ID")

        if not self.api_key:
            raise ValueError("NOTION_API_KEY가 설정되지 않았습니다.")
        if not self.database_id:
            raise ValueError("NOTION_DATABASE_ID가 설정되지 않았습니다.")

        self.client = Client(auth=self.api_key)

    def create_report_page(
        self,
        title: str,
        markdown_content: str,
        start_date: str,
        end_date: str
    ) -> Dict[str, Any]:
        """
        리포트 페이지 생성

        Args:
            title: 페이지 제목
            markdown_content: 마크다운 형식의 리포트 내용
            start_date: 시작 날짜 (YYYY-MM-DD)
            end_date: 종료 날짜 (YYYY-MM-DD)

        Returns:
            생성된 페이지 정보 {"id": str, "url": str}
        """
        # 마크다운을 Notion 블록으로 변환
        blocks = self._markdown_to_blocks(markdown_content)

        # 페이지 생성
        page = self.client.pages.create(
            parent={"database_id": self.database_id},
            properties={
                "이름": {
                    "title": [
                        {
                            "text": {
                                "content": title
                            }
                        }
                    ]
                }
            },
            children=blocks[:100]  # Notion API는 한 번에 100개 블록까지만 허용
        )

        page_id = page["id"]
        page_url = page["url"]

        # 100개 초과 블록이 있으면 추가로 append
        if len(blocks) > 100:
            for i in range(100, len(blocks), 100):
                chunk = blocks[i:i+100]
                self.client.blocks.children.append(
                    block_id=page_id,
                    children=chunk
                )

        return {
            "id": page_id,
            "url": page_url
        }

    def _markdown_to_blocks(self, markdown: str) -> List[Dict[str, Any]]:
        """
        마크다운을 Notion 블록 형식으로 변환

        Args:
            markdown: 마크다운 텍스트

        Returns:
            Notion 블록 리스트
        """
        blocks = []
        lines = markdown.split('\n')
        i = 0

        while i < len(lines):
            line = lines[i]

            # 빈 줄 건너뛰기
            if not line.strip():
                i += 1
                continue

            # 구분선 (---)
            if line.strip() == '---':
                blocks.append({"type": "divider", "divider": {}})
                i += 1
                continue

            # 헤더 (# ~ ###)
            if line.startswith('# '):
                blocks.append(self._create_heading(line[2:].strip(), 1))
                i += 1
                continue
            if line.startswith('## '):
                blocks.append(self._create_heading(line[3:].strip(), 2))
                i += 1
                continue
            if line.startswith('### '):
                blocks.append(self._create_heading(line[4:].strip(), 3))
                i += 1
                continue

            # 테이블 (| ... |)
            if line.strip().startswith('|'):
                table_lines = []
                while i < len(lines) and lines[i].strip().startswith('|'):
                    table_lines.append(lines[i])
                    i += 1
                table_block = self._create_table(table_lines)
                if table_block:
                    blocks.append(table_block)
                continue

            # 인용문 (> ...)
            if line.startswith('> '):
                blocks.append(self._create_quote(line[2:].strip()))
                i += 1
                continue

            # 불릿 리스트 (- ...)
            if line.startswith('- '):
                blocks.append(self._create_bullet(line[2:].strip()))
                i += 1
                continue

            # 볼드 텍스트로 시작하는 경우 (**...**)
            if line.startswith('**'):
                blocks.append(self._create_paragraph(line))
                i += 1
                continue

            # 기타 텍스트는 paragraph로
            blocks.append(self._create_paragraph(line))
            i += 1

        return blocks

    def _create_heading(self, text: str, level: int) -> Dict[str, Any]:
        """헤딩 블록 생성"""
        # 이모지 제거 (Notion에서 자체 아이콘 사용)
        text = re.sub(r'^[^\w\s]+\s*', '', text).strip()

        heading_type = f"heading_{level}"
        return {
            "type": heading_type,
            heading_type: {
                "rich_text": self._parse_rich_text(text)
            }
        }

    def _create_paragraph(self, text: str) -> Dict[str, Any]:
        """문단 블록 생성"""
        return {
            "type": "paragraph",
            "paragraph": {
                "rich_text": self._parse_rich_text(text)
            }
        }

    def _create_bullet(self, text: str) -> Dict[str, Any]:
        """불릿 리스트 블록 생성"""
        return {
            "type": "bulleted_list_item",
            "bulleted_list_item": {
                "rich_text": self._parse_rich_text(text)
            }
        }

    def _create_quote(self, text: str) -> Dict[str, Any]:
        """인용문 블록 생성"""
        return {
            "type": "quote",
            "quote": {
                "rich_text": self._parse_rich_text(text)
            }
        }

    def _create_table(self, lines: List[str]) -> Dict[str, Any]:
        """테이블 블록 생성"""
        if len(lines) < 2:
            return None

        # 구분선(| --- | --- |) 제거
        data_lines = [l for l in lines if not re.match(r'^\|[\s\-\|]+\|$', l.strip())]

        if not data_lines:
            return None

        rows = []
        for line in data_lines:
            # | 로 분리하고 앞뒤 빈 요소 제거
            cells = [c.strip() for c in line.split('|')]
            cells = [c for c in cells if c]  # 빈 문자열 제거
            rows.append(cells)

        if not rows:
            return None

        # 열 개수
        col_count = max(len(row) for row in rows)

        table_rows = []
        for row in rows:
            # 열 개수 맞추기
            while len(row) < col_count:
                row.append("")

            cells = []
            for cell in row:
                cells.append(self._parse_rich_text(cell))

            table_rows.append({
                "type": "table_row",
                "table_row": {
                    "cells": cells
                }
            })

        return {
            "type": "table",
            "table": {
                "table_width": col_count,
                "has_column_header": True,
                "has_row_header": False,
                "children": table_rows
            }
        }

    def _parse_rich_text(self, text: str) -> List[Dict[str, Any]]:
        """텍스트를 Notion rich_text 형식으로 파싱"""
        if not text:
            return []

        result = []
        # 간단한 파싱: 볼드(**text**), 이탤릭(_text_) 처리
        # 정규식으로 분리
        pattern = r'(\*\*[^*]+\*\*|_[^_]+_)'
        parts = re.split(pattern, text)

        for part in parts:
            if not part:
                continue

            if part.startswith('**') and part.endswith('**'):
                # 볼드
                result.append({
                    "type": "text",
                    "text": {"content": part[2:-2]},
                    "annotations": {"bold": True}
                })
            elif part.startswith('_') and part.endswith('_'):
                # 이탤릭
                result.append({
                    "type": "text",
                    "text": {"content": part[1:-1]},
                    "annotations": {"italic": True}
                })
            else:
                # 일반 텍스트
                result.append({
                    "type": "text",
                    "text": {"content": part}
                })

        return result if result else [{"type": "text", "text": {"content": text}}]

    def get_week_number_korean(self, date_str: str) -> str:
        """
        날짜를 한글 주차로 변환
        예: 2025-12-22 -> 12월 넷째 주

        Args:
            date_str: YYYY-MM-DD 형식의 날짜

        Returns:
            "12월 넷째 주" 형식의 문자열
        """
        date = datetime.strptime(date_str, '%Y-%m-%d')
        month = date.month

        # 해당 월의 몇 번째 주인지 계산
        first_day = date.replace(day=1)
        week_of_month = (date.day + first_day.weekday()) // 7 + 1

        week_names = ["첫째", "둘째", "셋째", "넷째", "다섯째"]
        week_name = week_names[min(week_of_month - 1, 4)]

        return f"{month}월 {week_name} 주"
