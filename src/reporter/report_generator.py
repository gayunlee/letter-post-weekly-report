"""주간 리포트 생성 모듈"""
import os
from typing import Dict, Any
from datetime import datetime
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()


class ReportGenerator:
    """주간 리포트 생성기"""

    def __init__(self, api_key: str = None):
        """
        ReportGenerator 초기화

        Args:
            api_key: Anthropic API 키
        """
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY가 설정되지 않았습니다.")

        self.client = Anthropic(api_key=self.api_key)

    def generate_report(
        self,
        stats: Dict[str, Any],
        start_date: str,
        end_date: str,
        output_path: str = None
    ) -> str:
        """
        주간 리포트 생성

        Args:
            stats: 통계 분석 결과
            start_date: 시작 날짜 (YYYY-MM-DD)
            end_date: 종료 날짜 (YYYY-MM-DD)
            output_path: 저장 경로 (선택, 지정하지 않으면 저장하지 않음)

        Returns:
            생성된 마크다운 리포트
        """
        # 리포트 헤더 생성
        report = self._generate_header(start_date, end_date, stats)

        # 핵심 요약 생성
        report += self._generate_summary(stats)

        # 마스터별 상세 생성
        report += self._generate_master_details(stats)

        # 서비스 피드백 요약
        report += self._generate_service_feedback_summary(stats)

        # 파일로 저장
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report)

        return report

    def _generate_header(
        self,
        start_date: str,
        end_date: str,
        stats: Dict[str, Any]
    ) -> str:
        """리포트 헤더 생성"""
        # 날짜 포맷 변환 (YYYY-MM-DD -> MM.DD)
        start_formatted = datetime.strptime(start_date, '%Y-%m-%d').strftime('%m.%d')
        end_formatted = datetime.strptime(end_date, '%Y-%m-%d').strftime('%m.%d')

        return f"""# 📌 이번 주 이용자 반응 리포트 ({start_formatted} ~ {end_formatted})

(편지 + 게시글 기준)

---

"""

    def _generate_summary(self, stats: Dict[str, Any]) -> str:
        """핵심 요약 생성"""
        total = stats["total_stats"]
        this_week = total["this_week"]
        last_week = total["last_week"]
        change = total["change"]

        summary = f"""# 0. 핵심 요약

| 구분             | 이번 주 | 전주 | 증감 |
| ---------------- | ------- | ---- | ---- |
| 전체 편지 건수   | {this_week['letters']} | {last_week['letters']} | {self._format_change(change['letters'])} |
| 전체 게시글 건수 | {this_week['posts']} | {last_week['posts']} | {self._format_change(change['posts'])} |
| 전체 총합        | {this_week['total']} | {last_week['total']} | {self._format_change(change['total'])} |

"""

        # Claude API로 인사이트 생성
        insight = self._generate_insight_summary(stats)
        summary += f"{insight}\n"

        summary += f"""(총합: 편지 {this_week['letters']}건 / 게시글 {this_week['posts']}건)

---

"""
        return summary

    def _generate_insight_summary(self, stats: Dict[str, Any]) -> str:
        """Claude API를 사용한 인사이트 요약 생성"""
        total = stats["total_stats"]
        category_stats = stats["category_stats"]

        prompt = f"""다음은 금융 콘텐츠 플랫폼의 주간 이용자 반응 통계입니다:

[전체 통계]
- 이번 주: 편지 {total['this_week']['letters']}건, 게시글 {total['this_week']['posts']}건
- 전주: 편지 {total['last_week']['letters']}건, 게시글 {total['last_week']['posts']}건
- 증감: 편지 {total['change']['letters']}, 게시글 {total['change']['posts']}

[카테고리별 통계]
{chr(10).join([f"- {cat}: {count}건" for cat, count in category_stats.items()])}

위 데이터를 바탕으로 2-3문장으로 핵심 인사이트를 작성해주세요. 다음 사항을 포함하세요:
1. 전주 대비 증감 추세
2. 가장 눈에 띄는 특징이나 변화

markdown 불릿 포인트 형식으로 작성해주세요."""

        try:
            message = self.client.messages.create(
                model="claude-sonnet-4-5-20250929",
                max_tokens=500,
                temperature=0.3,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            return message.content[0].text.strip()

        except Exception as e:
            # API 오류시 기본 텍스트 반환
            return f"- 이번 주 전체 이용자 반응 규모는 총 {total['this_week']['total']}건입니다."

    def _generate_master_details(self, stats: Dict[str, Any]) -> str:
        """마스터별 상세 리포트 생성"""
        master_stats = stats["master_stats"]

        # 총 건수로 정렬
        sorted_masters = sorted(
            master_stats.items(),
            key=lambda x: x[1]["this_week"]["total"],
            reverse=True
        )

        details = "# 1. 마스터별 상세\n\n"

        for i, (master_group_name, data) in enumerate(sorted_masters, 1):
            if data["this_week"]["total"] == 0:
                continue

            this_week = data["this_week"]
            last_week = data["last_week"]
            change_data = data["change"]

            # 마스터별 요약 문구 생성
            summary_text = self._generate_master_summary(data)

            # 마스터 그룹명 사용 (숫자 제거된 이름)
            master_name = master_group_name

            # 클럽명은 data["club_names"]에서 가져옴 (analytics에서 수집)
            club_names = data.get("club_names", set())

            # 클럽명 리스트
            clubs_text = ", ".join(sorted(club_names)) if club_names else "정보 없음"

            details += f"""## {i}. {master_name}

**소속 클럽**: {clubs_text}

> {summary_text}

| 구분   | 이번 주 | 전주 | 증감 |
| ------ | ------- | ---- | ---- |
| 편지   | {this_week['letters']} | {last_week['letters']} | {self._format_change(change_data['letters'])} |
| 게시글 | {this_week['posts']} | {last_week['posts']} | {self._format_change(change_data['posts'])} |
| 총합   | {this_week['total']} | {last_week['total']} | {self._format_change(change_data['total'])} |

■ 주요 내용

"""

            # 카테고리별 주요 내용
            categories = data["categories"]
            for category, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
                if count > 0:
                    details += f"- {category}: {count}건\n"

            details += "\n"

            # 샘플 콘텐츠
            if data["contents"]:
                details += "샘플 콘텐츠:\n\n"
                for content in data["contents"][:3]:
                    text = content['content']
                    if len(text) > 100:
                        text = text[:100] + "..."
                    details += f"  _\"{text}\"_\n\n"

            details += "---\n\n"

        return details

    def _generate_master_summary(self, master_data: Dict[str, Any]) -> str:
        """마스터별 요약 문구 생성"""
        categories = master_data["categories"]
        top_category = max(categories.items(), key=lambda x: x[1])[0] if categories else "없음"

        change_total = master_data["change"]["total"]

        if change_total > 0:
            trend = "증가"
        elif change_total < 0:
            trend = "감소"
        else:
            trend = "유지"

        return f"{trend} 추세이며, {top_category} 중심의 주간입니다."

    def _generate_service_feedback_summary(self, stats: Dict[str, Any]) -> str:
        """서비스 피드백 요약 생성"""
        feedbacks = stats.get("service_feedbacks", [])

        if not feedbacks:
            return "# 2. 서비스 피드백\n\n서비스 피드백이 없습니다.\n\n---\n\n"

        summary = f"# 2. 서비스 피드백\n\n총 {len(feedbacks)}건의 서비스 피드백이 접수되었습니다.\n\n"

        for i, feedback in enumerate(feedbacks[:10], 1):
            content = feedback['content']
            if len(content) > 150:
                content = content[:150] + "..."

            summary += f"""### {i}. {feedback.get('title', '피드백')}

{content}

**분류 이유**: {feedback['reason']}

---

"""

        return summary

    def _format_change(self, value: int) -> str:
        """증감 값 포맷팅"""
        if value > 0:
            return f"+{value}"
        elif value < 0:
            return str(value)
        else:
            return "±0"
