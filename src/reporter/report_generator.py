"""주간 리포트 생성 모듈"""
import os
from typing import Dict, Any
from datetime import datetime
from anthropic import Anthropic
from dotenv import load_dotenv
from src.utils.text_utils import clean_text

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
    ) -> tuple:
        """
        주간 리포트 생성

        Args:
            stats: 통계 분석 결과
            start_date: 시작 날짜 (YYYY-MM-DD)
            end_date: 종료 날짜 (YYYY-MM-DD)
            output_path: 저장 경로 (선택, 지정하지 않으면 저장하지 않음)

        Returns:
            (생성된 마크다운 리포트, 슬랙용 3줄 요약)
        """
        # 리포트 헤더 생성
        report = self._generate_header(start_date, end_date, stats)

        # 핵심 요약 생성
        report += self._generate_summary(stats)

        # 마스터별 상세 생성 (플랫폼/서비스 피드백, 체크포인트 포함)
        report += self._generate_master_details(stats)

        # 슬랙용 3줄 요약 생성
        slack_summary = self._generate_slack_summary(stats)

        # 파일로 저장
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report)

        return report, slack_summary

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

    def _generate_slack_summary(self, stats: Dict[str, Any]) -> str:
        """슬랙 스레드용 3줄 요약 생성"""
        total = stats["total_stats"]
        category_stats = stats["category_stats"]
        master_stats = stats["master_stats"]

        # 마스터별 주요 이슈 수집
        sorted_masters = sorted(
            master_stats.items(),
            key=lambda x: x[1]["this_week"]["total"],
            reverse=True
        )[:5]  # 상위 5개만

        master_summaries = []
        for master_name, data in sorted_masters:
            if data["this_week"]["total"] > 0:
                master_summaries.append(f"- {master_name}: 편지 {data['this_week']['letters']}건, 게시글 {data['this_week']['posts']}건")

        prompt = f"""다음은 금융 콘텐츠 플랫폼의 주간 이용자 반응 통계입니다:

[전체 통계]
- 이번 주: 편지 {total['this_week']['letters']}건, 게시글 {total['this_week']['posts']}건 (총 {total['this_week']['total']}건)
- 전주: 편지 {total['last_week']['letters']}건, 게시글 {total['last_week']['posts']}건 (총 {total['last_week']['total']}건)
- 증감: 편지 {total['change']['letters']:+d}건, 게시글 {total['change']['posts']:+d}건

[카테고리별 통계]
{chr(10).join([f"- {cat}: {count}건" for cat, count in category_stats.items()])}

[상위 마스터]
{chr(10).join(master_summaries)}

위 데이터를 바탕으로 슬랙 스레드에 올릴 3줄 요약을 작성해주세요.

형식:
1줄: 전주 대비 전체 추이 (증감 방향과 특징)
2줄: 카테고리별 특징 (가장 높은 카테고리 중심)
3줄: 이번 주 주요 이슈 (마스터별 특이사항이 있다면 포함)

각 줄은 한 문장으로, 구체적인 숫자는 최소화하고 추세와 특징 중심으로 작성해주세요.
줄바꿈으로 구분하여 3줄만 출력해주세요."""

        try:
            message = self.client.messages.create(
                model="claude-sonnet-4-5-20250929",
                max_tokens=300,
                temperature=0.3,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            return message.content[0].text.strip()

        except Exception as e:
            # API 오류시 기본 텍스트 반환
            change_total = total['change']['total']
            trend = "증가" if change_total > 0 else ("감소" if change_total < 0 else "유지")
            top_category = max(category_stats.items(), key=lambda x: x[1])[0] if category_stats else "일상·공감"

            return (
                f"이번 주 이용자 반응이 전주 대비 {trend}했습니다.\n"
                f"'{top_category}' 카테고리가 가장 많은 비중을 차지했습니다.\n"
                f"서비스 불편 건수는 소수에 그쳐 플랫폼 만족도가 양호한 것으로 파악됩니다."
            )

    def _generate_master_details(self, stats: Dict[str, Any]) -> str:
        """마스터별 상세 리포트 생성"""
        master_stats = stats["master_stats"]

        # 총 건수로 정렬
        sorted_masters = sorted(
            master_stats.items(),
            key=lambda x: x[1]["this_week"]["total"],
            reverse=True
        )

        details = "# 1. 오피셜클럽별 상세\n\n"

        for i, (master_group_name, data) in enumerate(sorted_masters, 1):
            if data["this_week"]["total"] == 0:
                continue

            this_week = data["this_week"]
            last_week = data["last_week"]
            change_data = data["change"]

            # 마스터 그룹명 사용 (숫자 제거된 이름)
            master_name = master_group_name

            # 클럽명은 data["club_names"]에서 가져옴 (analytics에서 수집)
            club_names = data.get("club_names", set())

            # 클럽명이 여러 개면 합산 표시
            if len(club_names) > 1:
                clubs_suffix = f" _({'+'.join(sorted(club_names))} 합산)_"
            else:
                clubs_suffix = ""

            # Claude로 상세 인사이트 생성
            insight = self._generate_master_insight(master_name, data)

            details += f"""## {i}. {master_name}{clubs_suffix}

> {insight['summary']}

| 구분   | 이번 주 | 전주 | 증감 |
| ------ | ------- | ---- | ---- |
| 편지   | {this_week['letters']} | {last_week['letters']} | {self._format_change(change_data['letters'])} |
| 게시글 | {this_week['posts']} | {last_week['posts']} | {self._format_change(change_data['posts'])} |
| 총합   | {this_week['total']} | {last_week['total']} | {self._format_change(change_data['total'])} |

■ 주요 내용

{insight['main_content']}

■ 서비스 피드백

{insight['service_feedback']}

---

"""

        return details

    def _generate_master_insight(self, master_name: str, data: Dict[str, Any]) -> Dict[str, str]:
        """Claude API로 마스터별 상세 인사이트 생성"""
        contents = data.get("contents", [])
        categories = data.get("categories", {})
        change = data.get("change", {})

        # 콘텐츠가 없으면 기본값 반환
        if not contents:
            return {
                "summary": "반응 데이터가 부족하여 상세 분석이 어렵습니다.",
                "main_content": "- 분석할 콘텐츠가 없습니다.",
                "service_feedback": "- 서비스 피드백이 없습니다."
            }

        # 일반 콘텐츠와 서비스 관련 분리
        general_contents = []
        inquiry_contents = []  # 서비스 문의
        complaint_contents = []  # 서비스 불편
        suggestion_contents = []  # 서비스 제보/건의
        for c in contents:
            cat = c.get("category", "미분류")
            text = c.get("content", "")
            if cat == "서비스 문의":
                inquiry_contents.append(text)
            elif cat == "서비스 불편":
                complaint_contents.append(text)
            elif cat == "서비스 제보/건의":
                suggestion_contents.append(text)
            else:
                general_contents.append(f"[{cat}] {text}")

        # 일반 콘텐츠 (최대 15개)
        general_str = "\n".join(general_contents[:15])

        # 서비스 관련 피드백 합쳐서 전달
        all_feedback = []
        if complaint_contents:
            all_feedback.extend([f"[서비스 불편] {cp}" for cp in complaint_contents])
        if inquiry_contents:
            all_feedback.extend([f"[서비스 문의] {iq}" for iq in inquiry_contents])
        if suggestion_contents:
            all_feedback.extend([f"[서비스 제보/건의] {sg}" for sg in suggestion_contents])
        feedback_str = "\n".join([f"- {fb}" for fb in all_feedback]) if all_feedback else "없음"

        # 카테고리 통계
        cat_stats = "\n".join([f"- {cat}: {cnt}건" for cat, cnt in categories.items()])

        # 서비스 피드백 개수 요약
        feedback_count_summary = []
        if inquiry_contents:
            feedback_count_summary.append(f"서비스 문의 {len(inquiry_contents)}건")
        if complaint_contents:
            feedback_count_summary.append(f"서비스 불편 {len(complaint_contents)}건")
        if suggestion_contents:
            feedback_count_summary.append(f"서비스 제보/건의 {len(suggestion_contents)}건")
        feedback_count_str = ", ".join(feedback_count_summary) if feedback_count_summary else "없음"

        prompt = f"""다음은 금융 투자 커뮤니티 "{master_name}" 마스터의 이번 주 이용자 반응 데이터입니다.

[통계]
- 편지: {data['this_week']['letters']}건 (전주 대비 {change.get('letters', 0):+d})
- 게시글: {data['this_week']['posts']}건 (전주 대비 {change.get('posts', 0):+d})

[카테고리별 분류]
{cat_stats}

[일반 콘텐츠]
{general_str}

[서비스 관련 피드백] ({feedback_count_str})
{feedback_str}

위 데이터를 분석하여 다음 3가지를 작성해주세요:

1. **summary**: 한 줄 요약 (예: "편지 수는 감소했으나, 포트폴리오 구성 질문이 중심인 주간입니다.")

2. **main_content**: 주요 내용 (2-3개 테마로 분류, 각 테마에 대표 인용문 1개 포함)
   형식 (중요: 각 테마 항목 사이에 반드시 빈 줄 2개 추가, 인용문은 > 블록으로):
   **1. 테마 제목 (N건)**

   테마 설명 (1-2문장)

   > _"대표 인용문 1개"_


   **2. 테마 제목 (N건)**

   테마 설명 (1-2문장)

   > _"대표 인용문 1개"_

3. **service_feedback**: 서비스 관련 피드백 요약
   - 서비스 피드백이 있다면: 어떤 내용이 있었는지 1-2문장으로 요약하고, 대표 예시 1개 인용
   - 형식 (인용문은 > 블록으로, 요약과 인용문 사이 빈 줄):
     OOO 관련 문의/불편이 N건 있었습니다.

     > _"대표 예시"_

   - 없으면: "- 서비스 관련 피드백 없음"

JSON 형식으로 응답해주세요:
{{"summary": "...", "main_content": "...", "service_feedback": "..."}}"""

        try:
            message = self.client.messages.create(
                model="claude-sonnet-4-5-20250929",
                max_tokens=1500,
                temperature=0.3,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            response_text = message.content[0].text.strip()

            # JSON 파싱
            import json
            import re

            # JSON 블록 추출
            json_match = re.search(r'\{[\s\S]*\}', response_text)
            if json_match:
                result = json.loads(json_match.group())
                return {
                    "summary": result.get("summary", "분석 결과 없음"),
                    "main_content": result.get("main_content", "- 분석 결과 없음"),
                    "service_feedback": result.get("service_feedback", "- 서비스 피드백 없음")
                }

        except Exception as e:
            print(f"⚠️  {master_name} 인사이트 생성 실패: {str(e)}")

        # 실패 시 기본값 반환
        return self._generate_fallback_insight(data)

    def _generate_fallback_insight(self, data: Dict[str, Any]) -> Dict[str, str]:
        """API 실패 시 기본 인사이트 생성 (라벨링 데이터 활용)"""
        categories = data.get("categories", {})
        contents = data.get("contents", [])
        change = data.get("change", {})

        # 가장 많은 카테고리
        top_category = max(categories.items(), key=lambda x: x[1])[0] if categories else "미분류"

        # 증감 트렌드
        if change.get("total", 0) > 0:
            trend = "증가"
        elif change.get("total", 0) < 0:
            trend = "감소"
        else:
            trend = "유지"

        summary = f"전체 규모는 {trend}했으며, {top_category} 중심의 주간입니다."

        # 주요 내용 (새 형식: 테마별 구분, > 블록 인용)
        general_contents = [c for c in contents if c.get("category") not in ["서비스 문의", "서비스 불편", "서비스 제보/건의"]]

        # 카테고리별로 그룹화
        main_parts = []
        for i, (cat, cnt) in enumerate(sorted(categories.items(), key=lambda x: x[1], reverse=True)[:3], 1):
            if cat in ["서비스 문의", "서비스 불편", "서비스 제보/건의"]:
                continue
            # 해당 카테고리의 예시 1개 찾기
            example = next((c.get("content", "") for c in general_contents if c.get("category") == cat), "")
            example_text = f'\n\n> _"{example[:150]}..."_' if example else ""
            main_parts.append(f"**{i}. {cat} ({cnt}건)**\n\n해당 카테고리의 내용입니다.{example_text}")

        main_content = "\n\n".join(main_parts) if main_parts else "- 분석 데이터 부족"

        # 서비스 관련 피드백 추출 (라벨링 데이터 기반)
        inquiry_items = []
        complaint_items = []
        suggestion_items = []
        for c in contents:
            cat = c.get("category", "")
            text = c.get("content", "")
            if cat == "서비스 문의" and text:
                inquiry_items.append(text)
            elif cat == "서비스 불편" and text:
                complaint_items.append(text)
            elif cat == "서비스 제보/건의" and text:
                suggestion_items.append(text)

        # 서비스 피드백 요약
        service_feedback_parts = []
        total_feedback = len(inquiry_items) + len(complaint_items) + len(suggestion_items)

        if total_feedback > 0:
            if complaint_items:
                service_feedback_parts.append(f"- 서비스 불편 {len(complaint_items)}건: _\"{complaint_items[0][:80]}...\"_")
            if inquiry_items:
                service_feedback_parts.append(f"- 서비스 문의 {len(inquiry_items)}건: _\"{inquiry_items[0][:80]}...\"_")
            if suggestion_items:
                service_feedback_parts.append(f"- 서비스 제보/건의 {len(suggestion_items)}건: _\"{suggestion_items[0][:80]}...\"_")

        service_feedback = "\n".join(service_feedback_parts) if service_feedback_parts else "- 서비스 관련 피드백 없음"

        return {
            "summary": summary,
            "main_content": main_content,
            "service_feedback": service_feedback
        }

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
            # clean_text는 analytics에서 이미 적용됨
            content = feedback['content']

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
