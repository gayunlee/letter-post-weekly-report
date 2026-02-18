"""주간 리포트 생성 모듈"""
import os
from typing import Dict, Any, List, Tuple
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
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

        # 마스터별 상세 생성 (플랫폼/서비스 피드백, 체크포인트 포함)
        report += self._generate_master_details(stats)

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
        from datetime import timedelta
        # 날짜 포맷 변환 (YYYY-MM-DD -> MM.DD)
        # end_date는 exclusive이므로 하루 빼서 표시
        start_formatted = datetime.strptime(start_date, '%Y-%m-%d').strftime('%m.%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=1)
        end_formatted = end_dt.strftime('%m.%d')

        return f"""# 📌 이번 주 이용자 반응 리포트 ({start_formatted} ~ {end_formatted})

(편지 + 게시글 기준)

---

### 분류 기준

| 카테고리 | 설명 |
| -------- | ---- |
| 감사·후기 | 마스터에 대한 피드백 (긍정적/부정적 포함) |
| 질문·토론 | 포트폴리오, 종목, 투자 전략에 대한 질문 및 토론 |
| 정보성 글 | 투자 경험 공유, 종목 분석, 뉴스/정보 공유 |
| 일상·공감 | 안부, 축하, 가입인사, 일상 이야기, 공감 표현 |
| 서비스 피드백 | 플랫폼/서비스 기능 문의, 일반적인 서비스 관련 질문 |
| 서비스 불편사항 | 플랫폼 오류, 버그, 장애 등 기술적 문제 제보 |
| 서비스 제보/건의 | 서비스 개선 제안, 신규 기능 요청 |

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
        """마스터별 상세 리포트 생성 (Claude API 병렬 호출)"""
        master_stats = stats["master_stats"]

        # 총 건수로 정렬
        sorted_masters = sorted(
            master_stats.items(),
            key=lambda x: x[1]["this_week"]["total"],
            reverse=True
        )

        # 활성 마스터만 필터링
        active_masters = [
            (name, data) for name, data in sorted_masters
            if data["this_week"]["total"] > 0
        ]

        # Claude API 병렬 호출로 인사이트 생성
        insights = {}
        max_workers = min(5, len(active_masters))  # 최대 5개 동시 호출
        print(f"  마스터 인사이트 생성 중... ({len(active_masters)}명, 병렬 {max_workers}개)")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_master = {
                executor.submit(self._generate_master_insight, name, data): name
                for name, data in active_masters
            }
            for future in as_completed(future_to_master):
                master_name = future_to_master[future]
                try:
                    insights[master_name] = future.result()
                except Exception as e:
                    print(f"  ⚠️ {master_name} 인사이트 생성 실패: {e}")
                    insights[master_name] = self._generate_fallback_insight(
                        dict(active_masters).get(master_name, {})
                    )

        # 정렬 순서대로 리포트 조합
        details = "# 1. 오피셜클럽별 상세\n\n"

        for i, (master_group_name, data) in enumerate(active_masters, 1):
            this_week = data["this_week"]
            last_week = data["last_week"]
            change_data = data["change"]

            master_name = master_group_name
            club_names = data.get("club_names", set())

            if len(club_names) > 1:
                clubs_suffix = f" _({'+'.join(sorted(club_names))} 합산)_"
            else:
                clubs_suffix = ""

            insight = insights.get(master_name, self._generate_fallback_insight(data))

            details += f"""## {i}. {master_name}{clubs_suffix}

> {insight['summary']}

| 구분   | 이번 주 | 전주 | 증감 |
| ------ | ------- | ---- | ---- |
| 편지   | {this_week['letters']} | {last_week['letters']} | {self._format_change(change_data['letters'])} |
| 게시글 | {this_week['posts']} | {last_week['posts']} | {self._format_change(change_data['posts'])} |
| 총합   | {this_week['total']} | {last_week['total']} | {self._format_change(change_data['total'])} |

■ 주요 내용

{insight['main_content']}

■ 플랫폼/서비스 피드백

{insight['service_feedback']}

■ 체크 포인트

{insight['checkpoints']}

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
                "service_feedback": "- 서비스 피드백이 없습니다.",
                "checkpoints": "- 특이사항 없음."
            }

        # 일반 콘텐츠와 서비스 관련 콘텐츠 분리
        general_contents = []
        feedback_contents = []
        complaint_contents = []
        suggestion_contents = []
        for c in contents:
            cat = c.get("category", "미분류")
            text = c.get("content", "")
            if cat == "서비스 피드백":
                feedback_contents.append(text)
            elif cat in ("불편사항", "서비스 불편사항"):
                complaint_contents.append(text)
            elif cat == "서비스 제보/건의":
                suggestion_contents.append(text)
            else:
                general_contents.append(f"[{cat}] {text}")

        # 일반 콘텐츠 (최대 15개)
        general_str = "\n".join(general_contents[:15])

        # 서비스 피드백 + 불편사항 + 제보/건의 합쳐서 전달
        all_feedback = []
        if feedback_contents:
            all_feedback.extend([f"[서비스 피드백] {fb}" for fb in feedback_contents])
        if complaint_contents:
            all_feedback.extend([f"[서비스 불편사항] {cp}" for cp in complaint_contents])
        if suggestion_contents:
            all_feedback.extend([f"[서비스 제보/건의] {sg}" for sg in suggestion_contents])
        feedback_str = "\n".join([f"- {fb}" for fb in all_feedback]) if all_feedback else "없음"

        # 카테고리 통계
        cat_stats = "\n".join([f"- {cat}: {cnt}건" for cat, cnt in categories.items()])

        prompt = f"""다음은 금융 투자 커뮤니티 "{master_name}" 마스터의 이번 주 이용자 반응 데이터입니다.

[통계]
- 편지: {data['this_week']['letters']}건 (전주 대비 {change.get('letters', 0):+d})
- 게시글: {data['this_week']['posts']}건 (전주 대비 {change.get('posts', 0):+d})

[카테고리별 분류]
{cat_stats}

[일반 콘텐츠 샘플]
{general_str}

[서비스 피드백 및 불편사항]
{feedback_str}

위 데이터를 분석하여 다음 4가지를 작성해주세요:

1. **summary**: 한 줄 요약 (예: "편지 수는 감소했으나, 포트폴리오 구성과 종목 관련 질문이 중심인 주간입니다.")

2. **main_content**: 주요 내용을 테마별로 정리 (2-4개 테마)
   반드시 아래 형식을 따라주세요:

**1. [테마 제목]**

[테마에 대한 1-2문장 요약 설명]

> _"대표적인 인용문 1"_


**2. [테마 제목]**

[테마에 대한 1-2문장 요약 설명]

> _"대표적인 인용문 2"_


3. **service_feedback**: 플랫폼/서비스 피드백 분석 (서비스 관련 내용이 있을 경우에만)
   반드시 아래 형식을 따라주세요:

**[서비스 불편사항]** N건

> _"대표 인용문"_

_→ 권고사항_


**[서비스 제보/건의]** N건

> _"대표 인용문"_

_→ 권고사항_


(서비스 관련 내용이 없으면 "- 서비스 피드백 없음"으로 작성)

4. **checkpoints**: 체크 포인트 (운영 관점에서 주의할 점 - 불릿 포인트로 1-3개)

JSON 형식으로 응답해주세요:
{{"summary": "...", "main_content": "...", "service_feedback": "...", "checkpoints": "..."}}"""

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
                    "service_feedback": result.get("service_feedback", "- 서비스 피드백 없음"),
                    "checkpoints": result.get("checkpoints", "- 특이사항 없음")
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

        # 카테고리별로 콘텐츠 그룹핑
        service_categories = ["서비스 피드백", "서비스 불편사항", "서비스 제보/건의", "불편사항"]
        category_contents = {}
        for c in contents:
            cat = c.get("category", "미분류")
            if cat not in service_categories:
                if cat not in category_contents:
                    category_contents[cat] = []
                category_contents[cat].append(c.get("content", ""))

        # 주요 내용을 테마 형식으로
        main_parts = []
        theme_num = 1
        for cat, cnt in sorted(categories.items(), key=lambda x: x[1], reverse=True):
            if cat in service_categories:
                continue
            if cnt < 3:  # 3건 미만은 스킵
                continue
            if theme_num > 3:  # 최대 3개 테마
                break

            main_parts.append(f"**{theme_num}. {cat} ({cnt}건)**\n")

            # 해당 카테고리의 대표 인용문
            cat_texts = category_contents.get(cat, [])
            if cat_texts:
                sample = cat_texts[0][:150]
                main_parts.append(f"> _\"{sample}{'...' if len(cat_texts[0]) > 150 else ''}\"_\n\n")

            theme_num += 1

        main_content = "\n".join(main_parts) if main_parts else "- 분석 데이터 부족"

        # 서비스 피드백, 불편사항, 제보/건의 추출
        feedback_items = []
        complaint_items = []
        suggestion_items = []
        for c in contents:
            cat = c.get("category", "")
            text = c.get("content", "")
            if cat == "서비스 피드백" and text:
                feedback_items.append(text)
            elif cat in ("불편사항", "서비스 불편사항") and text:
                complaint_items.append(text)
            elif cat == "서비스 제보/건의" and text:
                suggestion_items.append(text)

        service_feedback_parts = []
        if complaint_items:
            service_feedback_parts.append(f"**[서비스 불편사항]** {len(complaint_items)}건\n")
            sample = complaint_items[0][:120]
            service_feedback_parts.append(f"> _\"{sample}{'...' if len(complaint_items[0]) > 120 else ''}\"_\n")
        if suggestion_items:
            service_feedback_parts.append(f"**[서비스 제보/건의]** {len(suggestion_items)}건\n")
            sample = suggestion_items[0][:120]
            service_feedback_parts.append(f"> _\"{sample}{'...' if len(suggestion_items[0]) > 120 else ''}\"_\n")
        if feedback_items:
            service_feedback_parts.append(f"**[서비스 피드백]** {len(feedback_items)}건\n")
            sample = feedback_items[0][:120]
            service_feedback_parts.append(f"> _\"{sample}{'...' if len(feedback_items[0]) > 120 else ''}\"_\n")

        service_feedback = "\n".join(service_feedback_parts) if service_feedback_parts else "- 서비스 피드백 없음"

        # 체크포인트
        checkpoints = []
        if complaint_items:
            checkpoints.append(f"- 서비스 불편사항 {len(complaint_items)}건 접수됨 - 확인 필요")
        if suggestion_items:
            checkpoints.append(f"- 서비스 제보/건의 {len(suggestion_items)}건 접수됨")
        if feedback_items:
            checkpoints.append(f"- 서비스 피드백 {len(feedback_items)}건 접수됨")
        checkpoint_str = "\n".join(checkpoints) if checkpoints else "- 특이사항 없음"

        return {
            "summary": summary,
            "main_content": main_content,
            "service_feedback": service_feedback,
            "checkpoints": checkpoint_str
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
