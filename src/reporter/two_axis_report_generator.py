"""2축 분류 체계 기반 주간 리포트 생성 모듈

1축 report_generator.py와 독립적으로 동작합니다.
리포트 구조:
  0. 핵심 요약 — 전체 통계 + Topic×Sentiment 매트릭스 + 부정 급증 하이라이트
  1. 오피셜클럽별 상세 — 감성 분포, 주제별 내용, 부정 콘텐츠 샘플
  2. 서비스 이슈 모아보기 — 서비스 이슈×부정 전체 목록
"""
import os
import json
import re
from typing import Dict, Any, List
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from dotenv import load_dotenv
from src.utils.text_utils import clean_text

load_dotenv()

DEFAULT_MODEL = "gpt-4o-mini"

TOPICS = ["콘텐츠 반응", "투자 이야기", "서비스 이슈", "커뮤니티 소통"]
SENTIMENTS = ["긍정", "부정", "중립"]


class TwoAxisReportGenerator:
    """2축 기반 주간 리포트 생성기"""

    def __init__(self, api_key: str = None, model: str = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("CALLME_OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY 또는 CALLME_OPENAI_API_KEY가 설정되지 않았습니다.")
        self.model = model or DEFAULT_MODEL
        self.client = OpenAI(api_key=self.api_key)

    def generate_report(
        self,
        stats: Dict[str, Any],
        start_date: str,
        end_date: str,
        output_path: str = None,
    ) -> str:
        report = self._generate_header(start_date, end_date)
        report += self._generate_summary(stats)
        report += self._generate_category_tags_section(stats)
        report += self._generate_master_details(stats)
        report += self._generate_service_issues_section(stats)
        report += self._generate_sub_theme_section(stats)

        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(report)

        return report

    # ── 헤더 ──────────────────────────────────────────────────────────

    def _generate_header(self, start_date: str, end_date: str) -> str:
        start_fmt = datetime.strptime(start_date, "%Y-%m-%d").strftime("%m.%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=1)
        end_fmt = end_dt.strftime("%m.%d")

        return f"""# 이번 주 이용자 반응 리포트 ({start_fmt} ~ {end_fmt})

(2축 분류: Topic × Sentiment)

---

### 분류 기준

| 축 | 카테고리 | 설명 |
| -- | -------- | ---- |
| Topic | 콘텐츠 반응 | 마스터 콘텐츠(강의, 리포트, 방송)에 대한 반응 |
| Topic | 투자 이야기 | 투자 전략, 종목, 포트폴리오, 시장 분석 |
| Topic | 서비스 이슈 | 플랫폼/앱 기능, 결제, 배송, 구독 등 |
| Topic | 커뮤니티 소통 | 인사, 안부, 축하, 일상 공유 |
| Sentiment | 긍정 | 감사, 만족, 칭찬, 격려 |
| Sentiment | 부정 | 불만, 실망, 걱정, 비판, 답답함 |
| Sentiment | 중립 | 질문, 정보 전달, 사실 기술, 요청 |

---

"""

    # ── 0. 핵심 요약 ──────────────────────────────────────────────────

    def _generate_summary(self, stats: Dict[str, Any]) -> str:
        total = stats["total_stats"]
        tw = total["this_week"]
        lw = total["last_week"]
        ch = total["change"]

        # 전체 통계 테이블
        summary = f"""# 0. 핵심 요약

| 구분 | 이번 주 | 전주 | 증감 |
| ---- | ------- | ---- | ---- |
| 편지 | {tw['letters']} | {lw['letters']} | {self._fmt_change(ch['letters'])} |
| 게시글 | {tw['posts']} | {lw['posts']} | {self._fmt_change(ch['posts'])} |
| 총합 | {tw['total']} | {lw['total']} | {self._fmt_change(ch['total'])} |

"""

        # 감성 분포 테이블
        sd = total["sentiment_distribution"]
        psd = total["prev_sentiment_distribution"]
        summary += "### 전체 감성 분포\n\n"
        summary += "| 감성 | 이번 주 | 전주 | 증감 |\n"
        summary += "| ---- | ------- | ---- | ---- |\n"
        for s in SENTIMENTS:
            this_cnt = sd.get(s, 0)
            prev_cnt = psd.get(s, 0)
            summary += f"| {s} | {this_cnt} | {prev_cnt} | {self._fmt_change(this_cnt - prev_cnt)} |\n"
        summary += "\n"

        # Topic × Sentiment 매트릭스
        matrix = stats["topic_sentiment_matrix"]
        summary += "### Topic × Sentiment 매트릭스\n\n"
        summary += "| Topic | 긍정 | 부정 | 중립 | 합계 |\n"
        summary += "| ----- | ---- | ---- | ---- | ---- |\n"
        for t in TOPICS:
            row = matrix.get(t, {})
            pos = row.get("긍정", 0)
            neg = row.get("부정", 0)
            neu = row.get("중립", 0)
            total_row = pos + neg + neu
            summary += f"| {t} | {pos} | {neg} | {neu} | {total_row} |\n"
        summary += "\n"

        # 부정 감성 증감 하이라이트
        spikes = stats.get("negative_spike_masters", [])
        drops = stats.get("negative_drop_masters", [])

        if spikes:
            summary += "### 부정 감성 급증 마스터\n\n"
            for sp in spikes:
                summary += (
                    f"- **{sp['master']}**: 부정 비율 "
                    f"{sp['prev_negative_ratio']}% → {sp['negative_ratio']}% "
                    f"(**+{sp['change_pp']}%p**), "
                    f"부정 {sp['negative_count']}건/{sp['total_count']}건\n"
                )
            summary += "\n"

        if drops:
            summary += "### 부정 감성 개선 마스터\n\n"
            for dp in drops:
                summary += (
                    f"- **{dp['master']}**: 부정 비율 "
                    f"{dp['prev_negative_ratio']}% → {dp['negative_ratio']}% "
                    f"(**{dp['change_pp']}%p**), "
                    f"부정 {dp['negative_count']}건/{dp['total_count']}건\n"
                )
            summary += "\n"

        # LLM 인사이트
        insight = self._generate_insight_summary(stats)
        summary += f"{insight}\n\n"

        summary += f"(총합: 편지 {tw['letters']}건 / 게시글 {tw['posts']}건)\n\n---\n\n"
        return summary

    def _generate_insight_summary(self, stats: Dict[str, Any]) -> str:
        total = stats["total_stats"]
        matrix = stats["topic_sentiment_matrix"]
        spikes = stats.get("negative_spike_masters", [])

        matrix_str = ""
        for t in TOPICS:
            row = matrix.get(t, {})
            matrix_str += f"  {t}: 긍정 {row.get('긍정', 0)}, 부정 {row.get('부정', 0)}, 중립 {row.get('중립', 0)}\n"

        drops = stats.get("negative_drop_masters", [])

        spike_str = ""
        if spikes:
            for sp in spikes:
                spike_str += f"  - {sp['master']}: 부정 {sp['prev_negative_ratio']}% → {sp['negative_ratio']}% (+{sp['change_pp']}%p)\n"
        else:
            spike_str = "  없음\n"

        drop_str = ""
        if drops:
            for dp in drops:
                drop_str += f"  - {dp['master']}: 부정 {dp['prev_negative_ratio']}% → {dp['negative_ratio']}% ({dp['change_pp']}%p)\n"
        else:
            drop_str = "  없음\n"

        prompt = f"""다음은 금융 콘텐츠 플랫폼의 주간 이용자 반응 통계입니다 (2축 분류: Topic × Sentiment).

[전체 통계]
- 이번 주: 편지 {total['this_week']['letters']}건, 게시글 {total['this_week']['posts']}건
- 전주: 편지 {total['last_week']['letters']}건, 게시글 {total['last_week']['posts']}건

[감성 분포]
- 긍정: {total['sentiment_distribution'].get('긍정', 0)}건
- 부정: {total['sentiment_distribution'].get('부정', 0)}건
- 중립: {total['sentiment_distribution'].get('중립', 0)}건

[Topic × Sentiment]
{matrix_str}

[부정 급증 마스터]
{spike_str}
[부정 개선 마스터]
{drop_str}

위 데이터를 바탕으로 2-3문장으로 핵심 인사이트를 작성해주세요:
1. 전주 대비 감성 변화 추세 (부정 급증/개선 마스터 포함)
2. 주목할 패턴 또는 주의가 필요한 사항

markdown 불릿 포인트 형식으로 작성해주세요."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=500,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content.strip()
        except Exception:
            return f"- 이번 주 전체 이용자 반응 규모는 총 {total['this_week']['total']}건입니다."

    # ── 0.5 카테고리 태그 집계 ────────────────────────────────────────

    def _generate_category_tags_section(self, stats: Dict[str, Any]) -> str:
        """카테고리 태그 기반 세부 집계 섹션"""
        tag_agg = stats.get("category_tag_aggregation")
        if not tag_agg:
            return ""

        overall = tag_agg.get("overall", {})
        by_topic = tag_agg.get("by_topic", {})
        coverage = tag_agg.get("tag_coverage", 0)

        if not overall:
            return ""

        section = f"## 카테고리 태그 집계\n\n"
        section += f"태그 부착률: {coverage}% ({tag_agg['tagged_items']}/{tag_agg['total_items']}건)\n\n"

        # Topic별 태그 테이블
        for topic in TOPICS:
            topic_tags = by_topic.get(topic, {})
            if not topic_tags:
                continue
            section += f"**{topic}**\n\n"
            section += "| 카테고리 태그 | 건수 |\n"
            section += "| ------------- | ---- |\n"
            for tag, count in sorted(topic_tags.items(), key=lambda x: -x[1]):
                section += f"| {tag} | {count} |\n"
            section += "\n"

        section += "---\n\n"
        return section

    # ── 1. 마스터별 상세 ──────────────────────────────────────────────

    def _generate_master_details(self, stats: Dict[str, Any]) -> str:
        master_stats = stats["master_stats"]

        sorted_masters = sorted(
            master_stats.items(),
            key=lambda x: x[1]["this_week"]["total"],
            reverse=True,
        )
        active_masters = [
            (name, data) for name, data in sorted_masters
            if data["this_week"]["total"] > 0
        ]

        # LLM 병렬 호출
        insights = {}
        max_workers = min(5, len(active_masters))
        print(f"  마스터 인사이트 생성 중... ({len(active_masters)}명, 병렬 {max_workers}개)")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {
                executor.submit(self._generate_master_insight, name, data): name
                for name, data in active_masters
            }
            for future in as_completed(future_map):
                master = future_map[future]
                try:
                    insights[master] = future.result()
                except Exception as e:
                    print(f"  {master} 인사이트 실패: {e}")
                    insights[master] = self._fallback_insight(dict(active_masters).get(master, {}))

        details = "# 1. 오피셜클럽별 상세\n\n"

        for i, (master, data) in enumerate(active_masters, 1):
            tw = data["this_week"]
            lw = data["last_week"]
            ch = data["change"]
            sent = data["sentiment"]
            club_names = data.get("club_names", set())

            clubs_suffix = ""
            if len(club_names) > 1:
                clubs_suffix = f" _({'+'.join(sorted(club_names))} 합산)_"

            insight = insights.get(master, self._fallback_insight(data))

            # 감성 바 (텍스트 기반)
            total = tw["total"]
            pos_pct = round(sent.get("긍정", 0) / total * 100) if total else 0
            neg_pct = round(sent.get("부정", 0) / total * 100) if total else 0
            neu_pct = 100 - pos_pct - neg_pct

            # 부정 증감 표시
            neg_change = data.get("negative_change_pp", 0)
            neg_indicator = ""
            if neg_change >= 10:
                neg_indicator = " **주의**"
            elif neg_change >= 5:
                neg_indicator = " _관찰 필요_"

            details += f"""## {i}. {master}{clubs_suffix}

> {insight['summary']}

| 구분 | 이번 주 | 전주 | 증감 |
| ---- | ------- | ---- | ---- |
| 편지 | {tw['letters']} | {lw['letters']} | {self._fmt_change(ch['letters'])} |
| 게시글 | {tw['posts']} | {lw['posts']} | {self._fmt_change(ch['posts'])} |
| 총합 | {tw['total']} | {lw['total']} | {self._fmt_change(ch['total'])} |

**감성 분포**: 긍정 {sent.get('긍정', 0)}건({pos_pct}%) · 부정 {sent.get('부정', 0)}건({neg_pct}%){neg_indicator} · 중립 {sent.get('중립', 0)}건({neu_pct}%)

{insight['main_content']}

{insight['service_feedback']}

{insight['checkpoints']}

---

"""

        return details

    def _generate_master_insight(self, master: str, data: Dict[str, Any]) -> Dict[str, str]:
        """LLM으로 마스터별 인사이트 생성 (2축 기반)"""
        contents = data.get("contents", [])
        if not contents:
            return self._fallback_insight(data)

        # 주제별 콘텐츠 분류
        topic_contents = {}
        service_neg = []
        for c in contents:
            t, s = c["topic"], c["sentiment"]
            if t not in topic_contents:
                topic_contents[t] = []
            topic_contents[t].append(f"[{s}] {c['content']}")

            if t == "서비스 이슈" and s == "부정":
                service_neg.append(c["content"])

        # 주제별 요약
        topic_str = ""
        for t in TOPICS:
            items = topic_contents.get(t, [])
            if items:
                topic_str += f"\n[{t}] {len(items)}건\n"
                for item in items[:5]:
                    topic_str += f"  - {item}\n"

        sent = data["sentiment"]
        change = data.get("change", {})
        neg_change = data.get("negative_change_pp", 0)

        service_str = "\n".join([f"  - {s}" for s in service_neg[:5]]) if service_neg else "없음"

        prompt = f"""다음은 금융 투자 커뮤니티 "{master}" 마스터의 이번 주 이용자 반응 데이터입니다. (2축 분류)

[통계]
- 편지: {data['this_week']['letters']}건 (전주 대비 {change.get('letters', 0):+d})
- 게시글: {data['this_week']['posts']}건 (전주 대비 {change.get('posts', 0):+d})

[감성 분포]
- 긍정: {sent.get('긍정', 0)}건, 부정: {sent.get('부정', 0)}건, 중립: {sent.get('중립', 0)}건
- 부정 비율 변화: {neg_change:+.1f}%p (전주 대비)

[주제별 콘텐츠]{topic_str}

[서비스 이슈 (부정)]
{service_str}

위 데이터를 분석하여 다음 4가지를 JSON으로 작성해주세요:

1. "summary": 한 줄 요약 (감성 추세 포함)
2. "main_content": 주제별 주요 내용 정리 (markdown 형식, 테마 2-4개)
   형식:
   **1. [테마 제목]**
   [1-2문장 설명]
   > _"대표 인용문"_

3. "service_feedback": 서비스 관련 피드백 (없으면 "■ 서비스 피드백\\n\\n- 서비스 피드백 없음")
   형식:
   ■ 서비스 피드백
   **[서비스 이슈 - 부정]** N건
   > _"대표 인용문"_
   _→ 권고사항_

4. "checkpoints": 체크 포인트 (불릿 1-3개, 감성 변화 관점 포함)
   형식:
   ■ 체크 포인트
   - 내용

{{"summary": "...", "main_content": "...", "service_feedback": "...", "checkpoints": "..."}}"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=1500,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}],
            )

            text = response.choices[0].message.content.strip()
            match = re.search(r'\{[\s\S]*\}', text)
            if match:
                result = json.loads(match.group())
                return {
                    "summary": result.get("summary", ""),
                    "main_content": result.get("main_content", ""),
                    "service_feedback": result.get("service_feedback", "■ 서비스 피드백\n\n- 서비스 피드백 없음"),
                    "checkpoints": result.get("checkpoints", "■ 체크 포인트\n\n- 특이사항 없음"),
                }
        except Exception as e:
            print(f"  {master} LLM 실패: {e}")

        return self._fallback_insight(data)

    def _fallback_insight(self, data: Dict[str, Any]) -> Dict[str, str]:
        """LLM 실패시 기본 인사이트"""
        sent = data.get("sentiment", {})
        change = data.get("change", {})
        contents = data.get("contents", [])

        total_change = change.get("total", 0)
        trend = "증가" if total_change > 0 else ("감소" if total_change < 0 else "유지")

        # 지배적 감성
        dominant = max(sent.items(), key=lambda x: x[1])[0] if sent else "중립"
        summary = f"전체 규모 {trend}, {dominant} 감성이 지배적인 주간입니다."

        # 주제별 그룹핑
        topic_groups = {}
        service_items = []
        for c in contents:
            t = c.get("topic", "미분류")
            if t == "서비스 이슈" and c.get("sentiment") == "부정":
                service_items.append(c["content"])
            if t not in topic_groups:
                topic_groups[t] = []
            topic_groups[t].append(c)

        main_parts = []
        idx = 1
        for t in TOPICS:
            items = topic_groups.get(t, [])
            if not items or len(items) < 2:
                continue
            main_parts.append(f"**{idx}. {t} ({len(items)}건)**\n")
            sample = items[0].get("content", "")[:150]
            if sample:
                main_parts.append(f'> _"{sample}"_\n\n')
            idx += 1

        main_content = "\n".join(main_parts) if main_parts else "- 분석 데이터 부족"

        service_feedback = "■ 서비스 피드백\n\n"
        if service_items:
            service_feedback += f"**[서비스 이슈 - 부정]** {len(service_items)}건\n\n"
            service_feedback += f'> _"{service_items[0][:120]}"_\n'
        else:
            service_feedback += "- 서비스 피드백 없음"

        checkpoints = "■ 체크 포인트\n\n"
        if service_items:
            checkpoints += f"- 서비스 이슈(부정) {len(service_items)}건 접수 — 확인 필요\n"
        neg_change = data.get("negative_change_pp", 0)
        if neg_change >= 5:
            checkpoints += f"- 부정 비율 {neg_change:+.1f}%p 변화 — 모니터링 필요\n"
        if not service_items and neg_change < 5:
            checkpoints += "- 특이사항 없음\n"

        return {
            "summary": summary,
            "main_content": main_content,
            "service_feedback": service_feedback,
            "checkpoints": checkpoints,
        }

    # ── 2. 서비스 이슈 모아보기 ───────────────────────────────────────

    def _generate_service_issues_section(self, stats: Dict[str, Any]) -> str:
        """서비스 이슈 모아보기 — 클러스터 기반 그룹핑"""
        sub_themes = stats.get("sub_themes", {})
        clusters = sub_themes.get("service_clusters", {}) if sub_themes else {}
        issues = stats.get("service_issues", [])

        if not clusters and not issues:
            return "# 2. 서비스 이슈 모아보기\n\n서비스 관련 부정 피드백이 없습니다.\n\n"

        # 클러스터 데이터가 있으면 클러스터 기반으로 표시
        if clusters:
            sorted_clusters = sorted(
                clusters.items(),
                key=lambda x: x[1]["count"],
                reverse=True,
            )
            total_items = sum(c["count"] for c in clusters.values())
            section = f"# 2. 서비스 이슈 모아보기\n\n서비스 이슈 총 {total_items}건 — {len(clusters)}개 유형\n\n"

            # 요약 테이블
            section += "| # | 이슈 유형 | 건수 | 주요 마스터 |\n"
            section += "| - | --------- | ---- | ----------- |\n"
            for i, (cid, info) in enumerate(sorted_clusters, 1):
                masters_str = ", ".join(
                    f"{m}({c})" for m, c in info.get("top_masters", [])[:2]
                )
                section += f"| {i} | {info['label']} | {info['count']} | {masters_str} |\n"
            section += "\n"

            # 3건 이상인 클러스터만 상세 표시
            for cid, info in sorted_clusters:
                if info["count"] < 3:
                    continue
                section += f"**{info['label']}** ({info['count']}건)\n\n"
                for sample in info.get("samples", [])[:2]:
                    clean = sample[:150].replace("\n", " ")
                    section += f'> _"{clean}"_\n\n'

            section += "---\n\n"
            return section

        # 클러스터 없으면 기존 방식 (건건이 나열)
        section = f"# 2. 서비스 이슈 모아보기\n\n서비스 이슈(부정) 총 {len(issues)}건\n\n"
        for i, issue in enumerate(issues, 1):
            master = issue.get("master", "Unknown")
            content = issue.get("content", "")
            title = issue.get("title", "")
            type_label = "편지" if issue["type"] == "letter" else "게시글"
            title_str = f" — {title}" if title else ""

            section += f"{i}. [{type_label}] **{master}**{title_str}\n"
            section += f'   > _"{content}"_\n\n'

        section += "---\n\n"
        return section

    # ── 3. 서브 테마 분석 ─────────────────────────────────────────────

    def _generate_sub_theme_section(self, stats: Dict[str, Any]) -> str:
        """주요 부정 테마 요약 섹션 (서비스 이슈 제외 — 이미 섹션 2에서 클러스터 기반 표시)"""
        sub_themes = stats.get("sub_themes", {})
        if not sub_themes:
            return ""

        patterns = sub_themes.get("notable_patterns", [])
        if not patterns:
            return ""

        section = "# 3. 주요 부정 테마 요약\n\n"
        for p in patterns:
            masters_str = ", ".join(f"{m}({c})" for m, c in p.get("top_masters", [])[:3])
            section += (
                f"**[{p['topic']}]** 부정 {p['negative_count']}건 "
                f"/ 전체 {p['total_in_topic']}건 ({p['negative_ratio']}%)\n"
            )
            if masters_str:
                section += f"주요 마스터: {masters_str}\n\n"
            section += f"{p['summary']}\n\n"

        section += "---\n\n"
        return section

    # ── 유틸 ──────────────────────────────────────────────────────────

    def _fmt_change(self, value: int) -> str:
        if value > 0:
            return f"+{value}"
        elif value < 0:
            return str(value)
        return "±0"
