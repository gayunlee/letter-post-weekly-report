"""VOC 피드백 뷰어 — 클러스터링 + 원문 조회

실행: streamlit run dashboard/tech_issues_viewer.py
"""
import json
from pathlib import Path
from collections import Counter

import streamlit as st
import pandas as pd

st.set_page_config(page_title="VOC 피드백 뷰어", layout="wide")

# ── 데이터 로드 ──
EXPORTS_DIR = Path("exports")


@st.cache_data
def load_excel(path):
    tasks = pd.read_excel(path, sheet_name="Jira Tasks")
    detail = pd.read_excel(path, sheet_name="전체 상세")
    stats = pd.read_excel(path, sheet_name="통계")
    return tasks, detail, stats


@st.cache_data
def load_classified_data(path):
    with open(path) as f:
        data = json.load(f)
    items = data if isinstance(data, list) else data.get('items', [])
    return pd.DataFrame(items)


# ── 사이드바: 데이터 선택 ──
excel_files = sorted(EXPORTS_DIR.glob("tech_issues_clustered_*.xlsx"), reverse=True)
json_files = sorted(EXPORTS_DIR.glob("*classified*.json"), reverse=True)
json_files += sorted(Path("data/classified_data").glob("v5_*.json"), reverse=True)

st.sidebar.header("데이터 선택")

selected_excel = st.sidebar.selectbox(
    "클러스터링 데이터",
    excel_files if excel_files else [None],
    format_func=lambda x: x.stem.replace("tech_issues_clustered_", "") if x else "없음",
)

selected_json = st.sidebar.selectbox(
    "원문 데이터",
    json_files if json_files else [None],
    format_func=lambda x: x.stem if x else "없음",
)

# ── 탭 구조 ──
main_tab1, main_tab2 = st.tabs(["클러스터링", "원문 조회"])

# ══════════════════════════════════════
# 탭 1: 클러스터링
# ══════════════════════════════════════
with main_tab1:
    if not selected_excel:
        st.warning("클러스터링 엑셀 파일이 없습니다. cluster_tech_issues.py를 먼저 실행하세요.")
    else:
        tasks_df, detail_df, stats_df = load_excel(selected_excel)

        col1, col2, col3 = st.columns(3)
        col1.metric("전체 이슈", f"{len(detail_df)}건")
        col2.metric("클러스터 (Jira 태스크)", f"{len(tasks_df)}개")
        col3.metric("2건 이상 클러스터", f"{len(tasks_df[tasks_df['건수'] >= 2])}개")

        st.divider()

        # 필터
        st.sidebar.divider()
        st.sidebar.subheader("클러스터 필터")

        min_count = st.sidebar.slider("최소 건수", 1, int(tasks_df['건수'].max()), 1)

        if '작성일' in detail_df.columns:
            detail_df['월'] = detail_df['작성일'].astype(str).str[:7]
            months = sorted(detail_df['월'].dropna().unique())
            selected_months = st.sidebar.multiselect("월 선택", months, default=months)
            if selected_months:
                filtered_clusters = detail_df[detail_df['월'].isin(selected_months)]['클러스터'].unique()
                filtered_tasks = tasks_df[tasks_df['클러스터'].isin(filtered_clusters)]
            else:
                filtered_tasks = tasks_df
        else:
            filtered_tasks = tasks_df

        filtered_tasks = filtered_tasks[filtered_tasks['건수'] >= min_count]

        # 클러스터 목록
        st.subheader(f"클러스터 목록 ({len(filtered_tasks)}개)")

        for _, row in filtered_tasks.iterrows():
            cluster_id = row['클러스터']
            count = row['건수']
            desc = row['대표 증상']
            communities = row.get('영향 커뮤니티', '')
            date_range = row.get('날짜 범위', '')
            sentiment = row.get('감성 분포', '')

            if count >= 5:
                badge = "🔴"
            elif count >= 3:
                badge = "🟡"
            else:
                badge = "🟢"

            with st.expander(f"{badge} {cluster_id} — {desc} ({count}건)", expanded=(count >= 5)):
                cols = st.columns([1, 1, 1])
                cols[0].write(f"**커뮤니티:** {communities}")
                cols[1].write(f"**기간:** {date_range}")
                cols[2].write(f"**감성:** {sentiment}")

                cluster_items = detail_df[detail_df['클러스터'] == cluster_id]

                if not cluster_items.empty:
                    for _, item in cluster_items.iterrows():
                        border_color = '#dc3545' if item.get('감성') == '부정' else '#6c757d'
                        st.markdown(f"""
<div style="background: #f8f9fa; padding: 10px; border-radius: 5px; margin: 5px 0; border-left: 3px solid {border_color};">
<small><b>{item.get('출처', '')}</b> | {item.get('마스터', '')} | {item.get('작성일', '')}</small><br>
<b>{item.get('증상 서술', '')}</b><br>
<span style="color: #666;">{str(item.get('원문', ''))[:200]}</span>
</div>
""", unsafe_allow_html=True)

                jira_title = row.get('Jira 제목 제안', '')
                if jira_title:
                    st.code(jira_title, language=None)

        st.divider()
        st.subheader("전체 데이터 테이블")

        tab1, tab2 = st.tabs(["클러스터 요약", "전체 상세"])
        with tab1:
            st.dataframe(
                filtered_tasks[['클러스터', '건수', '대표 증상', '영향 커뮤니티', '날짜 범위', 'Jira 제목 제안']],
                width='stretch', height=400,
            )
        with tab2:
            st.dataframe(
                detail_df[['클러스터', '증상 서술', 'subtag', '마스터', '감성', '원문', '작성일']],
                width='stretch', height=500,
            )


# ══════════════════════════════════════
# 탭 2: 원문 조회
# ══════════════════════════════════════
with main_tab2:
    if not selected_json:
        st.warning("원문 데이터가 없습니다.")
    else:
        raw_df = load_classified_data(selected_json)
        total = len(raw_df)

        st.subheader(f"원문 데이터 ({total:,}건)")

        # 필터 영역
        st.sidebar.divider()
        st.sidebar.subheader("원문 필터")

        # topic 필터
        if 'topic' in raw_df.columns:
            topics = sorted(raw_df['topic'].dropna().unique())
            sel_topics = st.sidebar.multiselect("topic", topics, default=topics)
            filtered = raw_df[raw_df['topic'].isin(sel_topics)]
        else:
            filtered = raw_df

        # subtag 필터
        if 'subtag' in filtered.columns:
            subtags = sorted(filtered['subtag'].dropna().unique())
            sel_subtags = st.sidebar.multiselect("subtag", subtags, default=subtags)
            filtered = filtered[filtered['subtag'].isin(sel_subtags)]

        # 마스터 필터
        if 'masterName' in filtered.columns:
            masters = sorted(filtered['masterName'].dropna().unique())
            sel_masters = st.sidebar.multiselect("마스터", masters, default=[])
            if sel_masters:
                filtered = filtered[filtered['masterName'].isin(sel_masters)]

        # 감성 필터
        if 'sentiment' in filtered.columns:
            sentiments = sorted(filtered['sentiment'].dropna().unique())
            sel_sentiments = st.sidebar.multiselect("감성", sentiments, default=sentiments)
            filtered = filtered[filtered['sentiment'].isin(sel_sentiments)]

        # 키워드 검색
        keyword = st.text_input("키워드 검색 (원문에서)", "")
        if keyword:
            text_col = 'text' if 'text' in filtered.columns else 'message'
            if text_col in filtered.columns:
                filtered = filtered[filtered[text_col].astype(str).str.contains(keyword, case=False, na=False)]

        # 통계 표시
        st.write(f"**필터 결과: {len(filtered):,}건** / 전체 {total:,}건")

        col1, col2, col3, col4 = st.columns(4)
        if 'topic' in filtered.columns:
            topic_counts = filtered['topic'].value_counts()
            for i, (topic, count) in enumerate(topic_counts.items()):
                [col1, col2, col3, col4][i % 4].metric(topic, f"{count}건")

        st.divider()

        # subtag 분포
        if 'subtag' in filtered.columns:
            with st.expander("subtag 분포", expanded=False):
                subtag_counts = filtered['subtag'].value_counts()
                st.bar_chart(subtag_counts)

        # 마스터별 분포
        if 'masterName' in filtered.columns:
            with st.expander("마스터별 분포", expanded=False):
                master_counts = filtered['masterName'].value_counts().head(20)
                st.bar_chart(master_counts)

        st.divider()

        # 원문 테이블
        text_col = 'text' if 'text' in filtered.columns else 'message'
        display_cols = []
        for col in ['topic', 'subtag', 'type', 'masterName', 'sentiment', 'summary', text_col, 'createdAt']:
            if col in filtered.columns:
                display_cols.append(col)

        st.dataframe(
            filtered[display_cols].sort_values('createdAt', ascending=False) if 'createdAt' in filtered.columns else filtered[display_cols],
            width='stretch',
            height=600,
        )

        # 원문 카드 뷰
        st.divider()
        st.subheader("원문 상세 (최근 50건)")

        recent = filtered.sort_values('createdAt', ascending=False).head(50) if 'createdAt' in filtered.columns else filtered.head(50)

        for _, item in recent.iterrows():
            sentiment = item.get('sentiment', '중립')
            border_color = '#dc3545' if sentiment == '부정' else '#28a745' if sentiment == '긍정' else '#6c757d'
            topic = item.get('topic', '')
            subtag = item.get('subtag', '')
            master = item.get('masterName', '')
            date = str(item.get('createdAt', ''))[:10]
            text = str(item.get(text_col, ''))[:400]
            summary = item.get('summary', '')
            tags = item.get('tags', '')
            if isinstance(tags, list):
                tags = ', '.join(tags)

            st.markdown(f"""
<div style="background: #f8f9fa; padding: 12px; border-radius: 5px; margin: 8px 0; border-left: 4px solid {border_color};">
<small><b>{topic}</b> > {subtag} | {item.get('type', '')} | <b>{master}</b> | {date} | {sentiment}</small><br>
{f'<small style="color: #888;">요약: {summary}</small><br>' if summary else ''}
<div style="margin-top: 6px;">{text}</div>
{f'<small style="color: #aaa;">tags: {tags}</small>' if tags else ''}
</div>
""", unsafe_allow_html=True)
