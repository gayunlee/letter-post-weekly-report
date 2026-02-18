"""2축 VOC 데이터 분석 대시보드

실행: streamlit run dashboard/app.py
"""
import json
import re
import sys
import os
from pathlib import Path
from datetime import datetime

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# 프로젝트 루트를 path에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

# ── 상수 ──────────────────────────────────────────────────────────────
TOPICS = ["콘텐츠 반응", "투자 이야기", "서비스 이슈", "커뮤니티 소통"]
SENTIMENTS = ["긍정", "부정", "중립"]
SENTIMENT_COLORS = {"긍정": "#2ecc71", "부정": "#e74c3c", "중립": "#95a5a6"}
TOPIC_COLORS = {
    "콘텐츠 반응": "#3498db",
    "투자 이야기": "#e67e22",
    "서비스 이슈": "#e74c3c",
    "커뮤니티 소통": "#9b59b6",
}

TWO_AXIS_DIR = Path("data/classified_data_two_axis")
ONE_AXIS_DIR = Path("data/classified_data")


# ── 데이터 로드 ──────────────────────────────────────────────────────

@st.cache_data
def load_week_data(file_path: str) -> dict:
    with open(file_path, encoding="utf-8") as f:
        return json.load(f)


def get_available_weeks(data_dir: Path) -> list:
    """사용 가능한 주차 목록 반환 (최신순)"""
    if not data_dir.exists():
        return []
    files = sorted(data_dir.glob("*.json"), reverse=True)
    return [f.stem for f in files]


def items_to_dataframe(letters: list, posts: list, axis: str = "two") -> pd.DataFrame:
    """편지+게시글을 DataFrame으로 변환"""
    rows = []

    for letter in letters:
        cls = letter.get("classification", {})
        master_name = letter.get("masterName", "Unknown")
        # 이름 끝 숫자 제거
        master_group = re.sub(r'\d+$', '', master_name).strip()

        row = {
            "유형": "편지",
            "마스터": master_group,
            "마스터_원본": master_name,
            "클럽": letter.get("masterClubName", ""),
            "내용": letter.get("message", ""),
            "날짜": _parse_date(letter.get("createdAt", "")),
        }

        if axis == "two":
            row["주제"] = cls.get("topic", "미분류")
            row["감성"] = cls.get("sentiment", "미분류")
            row["주제_신뢰도"] = cls.get("topic_confidence", 0)
            row["감성_신뢰도"] = cls.get("sentiment_confidence", 0)
        else:
            row["카테고리"] = cls.get("category", "미분류")
            row["신뢰도"] = cls.get("confidence", 0)

        rows.append(row)

    for post in posts:
        cls = post.get("classification", {})
        master_name = post.get("masterName", "Unknown")
        master_group = re.sub(r'\d+$', '', master_name).strip()
        content = post.get("textBody") or post.get("body", "")

        row = {
            "유형": "게시글",
            "마스터": master_group,
            "마스터_원본": master_name,
            "클럽": post.get("masterClubName", ""),
            "내용": content,
            "제목": post.get("title", ""),
            "날짜": _parse_date(post.get("createdAt", "")),
        }

        if axis == "two":
            row["주제"] = cls.get("topic", "미분류")
            row["감성"] = cls.get("sentiment", "미분류")
            row["주제_신뢰도"] = cls.get("topic_confidence", 0)
            row["감성_신뢰도"] = cls.get("sentiment_confidence", 0)
        else:
            row["카테고리"] = cls.get("category", "미분류")
            row["신뢰도"] = cls.get("confidence", 0)

        rows.append(row)

    return pd.DataFrame(rows)


def _parse_date(date_str: str):
    if not date_str:
        return None
    try:
        if "T" in date_str:
            return datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        return datetime.strptime(date_str, "%Y-%m-%d")
    except Exception:
        return None


# ── 페이지 설정 ──────────────────────────────────────────────────────

st.set_page_config(
    page_title="VOC 분석 대시보드",
    page_icon="📊",
    layout="wide",
)

st.title("VOC 데이터 분석 대시보드")

# ── 사이드바: 데이터 소스 선택 ────────────────────────────────────────

with st.sidebar:
    st.header("데이터 설정")

    # 분류 체계 선택
    axis_mode = st.radio(
        "분류 체계",
        ["2축 (Topic × Sentiment)", "1축 (카테고리)"],
        index=0,
    )
    is_two_axis = axis_mode.startswith("2축")
    data_dir = TWO_AXIS_DIR if is_two_axis else ONE_AXIS_DIR

    weeks = get_available_weeks(data_dir)

    if not weeks:
        st.warning(
            f"{'2축' if is_two_axis else '1축'} 데이터가 없습니다.\n\n"
            f"{'`python3 scripts/generate_two_axis_report.py`' if is_two_axis else '`python3 scripts/generate_custom_week_report.py`'}"
            " 를 먼저 실행하세요."
        )
        # 다른 축으로 전환 시도
        alt_dir = ONE_AXIS_DIR if is_two_axis else TWO_AXIS_DIR
        alt_weeks = get_available_weeks(alt_dir)
        if alt_weeks:
            st.info(f"{'1축' if is_two_axis else '2축'} 데이터는 {len(alt_weeks)}주차 존재합니다.")
        st.stop()

    selected_weeks = st.multiselect(
        "분석 주차 선택",
        weeks,
        default=[weeks[0]] if weeks else [],
    )

    if not selected_weeks:
        st.info("주차를 선택하세요.")
        st.stop()

# ── 데이터 로드 ──────────────────────────────────────────────────────

all_dfs = []
for week in selected_weeks:
    file_path = data_dir / f"{week}.json"
    data = load_week_data(str(file_path))
    df = items_to_dataframe(
        data.get("letters", []),
        data.get("posts", []),
        axis="two" if is_two_axis else "one",
    )
    df["주차"] = week
    all_dfs.append(df)

df = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()

if df.empty:
    st.warning("선택한 주차에 데이터가 없습니다.")
    st.stop()

# ── 사이드바: 필터 ────────────────────────────────────────────────────

with st.sidebar:
    st.header("필터")

    masters = sorted(df["마스터"].unique())
    selected_masters = st.multiselect("마스터", masters, default=masters)

    types = st.multiselect("유형", ["편지", "게시글"], default=["편지", "게시글"])

    if is_two_axis:
        selected_topics = st.multiselect("주제", TOPICS, default=TOPICS)
        selected_sentiments = st.multiselect("감성", SENTIMENTS, default=SENTIMENTS)

# 필터 적용
mask = df["마스터"].isin(selected_masters) & df["유형"].isin(types)
if is_two_axis:
    mask = mask & df["주제"].isin(selected_topics) & df["감성"].isin(selected_sentiments)
df_filtered = df[mask]

# ── 탭 구성 ──────────────────────────────────────────────────────────

if is_two_axis:
    tab_overview, tab_master, tab_alerts, tab_explorer = st.tabs(
        ["개요", "마스터 분석", "감성 알림", "콘텐츠 탐색"]
    )
else:
    tab_overview, tab_master, tab_explorer = st.tabs(
        ["개요", "마스터 분석", "콘텐츠 탐색"]
    )

# ═══════════════════════════════════════════════════════════════════════
# 탭 1: 개요
# ═══════════════════════════════════════════════════════════════════════

with tab_overview:
    # 상단 메트릭
    col1, col2, col3 = st.columns(3)
    col1.metric("전체", len(df_filtered))
    col2.metric("편지", len(df_filtered[df_filtered["유형"] == "편지"]))
    col3.metric("게시글", len(df_filtered[df_filtered["유형"] == "게시글"]))

    if is_two_axis:
        # 감성 메트릭
        st.subheader("감성 분포")
        sent_cols = st.columns(3)
        for i, s in enumerate(SENTIMENTS):
            cnt = len(df_filtered[df_filtered["감성"] == s])
            pct = cnt / len(df_filtered) * 100 if len(df_filtered) > 0 else 0
            sent_cols[i].metric(s, f"{cnt}건 ({pct:.1f}%)")

        # Topic × Sentiment 히트맵
        st.subheader("Topic × Sentiment 매트릭스")
        cross = pd.crosstab(df_filtered["주제"], df_filtered["감성"])
        # 순서 맞추기
        for s in SENTIMENTS:
            if s not in cross.columns:
                cross[s] = 0
        cross = cross[SENTIMENTS]
        for t in TOPICS:
            if t not in cross.index:
                cross.loc[t] = 0
        cross = cross.loc[[t for t in TOPICS if t in cross.index]]

        fig_heatmap = px.imshow(
            cross,
            text_auto=True,
            color_continuous_scale="RdYlGn_r",
            labels=dict(x="감성", y="주제", color="건수"),
        )
        fig_heatmap.update_layout(height=350)
        st.plotly_chart(fig_heatmap, use_container_width=True)

        # 주제별 감성 비율 스택 바
        st.subheader("주제별 감성 비율")
        topic_sent = df_filtered.groupby(["주제", "감성"]).size().reset_index(name="건수")
        fig_stack = px.bar(
            topic_sent,
            x="주제",
            y="건수",
            color="감성",
            color_discrete_map=SENTIMENT_COLORS,
            barmode="stack",
        )
        fig_stack.update_layout(height=350)
        st.plotly_chart(fig_stack, use_container_width=True)

    else:
        # 1축: 카테고리 분포
        st.subheader("카테고리 분포")
        cat_counts = df_filtered["카테고리"].value_counts().reset_index()
        cat_counts.columns = ["카테고리", "건수"]
        fig_cat = px.bar(cat_counts, x="카테고리", y="건수", color="카테고리")
        fig_cat.update_layout(height=350)
        st.plotly_chart(fig_cat, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════
# 탭 2: 마스터 분석
# ═══════════════════════════════════════════════════════════════════════

with tab_master:
    st.subheader("마스터별 통계")

    if is_two_axis:
        # 마스터별 감성 분포
        master_sent = df_filtered.groupby(["마스터", "감성"]).size().reset_index(name="건수")

        # 마스터별 총 건수 계산 후 정렬
        master_totals = df_filtered.groupby("마스터").size().reset_index(name="총건수")
        master_order = master_totals.sort_values("총건수", ascending=False)["마스터"].tolist()

        fig_master = px.bar(
            master_sent,
            x="마스터",
            y="건수",
            color="감성",
            color_discrete_map=SENTIMENT_COLORS,
            barmode="stack",
            category_orders={"마스터": master_order},
        )
        fig_master.update_layout(height=500, xaxis_tickangle=-45)
        st.plotly_chart(fig_master, use_container_width=True)

        # 마스터별 부정 비율 테이블
        st.subheader("마스터별 부정 비율")
        master_neg = []
        for master in master_order:
            m_df = df_filtered[df_filtered["마스터"] == master]
            total = len(m_df)
            neg = len(m_df[m_df["감성"] == "부정"])
            pos = len(m_df[m_df["감성"] == "긍정"])
            neu = len(m_df[m_df["감성"] == "중립"])
            master_neg.append({
                "마스터": master,
                "총건수": total,
                "긍정": pos,
                "부정": neg,
                "중립": neu,
                "부정비율(%)": round(neg / total * 100, 1) if total > 0 else 0,
            })
        df_master_neg = pd.DataFrame(master_neg)
        df_master_neg = df_master_neg.sort_values("부정비율(%)", ascending=False)

        st.dataframe(
            df_master_neg.style.background_gradient(
                subset=["부정비율(%)"],
                cmap="Reds",
            ),
            use_container_width=True,
            hide_index=True,
        )

        # 주차별 감성 추이 (여러 주 선택 시)
        if len(selected_weeks) > 1:
            st.subheader("주차별 감성 추이")

            # 마스터 선택
            trend_master = st.selectbox(
                "마스터 선택 (추이 확인)",
                ["전체"] + master_order,
            )

            if trend_master == "전체":
                trend_df = df_filtered
            else:
                trend_df = df_filtered[df_filtered["마스터"] == trend_master]

            week_sent = trend_df.groupby(["주차", "감성"]).size().reset_index(name="건수")
            fig_trend = px.line(
                week_sent,
                x="주차",
                y="건수",
                color="감성",
                color_discrete_map=SENTIMENT_COLORS,
                markers=True,
            )
            fig_trend.update_layout(height=350)
            st.plotly_chart(fig_trend, use_container_width=True)

            # 부정 비율 추이
            week_neg_ratio = []
            for week in sorted(selected_weeks):
                w_df = trend_df[trend_df["주차"] == week]
                total = len(w_df)
                neg = len(w_df[w_df["감성"] == "부정"])
                week_neg_ratio.append({
                    "주차": week,
                    "부정비율(%)": round(neg / total * 100, 1) if total > 0 else 0,
                    "건수": total,
                })
            df_neg_ratio = pd.DataFrame(week_neg_ratio)
            fig_neg = px.bar(
                df_neg_ratio,
                x="주차",
                y="부정비율(%)",
                text="부정비율(%)",
                color_discrete_sequence=["#e74c3c"],
            )
            fig_neg.update_layout(height=300, title="부정 비율 추이")
            st.plotly_chart(fig_neg, use_container_width=True)

    else:
        # 1축: 마스터별 카테고리 분포
        master_cat = df_filtered.groupby(["마스터", "카테고리"]).size().reset_index(name="건수")
        master_totals = df_filtered.groupby("마스터").size().reset_index(name="총건수")
        master_order = master_totals.sort_values("총건수", ascending=False)["마스터"].tolist()

        fig_master_cat = px.bar(
            master_cat,
            x="마스터",
            y="건수",
            color="카테고리",
            barmode="stack",
            category_orders={"마스터": master_order},
        )
        fig_master_cat.update_layout(height=500, xaxis_tickangle=-45)
        st.plotly_chart(fig_master_cat, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════
# 탭 3: 감성 알림 (2축 전용)
# ═══════════════════════════════════════════════════════════════════════

if is_two_axis:
    with tab_alerts:
        if len(selected_weeks) < 2:
            st.info("2주 이상 선택하면 전주 대비 부정 감성 증감을 확인할 수 있습니다.")
        else:
            sorted_weeks = sorted(selected_weeks)
            this_week = sorted_weeks[-1]
            prev_week = sorted_weeks[-2]

            df_this = df_filtered[df_filtered["주차"] == this_week]
            df_prev = df_filtered[df_filtered["주차"] == prev_week]

            st.subheader(f"부정 감성 변화 ({prev_week} → {this_week})")

            alerts = []
            all_masters = set(df_this["마스터"].unique()) | set(df_prev["마스터"].unique())

            for master in all_masters:
                m_this = df_this[df_this["마스터"] == master]
                m_prev = df_prev[df_prev["마스터"] == master]

                total_this = len(m_this)
                total_prev = len(m_prev)
                neg_this = len(m_this[m_this["감성"] == "부정"])
                neg_prev = len(m_prev[m_prev["감성"] == "부정"])

                ratio_this = neg_this / total_this * 100 if total_this > 0 else 0
                ratio_prev = neg_prev / total_prev * 100 if total_prev > 0 else 0
                change_pp = ratio_this - ratio_prev

                alerts.append({
                    "마스터": master,
                    "이번주_총건수": total_this,
                    "전주_총건수": total_prev,
                    "이번주_부정": neg_this,
                    "전주_부정": neg_prev,
                    "이번주_부정비율(%)": round(ratio_this, 1),
                    "전주_부정비율(%)": round(ratio_prev, 1),
                    "변화(%p)": round(change_pp, 1),
                })

            df_alerts = pd.DataFrame(alerts)
            df_alerts = df_alerts.sort_values("변화(%p)", ascending=False)

            # 급증 (10%p 이상 증가 + 5건 이상)
            spikes = df_alerts[
                (df_alerts["변화(%p)"] >= 10) & (df_alerts["이번주_총건수"] >= 5)
            ]
            if not spikes.empty:
                st.error(f"부정 급증: {len(spikes)}명")
                st.dataframe(
                    spikes.style.background_gradient(subset=["변화(%p)"], cmap="Reds"),
                    use_container_width=True,
                    hide_index=True,
                )

                # 급증 마스터의 부정 콘텐츠 샘플
                for _, row in spikes.iterrows():
                    master = row["마스터"]
                    with st.expander(f"{master} — 부정 콘텐츠 샘플"):
                        neg_items = df_this[
                            (df_this["마스터"] == master) & (df_this["감성"] == "부정")
                        ]
                        for _, item in neg_items.head(5).iterrows():
                            content = item.get("내용", "")[:200]
                            topic = item.get("주제", "")
                            st.markdown(f"- [{topic}] {content}")
            else:
                st.success("부정 급증 마스터 없음")

            st.divider()

            # 개선 (10%p 이상 감소 + 전주 5건 이상)
            drops = df_alerts[
                (df_alerts["변화(%p)"] <= -10) & (df_alerts["전주_총건수"] >= 5)
            ]
            if not drops.empty:
                st.success(f"부정 개선: {len(drops)}명")
                st.dataframe(
                    drops.style.background_gradient(subset=["변화(%p)"], cmap="Greens_r"),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info("부정 개선 마스터 없음")

            st.divider()

            # 전체 변화 테이블
            st.subheader("전체 마스터 부정 변화")
            st.dataframe(
                df_alerts.style.background_gradient(subset=["변화(%p)"], cmap="RdYlGn_r"),
                use_container_width=True,
                hide_index=True,
            )

# ═══════════════════════════════════════════════════════════════════════
# 탭 4 (2축) / 탭 3 (1축): 콘텐츠 탐색
# ═══════════════════════════════════════════════════════════════════════

with tab_explorer:
    st.subheader("콘텐츠 탐색기")

    # 검색
    search_query = st.text_input("내용 검색", placeholder="키워드 입력...")

    display_df = df_filtered.copy()

    if search_query:
        mask = display_df["내용"].str.contains(search_query, case=False, na=False)
        if "제목" in display_df.columns:
            mask = mask | display_df["제목"].str.contains(search_query, case=False, na=False)
        display_df = display_df[mask]

    st.caption(f"총 {len(display_df)}건")

    # 정렬
    if is_two_axis:
        sort_col = st.selectbox("정렬 기준", ["날짜", "마스터", "주제", "감성"])
    else:
        sort_col = st.selectbox("정렬 기준", ["날짜", "마스터", "카테고리"])

    display_df = display_df.sort_values(sort_col, ascending=sort_col != "날짜")

    # 표시할 컬럼 선택
    if is_two_axis:
        show_cols = ["주차", "유형", "마스터", "주제", "감성", "내용", "날짜"]
    else:
        show_cols = ["주차", "유형", "마스터", "카테고리", "내용", "날짜"]

    available_cols = [c for c in show_cols if c in display_df.columns]

    # 내용 길이 제한 (테이블 뷰)
    view_df = display_df[available_cols].copy()
    view_df["내용"] = view_df["내용"].str[:200]

    st.dataframe(
        view_df,
        use_container_width=True,
        hide_index=True,
        height=600,
    )

    # CSV 다운로드
    csv = df_filtered.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "CSV 다운로드",
        csv,
        file_name=f"voc_data_{'_'.join(selected_weeks)}.csv",
        mime="text/csv",
    )
