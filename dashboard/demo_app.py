"""VOC Intelligence Demo — CTO 설득용 데모

실행: streamlit run dashboard/demo_app.py
"""
import json
import re
import sys
import time
from pathlib import Path
from collections import Counter, defaultdict

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).parent.parent))

DATA_DIR = Path("data")
TWO_AXIS_DIR = DATA_DIR / "classified_data_two_axis"
CHANNEL_DIR = DATA_DIR / "channel_io" / "golden"

# BigQuery 설정
import os
os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS", str(Path(__file__).parent.parent / "accountKey.json"))
BQ_PROJECT = "us-service-data"
BQ_DATASET = "voc_labelled"


# ── 데이터 로드 ──────────────────────────────────────────────────────

@st.cache_data(ttl=600)
def load_two_axis_data():
    """BigQuery voc_labelled에서 분류 데이터 로드"""
    try:
        from google.cloud import bigquery
        client = bigquery.Client(project=BQ_PROJECT)
        query = f"""
        SELECT
            id, source_type, master_name, user_id, content,
            created_at, topic, sentiment, summary, tags,
            confidence, pipeline_date
        FROM `{BQ_PROJECT}.{BQ_DATASET}.letters_posts`
        ORDER BY pipeline_date DESC, created_at DESC
        LIMIT 10000
        """
        df = client.query(query).to_dataframe()
        items = []
        for _, row in df.iterrows():
            item = {
                "_id": row.get("id", ""),
                "_type": "편지" if row.get("source_type") == "letter" else "게시글",
                "_week": str(row.get("pipeline_date", ""))[:10],
                "masterName": row.get("master_name", ""),
                "message": row.get("content", ""),
                "textBody": row.get("content", ""),
                "body": row.get("content", ""),
                "createdAt": str(row.get("created_at", "")),
                "classification": {
                    "topic": row.get("topic", ""),
                    "sentiment": row.get("sentiment", ""),
                },
                "detail_tags": {
                    "category_tags": list(row.get("tags", [])) if row.get("tags") is not None else [],
                    "free_tags": list(row.get("tags", [])) if row.get("tags") is not None else [],
                    "summary": row.get("summary", ""),
                },
            }
            items.append(item)
        st.sidebar.success(f"BigQuery: {len(items)}건 로드")
        return items
    except Exception as e:
        st.sidebar.warning(f"BigQuery 실패: {e}\n로컬 파일 사용")
        # fallback: 로컬 파일
        items = []
        for f in sorted(TWO_AXIS_DIR.glob("*.json")):
            with open(f, encoding="utf-8") as fh:
                data = json.load(fh)
            week = f.stem
            for item in data.get("letters", []):
                item["_type"] = "편지"
                item["_week"] = week
                items.append(item)
            for item in data.get("posts", []):
                item["_type"] = "게시글"
                item["_week"] = week
                items.append(item)
        return items


@st.cache_data
def load_prev_week_simple():
    """전주 데이터 — 마스터별 감성 통계 (detail_tags 불필요)"""
    path = TWO_AXIS_DIR / "2026-02-02.json"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    stats = {}
    for item in data.get("letters", []) + data.get("posts", []):
        master = re.sub(r"\d+$", "", item.get("masterName", "Unknown")).strip()
        cls = item.get("classification", {})
        if master not in stats:
            stats[master] = {"total": 0, "neg": 0, "pos": 0}
        stats[master]["total"] += 1
        s = cls.get("sentiment", "중립")
        if s == "부정":
            stats[master]["neg"] += 1
        elif s == "긍정":
            stats[master]["pos"] += 1
    for m in stats:
        t = stats[m]["total"]
        stats[m]["neg_pct"] = round(stats[m]["neg"] / t * 100, 1) if t else 0
        stats[m]["pos_pct"] = round(stats[m]["pos"] / t * 100, 1) if t else 0
    total = sum(s["total"] for s in stats.values())
    return {"masters": stats, "total": total, "week": "2026-02-02"}


def infer_master_from_channel_text(text):
    """채널톡 텍스트에서 마스터 이름 추론"""
    master_map = {
        "박두환": ["박두환", "투자동행학교"],
        "서재형": ["서재형", "담임쌤", "담쌤", "센트러스"],
        "미과장": ["미과장"],
        "이정윤": ["이정윤", "이정윤세무사"],
        "돈깡": ["돈깡"],
        "오종태": ["오종태"],
        "김기훈": ["김기훈"],
        "체슬리": ["체슬리", "1등매니저"],
    }
    for master, keywords in master_map.items():
        if any(kw in text for kw in keywords):
            return master
    return None


@st.cache_data
def load_channel_data():
    path = CHANNEL_DIR / "golden_multilabel_270_with_subtags.json"
    if not path.exists():
        path = CHANNEL_DIR / "golden_multilabel_270.json"
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def to_df(items):
    rows = []
    for item in items:
        if "detail_tags" not in item:
            continue
        cls = item.get("classification", {})
        dt = item.get("detail_tags", {})
        master = re.sub(r"\d+$", "", item.get("masterName", "Unknown")).strip()
        content = item.get("message") or item.get("textBody") or item.get("body", "")
        rows.append({
            "유형": item["_type"],
            "주차": item["_week"],
            "마스터": master,
            "클럽": item.get("masterClubName", ""),
            "주제": cls.get("topic", "미분류"),
            "감성": cls.get("sentiment", "중립"),
            "내용": content,
            "summary": dt.get("summary", ""),
            "category_tags": dt.get("category_tags", []),
            "free_tags": dt.get("free_tags", []),
        })
    return pd.DataFrame(rows)


@st.cache_data
def compute_insights(_df, _df_channel):
    df = _df
    df_channel = _df_channel

    pay_kw = ["환불", "결제", "구독해지", "자동결제"]
    letter_pay = df[df["내용"].str.contains("|".join(pay_kw), na=False)]

    ch_pay = 0
    if not df_channel.empty:
        for _, row in df_channel.iterrows():
            if any(s.get("subtag") in ["환불", "구독해지", "자동결제항의", "상품변경"]
                   for s in row.get("subtags", [])):
                ch_pay += 1

    master_health = []
    for master, m_df in df.groupby("마스터"):
        t = len(m_df)
        if t < 10:
            continue
        neg = len(m_df[m_df["감성"] == "부정"])
        pos = len(m_df[m_df["감성"] == "긍정"])
        svc = len(m_df[m_df["주제"] == "서비스 이슈"])
        neg_r = round(neg / t * 100, 1)
        pos_r = round(pos / t * 100, 1)
        health = round(pos_r - neg_r * 1.5 - (svc / t * 100 * 0.5), 1)
        status = "🔴 긴급" if neg_r >= 20 else "🟡 주의" if neg_r >= 12 else "🟢 양호"
        master_health.append({
            "마스터": master, "총건수": t, "긍정%": pos_r,
            "부정%": neg_r, "서비스이슈": svc, "건강지수": health, "상태": status
        })

    subtag_cnt = Counter()
    if not df_channel.empty:
        for _, row in df_channel.iterrows():
            for s in row.get("subtags", []):
                if s.get("subtag") != "기타":
                    subtag_cnt[f"{s['topic']} > {s['subtag']}"] += 1

    # ── 전주 대비 WoW ───────────────────────────────────────────
    prev = load_prev_week_simple()
    prev_masters = prev.get("masters", {})
    prev_total = prev.get("total", 0)
    curr_total = len(_df)

    master_wow = {}
    for row in master_health:
        m = row["마스터"]
        if m in prev_masters:
            prev_neg = prev_masters[m]["neg_pct"]
            delta = round(row["부정%"] - prev_neg, 1)
            trend = "↑" if delta > 1 else "↓" if delta < -1 else "→"
            master_wow[m] = {"prev_neg_pct": prev_neg, "delta": delta, "trend": trend}

    mhdf = pd.DataFrame(master_health).sort_values("건강지수", ascending=False)
    if not mhdf.empty and master_wow:
        mhdf["전주부정%"] = mhdf["마스터"].map(lambda m: master_wow.get(m, {}).get("prev_neg_pct", None))
        mhdf["변화"] = mhdf["마스터"].map(lambda m: master_wow.get(m, {}).get("delta", None))
        mhdf["추세"] = mhdf["마스터"].map(lambda m: master_wow.get(m, {}).get("trend", "?"))

    # ── 채널톡 마스터별 분류 ────────────────────────────────────
    ch_master_cnt = Counter()
    ch_master_topic = defaultdict(Counter)
    for _, row in df_channel.iterrows():
        text = row.get("text", "")
        m = infer_master_from_channel_text(text)
        if m:
            ch_master_cnt[m] += 1
            for s in row.get("subtags", []):
                if s.get("subtag") != "기타":
                    ch_master_topic[m][s["topic"]] += 1

    return {
        "letter_pay": len(letter_pay),
        "ch_pay": ch_pay,
        "cross_pay_total": len(letter_pay) + ch_pay,
        "master_health_df": mhdf,
        "subtag_cnt": subtag_cnt,
        "master_wow": master_wow,
        "wow_total_delta": curr_total - prev_total,
        "prev_total": prev_total,
        "ch_master_cnt": ch_master_cnt,
        "ch_master_topic": ch_master_topic,
    }


def generate_report(df):
    week = df["주차"].iloc[0] if not df.empty else "2026-02-09"
    letters = len(df[df["유형"] == "편지"])
    posts = len(df[df["유형"] == "게시글"])
    total = len(df)

    report = f"# 주간 VOC 리포트 ({week} 주차)\n\n## 0. 핵심 요약\n\n"
    report += "| 구분 | 이번 주 |\n|------|--------|\n"
    report += f"| 전체 편지 건수 | {letters}건 |\n| 전체 게시글 건수 | {posts}건 |\n| 전체 총합 | {total}건 |\n\n"

    top3l = df[df["유형"] == "편지"].groupby("마스터").size().sort_values(ascending=False).head(3)
    top3p = df[df["유형"] == "게시글"].groupby("마스터").size().sort_values(ascending=False).head(3)
    report += f"편지 Top3 ({'+'.join(top3l.index)}): **약 {round(top3l.sum()/letters*100)}%**\n\n"
    report += f"게시글 Top3 ({'+'.join(top3p.index)}): **약 {round(top3p.sum()/posts*100)}%**\n\n---\n\n"
    report += "## 1. 오피셜클럽별 상세\n\n"

    for idx, master in enumerate(df.groupby("마스터").size().sort_values(ascending=False).head(7).index, 1):
        m_df = df[df["마스터"] == master]
        m_l = len(m_df[m_df["유형"] == "편지"])
        m_p = len(m_df[m_df["유형"] == "게시글"])
        m_t = len(m_df)
        neg = len(m_df[m_df["감성"] == "부정"])
        neg_r = round(neg / m_t * 100, 1) if m_t else 0

        cat_cnt = Counter()
        for tags in m_df["category_tags"]:
            for t in tags:
                cat_cnt[t] += 1
        ftags = Counter()
        for tags in m_df[m_df["감성"] == "부정"]["free_tags"]:
            for t in tags:
                ftags[t] += 1

        clubs = [c for c in m_df["클럽"].dropna().unique() if c]
        club_str = f" *({' + '.join(clubs[:3])} 합산)*" if len(clubs) > 1 else ""
        top_neg = [f'"{t}"' for t, _ in ftags.most_common(2)]

        report += f"## {idx}. {master}{club_str}\n\n"
        report += f"> {m_t}건 | 부정 {neg_r}%" + (f" | 주요 부정 키워드: {', '.join(top_neg)}" if top_neg else "") + "\n\n"
        report += f"| 편지 | 게시글 | 총합 |\n|------|--------|------|\n| {m_l}건 | {m_p}건 | {m_t}건 |\n\n"
        report += "■ 주요 내용\n\n"

        for rank, (cat, cnt) in enumerate(cat_cnt.most_common(3), 1):
            cat_items = m_df[m_df["category_tags"].apply(lambda tags: cat in tags)]
            summaries = cat_items["summary"].dropna().tolist()
            quotes = cat_items["내용"].dropna().tolist()
            report += f"**{rank}. {cat} ({cnt}건)**\n\n"
            if summaries:
                report += summaries[0] + "\n\n"
            if quotes:
                q = quotes[0][:200].replace("\n", " ").strip()
                report += f'> *"{q}..."*\n\n'

        svc = m_df[m_df["주제"] == "서비스 이슈"]
        report += "■ 서비스 피드백\n\n"
        if svc.empty:
            report += "- 서비스 관련 피드백 없음\n\n"
        else:
            sc = Counter()
            for tags in svc["category_tags"]:
                for t in tags:
                    sc[t] += 1
            report += f"{', '.join([f'{t}({c}건)' for t, c in sc.most_common(2)])} 관련 {len(svc)}건\n\n"
            q = svc["내용"].iloc[0][:150].replace("\n", " ").strip()
            report += f'> _"{q}"_\n\n'
        report += "---\n\n"
    return report


@st.cache_resource
def _load_vectorstore():
    """ChromaDB + 임베딩 모델 로드 (앱 시작 시 1회)"""
    try:
        import chromadb
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
        ef = SentenceTransformerEmbeddingFunction(model_name="jhgan/ko-sroberta-multitask")
        client = chromadb.PersistentClient(path="chroma_db")
        col = client.get_collection("voc_demo_2026_02_09", embedding_function=ef)
        return col
    except Exception:
        return None


def _retrieve_relevant_docs(query, n=12):
    """질문과 유사한 VOC 문서 검색"""
    col = _load_vectorstore()
    if col is None:
        return []
    try:
        result = col.query(query_texts=[query], n_results=n,
                           include=["documents", "metadatas"])
        docs = []
        for doc, meta in zip(result["documents"][0], result["metadatas"][0]):
            docs.append({"text": doc, "meta": meta})
        return docs
    except Exception:
        return []


def _build_data_context(df, df_channel, insights, query=None):
    """Claude에 넘길 VOC 데이터 컨텍스트 — 질문 있으면 벡터 검색 우선"""
    lines = ["# 2026-02-09 주차 VOC 데이터 요약"]

    # 전체 통계 (항상 포함)
    if not df.empty:
        total = len(df)
        neg = len(df[df["감성"] == "부정"])
        pos = len(df[df["감성"] == "긍정"])
        lines.append(f"\n## 편지·게시글 전체\n총 {total}건 | 긍정 {pos}건({round(pos/total*100,1)}%) | 부정 {neg}건({round(neg/total*100,1)}%)")
        lines.append("\n## 마스터별 현황")
        for master, m_df in df.groupby("마스터"):
            t = len(m_df)
            if t < 10:
                continue
            m_neg = len(m_df[m_df["감성"] == "부정"])
            m_pos = len(m_df[m_df["감성"] == "긍정"])
            cat_cnt = Counter()
            for tags in m_df["category_tags"]:
                for tag in tags:
                    cat_cnt[tag] += 1
            top_tags = ", ".join([f"{tg}({c}건)" for tg, c in cat_cnt.most_common(3)])
            lines.append(f"- {master}: {t}건, 긍정 {round(m_pos/t*100,1)}%, 부정 {round(m_neg/t*100,1)}%, 주요 태그: {top_tags}")

    if not df_channel.empty:
        lines.append(f"\n## 채널톡 CS\n총 {len(df_channel)}건")
        if insights and insights.get("subtag_cnt"):
            lines.append("주요 문의 유형:")
            for subtag, cnt in insights["subtag_cnt"].most_common(8):
                lines.append(f"- {subtag}: {cnt}건")
        # 채널톡 원문 — 질문 키워드 필터
        kw = query.lower() if query else ""
        ch_filtered = df_channel
        topic_map = {"환불": "결제·환불", "결제": "결제·환불", "구독": "구독·멤버십",
                     "해지": "구독·멤버십", "오류": "기술·오류", "수강": "콘텐츠·수강"}
        for k, topic in topic_map.items():
            if k in kw:
                ch_filtered = df_channel[df_channel["topics"].apply(
                    lambda ts: any(topic in t for t in ts))]
                break
        lines.append(f"\n## 채널톡 원문 샘플 ({min(len(ch_filtered),30)}건)")
        for _, row in ch_filtered.head(30).iterrows():
            text = row.get("text", "")[:150].replace("\n", " ").strip()
            topics = ", ".join(row.get("topics", []))
            lines.append(f'- [{topics}] {text}')

    # 질문 관련 편지글 벡터 검색 결과
    if query:
        docs = _retrieve_relevant_docs(query, n=12)
        if docs:
            lines.append(f"\n## 질문과 관련된 편지·게시글 원문 (벡터 검색 Top {len(docs)}건)")
            for d in docs:
                meta = d["meta"]
                master = meta.get("master", "")
                topic = meta.get("topic", "")
                sentiment = meta.get("sentiment", "")
                tags = meta.get("category_tags", "")
                lines.append(f'- [{master} | {topic} | {sentiment}] {d["text"][:150]}' +
                             (f' (태그: {tags})' if tags else ''))

    return "\n".join(lines)


def _generate_agent_response(query, df, df_channel, insights):
    """Claude Code CLI subprocess로 실시간 답변"""
    import subprocess
    context = _build_data_context(df, df_channel, insights, query=query)
    prompt = f"""당신은 VOC 분석 봇입니다. 아래 실제 데이터를 기반으로 질문에 답하세요.
Slack 메시지 형식으로 간결하게 (3-5줄), 실제 수치를 인용하고, 액션 가능한 결론으로 마무리하세요. 이모지 적절히 사용.

{context}

질문: {query}"""
    try:
        result = subprocess.run(
            ["claude", "-p", prompt],
            capture_output=True, text=True, timeout=40,
            cwd=str(Path(__file__).parent.parent),
        )
        return result.stdout.strip() or result.stderr.strip() or "응답 없음"
    except subprocess.TimeoutExpired:
        return "⚠️ 응답 시간 초과 (40초)"
    except FileNotFoundError:
        return "⚠️ claude CLI를 찾을 수 없습니다."
    except Exception as e:
        return f"⚠️ 오류: {e}"


SLACK_QA = {
    "이번 주 위험 신호 있는 마스터 있어?": """\
🚨 **위험 신호 — 2026-02-09 주차**

**서재형 클럽** 부정 21.4% (308건 중 66건)
→ 보유 종목 급락 직후 구독자 불안 급증. 마스터 대응 없으면 구독 취소 전환 위험.
→ **사업팀 즉시:** 마스터에게 센트러스 입장 콘텐츠 48h 이내 요청

실제 목소리:
> _"막판에 센트러스 폭락으로 마음이 편하지 않아요. 담쌤 명절 편하게 보내셨으면..."_

**미과장 클럽** 부정 14.6% (1,382건 중 202건)
→ 포트폴리오 하락 + 소통 부족 이중 불만. 커뮤니티 분열 징후 동반.
→ **사업팀:** 하락장 대응 전략 콘텐츠 요청. **운영팀:** 커뮤니티 가이드라인 공지""",

    "서재형 클럽 부정 여론 원인이 뭐야?": """\
📊 **서재형 클럽 부정 분석 — 66건**

**공통 원인:** 보유 종목 급락 → 마스터 분석이 없어 답답함

🔴 **센트러스 에너지 실적 부진 (이번 주 최다)**
> _"책임감 강하신 담임 쌤께 오랜만에 글 올립니다. 매번 하는 손절이지만 손실액이 클 때 마음이 아픔은 어쩔 수 없네요. 경험상 버티다가..."_

🔴 **팔란티어 PER 해석 혼란**
Trailing/Forward 기준이 강의마다 달라 투자 판단 어려움.

🟡 **명절 연휴 중 마스터 부재 불안**
> _"익절도 하고 손절도 하고 배워나가는 거지 너무 걱정 마세요. 설 명절 근심걱정 내려놓으시고..."_

**결론:** 종목 급락 시 마스터 빠른 대응이 부정 확산 차단 핵심
→ **SLA 제안:** 보유 종목 -10% 이상 시 48h 이내 대응 콘텐츠""",

    "채널톡 반복 문의 TOP5 알려줘": """\
📋 **채널톡 CS 반복 문의 (270건)**

| 순위 | 유형 | 건수 | 근본 원인 |
|------|------|------|-----------|
| 1 | 환불 요청 | 54건 | 자동결제 인지 못함 |
| 2 | 구독 해지 | 40건 | 해지 경로 불명확 |
| 3 | 환불(구독) | 25건 | 해지≠환불 혼동 |
| 4 | 상품 변경 | 21건 | 셀프 변경 불가 |
| 5 | 수강 방법 | 16건 | 온보딩 없음 |

**환불+해지+변경 = 115건(43%) → 공통 원인: 셀프서비스 부재**
→ **제품팀:** 마이페이지 1클릭 해지/환불 버튼 = CS 40% 감소
→ **제품팀:** 첫 로그인 온보딩 = 수강방법 문의 90% 감소""",

    "이번 주 서비스 피드백 요약해줘": """\
🛠 **서비스 피드백 — 2026-02-09 주차**

편지·게시글 **104건** + 채널톡 **270건** = 통합 **374건**

**🔴 즉시 (제품팀):**
콘텐츠 접근 오류 29건 — 음성 파일 누락, 영상 접근 불가 반복
> _"2/11 주간 라이브 녹화본 업로드 2/13일로 공지됐는데 2/15일인데 아직 안 올라왔네요"_

**🟡 이번 주 안에 (제품팀):**
결제/환불 양 채널 교차 감지 — 편지글 50건 + 채널톡 79건 = **129건 동일 이슈**
→ 단순 CS가 아닌 UX 구조 문제

**💡 인사이트:**
미과장 클럽이 서비스 이슈 104건 중 53건(51%) 집중
→ 가장 큰 클럽이 이슈도 집중 — 모니터링 1순위""",

    "미과장 커뮤니티 이번 주 분위기 알려줘": """\
📌 **미과장 커뮤니티 — 2026-02-09 주차**

총 **1,382건** (전체의 54%) | 부정 14.6% 🟡 주의

**부정 여론의 본질: 투자 손실 불안 (서비스 불만 아님)**
> _"소통이 안 되네요. 제가 질문을 두 번이나 했는데 그 답을 듣지 못했어요. 매일 내용은 바뀌지만 큰 흐름은 같은 브리핑을..."_

> _"투자 잘 모르고 시작해서 자금이 작년 12월 기준 다 쓰고 기다리는 입장입니다. 오늘 보니까 -12% 달리고 있네요."_

**🟡 커뮤니티 분열 징후:**
> _"요 며칠간 커뮤니티가 소수 멤버의 싸움터가 되어버린 것 같습니다"_

**서비스 이슈 53건 집중 — 제품팀 확인 필요**

→ **사업팀:** 마스터에게 하락장 대응 전략 콘텐츠 요청
→ **운영팀:** 커뮤니티 가이드라인 리마인드 공지""",
}


# ══════════════════════════════════════════════════════════════════════
# 페이지 설정
# ══════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="VOC Intelligence Demo",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
.signal-card-red {
    border-left: 5px solid #e74c3c;
    background: #2a0f0f;
    padding: 20px 24px;
    border-radius: 0 10px 10px 0;
    margin: 10px 0;
    color: #f0d0d0;
}
.signal-card-yellow {
    border-left: 5px solid #f39c12;
    background: #2a2000;
    padding: 20px 24px;
    border-radius: 0 10px 10px 0;
    margin: 10px 0;
    color: #f0e0c0;
}
.signal-card-green {
    border-left: 5px solid #2ecc71;
    background: #0f2a18;
    padding: 20px 24px;
    border-radius: 0 10px 10px 0;
    margin: 10px 0;
    color: #d0f0d8;
}
.signal-card-red b, .signal-card-yellow b, .signal-card-green b { color: #ffffff; }
.signal-card-red small, .signal-card-yellow small, .signal-card-green small { color: #ddd; }
.quote-block {
    background: #1a1a1a;
    border-left: 3px solid #777;
    padding: 12px 16px;
    margin: 10px 0;
    border-radius: 0 6px 6px 0;
    font-style: italic;
    color: #e8e8e8;
    font-size: 0.92rem;
}
.action-row {
    background: #0f1e35;
    border: 1px solid #3a5a8a;
    border-radius: 6px;
    padding: 10px 14px;
    margin: 6px 0;
    font-size: 0.9rem;
    color: #d0e0f0;
}
.owner-tag {
    display: inline-block;
    background: #1a3a5c;
    color: #7ab3e0;
    border-radius: 4px;
    padding: 1px 8px;
    font-size: 0.8rem;
    font-weight: bold;
    margin-right: 6px;
}
</style>
""", unsafe_allow_html=True)


# ── 데이터 로드 ──────────────────────────────────────────────────────
all_items = load_two_axis_data()
df_all = to_df(all_items)
df = df_all[df_all["주차"] == "2026-02-09"].copy() if not df_all.empty else df_all
channel_items = load_channel_data()
ch_rows = [{"route": x.get("route", ""), "topics": x.get("topics", []),
             "subtags": x.get("subtags", []), "text": x.get("text", "")}
           for x in channel_items]
df_channel = pd.DataFrame(ch_rows) if ch_rows else pd.DataFrame()
insights = compute_insights(df, df_channel) if not df.empty else {}


# ── 탭 ───────────────────────────────────────────────────────────────
tabs = st.tabs([
    "🏠 이번 주 브리핑",
    "📄 리포트 자동생성",
    "💼 사업팀",
    "🔧 제품/개발팀",
    "🔍 원문 탐색",
    "💬 슬랙봇 데모",
])


# ══════════════════════════════════════════════════════════════════════
# 탭1: 이번 주 브리핑
# ══════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.title("이번 주 VOC 브리핑")
    st.caption("2026-02-09 주차 | 편지·게시글 2,542건 + 채널톡 CS 270건 자동 분석")

    wow_delta = insights.get("wow_total_delta", 0) if insights else 0
    prev_total = insights.get("prev_total", 0) if insights else 0
    wow_pct = f"{wow_delta:+,}건 ({round(wow_delta/prev_total*100,1):+.1f}%)" if prev_total else ""

    col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
    col1.metric("분석 건수", f"{len(df)+len(df_channel):,}건", wow_pct)
    col2.metric("리포트 생성", "< 2분", "기존 수동 8시간 → 98% 단축")
    col3.metric("데이터 소스", "2개 채널", "편지·게시글 + 채널톡 CS")
    with col4:
        if not df.empty:
            try:
                from io import BytesIO
                buf = BytesIO()
                df.drop(columns=["category_tags", "free_tags"], errors="ignore").to_excel(buf, index=False, engine="openpyxl")
                st.download_button("📥 엑셀", buf.getvalue(),
                                   file_name=f"voc_전체_{df['주차'].iloc[0] if not df.empty else 'data'}.xlsx",
                                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                   use_container_width=True)
            except Exception:
                pass

    st.markdown("---")
    st.subheader("📌 이번 주 3가지 신호")
    st.caption("무신사·아임웹 사례처럼 — 숫자 → 실제 목소리 → 액션 (오너십 명시)")

    if not df.empty:
        # ── 신호 1: 서재형 ──
        sj = df[df["마스터"] == "서재형"]
        sj_neg = sj[sj["감성"] == "부정"]
        sj_neg_r = round(len(sj_neg) / len(sj) * 100, 1) if len(sj) else 0
        sj_quote = sj_neg["내용"].dropna().tolist()
        sj_q = sj_quote[0][:120].replace("\n", " ").strip() + "..." if sj_quote else ""

        master_wow = insights.get("master_wow", {}) if insights else {}
        sj_wow = master_wow.get("서재형", {})
        sj_delta_str = f"전주 {sj_wow['prev_neg_pct']}% → 이번 주 {sj_neg_r}% ({sj_wow['delta']:+.1f}%p {sj_wow['trend']})" if sj_wow else ""

        st.markdown(f"""
<div class="signal-card-red">
<b>🔴 긴급 — 서재형 클럽 부정 여론 {sj_neg_r}% ({len(sj_neg)}건)</b>
{"<br><small style='color:#e87070'>" + sj_delta_str + "</small>" if sj_delta_str else ""}
<br><small>센트러스 에너지 급락 직후 구독자 불안 급증. 방치 시 구독 취소 전환 위험.</small>
<div class="quote-block">"{sj_q}"</div>
<div class="action-row"><span class="owner-tag">사업팀</span> 마스터에게 센트러스 입장 콘텐츠 48h 이내 요청</div>
</div>
""", unsafe_allow_html=True)

        # ── 신호 2: 교차 이슈 ──
        cross_total = insights.get("cross_pay_total", 0)
        letter_pay = insights.get("letter_pay", 0)
        ch_pay = insights.get("ch_pay", 0)

        st.markdown(f"""
<div class="signal-card-yellow">
<b>🟡 주의 — 결제·환불 이슈 두 채널에서 동시 감지 {cross_total}건</b><br>
<small>편지·게시글 {letter_pay}건 + 채널톡 CS {ch_pay}건 — 동일 이슈가 두 채널에서 반복 = 서비스 구조 문제</small>
<div class="quote-block">"6개월 서비스 신청 후 재계약 안 했는데 오늘 50만원이 결제됐네요. 환불하고 싶은데 전화도 안 되고 황당하네요"</div>
<div class="action-row"><span class="owner-tag">제품팀</span> 마이페이지 셀프 해지/환불 버튼 추가 → CS 40% 감소 예상</div>
<div class="action-row"><span class="owner-tag">운영팀</span> 자동결제 갱신 전 이메일 사전 고지 강화</div>
</div>
""", unsafe_allow_html=True)

        # ── 신호 3: 긍정 인사이트 ──
        cnt_pos = df[(df["주제"] == "콘텐츠 반응") & (df["감성"] == "긍정")]
        cnt_total = df[df["주제"] == "콘텐츠 반응"]
        pos_r = round(len(cnt_pos) / len(cnt_total) * 100) if len(cnt_total) else 0
        pos_quotes = cnt_pos["내용"].dropna().tolist()
        pos_q = pos_quotes[0][:120].replace("\n", " ").strip() + "..." if pos_quotes else ""

        st.markdown(f"""
<div class="signal-card-green">
<b>🟢 양호 — 콘텐츠 품질 긍정 반응 {pos_r}% ({len(cnt_pos)}건)</b><br>
<small>부정 여론의 원인은 콘텐츠 품질이 아닌 <b>투자 손실 불안과 서비스 UX</b>. 콘텐츠 개선보다 대응 속도 개선이 우선.</small>
<div class="quote-block">"{pos_q}"</div>
<div class="action-row"><span class="owner-tag">사업팀</span> 콘텐츠 개선보다 하락장 대응 가이드 보강에 집중 권고</div>
</div>
""", unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("주제 × 감성 — 어디서 불만이 터지는가")
    st.caption("아임웹 사례처럼 이상 패턴을 한눈에")

    if not df.empty:
        pivot = df.groupby(["주제", "감성"]).size().unstack(fill_value=0)
        for c in ["긍정", "부정", "중립"]:
            if c not in pivot.columns:
                pivot[c] = 0
        pivot = pivot[["긍정", "중립", "부정"]].reset_index()
        pivot["부정률(%)"] = (pivot["부정"] / (pivot["긍정"] + pivot["중립"] + pivot["부정"]) * 100).round(1)

        col1, col2 = st.columns([3, 2])
        with col1:
            fig = go.Figure()
            fig.add_trace(go.Bar(name="긍정", x=pivot["주제"], y=pivot["긍정"], marker_color="#2ecc71"))
            fig.add_trace(go.Bar(name="중립", x=pivot["주제"], y=pivot["중립"], marker_color="#555"))
            fig.add_trace(go.Bar(name="부정", x=pivot["주제"], y=pivot["부정"], marker_color="#e74c3c"))
            fig.update_layout(
                barmode="stack",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                height=300, legend=dict(orientation="h", y=1.12),
                margin=dict(l=10, r=10, t=30, b=10),
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("**읽는 법:**")
            for _, row in pivot.iterrows():
                t = row["주제"]
                nr = row["부정률(%)"]
                icon = "🔴" if nr >= 15 else "🟡" if nr >= 8 else "🟢"
                total = row["긍정"] + row["중립"] + row["부정"]
                st.markdown(f"{icon} **{t}** — 부정 {nr}% ({int(row['부정'])}건/{total}건)")
            st.markdown("")
            st.info("💡 투자 이야기 부정 17% = 시장 하락 때 커뮤니티 불안 직결\n\n콘텐츠 반응 부정 6% = 콘텐츠 자체는 우수")


# ══════════════════════════════════════════════════════════════════════
# 탭2: 리포트 자동생성
# ══════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.title("주간 리포트 자동생성")
    st.caption("분류된 VOC 데이터 기반으로 마크다운 리포트를 자동 생성합니다.")

    col1, col2 = st.columns([3, 1])
    with col1:
        st.info("📅 **2026-02-09 주차** | 편지 1,233건 + 게시글 1,309건 = 총 2,542건")
    with col2:
        gen_btn = st.button("📄 리포트 생성", type="primary", use_container_width=True)

    st.markdown("> **기존:** 담당자가 raw 데이터 수동 분류 → **4~8시간** | **자동화:** 분류 + 생성 → **2분 이내**")

    if gen_btn:
        with st.spinner("2,542건 분류 데이터 기반 리포트 생성 중..."):
            time.sleep(1.5)
            report_md = generate_report(df)
        st.success("✅ 생성 완료")
        st.markdown("---")
        st.markdown(report_md)
        st.download_button("📥 마크다운 다운로드", report_md,
                           file_name="weekly_report_2026-02-09.md", mime="text/markdown")


# ══════════════════════════════════════════════════════════════════════
# 탭3: 사업팀
# ══════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.title("사업팀 — 콘텐츠·커뮤니티 인사이트")

    if df.empty or not insights:
        st.warning("데이터 없음")
    else:
        mhdf = insights["master_health_df"]

        # 마스터 건강 지수
        st.subheader("마스터별 커뮤니티 건강 지수")
        st.caption("긍정 활성도 − 부정 비율 × 1.5 − 서비스 이슈 가중치. 당근마켓식 '지표가 팀 전체 지식' 원칙.")

        col1, col2 = st.columns([3, 2])
        with col1:
            if not mhdf.empty:
                fig = px.bar(
                    mhdf.sort_values("건강지수"),
                    x="건강지수", y="마스터", orientation="h",
                    color="건강지수",
                    color_continuous_scale=["#e74c3c", "#f39c12", "#2ecc71"],
                    text="건강지수",
                )
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    height=420, coloraxis_showscale=False,
                    margin=dict(l=10, r=10, t=10, b=10),
                )
                fig.add_vline(x=0, line_dash="dash", line_color="#555")
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("**🔴 즉시 대응:**")
            red = mhdf[mhdf["상태"] == "🔴 긴급"]
            for _, r in red.iterrows():
                wow = insights.get("master_wow", {}).get(r["마스터"], {})
                delta_str = f" ({wow['delta']:+.1f}%p {wow['trend']})" if wow else ""
                st.markdown(f"**{r['마스터']}** — 부정 {r['부정%']}%{delta_str}, 건강 {r['건강지수']:+.1f}")
            st.markdown("**🟡 모니터링:**")
            yel = mhdf[mhdf["상태"] == "🟡 주의"]
            for _, r in yel.iterrows():
                wow = insights.get("master_wow", {}).get(r["마스터"], {})
                delta_str = f" ({wow['delta']:+.1f}%p {wow['trend']})" if wow else ""
                st.markdown(f"**{r['마스터']}** — 부정 {r['부정%']}%{delta_str}")
            st.markdown("**🟢 양호:**")
            grn = mhdf[mhdf["상태"] == "🟢 양호"]
            for _, r in grn.iterrows():
                st.markdown(f"**{r['마스터']}** — 긍정 {r['긍정%']}%")

        st.markdown("---")

        # 마스터 드릴다운
        st.subheader("마스터 상세 — 실제 목소리")
        masters = df.groupby("마스터").size().sort_values(ascending=False).index.tolist()
        selected = st.selectbox("마스터 선택", masters)
        m_df = df[df["마스터"] == selected]
        total = len(m_df)
        neg = len(m_df[m_df["감성"] == "부정"])
        pos = len(m_df[m_df["감성"] == "긍정"])
        svc_cnt = len(m_df[m_df["주제"] == "서비스 이슈"])

        m_wow = insights.get("master_wow", {}).get(selected, {}) if insights else {}
        neg_pct = round(neg/total*100,1) if total else 0
        delta_str = f"{m_wow['delta']:+.1f}%p {m_wow['trend']} 전주 {m_wow['prev_neg_pct']}%" if m_wow else None

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("총 건수", f"{total}건")
        col2.metric("긍정 반응", f"{pos}건", f"{round(pos/total*100,1)}%")
        col3.metric("부정 여론", f"{neg}건 ({neg_pct}%)", delta_str, delta_color="inverse")
        col4.metric("서비스 이슈", f"{svc_cnt}건")

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**이번 주 가장 많이 이야기한 것:**")
            cat_cnt = Counter()
            for tags in m_df["category_tags"]:
                for t in tags:
                    cat_cnt[t] += 1
            for cat, cnt in cat_cnt.most_common(5):
                pct = round(cnt / total * 100, 1)
                st.progress(min(pct / 30, 1.0), text=f"{cat} — {cnt}건 ({pct}%)")

        with col_b:
            st.markdown("**부정 여론 키워드 (실제 원인):**")
            neg_df = m_df[m_df["감성"] == "부정"]
            ftags = Counter()
            for tags in neg_df["free_tags"]:
                for t in tags:
                    ftags[t] += 1
            if ftags:
                for tag, cnt in ftags.most_common(5):
                    st.markdown(f"🔴 **{tag}** — {cnt}건")
            else:
                st.success("이번 주 부정 키워드 없음")

        # 실제 목소리 (expander 밖에 바로 노출 — 무신사 원칙)
        if not neg_df.empty:
            st.markdown("**📢 대응 필요 — 실제 목소리:**")
            for _, row in neg_df.head(2).iterrows():
                quote = row["내용"][:160].replace("\n", " ").strip()
                summary = row["summary"][:60] if row["summary"] else ""
                st.markdown(f"""
<div class="quote-block">
<small style="color:#888">[{row['유형']} | {summary}]</small><br>
"{quote}..."
</div>
""", unsafe_allow_html=True)

        svc_df = m_df[m_df["주제"] == "서비스 이슈"]
        if not svc_df.empty:
            st.markdown("**⚠️ 서비스·운영 피드백:**")
            for _, row in svc_df.head(3).iterrows():
                tags_str = " · ".join(row["category_tags"])
                quote = row["내용"][:160].replace("\n", " ").strip()
                st.markdown(f"""
<div class="quote-block">
<small style="color:#f39c12">[{tags_str}]</small><br>
"{quote}..."
</div>
""", unsafe_allow_html=True)

        st.markdown("---")

        # 채널톡 × 편지글 마스터별 교차 분석
        ch_master_cnt = insights.get("ch_master_cnt", Counter()) if insights else Counter()
        ch_master_topic = insights.get("ch_master_topic", {}) if insights else {}
        if ch_master_cnt:
            st.subheader("채널톡 CS × 편지글 — 마스터별 교차")
            st.caption(f"채널톡 270건 중 {sum(ch_master_cnt.values())}건({round(sum(ch_master_cnt.values())/270*100)}%)에서 마스터 텍스트 추론 성공")

            cross_rows = []
            for m, ch_cnt in ch_master_cnt.most_common():
                letter_cnt = len(df[df["마스터"] == m]) if not df.empty else 0
                letter_neg = len(df[(df["마스터"] == m) & (df["감성"] == "부정")]) if not df.empty else 0
                top_topics = ", ".join([f"{t}({c}건)" for t, c in ch_master_topic.get(m, Counter()).most_common(2)])
                cross_rows.append({
                    "마스터": m,
                    "편지·게시글": letter_cnt,
                    "채널톡 CS": ch_cnt,
                    "편지 부정": f"{letter_neg}건 ({round(letter_neg/letter_cnt*100,1) if letter_cnt else 0}%)",
                    "CS 주요 유형": top_topics or "—",
                })
            cross_df = pd.DataFrame(cross_rows)
            st.dataframe(cross_df, use_container_width=True, hide_index=True)
            st.caption("💡 편지 부정 높음 + CS 건수 높음 = 이탈 위험 마스터. 두 지표 모두 높은 마스터 최우선 대응.")

        st.markdown("---")

        # 이번 주 전체 화제 클러스터
        st.subheader("이번 주 전체 커뮤니티 화제 클러스터")
        st.caption("카카오뱅크식 — 데이터에 스토리 입히기. 크기 = 언급 건수.")
        all_cats = Counter()
        for tags in df["category_tags"]:
            for t in tags:
                all_cats[t] += 1
        cat_df = pd.DataFrame(all_cats.most_common(15), columns=["태그", "건수"])
        fig = px.treemap(cat_df, path=["태그"], values="건수",
                         color="건수", color_continuous_scale=["#1a1a2e", "#3498db"])
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=360, margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════
# 탭4: 제품/개발팀
# ══════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.title("제품/개발팀 — 자동 생성 개선 백로그")
    st.caption("편지·게시글 서비스 피드백 + 채널톡 CS 패턴 → 제품 개선 과제 자동 도출")

    svc_total = len(df[df["주제"] == "서비스 이슈"]) if not df.empty else 0
    ch_total = len(df_channel)
    cross = insights.get("cross_pay_total", 0) if insights else 0

    col1, col2, col3 = st.columns(3)
    col1.metric("편지글 서비스 피드백", f"{svc_total}건")
    col2.metric("채널톡 CS", f"{ch_total}건")
    col3.metric("교차 감지 (결제·환불)", f"{cross}건", "🔴 두 채널 동시 감지 = 구조 문제")

    st.markdown("---")
    st.subheader("📋 자동 생성 개선 백로그")

    backlog = [
        {
            "심각도": "🔴 Critical",
            "이슈": "결제/해지 셀프서비스 부재",
            "근거": f"채널톡 환불+해지+변경 115건(43%) + 편지글 {insights.get('letter_pay',0)}건 = {cross}건 교차 확인",
            "실제목소리": "\"6개월 서비스 신청 후 재계약 안 했는데 오늘 50만원이 결제됐네요. 황당하네요\"",
            "제안": "마이페이지 1클릭 해지/환불 신청",
            "owner": "제품팀",
            "기대효과": "CS 주당 ~40건 감소 / CS팀 ~8h 절감",
        },
        {
            "심각도": "🔴 Critical",
            "이슈": "콘텐츠 접근 오류 반복",
            "근거": "편지글 서비스 이슈 중 '콘텐츠 접근 문제' 29건 — 음성 파일 누락, 녹화본 미업로드 반복",
            "실제목소리": "\"2/11 주간 라이브 녹화본 업로드 2/13일로 공지됐는데 2/15일인데 아직 안 올라왔네요\"",
            "제안": "콘텐츠 업로드 후 자동 접근 검증 스크립트",
            "owner": "제품팀",
            "기대효과": "오류 발생 즉시 감지 → 대응 시간 단축",
        },
        {
            "심각도": "🟡 High",
            "이슈": "신규 구독자 온보딩 부재",
            "근거": "채널톡 수강방법 16건 + 편지글 온보딩/접근성 11건 — 신규 가입 직후 집중 발생",
            "실제목소리": "\"신규수강생입니다. 주간 매매전략은 어디서 볼 수 있나요?\"",
            "제안": "첫 로그인 시 마스터별 콘텐츠 위치 안내 팝업",
            "owner": "제품팀",
            "기대효과": "수강방법 CS 90% 감소 예상",
        },
        {
            "심각도": "🟡 High",
            "이슈": "앱 오류 집중 신고",
            "근거": "편지글 '앱/기능 오류' 19건 — iOS 관련 언급 다수",
            "실제목소리": "\"앱이 계속 튕겨요 아이폰 최신버전인데\"",
            "제안": "iOS 크래시 리포팅 강화 + 재현 경로 파악",
            "owner": "제품팀",
            "기대효과": "패치 우선순위 결정 가속",
        },
        {
            "심각도": "⚪ Medium",
            "이슈": "콘텐츠 일정 공지 불명확",
            "근거": "라이브 시작 시간, 자료 업로드 일정 문의 다수 마스터에서 반복",
            "실제목소리": "\"라방 며칠 몇시부터인지 공지가 제대로 안 되어 찾아 들어가기 어려운 것 같습니다\"",
            "제안": "마스터 콘텐츠 캘린더 UI — 예정 콘텐츠 스케줄 표시",
            "owner": "제품팀·운영팀",
            "기대효과": "일정 문의 CS 감소",
        },
    ]

    for item in backlog:
        with st.expander(f"{item['심각도']} — {item['이슈']}"):
            st.markdown(f"**근거:** {item['근거']}")
            st.markdown(f"""
<div class="quote-block">💬 실제 목소리: {item['실제목소리']}</div>
""", unsafe_allow_html=True)
            st.markdown(f"""
<div class="action-row"><span class="owner-tag">{item['owner']}</span> {item['제안']}</div>
""", unsafe_allow_html=True)
            st.success(f"기대효과: {item['기대효과']}")

    st.markdown("---")
    col_l, col_r = st.columns(2)
    with col_l:
        st.subheader("채널톡 반복 문의 분포")
        if insights and insights.get("subtag_cnt"):
            top = insights["subtag_cnt"].most_common(12)
            st_df = pd.DataFrame(top, columns=["유형", "건수"])
            fig = px.bar(st_df, x="건수", y="유형", orientation="h",
                         color_discrete_sequence=["#e74c3c"])
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                               height=380, yaxis=dict(autorange="reversed"),
                               margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.subheader("편지글 서비스 이슈 분포")
        if not df.empty:
            svc_df = df[df["주제"] == "서비스 이슈"]
            cat_cnt = Counter()
            for tags in svc_df["category_tags"]:
                for t in tags:
                    cat_cnt[t] += 1
            sc_df = pd.DataFrame(cat_cnt.most_common(8), columns=["카테고리", "건수"])
            fig2 = px.bar(sc_df, x="건수", y="카테고리", orientation="h",
                          color_discrete_sequence=["#e67e22"])
            fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                height=380, yaxis=dict(autorange="reversed"),
                                margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.subheader("교차 채널 이슈 — 두 채널에서 동시 감지 시 심각도 UP")
    st.markdown("""
| 이슈 | 편지·게시글 | 채널톡 | 합계 | 심각도 |
|------|------------|--------|------|--------|
| 결제/환불/구독해지 | ~50건 | 79건 | **~129건** | 🔴 구조 문제 |
| 콘텐츠 접근 오류 | 29건 | — | 29건 | 🟡 기술 이슈 |
| 온보딩/수강방법 | 11건 | 16건 | **27건** | 🟡 UX 이슈 |

> 💡 동일 이슈가 두 채널에서 동시 감지 = 개인적 불만이 아닌 **구조적 문제**
""")

    if not df_channel.empty:
        route_cnt = df_channel["route"].value_counts()
        route_labels = {
            "manager_resolved": "상담사 처리 🧑‍💼",
            "bot_resolved": "봇 자동 처리 🤖",
            "opened": "열림 상태",
        }
        cols = st.columns(len(route_cnt))
        for i, (route, cnt) in enumerate(route_cnt.items()):
            cols[i].metric(route_labels.get(route, route), f"{cnt}건",
                           f"{round(cnt/len(df_channel)*100)}%")

    st.markdown("---")
    st.subheader("채널톡 다중 주제 분석 — 하나의 문의에 여러 주제가 포함된 경우")
    st.caption("다중 주제가 반복되면 서비스 구조 문제. 유저가 한 번에 해결하지 못해서 여러 주제를 말하는 것.")

    if not df_channel.empty:
        # 주제 개수 분포
        topic_count_dist = Counter()
        combo_counts = Counter()
        for _, row in df_channel.iterrows():
            topics = row.get("topics", [])
            n = len(topics) if isinstance(topics, list) else 0
            topic_count_dist[n] += 1
            if n >= 2:
                combo_counts[" + ".join(sorted(topics))] += 1

        col1, col2, col3 = st.columns(3)
        single = topic_count_dist.get(1, 0)
        multi = sum(v for k, v in topic_count_dist.items() if k >= 2)
        col1.metric("단일 주제", f"{single}건", f"{round(single/len(df_channel)*100)}%")
        col2.metric("다중 주제", f"{multi}건", f"{round(multi/len(df_channel)*100)}%")
        col3.metric("가장 많은 조합", combo_counts.most_common(1)[0][0] if combo_counts else "—",
                    f"{combo_counts.most_common(1)[0][1]}건" if combo_counts else "")

        if combo_counts:
            st.markdown("**다중 주제 조합 분포:**")
            combo_rows = []
            for combo, cnt in combo_counts.most_common(8):
                pct = round(cnt / multi * 100, 1) if multi else 0
                combo_rows.append({"주제 조합": combo, "건수": cnt, "다중 주제 중 비율": f"{pct}%"})
            st.dataframe(pd.DataFrame(combo_rows), use_container_width=True, hide_index=True)

            top_combo = combo_counts.most_common(1)[0]
            if "결제" in top_combo[0] and "구독" in top_combo[0]:
                st.markdown(f"""
<div class="signal-card-yellow">
<b>🟡 구조 문제 — "{top_combo[0]}" {top_combo[1]}건 ({round(top_combo[1]/multi*100)}%)</b><br>
<small>환불과 해지를 유저가 구분하지 못함 → 한 대화에서 두 가지를 모두 요청. 셀프서비스 UX로 해결 가능.</small>
<div class="action-row"><span class="owner-tag">제품팀</span> 해지 시 환불 옵션 동시 제공 = 다중 문의 68% 감소 기대</div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# 탭5: 원문 탐색
# ══════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.title("원문 탐색")
    st.caption("분류 결과를 실제 원문으로 검증합니다.")

    if df.empty:
        st.warning("데이터 없음")
    else:
        col1, col2, col3 = st.columns(3)
        f_master = col1.selectbox("마스터", ["전체"] + df["마스터"].unique().tolist())
        f_topic = col2.selectbox("주제", ["전체"] + sorted(df["주제"].dropna().unique().tolist()))
        f_sent = col3.selectbox("감성", ["전체", "긍정", "부정", "중립"])
        keyword = st.text_input("키워드 검색")

        filtered = df.copy()
        if f_master != "전체":
            filtered = filtered[filtered["마스터"] == f_master]
        if f_topic != "전체":
            filtered = filtered[filtered["주제"] == f_topic]
        if f_sent != "전체":
            filtered = filtered[filtered["감성"] == f_sent]
        if keyword:
            mask = (filtered["내용"].str.contains(keyword, case=False, na=False) |
                    filtered["summary"].str.contains(keyword, case=False, na=False))
            filtered = filtered[mask]

        st.caption(f"검색 결과: {len(filtered)}건")
        display = filtered[["유형", "마스터", "주제", "감성", "summary", "내용"]].copy()
        display["내용"] = display["내용"].str[:200]
        st.dataframe(display, use_container_width=True, hide_index=True, height=450)
        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            csv = filtered.drop(columns=["category_tags", "free_tags"]).to_csv(index=False).encode("utf-8-sig")
            st.download_button("📥 CSV 다운로드", csv, file_name="voc_filtered.csv", mime="text/csv")
        with col_dl2:
            try:
                from io import BytesIO
                buf = BytesIO()
                export_df = filtered.drop(columns=["category_tags", "free_tags"]).copy()
                export_df.to_excel(buf, index=False, engine="openpyxl")
                st.download_button("📥 엑셀 다운로드", buf.getvalue(),
                                   file_name="voc_filtered.xlsx",
                                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            except Exception:
                st.caption("엑셀 다운로드는 openpyxl 필요")


# ══════════════════════════════════════════════════════════════════════
# 탭6: 슬랙봇 데모
# ══════════════════════════════════════════════════════════════════════
with tabs[5]:
    st.title("💬 슬랙봇 데모")
    st.caption("분류 데이터 기반으로 Slack에서 인사이트를 즉시 조회하는 시나리오 데모")
    st.info("예시 버튼을 클릭하거나 직접 질문을 입력하세요.")

    questions = list(SLACK_QA.keys())
    btn_cols = st.columns(3)
    selected_q = None
    for i, q in enumerate(questions):
        if btn_cols[i % 3].button(q, key=f"q_{i}", use_container_width=True):
            selected_q = q

    st.markdown("---")

    if "slack_history" not in st.session_state:
        st.session_state.slack_history = []

    user_input = st.chat_input("VOC Bot에게 질문하세요...")
    query = selected_q or user_input

    if query:
        st.session_state.slack_history.append(("user", query))
        # 버튼 클릭 = 하드코딩 (즉시), 직접 입력 = Claude Code (실시간)
        answer = SLACK_QA.get(query) if selected_q else None
        if not answer:
            with st.spinner("VOC 데이터 분석 중..."):
                answer = _generate_agent_response(query, df, df_channel, insights)
        st.session_state.slack_history.append(("bot", answer))

    for role, msg in st.session_state.slack_history:
        if role == "user":
            with st.chat_message("user"):
                st.write(msg)
        else:
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(msg)

    if not st.session_state.slack_history:
        st.markdown("""
**슬랙에서 이렇게 씁니다:**
```
/voc 이번 주 위험 신호 있는 마스터 있어?
→ 서재형 21.4% 경보 + 실제 구독자 목소리 + 액션 포인트

/voc 채널톡 반복 문의 TOP5 알려줘
→ 환불 54건 패턴 + 근본 원인 + 제품팀 제안

/voc 미과장 커뮤니티 이번 주 분위기 알려줘
→ 감성 구조 + 실제 인용 + 팀별 권고
```
> 운영팀·사업팀·제품팀이 각자 채널에서 필요한 인사이트를 바로 조회합니다.
""")

    if st.session_state.slack_history:
        if st.button("대화 초기화"):
            st.session_state.slack_history = []
            st.rerun()
