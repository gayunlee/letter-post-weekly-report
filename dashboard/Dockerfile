# VOC 대시보드 — Cloud Run Service 용 (Streamlit 웹 서버)
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8080

WORKDIR /app

# 의존성 (대시보드만 — 분류기/모델 제외)
COPY dashboard/requirements.txt .
RUN pip install -r requirements.txt

# 대시보드 코드만 복사 (src/ 등 무거운 부분 제외)
COPY dashboard ./dashboard

# Streamlit 헬스체크/CORS 설정
ENV STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ENABLE_CORS=false \
    STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

EXPOSE 8080

CMD streamlit run dashboard/voc_dashboard.py --server.port=$PORT --server.address=0.0.0.0
