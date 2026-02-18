# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

주간 리포트 자동 생성 시스템 - 금융 콘텐츠 플랫폼의 편지글/게시글을 분석하여 인사이트 리포트 생성

### Key Concepts

- **마스터 (Masters)**: 투자 교육 커뮤니티를 운영하는 금융 콘텐츠 크리에이터
- **편지글 (Letters)**: 구독자가 마스터에게 보내는 개인 메시지
- **게시글 (Posts)**: 각 마스터 커뮤니티 게시판의 게시글
- **오피셜클럽**: 각 마스터가 운영하는 개별 커뮤니티

## Commands

```bash
# 의존성 설치
pip install -r requirements.txt

# 주간 리포트 생성 (지난 주 데이터)
python scripts/generate_weekly_report.py

# 특정 기간 리포트 생성
python scripts/generate_custom_week_report.py  # 날짜 입력 필요

# 전주 데이터 생성 (비교용)
python scripts/generate_previous_week.py

# 개별 테스트
python scripts/test_classifier.py      # 분류기 테스트
python scripts/test_vectorstore.py     # 벡터 스토어 테스트
python scripts/explore_bigquery.py     # BigQuery 스키마 탐색
```

## Architecture

```
src/
├── bigquery/           # BigQuery 연동
│   ├── client.py       # BigQueryClient - 쿼리 실행
│   └── queries.py      # WeeklyDataQuery - 주간 데이터 조회 (letters, posts)
├── classifier/         # 콘텐츠 분류
│   ├── content_classifier.py   # ContentClassifier - Claude API 기반 few-shot 분류
│   └── vector_classifier.py    # VectorContentClassifier - 벡터 유사도 기반 분류
├── vectorstore/        # 벡터 저장소
│   └── chroma_store.py # ChromaVectorStore - ChromaDB 래퍼
├── reporter/           # 리포트 생성
│   ├── analytics.py    # WeeklyAnalytics - 통계 분석
│   └── report_generator.py  # ReportGenerator - 마크다운 리포트 생성
└── storage/            # 데이터 저장
    └── data_store.py   # ClassifiedDataStore - 분류된 데이터 캐싱
```

### Pipeline Flow

1. **BigQuery 조회**: `WeeklyDataQuery.get_weekly_data()` → 편지/게시글 조회
2. **콘텐츠 분류**: `VectorContentClassifier.classify_batch()` → 5개 카테고리로 분류
3. **벡터 저장**: `ChromaVectorStore.add_contents_batch()` → 시맨틱 검색용 저장
4. **통계 분석**: `WeeklyAnalytics.analyze_weekly_data()` → 전주 대비 통계
5. **리포트 생성**: `ReportGenerator.generate_report()` → 마크다운 출력

### Classification Categories

- 감사·후기: 마스터에 대한 감사, 긍정적 피드백
- 질문·토론: 포트폴리오, 종목, 투자 전략 질문
- 정보성 글: 투자 경험 공유, 종목 분석
- 서비스 피드백: 플랫폼 개선점, 불편사항
- 일상·공감: 안부, 축하, 공감 표현

## Environment Setup

```bash
# .env 파일 생성 (.env.example 참고)
ANTHROPIC_API_KEY=your_api_key
GOOGLE_APPLICATION_CREDENTIALS=./accountKey.json
BIGQUERY_PROJECT_ID=your_project
REPORT_OUTPUT_DIR=./reports
```

## Data Storage

- `data/classified_data/`: 분류된 주간 데이터 (JSON)
- `data/stats/`: 주간 통계 요약
- `chroma_db/`: ChromaDB 벡터 저장소
- `reports/`: 생성된 마크다운 리포트

## Report Format

`example.md` 형식을 따름:
- 핵심 요약: 전체 편지/게시글 통계 + 전주 대비 증감
- 마스터별 상세: 통계 테이블, 주요 내용, 서비스 피드백, 체크 포인트
- 직접 인용은 이탤릭 (_"인용문"_)
- 개선 권고는 화살표 표시 (_→ 권고사항_)

### /note 기본 설정
- **category**: `voc 데이터 라벨링`
- **trigger**: `coding`
- 이 프로젝트에서 `/note` 실행 시 위 category를 기본값으로 사용한다 (F.F.md 추론 생략)
