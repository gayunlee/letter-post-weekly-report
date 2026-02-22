# Design: Phase 1 — Confidence 기반 검수 라우팅

## Technical Approach

4축 분류 결과의 confidence score를 기반으로 auto/conditional/review 3단계 라우팅을 구축한다.
검수 대상은 CSV로 내보내고, 검수 결과는 `data/review/`에 독립 저장한다.

## Architecture Decisions

### Decision: 검수 포맷은 CSV
이유:
- 비개발 팀(운영/기획)이 Google Sheet에서 바로 편집 가능
- JSON보다 진입 장벽이 낮음
- 검수 결과 수집 시 CSV → JSON 변환은 간단

### Decision: 3단계 라우팅 (auto / conditional / review)
이유:
- 단순 2단계(pass/review)보다 검수 효율이 높음
- conditional 구간에서 전주 분포 비교로 이상 없는 건 자동 수용 가능
- 검수 비율을 단계적으로 줄여갈 수 있는 구조

### Decision: 검수 데이터는 `data/review/`에 독립 저장
이유:
- 원본 분류 데이터 오염 방지
- 검수 전/후 비교가 항상 가능
- Phase 2 재학습 시 학습 데이터로 직접 사용

### Decision: confidence는 4축 중 최저값 기준
이유:
- Topic confidence만 보면 Sentiment/Intent 오분류를 놓침
- min(topic_conf, sentiment_conf, intent_conf)로 보수적 판단
- Phase 2에서 축별 가중치 최적화 가능

## Data Flow

```
data/classified_data_two_axis/{date}.json
    ↓ ReviewRouter.route()
    ├─ auto (≥0.85)     → 자동 수용
    ├─ conditional (0.60~0.85) → 전주 분포 비교 → 수용 or 검수
    └─ review (<0.60)    → 검수 대상
    ↓
    ↓ export_for_review.py
data/review/export_{date}.csv          ← 검수 대상
    ↓ (사람 검수: Google Sheet 등)
data/review/reviewed_{date}.csv        ← 검수 결과
    ↓ import_review_results.py
data/review/accuracy_{date}.json       ← 정확도 리포트
```

## File Changes

- `src/classifier_v3/review_router.py` (new) — confidence 기반 3단계 라우팅
- `src/storage/review_store.py` (new) — 검수 데이터 관리 (내보내기/수집/통계)
- `scripts/export_for_review.py` (new) — 검수 대상 CSV 추출
- `scripts/import_review_results.py` (new) — 검수 결과 수집 + 정확도 리포트
- `scripts/generate_two_axis_report.py` (modified) — 검수 라우팅 통합
