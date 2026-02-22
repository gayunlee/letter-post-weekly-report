# Design: Phase 1 — 사람 검수 기반 구축

## Technical Approach

기존 파이프라인의 분류 결과 JSON에 검수 메타데이터를 추가하고,
별도 검수 워크플로우(추출 → 검수 → 수집 → 정확도 측정)를 구축한다.

## Architecture Decisions

### Decision: 검수 포맷은 CSV
이유:
- 비개발 팀(운영/기획)이 Google Sheet에서 바로 편집 가능
- JSON보다 진입 장벽이 낮음
- 검수 결과 수집 시 CSV → JSON 변환은 간단

### Decision: 검수 기준은 confidence score 기반
이유:
- 2축 분류는 이미 topic_confidence, sentiment_confidence 제공
- detail_tags는 LLM 응답이므로 별도 confidence 추정 필요 (또는 검증 모델)
- low-confidence 전량 + high-confidence 랜덤 20% → 편향 없는 기준선 측정

### Decision: 검수 데이터는 `data/review/`에 독립 저장
이유:
- 원본 분류 데이터 오염 방지
- 검수 전/후 비교가 항상 가능
- Phase 2 재학습 시 학습 데이터로 직접 사용

## Data Flow

```
data/classified_data_two_axis/{date}.json
    ↓ export_for_review.py
data/review/export_{date}.csv          ← 검수 대상
    ↓ (사람 검수: Google Sheet 등)
data/review/reviewed_{date}.csv        ← 검수 결과
    ↓ import_review_results.py
data/review/accuracy_{date}.json       ← 정확도 리포트
```

## File Changes

- `scripts/export_for_review.py` (new) — 검수 대상 추출
- `scripts/import_review_results.py` (new) — 검수 결과 수집 + 정확도 리포트
- `src/classifier_v2/confidence.py` (new) — detail_tags confidence 추정
- `scripts/generate_two_axis_report.py` (modified) — 정확도 섹션 추가 (선택)
