# Proposal: Phase 1 — Confidence 기반 검수 라우팅

## Intent

자동 분류 결과의 정확도 기준선을 확립하고, confidence 기반으로 검수 프로세스를 자동화한다.
현재 분류 품질이 "체감상 좋다"는 수준이지, 정량적 정확도가 측정되지 않았다.
다른 팀이 분류 데이터를 신뢰하고 활용하려면 정확도 수치가 필요하다.

## Scope

In scope:
- ReviewRouter: confidence 기반 auto/conditional/review 3단계 라우팅
- ReviewStore: 검수 대기/완료 데이터 관리 (`data/review/`)
- 검수용 CSV 내보내기 (비개발 팀 편집용)
- 검수 결과 수집 + 원본 대비 정확도 리포트
- 레이어별 정확도 기준선 측정 (Topic / Sentiment / Intent / Category Tag)
- 파이프라인 검수 라우팅 통합

Out of scope:
- Streamlit 검수 대시보드 (선택적 구현)
- 모델 재학습 (Phase 2)
- 채널톡 데이터 (2차 우선순위)
- 드리프트 모니터링 (Phase 4)

## Approach

1. 4축 분류의 confidence score 기반으로 3단계 라우팅
2. 주간 데이터에서 검수 대상 자동 추출 (low-confidence 전량 + conditional + high-confidence 랜덤 샘플)
3. 비개발 팀이 사용할 수 있는 CSV 형태로 내보내기
4. 검수 결과를 `data/review/`에 수집하고, 원본 대비 차이를 자동 집계
5. 2주간 30~50% 샘플 검수 → 레이어별 정확도 기준선 도출

## Confidence 임계값 (초기)

| 범위 | 처리 | 비고 |
|------|------|------|
| ≥ 0.85 | 자동 수용 | |
| 0.60 ~ 0.85 | 조건부 수용 | 전주 분포와 크게 다르지 않으면 수용 |
| < 0.60 | 반드시 사람 검수 | |

예상 검수 비율: Topic 69.3% 기준 ~30-50% → 주간 1,000~1,700건

## 의사결정 포인트

검수 완료 후 정확도에 따라 Phase 2 방향이 결정됨:

| Category Tag 정확도 | 다음 액션 |
|---------------------|----------|
| 85% 이상 | few-shot 유지, Phase 2에서 임계값만 설정 |
| 80~85% | 해당 카테고리 예제 보강 시도 |
| 80% 미만 | Phase 2에서 파인튜닝 검토 |
