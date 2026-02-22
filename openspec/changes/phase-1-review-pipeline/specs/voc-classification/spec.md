# Delta for VOC Classification

## ADDED Requirements

### Requirement: 검수 대상 추출

confidence score 기반으로 주간 분류 데이터에서 검수 대상을 자동 추출한다.

#### Scenario: Low-confidence 항목 전량 추출
- GIVEN 주간 분류 완료 데이터
- WHEN confidence < 0.7인 항목
- THEN 해당 항목은 검수 대상으로 전량 추출됨

#### Scenario: High-confidence 랜덤 샘플
- GIVEN 주간 분류 완료 데이터
- WHEN confidence >= 0.7인 항목 중 랜덤 20%
- THEN 해당 항목도 검수 대상으로 추출됨 (기준선 편향 방지)

#### Scenario: CSV 내보내기
- GIVEN 검수 대상이 추출됨
- WHEN 내보내기 실행
- THEN `data/review/export_{date}.csv` 파일이 생성됨
- AND 자동 분류 결과 + 빈 검수 컬럼이 포함됨

---

### Requirement: 검수 결과 수집 및 정확도 측정

검수자가 수정한 라벨을 수집하고, 자동 분류 대비 정확도를 측정한다.

#### Scenario: 검수 결과 반영
- GIVEN 검수 완료된 `data/review/reviewed_{date}.csv`
- WHEN 결과 수집 실행
- THEN 원본 분류 vs 검수 결과 비교 → 레이어별 정확도 산출
- AND `data/review/accuracy_{date}.json`에 저장됨

#### Scenario: 정확도 리포트
- GIVEN 정확도 데이터
- WHEN 리포트 생성
- THEN Topic / Sentiment / Category Tag 각각의 accuracy + confusion matrix 출력
