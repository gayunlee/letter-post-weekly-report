# Delta for VOC Classification — Phase 1

## ADDED Requirements

### Requirement: 검수 라우팅 (ReviewRouter)

4축 분류의 confidence score 기반으로 auto/conditional/review 3단계로 라우팅한다.

#### Scenario: Auto 수용
- GIVEN 분류 완료 항목의 min(topic_conf, sentiment_conf, intent_conf) ≥ 0.85
- WHEN 검수 라우팅 실행
- THEN 해당 항목은 자동 수용 (검수 불필요)

#### Scenario: Conditional 수용
- GIVEN 분류 완료 항목의 min confidence가 0.60 ~ 0.85
- WHEN 전주 분포와 비교하여 이상 없음
- THEN 해당 항목은 조건부 수용

#### Scenario: Review 대상
- GIVEN 분류 완료 항목의 min confidence < 0.60
- WHEN 검수 라우팅 실행
- THEN 해당 항목은 검수 대상으로 추출됨

---

### Requirement: 검수 대상 CSV 추출

검수 대상 항목을 비개발 팀이 편집할 수 있는 CSV로 내보낸다.

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
- THEN Topic / Sentiment / Intent / Category Tag 각각의 accuracy + confusion matrix 출력
