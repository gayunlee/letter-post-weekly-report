# Tasks

## 1. Confidence Score 통합
- [ ] 1.1 2축 분류의 기존 confidence score 확인 및 분포 분석
- [ ] 1.2 detail_tags용 confidence 추정 방식 결정 (LLM self-eval vs 검증 모델)
- [ ] 1.3 `src/classifier_v2/confidence.py` 모듈 작성

## 2. 검수 대상 추출
- [ ] 2.1 `scripts/export_for_review.py` 작성
- [ ] 2.2 추출 기준 구현: confidence < 0.7 전량 + 고신뢰 랜덤 20%
- [ ] 2.3 CSV 컬럼 설계: id, text, auto_topic, auto_sentiment, auto_category_tags, review_topic, review_sentiment, review_category_tags
- [ ] 2.4 1주치 데이터로 추출 테스트

## 3. 검수 결과 수집
- [ ] 3.1 `scripts/import_review_results.py` 작성
- [ ] 3.2 CSV 파싱 + 유효성 검증 (허용 라벨 범위 체크)
- [ ] 3.3 원본 분류 vs 검수 결과 비교 로직
- [ ] 3.4 정확도 리포트 자동 생성: Topic / Sentiment / Category Tag 별 accuracy, confusion matrix

## 4. 정확도 기준선 측정
- [ ] 4.1 첫 주간 데이터 검수 실행 (30~50% 샘플)
- [ ] 4.2 레이어별 정확도 수치 도출
- [ ] 4.3 Phase 2 방향 결정 (few-shot 유지 vs 파인튜닝)

## 5. 데이터 동기화 연동
- [ ] 5.1 `data/review/` 폴더가 sync_data.py에 이미 포함 확인
- [ ] 5.2 검수 결과 Vultr Object Storage 동기화 테스트
