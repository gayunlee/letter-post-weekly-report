# Tasks

## 1. ReviewRouter 구현
- [ ] 1.1 `src/classifier_v3/review_router.py` 작성
- [ ] 1.2 auto/conditional/review 3단계 라우팅 로직
- [ ] 1.3 conditional 구간: 전주 분포 비교 로직
- [ ] 1.4 confidence 기준: min(topic_conf, sentiment_conf, intent_conf)

## 2. ReviewStore 구현
- [ ] 2.1 `src/storage/review_store.py` 작성
- [ ] 2.2 검수 대상 추출 + CSV 내보내기
- [ ] 2.3 검수 결과 CSV 파싱 + 유효성 검증
- [ ] 2.4 원본 분류 vs 검수 결과 비교 로직

## 3. 검수 대상 추출
- [ ] 3.1 `scripts/export_for_review.py` 작성
- [ ] 3.2 CSV 컬럼: id, text, auto_topic, auto_sentiment, auto_intent, auto_category_tags, confidence_min, review_topic, review_sentiment, review_intent, review_category_tags
- [ ] 3.3 review 전량 + conditional 중 이상 감지 건 + auto 랜덤 5% 추출
- [ ] 3.4 1주치 데이터로 추출 테스트

## 4. 검수 결과 수집
- [ ] 4.1 `scripts/import_review_results.py` 작성
- [ ] 4.2 CSV 파싱 + 허용 라벨 범위 체크
- [ ] 4.3 정확도 리포트: Topic / Sentiment / Intent / Category Tag 별 accuracy, confusion matrix
- [ ] 4.4 `data/review/accuracy_{date}.json` 저장

## 5. 파이프라인 통합
- [ ] 5.1 `scripts/generate_two_axis_report.py`에 검수 라우팅 옵션 추가
- [ ] 5.2 검수 라우팅 통계 (auto/conditional/review 비율) 리포트 포함

## 6. 정확도 기준선 측정
- [ ] 6.1 첫 주간 데이터 검수 실행 (30~50% 샘플)
- [ ] 6.2 레이어별 정확도 수치 도출
- [ ] 6.3 Phase 2 방향 결정 (few-shot 유지 vs 파인튜닝)

## 7. 데이터 동기화
- [ ] 7.1 `data/review/`가 sync_data.py에 포함 확인
- [ ] 7.2 검수 결과 Vultr Object Storage 동기화 테스트
