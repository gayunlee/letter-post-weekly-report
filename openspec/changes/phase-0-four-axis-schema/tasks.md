# Tasks

## 1. 4축 Taxonomy 정의
- [ ] 1.1 `src/classifier_v3/taxonomy.py` 작성 — Topic/Sentiment/Intent/Urgency 정의
- [ ] 1.2 기존 v2 taxonomy와의 호환성 확인

## 2. 긴급도 규칙 엔진
- [ ] 2.1 `src/classifier_v3/urgency_rules.py` 작성
- [ ] 2.2 키워드 목록 정의 (결제오류, 접속불가, 사칭 등)
- [ ] 2.3 Topic + Sentiment + Intent + 키워드 조합 규칙 구현

## 3. 부서 라우팅
- [ ] 3.1 `src/classifier_v3/department_router.py` 작성
- [ ] 3.2 Topic × Intent → 부서 매핑 테이블 구현

## 4. Intent 통합 추출
- [ ] 4.1 `src/classifier_v2/detail_tag_extractor.py` 수정 — Haiku 프롬프트에 Intent 분류 추가
- [ ] 4.2 응답 파싱에 intent + intent_confidence 추가
- [ ] 4.3 기존 detail_tags만 있는 데이터와의 하위 호환성 유지

## 5. FourAxisClassifier 래퍼
- [ ] 5.1 `src/classifier_v3/four_axis_classifier.py` 작성
- [ ] 5.2 TwoAxisClassifier + DetailTagExtractor + UrgencyRules + DepartmentRouter 통합
- [ ] 5.3 classify_batch() 메서드 구현

## 6. 파이프라인 통합
- [ ] 6.1 `scripts/generate_two_axis_report.py` 수정 — v3 분류기 옵션 추가
- [ ] 6.2 기존 v2 모드와 v3 모드 양립 가능하게 --use-v3 플래그

## 7. 검증
- [ ] 7.1 기존 캐시 데이터로 Intent + Urgency 추가 테스트
- [ ] 7.2 부서 라우팅 정합성 확인
