# Design: Phase 0 — 4축 분류 스키마 + 기초 인프라

## Technical Approach

기존 2축 분류 파이프라인을 감싸는 v3 레이어를 구축한다.
기존 `src/classifier_v2/` 코드는 수정 최소화, `src/classifier_v3/`에 새 모듈을 추가한다.

## Architecture Decisions

### Decision: v2를 수정하지 않고 v3로 래핑
이유:
- 기존 파이프라인의 안정성 유지
- v2 → v3 전환을 점진적으로 진행 가능
- v3에서 문제 발생 시 v2로 즉시 롤백 가능

### Decision: Intent는 detail_tags Haiku 호출에 통합
이유:
- 별도 API 호출 없이 프롬프트 확장만으로 구현 (비용 증가 0)
- detail_tags 추출 시점에 이미 텍스트 + topic + sentiment 정보가 있어 판단에 충분
- 추후 파인튜닝 전환 시 분리하기 쉬운 구조

### Decision: Urgency는 규칙 기반
이유:
- 조건이 명확하고 예측 가능 (Topic + Sentiment + 키워드 조합)
- 모델 학습 불필요, 즉시 적용 가능
- 키워드 목록만 업데이트하면 되므로 운영 부담 최소

### Decision: 기존 캐시 데이터 구조 확장
이유:
- classification 객체에 intent, urgency, department_route 필드 추가
- 기존 topic, sentiment 필드는 그대로 유지
- 이미 분류된 항목에 intent가 없으면 detail_tags 추출 시 함께 보강

## Data Flow

```
BigQuery(us_plus)
    ↓
WeeklyDataQuery.get_weekly_data()
    ↓ {letters, posts}
    ↓
FourAxisClassifier.classify_batch()
    ├─ TwoAxisClassifier → topic, sentiment (기존)
    ├─ DetailTagExtractor → detail_tags + intent (Haiku, 통합)
    ├─ UrgencyRules → urgency (규칙 기반)
    └─ DepartmentRouter → department_route (매핑)
    ↓
classification: {
    topic, topic_confidence,
    sentiment, sentiment_confidence,
    intent, intent_confidence,
    urgency, urgency_method,
    department_route,
    needs_review
}
detail_tags: {category_tags, free_tags, summary}
```

## 분류 결과 데이터 구조

```json
{
  "classification": {
    "topic": "서비스 이슈",
    "topic_confidence": 0.92,
    "sentiment": "부정",
    "sentiment_confidence": 0.87,
    "intent": "제보/건의",
    "intent_confidence": 0.78,
    "urgency": "높음",
    "urgency_method": "rule_based",
    "department_route": ["개발팀"],
    "needs_review": false
  },
  "detail_tags": {
    "category_tags": ["앱/기능 오류"],
    "free_tags": ["앱 크래시", "iOS 업데이트 후"],
    "summary": "iOS 업데이트 후 앱 반복 종료 제보"
  }
}
```

## File Changes

- `src/classifier_v3/__init__.py` (new) — 패키지 초기화
- `src/classifier_v3/taxonomy.py` (new) — 4축 분류 체계 정의
- `src/classifier_v3/urgency_rules.py` (new) — 긴급도 규칙 엔진
- `src/classifier_v3/department_router.py` (new) — 부서 라우팅 매핑
- `src/classifier_v3/four_axis_classifier.py` (new) — 기존 2축 래핑 + Intent + Urgency
- `src/classifier_v2/detail_tag_extractor.py` (modified) — Intent 분류 프롬프트 통합
- `scripts/generate_two_axis_report.py` (modified) — v3 분류기 옵션 추가
