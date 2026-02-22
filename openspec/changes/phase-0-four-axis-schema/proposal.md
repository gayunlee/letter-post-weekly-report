# Proposal: Phase 0 — 4축 분류 스키마 + 기초 인프라

## Intent

기존 2축(Topic × Sentiment) 분류 체계를 4축(+ Intent × Urgency)으로 확장하여,
다부서가 자유롭게 쿼리·필터·분석할 수 있는 다층 분류 체계를 구축한다.

현재 문제:
1. 편지글/게시글 분류가 다른 부서에서 활용하기에 분석 자유도가 낮음
2. "이 글이 답변이 필요한 질문인지, 그냥 감상인지" 구분 불가
3. 긴급 서비스 이슈 자동 감지가 안 됨
4. 부서별 자동 라우팅 불가

## Scope

In scope:
- 4축 taxonomy 정의 (기존 2축 확장)
- Intent 분류 (Haiku — detail_tags 추출 시 함께 뽑음, 추가 API 호출 없음)
- Urgency 규칙 엔진 (키워드 + 조건 매칭)
- 부서 자동 라우팅 매핑 (Topic × Intent → 부서)
- FourAxisClassifier 래퍼 (기존 TwoAxisClassifier 래핑)
- 파이프라인 통합 (generate_two_axis_report.py 수정)

Out of scope:
- 검수 프로세스 (Phase 1)
- 채널톡 연동 (2차 우선순위)
- 모델 재학습 (Phase 2+)
- 드리프트 모니터링 (Phase 4)

## Approach

1. 기존 TwoAxisClassifier(Topic + Sentiment)를 그대로 재활용
2. Intent는 detail_tag_extractor.py의 Haiku 호출 프롬프트를 확장하여 동시 추출 (비용 증가 없음)
3. Urgency는 Topic + Sentiment + Intent + 키워드 조합으로 규칙 기반 판별
4. 부서 라우팅은 Topic × Intent 매핑 테이블로 결정
5. FourAxisClassifier가 이 모든 것을 조합하는 래퍼 역할

## 의사결정 포인트

- Intent 파인튜닝 전환 시점: Phase 2에서 검수 데이터 500건+ 축적 후 결정
- Urgency 키워드 목록: 운영팀과 협의하여 지속 업데이트
