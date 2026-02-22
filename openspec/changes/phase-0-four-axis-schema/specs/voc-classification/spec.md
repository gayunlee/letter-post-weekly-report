# Delta for VOC Classification — Phase 0

## ADDED Requirements

### Requirement: Intent 축 (의도 분류)

편지글/게시글의 작성 의도를 4가지로 분류한다. detail_tags 추출 시 Haiku 1회 호출에서 함께 추출.

#### Scenario: Intent 분류
- GIVEN 2축 분류(Topic + Sentiment)가 완료된 항목
- WHEN detail_tags + intent 추출 실행
- THEN intent: "질문/요청" | "피드백/의견" | "제보/건의" | "정보공유" 중 1개 부여
- AND intent_confidence: 0.0~1.0 수치 부여
- AND 추가 API 호출 없이 기존 Haiku 호출에서 통합 추출

---

### Requirement: Urgency 축 (긴급도 분류)

Topic + Sentiment + Intent + 키워드 조합으로 긴급도를 규칙 기반 판별한다.

#### Scenario: 긴급도 판별
- GIVEN 4축 중 topic, sentiment, intent가 부여된 항목
- WHEN 긴급도 규칙 엔진 실행
- THEN urgency: "긴급" | "높음" | "보통" | "낮음" 중 1개 부여
- AND urgency_method: "rule_based"로 표시

#### Scenario: 긴급 키워드 매칭
- GIVEN 서비스이슈 + 부정 감성 항목
- WHEN 텍스트에 "결제오류", "접속불가", "사칭" 등 긴급 키워드 포함
- THEN urgency: "긴급"으로 판별

---

### Requirement: 부서 자동 라우팅

Topic × Intent 조합에 따라 담당 부서를 자동 매핑한다.

#### Scenario: 라우팅 매핑
- GIVEN 4축 분류가 완료된 항목
- WHEN 부서 라우팅 실행
- THEN department_route: ["부서명", ...] 필드가 추가됨

---

## MODIFIED Requirements

### Requirement: Detail Tags 추출 (수정)

기존 detail_tags 추출에 Intent 분류를 통합. Haiku 프롬프트를 확장하여 category_tags + free_tags + summary + intent를 한 번에 추출.

#### Scenario: 통합 추출 (변경)
- GIVEN 2축 분류가 완료된 항목
- WHEN detail_tags 추출 실행
- THEN 기존 detail_tags에 더해 intent + intent_confidence가 함께 반환됨
- AND 이미 intent가 부여된 항목은 건너뜀

### Requirement: 주간 리포트 (수정)

엑셀 출력에 intent, urgency, department_route 컬럼을 추가한다.

#### Scenario: 엑셀 컬럼 확장 (변경)
- GIVEN 4축 분류 완료 데이터
- WHEN 엑셀 생성 실행
- THEN 기존 컬럼 + 의도, 긴급도, 담당부서 컬럼이 추가됨
