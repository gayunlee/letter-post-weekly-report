# VOC Classification Specification

## Purpose

금융 콘텐츠 플랫폼(오피셜클럽)의 편지글/게시글/채널톡 VOC 데이터를 다층 분류하여, 주간 인사이트 리포트 자동 생성 및 다른 팀의 자유 질의 응답을 가능하게 하는 시스템.

## Requirements

### Requirement: 4축 분류 (Topic × Sentiment × Intent × Urgency)

편지글/게시글을 4개 축으로 분류한다. Topic/Sentiment는 파인튜닝 KcBERT, Intent는 LLM(Haiku), Urgency는 규칙 기반.

#### 축1: Topic (파인튜닝 — 기구현)
| 값 | 설명 |
|----|------|
| 콘텐츠 반응 | 마스터 콘텐츠(강의, 리포트, 방송)에 대한 반응 |
| 투자 이야기 | 투자 전략, 종목, 포트폴리오, 시장 분석 |
| 서비스 이슈 | 플랫폼/앱 기능, 결제, 배송, 구독 등 |
| 커뮤니티 소통 | 인사, 안부, 축하, 일상 공유 |

#### 축2: Sentiment (파인튜닝 — 기구현)
| 값 | 설명 |
|----|------|
| 긍정 | 감사, 만족, 기쁨, 칭찬 |
| 부정 | 불만, 실망, 걱정, 비판 |
| 중립 | 질문, 정보 전달, 사실 기술 |

#### 축3: Intent (LLM → 파인튜닝 전환 예정)
| 값 | 설명 | 부서 활용 |
|----|------|----------|
| 질문/요청 | 답변이나 조치를 기대하는 글 | CS팀, 콘텐츠팀 |
| 피드백/의견 | 경험·감정 공유 (답변 불필요) | 사업팀, 경영진 |
| 제보/건의 | 문제 신고, 기능 요청, 정책 제안 | 개발팀, 운영팀 |
| 정보공유 | 뉴스, 분석, 경험 등 정보 전달 목적 | 콘텐츠팀 |

#### 축4: Urgency (규칙 기반)
| 값 | 조건 |
|----|------|
| 긴급 | 서비스이슈 + 부정 + 키워드(결제오류, 접속불가, 사칭 등) |
| 높음 | 서비스이슈 + (질문/요청 or 제보/건의) |
| 보통 | 부정 감성 or 질문/요청 |
| 낮음 | 기본값 (인사, 잡담 등) |

#### Scenario: 주간 데이터 일괄 분류
- GIVEN 1주치 편지글/게시글이 BigQuery에서 조회됨
- WHEN 4축 분류 파이프라인 실행
- THEN 각 항목에 topic, sentiment, intent, urgency + 각 confidence 필드가 부여됨
- AND urgency_method는 "rule_based"로 표시됨
- AND 결과가 `data/classified_data_two_axis/{date}.json`에 캐싱됨
- AND 이미 분류된 데이터는 건너뜀

#### Scenario: 전주 대비 통계
- GIVEN 대상 주간과 전주 분류 데이터가 존재
- WHEN 통계 분석 실행
- THEN Topic × Sentiment 교차 집계, 마스터별 분포, 전주 대비 증감이 산출됨

---

### Requirement: 부서 자동 라우팅

Topic × Intent 조합에 따라 담당 부서를 자동 매핑한다.

#### 라우팅 매핑
```
(서비스 이슈, 제보/건의) → [개발팀]
(서비스 이슈, 질문/요청) → [CS팀, 개발팀]
(서비스 이슈, 피드백/의견) → [운영팀]
(콘텐츠 반응, 질문/요청) → [콘텐츠팀]
(콘텐츠 반응, 피드백/의견) → [콘텐츠팀, 사업팀]
(투자 이야기, 질문/요청) → [콘텐츠팀]
(투자 이야기, 피드백/의견) → [사업팀]
(커뮤니티 소통, *) → [운영팀]
```

#### Scenario: 분류 결과에 부서 라우팅 부착
- GIVEN 4축 분류가 완료된 항목
- WHEN 라우팅 매핑 실행
- THEN 각 항목에 `department_route: ["부서명", ...]` 필드가 추가됨

---

### Requirement: Detail Tags 추출 (카테고리 태그 + 자유 태그)

2축 분류 후, LLM(Claude Haiku)으로 28개 controlled vocabulary 카테고리 태그 + 자유 태그 + 요약을 추출한다. Intent 분류도 동일 호출에서 함께 수행한다.

- **모델**: claude-haiku-4-5-20251001 ($3.12/주, 2,542건 기준)
- **카테고리 태그**: Topic별 6~8개, 총 28개
- **품질 기준**: 목록 외 태그 비율 < 5%, 태그 커버리지 > 90%

#### Scenario: 카테고리 태그 + Intent 동시 추출
- GIVEN 2축 분류가 완료된 항목
- WHEN detail_tags 추출 실행
- THEN 각 항목에 `detail_tags: {category_tags: [...], free_tags: [...], summary: "..."}` + `intent` + `intent_confidence`가 부여됨
- AND category_tags는 반드시 28개 허용 목록에서만 선택됨
- AND 이미 detail_tags가 있는 항목은 건너뜀
- AND 추가 API 호출 없이 기존 Haiku 1회 호출에서 통합 추출

#### Scenario: 태그 집계
- GIVEN detail_tags가 부착된 전체 데이터
- WHEN 카테고리 태그 집계 실행
- THEN Topic별 태그 분포가 산출됨
- AND "결제 관련 불만이 몇 건?" 같은 구체적 질의에 응답 가능

---

### Requirement: 서브 테마 분석

서비스 이슈 항목을 클러스터링하고, Topic별 부정 항목의 패턴을 LLM으로 요약한다.

#### Scenario: 서비스 이슈 클러스터링
- GIVEN 서비스 이슈로 분류된 항목들
- WHEN 임베딩 → K-Means 클러스터링 실행 (silhouette score로 최적 k 선택)
- THEN 각 클러스터에 LLM 생성 라벨 + 건수가 부여됨
- AND 결과가 `data/sub_themes/{date}.json`에 저장됨

#### Scenario: 부정 테마 요약
- GIVEN 콘텐츠 반응/투자 이야기/커뮤니티 소통의 부정 항목들
- WHEN LLM 패턴 요약 실행
- THEN Topic별 부정 의견의 주요 패턴과 대표 사례가 요약됨

---

### Requirement: 주간 리포트 생성

분류 + 통계 + 서브 테마를 종합하여 마크다운 리포트와 엑셀 파일을 자동 생성한다.

#### Scenario: 마크다운 리포트
- GIVEN 통계, 서브 테마, 마스터 목록이 준비됨
- WHEN 리포트 생성 실행
- THEN `reports/two_axis_report_{date}.md` 파일이 생성됨
- AND 핵심 요약, 카테고리 태그 분포, 서비스 이슈 클러스터, 마스터별 상세가 포함됨

#### Scenario: 엑셀 데이터
- GIVEN 분류 완료 데이터
- WHEN 엑셀 생성 실행
- THEN `reports/two_axis_data_{date}.xlsx` 파일이 생성됨
- AND 편지/게시글 시트에 카테고리 태그, 자유 태그, 요약, intent, urgency, department_route 컬럼이 포함됨

---

### Requirement: Notion/Slack 배포

생성된 리포트를 Notion에 업로드하고 Slack으로 알림을 전송한다.

#### Scenario: Notion 업로드
- GIVEN 마크다운 리포트가 생성됨
- WHEN Notion 업로드 실행
- THEN Notion 데이터베이스에 리포트 페이지가 생성됨

#### Scenario: Slack 알림
- GIVEN Notion 업로드 완료
- WHEN Slack 알림 실행
- THEN 지정 채널에 리포트 요약 + Notion 링크가 전송됨

---

### Requirement: 채널톡 CS 데이터 분류 _(2차 우선순위)_

채널톡 CS 문의 데이터를 BigQuery에서 조회하여 별도 분류 체계로 분류한다.

- **데이터소스**: `us-service-data.channel_io.messages`
- **분류 단위**: chatId 기준 1대화 = 1분류
- **전처리**: 사용자 메시지만 연결, 앞 500자 + 마지막 200자

#### 채널톡 분류 카테고리
| 축 | 값 |
|----|------|
| Intent | 결제/환불, 앱/서비스 장애, 계정/인증, 콘텐츠 접근, 멤버십/구독, 일반 문의 |
| Sentiment | 긍정/부정/중립 (편지글 Sentiment 모델 재활용) |
| Status | 봇_응답만 / 상담원_연결 / 미해결 (규칙 기반) |
| Resolution | 해결 / 진행중 / 미해결 (LLM 판별) |

#### Scenario: 채널톡 대화 분류
- GIVEN 1주치 채널톡 대화가 BigQuery에서 조회됨
- WHEN 채널톡 분류 파이프라인 실행
- THEN chatId별로 Intent, Sentiment, Status, Resolution이 부여됨

---

### Requirement: 사람 검수 프로세스 _(Phase 1)_

자동 분류 결과의 정확도를 측정하고, confidence 기반으로 검수 대상을 추출한다.

#### Confidence 임계값
| 범위 | 처리 |
|------|------|
| ≥ 0.85 | 자동 수용 |
| 0.60 ~ 0.85 | 조건부 수용 (전주 분포와 크게 다르지 않으면 수용) |
| < 0.60 | 반드시 사람 검수 |

#### Scenario: 검수 대상 추출
- GIVEN 주간 분류 완료 데이터
- WHEN confidence 기반 라우팅 실행
- THEN auto / conditional / review 3가지로 분류됨
- AND 검수 대상은 CSV로 추출 가능

#### Scenario: 검수 결과 반영
- GIVEN 검수자가 수정한 라벨
- WHEN 검수 결과 수집 실행
- THEN 원본 분류 vs 검수 결과 비교 → 정확도 리포트 자동 생성됨
- AND 검수 데이터가 `data/review/`에 축적됨

---

### Requirement: 신뢰 임계값 + Intent 파인튜닝 _(Phase 2)_

검수 데이터 기반으로 신뢰 임계값을 최적화하고, Intent 축을 LLM에서 파인튜닝 모델로 전환한다.

#### Scenario: 자동 통과 판정
- GIVEN 검수 데이터 기반 최적 임계값이 설정됨
- WHEN 분류 결과의 confidence가 임계값 이상
- THEN 해당 항목은 사람 검수 없이 자동 통과
- AND 자동 통과 건의 오분류율 5% 이하
- AND 검수 비율 15~25%로 감소

#### Scenario: Intent 모델 파인튜닝
- GIVEN 검수 데이터에서 Intent 라벨 500건+ 축적
- WHEN Intent 파인튜닝 실행
- THEN LLM 대비 정확도 비교 리포트 생성됨
- AND 성능 동등 이상이면 파인튜닝 모델로 전환

---

### Requirement: 재학습 파이프라인 _(Phase 3)_

검수 데이터를 활용하여 주기적으로 모델을 재학습하고, Vertex AI Custom Training으로 자동화한다.

#### Scenario: 월간 재학습
- GIVEN 검수 데이터 500건+ 축적 (4주치)
- WHEN 재학습 파이프라인 실행
- THEN 기존 학습 데이터 + 검수 데이터 병합 → 품질 검증 → 모델 학습
- AND 재학습 전후 테스트셋 A/B 성능 비교 리포트 생성됨
- AND 성능 향상시 배포 / 저하시 Slack 알림

---

### Requirement: 드리프트 모니터링 _(Phase 4)_

분류 분포 변화를 감지하고 알림을 발송한다.

#### Scenario: 드리프트 감지
- GIVEN 주간 분류 분포 이력
- WHEN chi-square 검정에서 p < 0.01 감지
- THEN Slack 알림 발생 + 검수 비율 일시적 상향

#### Scenario: 주간 품질 점검
- GIVEN 주간 분류 완료
- WHEN 품질 점검 스크립트 실행
- THEN Topic/Sentiment 분포 변화, 평균 confidence 추이, 마스터별 부정 비율 급증 감지
