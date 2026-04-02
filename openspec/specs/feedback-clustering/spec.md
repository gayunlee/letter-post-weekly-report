# 피드백 벡터 클러스터링 → Jira 태스크 자동화

## 목적
편지글/게시글 피드백을 벡터 임베딩 기반으로 클러스터링하여, 각 대응 부서가 바로 처리할 수 있는 구조로 자동 생성한다.

## 현재 완료 (Phase 1)

### 파이프라인
1. BigQuery에서 편지/게시글 조회
2. v5 Bedrock Haiku로 4분류 (topic + subtag + summary + tags)
3. 피드백 중 이용문의 제외
4. LLM이 각 건에 대해 세부 증상 자유 서술 (고정 카테고리 없음)
5. 서술 + 원문을 Bedrock Titan으로 임베딩
6. Agglomerative Clustering (코사인 거리 + 키워드 겹침 보정)
7. LLM 검수 (클러스터 단위 라벨링 + 대응 유형 판단 + 오분류 탐지)
8. 클러스터별 엑셀 출력 + 슬랙 전송

### 핵심 설계 결정
- **고정 카테고리 대신 벡터 클러스터링**: 기타/미분류 문제 해소, subtag 경계 넘는 이슈 자동 병합
- **임베딩 대상**: `LLM 서술 | 원문` concat → 서술 스타일 차이를 원문 유사도가 보완
- **키워드 보정**: tags 필드의 Jaccard 유사도로 거리 40% 감쇄 → 동의어 문제 완화
- **threshold 0.6**: Q1 기준 적정 수준
- **서술 톤**: 사용자가 겪은 현상/불편 중심, 특정 부서/담당자 지목 금지

### 프로덕션 운영 구조
```
BigQuery (voc_labelled.letters_posts)
  ├── 1차 분류 결과 저장 (topic, subtag, summary, tags, sentiment)
  │
  ├── 기술이슈 클러스터 → 엑셀로 개발팀 공유 (Jira 매핑)
  ├── 기능요청 클러스터 → 제품팀 검토
  ├── 운영/CS이슈 클러스터 → 운영팀/CS팀 확인
  ├── 나머지 → 주간 리포트 통계로 활용
  └── 원문 조회 → Streamlit 뷰어에서 필터/검색
```

### 관련 파일
- 1차 분류: `scripts/generate_tech_issues_report.py`
- 클러스터링: `scripts/cluster_tech_issues.py`
- LLM 검수: `scripts/review_clusters.py`
- Streamlit 뷰어: `dashboard/tech_issues_viewer.py`
- 출력: `exports/tech_issues_clustered_*.xlsx`, `exports/tech_issues_reviewed_*.json`

## Phase 2: 파이프라인 효율화 (다음)

### 2-1. 서술 단계 제거
- 현재: 1차 분류(summary 생성) → 별도 LLM 서술 → 임베딩
- 개선: 1차 분류의 **summary를 바로 임베딩** → LLM 호출 1단계 절감
- 검증 필요: summary 임베딩 vs 별도 서술 임베딩 품질 비교

### 2-2. subtag 격하
- subtag는 **tags 추출 보조용**으로만 유지
- 분류 단위로서의 역할은 벡터 클러스터링이 완전 대체
- 주간 리포트 통계에서 subtag 비율은 참고용으로만

### 2-3. 마스터 비판 클러스터링 추가
- `마스터 반응` topic 중 `비판·불신` subtag를 같은 방식으로 클러스터링
- 부정 여론 조기 감지 목적
- 임베딩 비용 미미 (Titan, 추가 ~1,500건)

## Phase 3: 주간 파이프라인 통합 (이후)

### 목표
매주 리포트 생성 시 자동으로:
1. BigQuery 조회 + 1차 분류 + BigQuery 업로드
2. 피드백 클러스터링 + 검수
3. 기술이슈 엑셀 생성 + 슬랙 전송
4. 주간 리포트 생성 + 노션/슬랙 전송

### 이전 주와 비교
- 이전 주 클러스터 임베딩을 ChromaDB에 저장
- 이번 주 새 건이 기존 클러스터에 매칭되는지 비교
- 반복 이슈 감지, 신규 이슈 하이라이트

## Phase 4: Jira API 연동 (이후)
- 기술이슈/기능요청 클러스터별 자동 티켓 생성
- 기존 티켓과 중복 체크

## Phase 5: Google Sheets / 외부 연동 (이후)
- OAuth로 개인 계정 Google Sheets 자동 출력
- MCP 서버 또는 gspread 라이브러리
