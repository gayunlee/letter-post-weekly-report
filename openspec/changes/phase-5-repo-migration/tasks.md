# Tasks: Phase 5 — Git 레포 이전

## Gayoon 결정 필요

- [ ] **레포 이름** 결정 (`letter-post-weekly-report` vs `voc-pipeline` vs 기타)
- [ ] **기존 개인 레포 처리** (archive vs delete)
- [ ] **추가 collaborators** 결정

## 실행 단계

- [ ] us-web 조직에 빈 레포 생성 (`gh repo create us-web/<NAME> --private`)
- [ ] mirror clone + push (히스토리 보존)
- [ ] 로컬 origin 갱신
- [ ] 새 레포 README/CLAUDE.md 의 개인 정보 흔적 검토 (예: Git user)
- [ ] Branch protection rules 적용 (main 직접 push 금지 등)
- [ ] 기존 개인 레포 archive/delete

## 영향 없는 것 (확인용)

- GCP 인프라 (us-service-data) — 코드 위치와 무관
- BigQuery 데이터 — 무관
- 이미 배포된 Cloud Run Jobs/Service — 무관
- 다음 코드 변경 시부터 새 레포로 push, Cloud Build 트리거 재연결만 필요 (현재는 수동 build 라 무관)
