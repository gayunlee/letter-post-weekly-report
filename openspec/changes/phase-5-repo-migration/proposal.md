# Proposal: Phase 5 — Git 레포지토리 us-web 조직 이전

## Intent

현재 개인 GitHub (`gayunlee/letter-post-weekly-report`) 에 있는 운영 코드를
회사 조직 (`us-web/<repo>`) 으로 통째 이전. 히스토리 + 브랜치 + 태그 보존.

## 동기

- 운영 인프라가 회사 GCP (`us-service-data`) 에서 돌아감
- 코드도 회사 자산으로 정리 (퇴사/인수인계 대비)
- 회사 organization secrets/branch protection 활용 가능
- 다른 팀원 협업 가능

## 이전 방식: Mirror clone (히스토리 100% 보존)

```bash
# 1. us-web 조직에 빈 레포 생성 (이름 결정 필요)
gh repo create us-web/<NEW_NAME> --private

# 2. mirror clone 후 push
cd /tmp
git clone --mirror git@github.com-personal:gayunlee/letter-post-weekly-report.git
cd letter-post-weekly-report.git
git remote set-url origin git@github.com-work:us-web/<NEW_NAME>.git
git push --mirror

# 3. 로컬 작업 디렉토리의 origin 갈아치우기
cd ~/Documents/ai/letter-post-weekly-report
git remote set-url origin git@github.com-work:us-web/<NEW_NAME>.git
git fetch origin
```

## 결정 필요

- **레포 이름**:
  - `letter-post-weekly-report` (현재 이름 그대로) — 검색성, 기존 링크 호환
  - `voc-pipeline` (운영 친화적) — 현재 GCP 리소스 이름과 일치
  - 기타
- **기존 개인 레포 처리**:
  - archive (읽기 전용 보존)
  - delete (완전 정리)
- **collaborators**: 누구에게 접근 권한 줄지

## 수동 정리 항목 (이전 후)

- [ ] GitHub Actions secrets (있다면 — 현재는 없음)
- [ ] Branch protection rules (main 보호 등)
- [ ] Webhook (있다면)
- [ ] CLAUDE.md 의 "Git user: gayunlee" 등 개인 정보 흔적 검토
- [ ] README/docs 의 레포 URL 참조 갱신
- [ ] 로컬 다른 머신에서 작업 중이면 origin 갱신

## 영향 없음

- GCP 인프라 (us-service-data) — 코드 위치와 무관
- BigQuery 데이터 — 무관
- 배포된 Cloud Run Jobs/Service — 이미 이미지 빌드됨, 다음 빌드부터 새 레포 사용
- Slack/Notion 통합 — 무관
