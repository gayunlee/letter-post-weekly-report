#!/bin/bash
# .env 파일에서 시크릿 값을 읽어 Secret Manager 에 신규 버전으로 업로드
# 사용: bash deploy/create_secrets.sh
#
# 주의: .env 파일 경로는 프로젝트 루트 기준. 커밋되지 않음.

set -euo pipefail

PROJECT_ID="${PROJECT_ID:-us-service-data}"
ENV_FILE="${ENV_FILE:-.env}"

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "❌ ${ENV_FILE} 이 없습니다."
  exit 1
fi

gcloud config set project "${PROJECT_ID}"

# Secret Manager 이름:env 키 매핑 (macOS bash 3.2 호환 — 연관배열 미사용)
MAPPINGS=(
  "aws-access-key-id:AWS_ACCESS_KEY_ID"
  "aws-secret-access-key:AWS_SECRET_ACCESS_KEY"
  "aws-region:AWS_REGION"
  "notion-api-key:NOTION_API_KEY"
  "notion-database-id:NOTION_DATABASE_ID"
  "slack-bot-token:SLACK_BOT_TOKEN"
  "slack-channel-id:SLACK_CHANNEL_ID"
)

for entry in "${MAPPINGS[@]}"; do
  secret_name="${entry%%:*}"
  env_key="${entry##*:}"
  value="$(grep -E "^${env_key}=" "${ENV_FILE}" | head -1 | sed -E "s/^${env_key}=//; s/^['\"]//; s/['\"]$//")"
  if [[ -z "${value}" ]]; then
    echo "⚠️  ${env_key} 값이 비어있음 — 건너뜀"
    continue
  fi
  printf '%s' "${value}" | gcloud secrets versions add "${secret_name}" --data-file=- \
      && echo "  + ${secret_name}"
done

echo
echo "완료. 현재 시크릿 목록:"
gcloud secrets list --filter="name ~ (aws-|notion-|slack-)" \
    --format="table(name.basename(),createTime)"
