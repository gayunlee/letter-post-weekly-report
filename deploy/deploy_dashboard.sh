#!/bin/bash
# VOC 대시보드 Cloud Run Service 배포
# 사용: bash deploy/deploy_dashboard.sh

set -euo pipefail

PROJECT_ID="${PROJECT_ID:-us-service-data}"
REGION="${REGION:-asia-northeast3}"
REPO_NAME="voc-pipeline"
SERVICE_NAME="voc-dashboard"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${SERVICE_NAME}:latest"
SA_EMAIL="voc-pipeline-sa@${PROJECT_ID}.iam.gserviceaccount.com"

gcloud config set project "${PROJECT_ID}"

# 1. 대시보드 이미지 빌드 (별도 Dockerfile 사용)
echo ">> 대시보드 이미지 빌드"
gcloud builds submit \
    --tag="${IMAGE}" \
    --machine-type=E2_HIGHCPU_8 \
    --file=dashboard/Dockerfile \
    .

# 2. Cloud Run Service 배포
echo ">> Cloud Run Service 배포"
gcloud run deploy "${SERVICE_NAME}" \
    --image="${IMAGE}" \
    --region="${REGION}" \
    --service-account="${SA_EMAIL}" \
    --set-env-vars="BIGQUERY_PROJECT_ID=${PROJECT_ID},PYTHONUNBUFFERED=1" \
    --memory=1Gi \
    --cpu=1 \
    --port=8080 \
    --min-instances=0 \
    --max-instances=3 \
    --timeout=300 \
    --allow-unauthenticated   # 운영 시작 후 IAP 로 전환 권장

URL=$(gcloud run services describe "${SERVICE_NAME}" --region="${REGION}" --format="value(status.url)")

echo
echo "=== 대시보드 배포 완료 ==="
echo "URL: ${URL}"
echo
echo "⚠️  현재 --allow-unauthenticated 상태. 운영 전에 IAP 또는 IAM 인증 적용 권장:"
echo "   gcloud run services update ${SERVICE_NAME} --region=${REGION} --no-allow-unauthenticated"
echo "   gcloud run services add-iam-policy-binding ${SERVICE_NAME} \\"
echo "       --region=${REGION} --member=domain:YOUR_DOMAIN.com --role=roles/run.invoker"
