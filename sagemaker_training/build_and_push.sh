#!/usr/bin/env bash
set -e

if [ -z "$1" ]; then
  echo "Usage: $0 <image-name>"
  exit 1
fi
IMAGE_NAME=$1
REGION=${AWS_DEFAULT_REGION:-us-east-1}
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
FULLNAME="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${IMAGE_NAME}:latest"

# create repo if not exists
aws ecr describe-repositories --repository-names "${IMAGE_NAME}" >/dev/null 2>&1 || aws ecr create-repository --repository-name "${IMAGE_NAME}"

# login
aws ecr get-login-password --region ${REGION} | docker login --username AWS --password-stdin ${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com

# build & push
docker build -t ${IMAGE_NAME} .
docker tag ${IMAGE_NAME} ${FULLNAME}
docker push ${FULLNAME}

echo "Image pushed: ${FULLNAME}"