#!/usr/bin/env bash

set -euo pipefail

SOME_UUID=$(uuidgen)

# Increment version manually
VERSION="1.0.1"

# TODO: modify paths for your usecase
DOCKERFILE_PATH="YOUR/PATH/TO/gpu_telemetry"
BASE_IMAGE="YOUR/DOCKER/IMAGE/PATH/HERE"

TAG="${VERSION}-${USER}-${SOME_UUID}"
IMAGE_FULL="${BASE_IMAGE}:${TAG}"

DOCKER_BUILDKIT=1 docker build -f $DOCKERFILE_PATH/GPUTelemetry.Dockerfile -t $IMAGE_FULL $DOCKERFILE_PATH

echo $IMAGE_FULL
docker push $IMAGE_FULL

echo "New GPU Telemetry tag: $TAG"