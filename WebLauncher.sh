#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="tiebreaker-api"
CONTAINER_NAME="tiebreaker-api"
HOST_PORT="8000"
CONTAINER_PORT="8000"

docker build -t "${IMAGE_NAME}" .

if docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  docker rm -f "${CONTAINER_NAME}" >/dev/null
fi

docker run --name "${CONTAINER_NAME}" --rm -p "${HOST_PORT}:${CONTAINER_PORT}" "${IMAGE_NAME}"
