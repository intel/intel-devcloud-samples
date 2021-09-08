#!/bin/bash

set -euo pipefail

SYSDIG_URL="https://us2.app.sysdig.com"
SYSDIG_TOKEN=
SYSDIG_IMAGE_NAME=

usage() { echo "Usage: $0 -t <SYSDIG_TOKEN> <SYSDIG_IMAGE_NAME>" 1>&2; exit 1; }

while getopts ":t:a" o; do
    case "${o}" in
        t)
            SYSDIG_TOKEN=${OPTARG}
            ;;
        *)
            usage
            ;;
    esac
done
shift $((OPTIND-1))

if [ -z "${SYSDIG_TOKEN}" ] || [ -z "${1:-}" ]; then
    usage
fi

SYSDIG_IMAGE_NAME=${1}

echo "Creating build directory"
mkdir -p build/reports

echo "Generating docker-archive of ${SYSDIG_IMAGE_NAME}"
docker save "${SYSDIG_IMAGE_NAME}" -o build/image.tar

echo "Scanning ${SYSDIG_IMAGE_NAME}"
docker run --rm \
    -v "${PWD}"/build/image.tar:/tmp/image.tar \
    -v "${PWD}"/build/reports:/tmp/reports \
    --env http_proxy="${http_proxy:-""}" \
    --env https_proxy="${https_proxy:-""}" \
    --env no_proxy="${no_proxy:-""}" \
    quay.io/sysdig/secure-inline-scan:2 \
    --sysdig-url "${SYSDIG_URL}" \
    --sysdig-token "${SYSDIG_TOKEN}" \
    --report-folder /tmp/reports \
    --storage-type docker-archive \
    --storage-path /tmp/image.tar \
    "${SYSDIG_IMAGE_NAME}" | tee ./build/reports/scan.log

echo "Done"
