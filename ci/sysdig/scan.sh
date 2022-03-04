#!/bin/bash

set -euo pipefail

SYSDIG_URL="https://us2.app.sysdig.com"
SYSDIG_TOKEN=
SYSDIG_IMAGE_NAME=

usage() { echo "Usage: $0 -t <SYSDIG_TOKEN> <SYSDIG_IMAGE_NAME>" 1>&2; exit 1; }

while getopts ":t:" o; do
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

echo "Scanning ${SYSDIG_IMAGE_NAME}"
docker run --rm \
    -v /var/run/docker.sock:/var/run/docker.sock \
    --env http_proxy="${http_proxy:-""}" \
    --env https_proxy="${https_proxy:-""}" \
    --env no_proxy="${no_proxy:-""}" \
    quay.io/sysdig/secure-inline-scan:2 \
    --sysdig-url "${SYSDIG_URL}" \
    --sysdig-token "${SYSDIG_TOKEN}" \
    --storage-type docker-daemon \
    --storage-path /var/run/docker.sock \
    "${SYSDIG_IMAGE_NAME}"

echo "Done"
