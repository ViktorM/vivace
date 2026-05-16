#!/usr/bin/env bash
# Tag the locally-built vivace:test image and push it to GHCR.
#
# Usage:
#   docker/push.sh v0.1.0                  # push :v0.1.0 and :<git-sha>
#   docker/push.sh v0.1.0 --latest         # also push :latest
#   docker/push.sh v0.1.0 --yes            # skip the uncommitted-changes prompt
#                                          #   (for unattended build && push chains)
#   docker/push.sh v0.1.0 --latest --yes
#   GH_USER=foo docker/push.sh v0.1.0      # different GitHub user
#
# Requires: vivace:test built locally, `docker login ghcr.io` already done.
set -euo pipefail

VERSION="${1:?Usage: $0 <version> [--latest] [--yes]  (e.g. $0 v0.1.0)}"
PUSH_LATEST=false
ASSUME_YES=false
for arg in "${@:2}"; do
    case "$arg" in
        --latest) PUSH_LATEST=true ;;
        --yes|-y) ASSUME_YES=true ;;
        *) echo "Unknown flag: $arg" >&2; exit 2 ;;
    esac
done

# GHCR image paths are lowercase-only, regardless of GitHub username casing.
# Force lowercase to survive env vars set for `docker login` (which is case-sensitive).
GH_USER="$(echo "${GH_USER:-viktorm}" | tr '[:upper:]' '[:lower:]')"
IMAGE="ghcr.io/${GH_USER}/vivace"
GITSHA=$(git rev-parse --short HEAD)

if ! git diff --quiet HEAD; then
    echo "Warning: working tree has uncommitted changes."
    echo "  ${IMAGE}:${GITSHA} will claim to be commit ${GITSHA}, but the build"
    echo "  context included un-committed edits."
    if $ASSUME_YES; then
        echo "  --yes given, continuing anyway."
    else
        read -rp "Continue? [y/N] " ans
        [[ "$ans" == "y" || "$ans" == "Y" ]] || { echo "Aborted."; exit 1; }
    fi
fi

if ! docker image inspect vivace:test >/dev/null 2>&1; then
    echo "Error: vivace:test not found locally. Build it first:"
    echo "  docker build -f docker/Dockerfile -t vivace:test ."
    exit 1
fi

docker tag vivace:test "${IMAGE}:${VERSION}"
docker tag vivace:test "${IMAGE}:${GITSHA}"
$PUSH_LATEST && docker tag vivace:test "${IMAGE}:latest"

docker push "${IMAGE}:${VERSION}"
docker push "${IMAGE}:${GITSHA}"
$PUSH_LATEST && docker push "${IMAGE}:latest"

echo
echo "Pushed:"
echo "  ${IMAGE}:${VERSION}"
echo "  ${IMAGE}:${GITSHA}"
$PUSH_LATEST && echo "  ${IMAGE}:latest"
