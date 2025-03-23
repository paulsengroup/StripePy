#!/usr/bin/env bash

# Copyright (c) 2024 Roberto Rossini <roberros@uio.no>
#
# SPDX-License-Identifier: MIT

set -e
set -u
set -o pipefail

ARGC=$#

if [ $ARGC -gt 2 ]; then
  1>&2 echo "Usage:   $0 [platform]"
  1>&2 echo "Example: $0 linux/amd64"
  1>&2 echo "Example: $0 linux/amd64,linux/arm64"
  exit 1
fi

IMAGE_NAME='stripepy'

if [ "$(uname)" == "Darwin" ]; then
  BUILD_USER="$USER"
else
  BUILD_USER='root'
fi

if [ $ARGC -eq 1 ]; then
  PLATFORM="$1"
else
  PLATFORM="$(sudo -u "$BUILD_USER" docker info --format '{{ .OSType }}/{{ .Architecture }}')"
fi

venv_dir="$(mktemp -d)"

# shellcheck disable=SC2064
trap "rm -rf '$venv_dir'" EXIT

python3 -m venv "$venv_dir"

"$venv_dir/bin/pip" install hatchling hatch_vcs
VERSION="$("$venv_dir/bin/hatchling" version)"

GIT_HASH="$(git rev-parse HEAD)"
GIT_SHORT_HASH="$(git rev-parse --short HEAD)"
CREATION_DATE="$(date -I)"

if [[ $(git status --porcelain -uno) ]]; then
  GIT_IS_DIRTY=1
else
  GIT_IS_DIRTY=0
fi

IMAGE_TAG="sha-$GIT_SHORT_HASH"
if [ $GIT_IS_DIRTY -ne 0 ]; then
  IMAGE_TAG+='-dirty'
fi

BASE_IMAGE='docker.io/library/python:3.12.7'
1>&2 echo "Building \"$IMAGE_NAME:$IMAGE_TAG\" (stripepy v$VERSION) for platform $PLATFORM..."

sudo -u "$BUILD_USER" docker pull "$BASE_IMAGE"
BASE_IMAGE_DIGEST="$(sudo -u "$BUILD_USER" docker inspect --format='{{index .RepoDigests 0}}' "$BASE_IMAGE" | cut -f 2 -d '@')"

sudo -u "$BUILD_USER" docker buildx build --platform "$PLATFORM" --load \
  --build-arg "BASE_IMAGE=$BASE_IMAGE" \
  --build-arg "BASE_IMAGE_DIGEST=$BASE_IMAGE_DIGEST" \
  --build-arg "GIT_HASH=$GIT_HASH" \
  --build-arg "CREATION_DATE=$CREATION_DATE" \
  --build-arg "VERSION=$VERSION" \
  -t "$IMAGE_NAME:latest" \
  -t "$IMAGE_NAME:$(echo "$CREATION_DATE" | tr -d '\-' )" \
  -t "$IMAGE_NAME:$IMAGE_TAG" \
  "$(git rev-parse --show-toplevel)"

 # sudo singularity build -F "${img_name}_v${ver}.sif" \
 #                           "docker-daemon://${img_name}:${ver}"
