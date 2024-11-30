#!/usr/bin/env bash

# Copyright (c) 2024 Roberto Rossini <roberros@uio.no>
#
# SPDX-License-Identifier: MIT

set -eu
set -o pipefail

if [ $# -ne 1 ]; then
  2>&1 echo "Usage: $0 stipepy:latest"
  exit 1
fi

for cmd in curl docker md5sum; do
  if ! command -v "$cmd" 2>&1 > /dev/null; then
    1>&2 echo "Unable to find $cmd in your PATH"
    1>&2 echo 'Please install curl, docker, and md5sum before running this script'
    exit 1
  fi
done

IMG="$1"

tmpdir="$(mktemp -d)"
trap "rm -rf '$tmpdir'" EXIT

TEST_DATASET='4DNFIOTPSS3L'
TEST_DATASET_URL='https://4dn-open-data-public.s3.amazonaws.com/fourfront-webprod/wfoutput/7386f953-8da9-47b0-acb2-931cba810544/4DNFIOTPSS3L.hic'
TEST_DATASET_MD5='d8b030bec6918bfbb8581c700990f49d'

if [ -f "$TEST_DATASET.hic" ]; then
  1>&2 echo "Copying test dataset to \"$tmpdir\"..."
  cp "$TEST_DATASET.hic" "$tmpdir/$TEST_DATASET.hic"
else
  1>&2 echo "Test dataset \"$TEST_DATASET\" not found"
  1>&2 echo "Downloading test dataset to \"$tmpdir\"..."
  curl -L "$TEST_DATASET_URL" -o "$tmpdir/$TEST_DATASET.hic"
fi

echo "$TEST_DATASET_MD5  $tmpdir/$TEST_DATASET.hic" > "$tmpdir/checksum.md5"
md5sum -c "$tmpdir/checksum.md5"

cat > "$tmpdir/runme.sh" <<- 'EOM'

set -eu

whereis -b stripepy
stripepy --version

mkdir /tmp/stripepy
cd /tmp/stripepy

TEST_DATASET="$1"

stripepy call \
  "$TEST_DATASET" \
  100000 \
  -o stripepy/ \
  --glob-pers-min 0.10 \
  --loc-pers-min 0.33 \
  --loc-trend-min 0.25 \
  --roi middle \
  --nproc "$(nproc)"

find stripepy/ -type f -exec ls -lah {} +

if [ ! -f stripepy/*/*/results.hdf5 ]; then
  1>&2 echo 'results.hdf5 is missing!'
  exit 1
fi

EOM

chmod 755 "$tmpdir/runme.sh"

if [ "$(uname)" == "Darwin" ]; then
  DOCKER_USER="$USER"
else
  DOCKER_USER='root'
fi


sudo -u "$DOCKER_USER" docker run --rm --entrypoint=/bin/bash \
  -v "$tmpdir/runme.sh:/tmp/runme.sh:ro" \
  -v "$tmpdir/$TEST_DATASET.hic:/data/$TEST_DATASET.hic:ro" \
  "$IMG" \
  /tmp/runme.sh "/data/$TEST_DATASET.hic"
