#!/usr/bin/env bash

# Copyright (c) 2024 Roberto Rossini <roberros@uio.no>
#
# SPDX-License-Identifier: MIT

set -eu
set -o pipefail

if [ $# -ne 1 ]; then
  1>&2 echo "Usage: $0 stripepy:latest"
  exit 1
fi

for cmd in curl docker md5sum; do
  if ! command -v "$cmd" > /dev/null 2>&1; then
    1>&2 echo "Unable to find $cmd in your PATH"
    1>&2 echo 'Please install curl, docker, and md5sum before running this script'
    exit 1
  fi
done

IMG="$1"

tmpdir="$(mktemp -d)"
# shellcheck disable=SC2064
trap "rm -rf '$tmpdir'" EXIT

TEST_DATASET='4DNFI9GMP2J8'
TEST_DATASET_URL='https://zenodo.org/records/14283922/files/4DNFI9GMP2J8.stripepy.mcool?download=1'
TEST_DATASET_MD5='a17d08460c03cf6c926e2ca5743e4888'

if [ -f "$TEST_DATASET.mcool" ]; then
  1>&2 echo "Copying test dataset to \"$tmpdir\"..."
  cp "$TEST_DATASET.mcool" "$tmpdir/$TEST_DATASET.mcool"
else
  1>&2 echo "Test dataset \"$TEST_DATASET\" not found"
  1>&2 echo "Downloading test dataset to \"$tmpdir\"..."
  curl -L "$TEST_DATASET_URL" -o "$tmpdir/$TEST_DATASET.mcool"
fi

echo "$TEST_DATASET_MD5  $tmpdir/$TEST_DATASET.mcool" > "$tmpdir/checksum.md5"
md5sum -c "$tmpdir/checksum.md5"

cat > "$tmpdir/runme.sh" <<- 'EOM'

set -eu

whereis -b stripepy
stripepy --version

mkdir /tmp/stripepy
cd /tmp/stripepy

TEST_DATASET="$1"

1>&2 echo '### testing stripepy call...'
stripepy call \
  "$TEST_DATASET" \
  20000 \
  -o out.hdf5 \
  --log-file out.log \
  --plot-dir plots/ \
  --roi middle \
  --nproc "$(nproc)"

ok=true

if [ ! -f out.hdf5 ]; then
  ok=false
  1>&2 echo 'out.hdf5 is missing!'
fi
if [ ! -f out.log ]; then
  ok=false
  1>&2 echo 'out.log is missing!'
fi
if [ ! -d plots/ ]; then
  ok=false
  1>&2 echo 'plots/ is missing!'
fi

if [ "$ok" != true ]; then
  1>&2 echo "### FAILURE!"
  exit 1
fi
1>&2 echo "### stripepy call: SUCCESS!"

1>&2 echo '### testing stripepy view...'
num_stripes="$(stripepy view out.hdf5 | wc -l)"

if [ "$num_stripes" -ne 7596 ]; then
  1>&2 echo "### FAILURE!"
  exit 1
fi
1>&2 echo "### stripepy view: SUCCESS!"

1>&2 echo '### testing stripepy plot...'
stripepy plot cm \
  "$TEST_DATASET" \
  20000 \
  out.png \
  --stripepy-hdf5 out.hdf5 \
  --highlight-stripes

if [ ! -f out.png ]; then
  1>&2 echo 'out.png is missing!'
  1>&2 echo "### FAILURE!"
  exit 1
fi
1>&2 echo "### stripepy plot: SUCCESS!"
1>&2 echo "### SUCCESS!"

EOM

chmod 755 "$tmpdir/runme.sh"

if [ "$(uname)" == "Darwin" ]; then
  DOCKER_USER="$USER"
else
  DOCKER_USER='root'
fi


sudo -u "$DOCKER_USER" docker run --rm --entrypoint=/bin/bash \
  -v "$tmpdir/runme.sh:/tmp/runme.sh:ro" \
  -v "$tmpdir/$TEST_DATASET.mcool:/data/$TEST_DATASET.mcool:ro" \
  "$IMG" \
  /tmp/runme.sh "/data/$TEST_DATASET.mcool"
