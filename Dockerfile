# Copyright (C) 2024 Roberto Rossini <roberros@uio.no>
#
# SPDX-License-Identifier: MIT

##### IMPORTANT #####
# This Dockerfile requires several build arguments to be defined through --build-arg
# See utils/devel/build_dockerfile.sh for an example of how to build this Dockerfile
#####################

ARG BASE_IMAGE
ARG BASE_IMAGE_DIGEST
FROM "${BASE_IMAGE}@${BASE_IMAGE_DIGEST}" AS builder

ARG GIT_HASH
ARG CREATION_DATE
ARG VERSION

RUN if [ -z "$GIT_HASH" ]; then echo "Missing GIT_HASH --build-arg" && exit 1; fi \
&&  if [ -z "$CREATION_DATE" ]; then echo "Missing CREATION_DATE --build-arg" && exit 1; fi \
&&  if [ -z "$VERSION" ]; then echo "Missing VERSION --build-arg" && exit 1; fi

ARG PYTHONDONTWRITEBYTECODE=1
ARG PIP_ROOT_USER_ACTION=ignore
ARG SETUPTOOLS_SCM_PRETEND_VERSION="$VERSION"
ARG src_dir='/root/stripepy'
ARG install_dir='/opt/stripepy'

COPY . "$src_dir/"

RUN python3 -m venv "$install_dir" \
&& "$install_dir/bin/pip" install "$src_dir" -v --no-compile


ARG BASE_IMAGE
ARG BASE_IMAGE_DIGEST
FROM "${BASE_IMAGE}@${BASE_IMAGE_DIGEST}" AS tester

ARG VERSION

ARG PYTHONDONTWRITEBYTECODE=1
ARG PIP_ROOT_USER_ACTION=ignore
ARG SETUPTOOLS_SCM_PRETEND_VERSION="$VERSION"
ARG src_dir='/root/stripepy'
ARG install_dir='/opt/stripepy'

COPY --from=builder "$src_dir" "$src_dir"
COPY --from=builder "$install_dir" "$install_dir"
COPY test/data/results_4DNFI9GMP2J8_v1.hdf5 "$src_dir/test/data/"

RUN "$install_dir/bin/pip" install "$src_dir[test]" -v --no-compile

RUN "$install_dir/bin/python3" -m pytest "$src_dir/test/" -v -m unit


ARG BASE_IMAGE
ARG BASE_IMAGE_DIGEST
FROM "${BASE_IMAGE}@${BASE_IMAGE_DIGEST}" AS base

ARG BASE_IMAGE
ARG BASE_IMAGE_DIGEST
ARG GIT_HASH
ARG CREATION_DATE
ARG VERSION

ARG src_dir='/root/stripepy'
ARG install_dir='/opt/stripepy'

COPY --from=builder "$install_dir" /opt/stripepy/
COPY --from=tester "$src_dir/LICENCE" /opt/stripepy/share/licenses/stripepy/LICENCE

WORKDIR /data
ENTRYPOINT ["/opt/stripepy/bin/stripepy"]
ENV PATH="$PATH:/opt/stripepy/bin"
ENV PYTHONDONTWRITEBYTECODE=1

RUN stripepy --help
RUN stripepy --version

# https://github.com/opencontainers/image-spec/blob/main/annotations.md#pre-defined-annotation-keys
LABEL org.opencontainers.image.authors='Andrea Raffo <andrea.raffo@ibv.uio.no>,Roberto Rossini <roberros@uio.no>'
LABEL org.opencontainers.image.url='https://github.com/paulsengroup/stripepy'
LABEL org.opencontainers.image.documentation='https://github.com/paulsengroup/stripepy'
LABEL org.opencontainers.image.source='https://github.com/paulsengroup/stripepy'
LABEL org.opencontainers.image.licenses='MIT'
LABEL org.opencontainers.image.title='StripePy'
LABEL org.opencontainers.image.description='StripePy recognizes architectural stripes in 3C and Hi-C contact maps using geometric reasoning'
LABEL org.opencontainers.image.base.digest="$BASE_IMAGE_DIGEST"
LABEL org.opencontainers.image.base.name="$BASE_IMAGE"

LABEL org.opencontainers.image.revision="$GIT_HASH"
LABEL org.opencontainers.image.created="$CREATION_DATE"
LABEL org.opencontainers.image.version="$VERSION"
