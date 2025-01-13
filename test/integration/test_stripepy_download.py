# Copyright (C) 2024 Roberto Rossini <roberroso@uio.no>
#
# SPDX-License-Identifier: MIT

import contextlib
import hashlib
import io
import json
import pathlib

import pytest

from stripepy import main


def _hash_file(path: pathlib.Path) -> str:
    hasher = hashlib.md5()
    hasher.update(path.read_bytes())
    return hasher.hexdigest()


@pytest.mark.end2end
class TestStripePyDownload:
    @staticmethod
    def test_list_only():
        args = ["download", "--list-only"]
        buff = io.StringIO()
        with contextlib.redirect_stdout(buff):
            main(args)

        data = json.loads(buff.getvalue())
        assert len(data) != 0

    @staticmethod
    def test_download_by_name(tmpdir):
        dest = pathlib.Path(tmpdir) / "out"

        args = ["download", "--name", "__results_v1", "--output", str(dest), "--include-private"]
        main(args)

        assert dest.is_file()

        assert _hash_file(dest) == "172872e8de9f35909f87ff33c185a07b"

    @staticmethod
    def test_download_random(tmpdir):
        dest = pathlib.Path(tmpdir) / "out"

        args = ["download", "--max-size", "2", "--output", str(dest), "--include-private"]
        main(args)

        assert dest.is_file()

        assert _hash_file(dest) == "adf60f386521f70b24936e53a6d11eab"
