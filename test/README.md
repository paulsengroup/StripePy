<!--
Copyright (C) 2024 Roberto Rossini <roberros@uio.no>

SPDX-License-Identifier: MIT
-->

# StripePy test instructions

The instructions in this README assume that StripePy and pytest have been installed in a virtual environment named `venv`.

The provided commands should work on any UNIX system. To run the test suites on Windows simply replace `venv/bin/` with `venv\Scripts\`.

## Downloading test files

```bash
venv/bin/stripepy download --unit-test
venv/bin/stripepy download --end2end
```

## Running the unit tests

```console
user@dev:/tmp/StripePy$ venv/bin/pytest -v -m unit

============================================== test session starts ==============================================
platform linux -- Python 3.13.1, pytest-8.3.4, pluggy-1.5.0 -- /tmp/StripePy/venv/bin/python3
cachedir: .pytest_cache
rootdir: /tmp/StripePy
configfile: pyproject.toml
plugins: cov-6.0.0
collected 31 items / 12 deselected / 19 selected

test/unit/test_IO.py::test_folders_for_plots PASSED                                                        [  5%]
test/unit/test_IO.py::TestRemoveAndCreateFolder::test_create_new_folder PASSED                             [ 10%]
test/unit/test_IO.py::TestRemoveAndCreateFolder::test_overwrite_existing_folder PASSED                     [ 15%]
test/unit/test_IO.py::test_create_folders_for_plots PASSED                                                 [ 21%]
test/unit/test_IO.py::TestResult::test_ctor PASSED                                                         [ 26%]
test/unit/test_IO.py::TestResult::test_setters PASSED                                                      [ 31%]
test/unit/test_IO.py::TestResult::test_getters PASSED                                                      [ 36%]
test/unit/test_IO.py::TestResult::test_stripe_getters PASSED                                               [ 42%]
test/unit/test_IO.py::TestResultFile::test_ctor PASSED                                                     [ 47%]
test/unit/test_IO.py::TestResultFile::test_properties PASSED                                               [ 52%]
test/unit/test_IO.py::TestResultFile::test_getters PASSED                                                  [ 57%]
test/unit/test_IO.py::TestResultFile::test_file_creation PASSED                                            [ 63%]
test/unit/test_others.py::TestOpenMatrixFileChecked::test_invalid_paths PASSED                             [ 68%]
test/unit/test_others.py::TestOpenMatrixFileChecked::test_invalid_formats PASSED                           [ 73%]
test/unit/test_others.py::TestOpenMatrixFileChecked::test_invalid_resolutions PASSED                       [ 78%]
test/unit/test_others.py::TestOpenMatrixFileChecked::test_valid_files PASSED                               [ 84%]
test/unit/test_stripepy.py::TestLogTransform::test_empty PASSED                                            [ 89%]
test/unit/test_stripepy.py::TestLogTransform::test_all_finite PASSED                                       [ 94%]
test/unit/test_stripepy.py::TestLogTransform::test_with_nans PASSED                                        [100%]

======================================= 19 passed, 12 deselected in 0.93s =======================================
```

## Running the integration tests

<!--
# To re-center the header/footer message

```python
x = " 5 passed, 11 deselected in 55.71s "
x.center(113, "=")
```
-->

```console
user@dev:/tmp/StripePy$ venv/bin/pytest -v -m end2end

============================================== test session starts ==============================================
platform linux -- Python 3.13.1, pytest-8.3.4, pluggy-1.5.0 -- /tmp/StripePy/venv/bin/python3
cachedir: .pytest_cache
rootdir: /tmp/StripePy
configfile: pyproject.toml
plugins: cov-6.0.0
collected 31 items / 19 deselected / 12 selected

test/integration/test_stripepy_call.py::TestStripePyCall::test_stripepy_call PASSED                        [  8%]
test/integration/test_stripepy_download.py::TestStripePyDownload::test_list_only PASSED                    [ 16%]
test/integration/test_stripepy_download.py::TestStripePyDownload::test_download_by_name PASSED             [ 25%]
test/integration/test_stripepy_download.py::TestStripePyDownload::test_download_random PASSED              [ 33%]
test/integration/test_stripepy_plot.py::TestStripePyPlot::test_contact_map PASSED                          [ 41%]
test/integration/test_stripepy_plot.py::TestStripePyPlot::test_contact_map_with_seeds PASSED               [ 50%]
test/integration/test_stripepy_plot.py::TestStripePyPlot::test_contact_map_with_stripes PASSED             [ 58%]
test/integration/test_stripepy_plot.py::TestStripePyPlot::test_contact_map_with_stripes_no_heights PASSED  [ 66%]
test/integration/test_stripepy_plot.py::TestStripePyPlot::test_pseudodistribution PASSED                   [ 75%]
test/integration/test_stripepy_plot.py::TestStripePyPlot::test_stripe_hist PASSED                          [ 83%]
test/integration/test_stripepy_plot.py::TestStripePyPlot::test_stripe_hist_gw PASSED                       [ 91%]
test/integration/test_stripepy_view.py::TestStripePyView::test_view PASSED                                 [100%]
======================================= 12 passed, 19 deselected in 53.73s ======================================
```

## For developers

If you need to collect coverage information use the following

```bash
venv/bin/pytest -v --cov --cov-report term --cov-report html -m unit
venv/bin/pytest -v --cov --cov-report term --cov-report html --cov-append -m end2end
```

The HTML coverage will be located under `coverage/html/index.html`.
