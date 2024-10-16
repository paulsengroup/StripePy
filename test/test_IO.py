import os
import pathlib
import shutil
import sys

import pytest

sys.path.insert(0, "./utils")


class TestIO:

    directory_name = 0
    while os.path.exists(str(directory_name)):
        directory_name += 1
    directory_name = str(directory_name)

    from collections import defaultdict

    errors = defaultdict(int)

    def _find_directory(self, dir) -> bool:
        return os.path.exists(dir)

    def designated_directory_exists(self) -> bool:
        return self._find_directory(self.directory_name)

    def test_list_folders_for_plots(self):
        # TODO: Add invalid characters in string cycle.
        from IO import list_folders_for_plots

        invalid_types_list = [int, float, bool]
        for types in invalid_types_list:
            with pytest.raises(TypeError) as exceptionFromOne:
                list_folders_for_plots(types(1))
            assert "argument should be a str" in str(exceptionFromOne.value)
            exceptionFromOne = None

            with pytest.raises(TypeError) as exceptionFromZero:
                list_folders_for_plots(types(0))
            assert "argument should be a str" in str(exceptionFromZero.value)
            exceptionFromZero = None
        assert list_folders_for_plots("1"), "Valid data type not accepted in test_list_folders_for_plots"
        assert list_folders_for_plots("0"), "Valid data type not accepted in test_list_folders_for_plots"

        assert list_folders_for_plots("1") == [
            pathlib.Path("1"),
            pathlib.Path("1") / "1_preprocessing",
            pathlib.Path("1") / "2_TDA",
            pathlib.Path("1") / "3_shape_analysis",
            pathlib.Path("1") / "4_biological_analysis",
            pathlib.Path("1") / "3_shape_analysis" / "local_pseudodistributions",
        ]
        assert len(list_folders_for_plots("1")) == 6
        return

    def test_remove_and_create_folder(self):
        from IO import remove_and_create_folder

        # Create directory
        assert not (self.designated_directory_exists())
        remove_and_create_folder(self.directory_name)

        # Directory already exists
        # TODO: Expand create-delete cycle when test object is given decision input
        assert self.designated_directory_exists()
        remove_and_create_folder(self.directory_name)

        # Remove directory
        assert self.designated_directory_exists()
        shutil.rmtree(self.directory_name)
        return

    def test_create_folders_for_plots(self):
        from IO import create_folders_for_plots

        assert not self.designated_directory_exists()
        result = create_folders_for_plots(self.directory_name)
        assert isinstance(result, list)

        assert self.designated_directory_exists()
        shutil.rmtree(self.directory_name)
        assert not self.designated_directory_exists()
        return

    def test_format_ticks(self):
        pass

    def test_HiC(self):
        pass

    def test_pseudodistrib(self):
        pass

    def test_pseudodistrib_and_HIoIs(self):
        pass

    def test_HiC_and_sites(self):
        pass

    def test_HiC_and_HIoIs(self):
        pass

    def test_plot_stripes(self):
        pass

    def test_plot_stripes_and_peaks(self):
        pass

    def test_save_candidates_bedpe(self):
        pass
