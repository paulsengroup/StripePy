import hictkpy as htk
import pandas as pd


class HiCObject:
    def __init__(self):
        self._path = ""
        self._resolution = 0
        self._chromosomes = {}
        self._attributes = {}
        self._normalization = ""
        self._region_of_interest = ""
        self._nnz = 0
        self._sum = 0

    @property
    def path(self):
        return self._path

    @path.setter
    def path(self, new_path):
        self._path = new_path
        if self._resolution:
            self._set_file()

    @property
    def resolution(self):
        return self._resolution

    @resolution.setter
    def resolution(self, new_resolution):
        self._resolution = new_resolution
        if self._path:
            self._set_file()

    @property
    def file(self):
        return self.f

    @property
    def normalization(self):
        return self._normalization

    @normalization.setter
    def normalization(self, algorithm):
        self._normalization = algorithm
        if self.region_of_interest:
            self.sel = self._fetch()

    @property
    def region_of_interest(self):
        return self._region_of_interest

    @region_of_interest.setter
    def region_of_interest(self, region):
        self._region_of_interest = region
        if self._normalization:
            self.sel = self._fetch()

    @property
    def frame(self):
        return pd.DataFrame(self.sel.to_numpy())

    @property
    def selector(self):
        selector = self.sel.to_numpy()
        return pd.DataFrame(selector)

    @property
    def nnz(self):
        return self._nnz

    @property
    def sum(self):
        return self._sum

    @property
    def attributes(self):
        return self._attributes

    def _set_file(self):
        self.f = htk.File(self._path, self._resolution)

        self._chromosomes = self.f.chromosomes()
        self._attributes = self.f.attributes()

    def _fetch(self):
        self.sel = self.f.fetch(self._region_of_interest, join=True, normalization=self._normalization)

        self._nnz = self.sel.nnz()
        self._sum = self.sel.sum()
