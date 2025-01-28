# Copyright (C) 2025 Roberto Rossini <roberros@uio.no>
#
# SPDX-License-Identifier: MIT

import ctypes
import gc
import multiprocessing as mp
import time
from typing import Optional, Union

import numpy as np
import numpy.typing as npt
import scipy.sparse as ss

from stripepy.utils.common import pretty_format_elapsed_time

# Generic SparseMatrix type used throughout the codebase
SparseMatrix = Union[ss.csr_matrix, ss.csc_matrix]


class _SharedTypedBuffer(object):
    """
    Wrapper class for mp.RawArray that supports efficient assignment and resize without re-allocation.
    """

    def __init__(
        self,
        data: Union[bytes, npt.NDArray],
        capacity: Optional[int] = None,
    ):
        if capacity is None:
            capacity = len(data)

        assert capacity >= len(data)

        self._dtype = bytes if isinstance(data, bytes) else data.dtype
        self._capacity = capacity
        self._size = mp.RawValue(ctypes.c_int, 0)

        if len(data) == 0:
            # Empty buffer optimization
            self._shared_buffer = None
        elif self._dtype == bytes:
            # Allocate a char buffer
            self._shared_buffer = mp.RawArray(ctypes.c_char, self._capacity)
        else:
            # Allocate a numeric buffer
            self._shared_buffer = mp.RawArray(np.ctypeslib.as_ctypes_type(self._dtype), self._capacity)

        self.assign(data)

    def __len__(self) -> int:
        return int(self._size.value)

    def can_assign(
        self,
        data: Union[bytes, npt.NDArray],
    ) -> bool:
        """
        Check whether the data can be assigned to the current buffer without re-allocation.
        """
        if len(data) > self._capacity:
            return False

        if isinstance(data, bytes):
            return self._dtype == bytes

        return data.dtype == self._dtype

    def assign(self, data: Union[bytes, npt.NDArray]):
        """
        Copy values from data into the shared buffer (shrinking the buffer if necessary).
        """
        assert self.can_assign(data)
        if len(data) == 0:
            self._resize_to_zero()
            return

        new_size = len(data)
        if self._dtype == bytes:
            self._shared_buffer[:new_size] = data
        else:
            dest = np.frombuffer(self._shared_buffer, dtype=self._dtype, count=new_size)
            np.copyto(dest, data, casting="safe")

        self.resize(new_size)

    def resize(self, new_size: int):
        """
        Resize underlying buffer without re-allocation.
        """
        if new_size == 0:
            self._resize_to_zero()
            return

        assert new_size <= self._capacity
        self._size.value = new_size

    @property
    def dtype(self) -> Union[np.dtype, type]:
        return self._dtype

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def data(self) -> Union[bytes, npt.NDArray]:
        """
        Get the underlying data as a read-only numpy array (without copying the data).
        If the data is an array of chars return it as a bytes object (with copy).
        """
        if self._dtype == bytes:
            return bytes(self._shared_buffer[: len(self)])
        v = np.frombuffer(self._shared_buffer, dtype=self._dtype, count=len(self))
        v.flags.writeable = False
        return v

    @property
    def raw_data(self) -> mp.RawArray:
        return self._shared_buffer

    def _resize_to_zero(self):
        self._size.value = 0


class _SharedTriangularSparseMatrixBase(object):  # noqa
    """
    Base class modeling a triangular sparse matrix stored in shared memory.
    """

    def __init__(
        self,
        chrom: Optional[str],
        m: Optional[SparseMatrix],
        logger=None,
        max_nnz: Optional[int] = None,
    ):
        assert isinstance(m, ss.csr_matrix) or isinstance(m, ss.csc_matrix) or m is None

        if chrom is None:
            # chrom should be None only if the object is to be initialized by a static function
            assert m is None
            assert max_nnz is None
            self._chrom = None
            self._data = None
            self._indices = None
            self._indptr = None
            self._shape = None
            self._matrix_type_str = None
            self._matrix_type = None
            return

        # hopefully nobody needs their chromosome names to be more than 1024 characters...
        capacity_chrom = max(1024, len(chrom))
        if max_nnz is None:
            capacity_data = len(m.data)
            capacity_indices = len(m.indices)
            capacity_indptr = len(m.indptr)
        else:
            spare_capacity_pct = max(1.0, max_nnz / m.nnz)
            capacity_data = int(np.ceil(len(m.data) * spare_capacity_pct))
            capacity_indices = int(np.ceil(len(m.indices) * spare_capacity_pct))
            capacity_indptr = int(np.ceil(len(m.indptr) * spare_capacity_pct))

        t0 = time.time()
        self._chrom = _SharedTypedBuffer(chrom.encode("ascii"), capacity_chrom)
        self._data = _SharedTypedBuffer(m.data, capacity_data)
        self._indices = _SharedTypedBuffer(m.indices, capacity_indices)
        self._indptr = _SharedTypedBuffer(m.indptr, capacity_indptr)
        self._shape = _SharedTypedBuffer(np.array(m.shape), 2)
        self._matrix_type_str = "CSR" if isinstance(m, ss.csr_matrix) else "CSC"
        self._matrix_type = type(m)

        if logger:
            logger.debug(
                "allocation and initialization of a %s matrix (%s; %d nnz) in shared memory took %s",
                self._matrix_type_str,
                chrom,
                m.nnz,
                pretty_format_elapsed_time(t0),
            )

    def get(self) -> SparseMatrix:
        """
        Get the sparse matrix without copying the data.
        """
        return self._matrix_type(
            (self._data.data, self._indices.data, self._indptr.data),
            shape=self._shape.data,
            copy=False,
        )

    def can_assign(self, m: SparseMatrix) -> bool:
        """
        Check whether the given sparse matrix can be assigned to the current matrix instance
        without incurring in shared memory re-allocation.
        """
        assert isinstance(m, self._matrix_type)

        return all(
            (
                self._data.can_assign(m.data),
                self._indices.can_assign(m.indices),
                self._indptr.can_assign(m.indptr),
            )
        )

    def assign(self, chrom: str, m: SparseMatrix, logger=None):
        """
        Assign the given SparseMatrix to the current matrix instance without re-allocations.
        """
        assert self.can_assign(m)

        chrom_ascii = chrom.encode("ascii")
        if len(chrom_ascii) > self._chrom.capacity:
            # in case the chromosome name is too long, just truncate it
            chrom_ascii = chrom_ascii[: self._chrom.capacity]

        t0 = time.time()
        self._chrom.assign(chrom_ascii)
        self._data.assign(m.data)
        self._indices.assign(m.indices)
        self._indptr.assign(m.indptr)
        self._shape.assign(np.array(m.shape))

        if logger:
            logger.debug(
                "assigning to a %s matrix (%s; %d nnz) in shared memory took %s",
                self._matrix_type_str,
                chrom,
                m.nnz,
                pretty_format_elapsed_time(t0),
            )

    @property
    def chrom(self) -> str:
        return bytes(self._chrom.data).decode("ascii")

    @property
    def shape(self) -> npt.NDArray[int]:
        return self._shape.data


class SharedTriangularCSRMatrix(_SharedTriangularSparseMatrixBase):
    """
    Class modeling a triangular CSR matrix stored in shared memory.
    """

    def __init__(
        self,
        chrom: str,
        m: ss.csr_matrix,
        logger=None,
        max_nnz: Optional[int] = None,
    ):
        """
        Parameters
        ----------
        chrom: str
            name of the chromosome to which the matrix refers to.
        m: ss.csr_matrix
            sparse matrix in CSR format.
        logger:
            logger
        max_nnz: Optional[int]
            when provided, use this number to attempt to allocate enough space to store a matrix
            with the given number of non-zero entries.
        """
        assert isinstance(m, ss.csr_matrix) or m is None
        super().__init__(chrom, m, logger, max_nnz)

    @property
    def T(self):  # noqa
        """
        Transpose the matrix without copying the data.

        Returns
        -------
        SharedTriangularCSCMatrix
            the transposed matrix in CSC format.
        """
        return SharedTriangularCSCMatrix._from_shared_buffers(  # noqa
            self._chrom,
            self._data,
            self._indices,
            self._indptr,
            self._shape,
        )

    @staticmethod
    def _from_shared_buffers(
        chrom: _SharedTypedBuffer,
        data: _SharedTypedBuffer,
        indices: _SharedTypedBuffer,
        indptr: _SharedTypedBuffer,
        shape: _SharedTypedBuffer,
    ):
        assert shape.data[0] == shape.data[1]  # noqa
        m = SharedTriangularCSRMatrix(None, None, None, None)  # noqa
        m._chrom = chrom
        m._data = data
        m._indices = indices
        m._indptr = indptr
        m._shape = shape
        m._matrix_type_str = "CSR"
        m._matrix_type = ss.csr_matrix

        # Assertion commented out for performance reasons
        # assert ss.triu(m.get(), k=1).sum() == 0 or ss.tril(m.get(), k=-1).sum() == 0

        return m


class SharedTriangularCSCMatrix(_SharedTriangularSparseMatrixBase):
    def __init__(
        self,
        chrom: str,
        m: ss.csc_matrix,
        logger=None,
        max_nnz: Optional[int] = None,
    ):
        """
        Parameters
        ----------
        chrom: str
            name of the chromosome to which the matrix refers to.
        m: ss.ccr_matrix
            sparse matrix in CSC format.
        logger:
            logger
        max_nnz: Optional[int]
            when provided, use this number to attempt to allocate enough space to store a matrix
            with the given number of non-zero entries.
        """
        assert isinstance(m, ss.csc_matrix) or m is None
        super().__init__(chrom, m, logger, max_nnz)

    @property
    def T(self):  # noqa
        """
        Transpose the matrix without copying the data.

        Returns
        -------
        SharedTriangularCSRMatrix
            the transposed matrix in CSR format.
        """
        return SharedTriangularCSRMatrix._from_shared_buffers(  # noqa
            self._chrom,
            self._data,
            self._indices,
            self._indptr,
            self._shape,
        )

    @staticmethod
    def _from_shared_buffers(
        chrom: _SharedTypedBuffer,
        data: _SharedTypedBuffer,
        indices: _SharedTypedBuffer,
        indptr: _SharedTypedBuffer,
        shape: _SharedTypedBuffer,
    ):
        assert shape.data[0] == shape.data[1]  # noqa
        m = SharedTriangularCSCMatrix(None, None, None, None)  # noqa
        m._chrom = chrom
        m._data = data
        m._indices = indices
        m._indptr = indptr
        m._shape = shape
        m._matrix_type_str = "CSC"
        m._matrix_type = ss.csc_matrix

        # Assertion commented out for performance reasons
        # assert ss.triu(m.get(), k=1).sum() == 0 or ss.tril(m.get(), k=-1).sum() == 0

        return m


class SharedTriangularSparseMatrix(object):  # noqa
    """
    Generic class used to represent a sparse matrix in CSR or CSC format using shared memory.
    Under the hood, this class uses SharedTriangularCSRMatrix and SharedTriangularCSCMatrix.
    """

    def __init__(
        self,
        chrom: str,
        m: SparseMatrix,
        logger=None,
        max_nnz: Optional[int] = None,
    ):
        """
        Parameters
        ----------
        chrom: str
            name of the chromosome to which the matrix refers to.
        m: SparseMatrix
            sparse matrix in CSR or CSC format.
            The format provided here will determine the class used to represent the matrix in shared memory.
        logger:
            logger
        max_nnz: Optional[int]
            when provided, use this number to attempt to allocate enough space to store a matrix
            with the given number of non-zero entries.
        """
        if m is None:
            self._m = SharedTriangularCSRMatrix(None, None, None, None)  # noqa
        elif isinstance(m, ss.csr_matrix):
            self._m = SharedTriangularCSRMatrix(chrom, m, logger, max_nnz)
        elif isinstance(m, ss.csc_matrix):
            self._m = SharedTriangularCSCMatrix(chrom, m, logger, max_nnz)
        else:
            self._m = SharedTriangularCSCMatrix(chrom, ss.csc_matrix(m), logger, max_nnz)

    def get(self) -> SparseMatrix:
        """
        Get the sparse matrix without copying the data.
        """
        return self._m.get()

    def can_assign(self, m) -> bool:
        """
        Check whether the given sparse matrix can be assigned to the current matrix instance
        without incurring in shared memory re-allocation.
        """
        if isinstance(m, ss.csr_matrix):
            return isinstance(self._m, SharedTriangularCSRMatrix) and self._m.can_assign(m)
        if isinstance(m, ss.csc_matrix):
            return isinstance(self._m, SharedTriangularCSCMatrix) and self._m.can_assign(m)

        raise NotImplementedError

    def assign(self, chrom: str, m: SparseMatrix, logger=None):
        """
        Assign the given SparseMatrix to the current matrix instance without re-allocations.
        """
        assert self.can_assign(m) and self._m.can_assign(m)
        self._m.assign(chrom, m, logger)

    @property
    def T(self):  # noqa
        """
        Transpose the matrix without copying the data.
        """
        if isinstance(self._m, SharedTriangularCSRMatrix):
            m = SharedTriangularSparseMatrix._initialize_empty("csc")
        else:
            m = SharedTriangularSparseMatrix._initialize_empty("csr")

        m._m = self._m.T
        return m

    @staticmethod
    def _initialize_empty(format: str):  # noqa
        if format == "csr":
            return SharedTriangularSparseMatrix(None, None, None, None)  # noqa
        if format == "csc":
            return SharedTriangularSparseMatrix(None, None, None, None)  # noqa

        raise NotImplementedError


def set_shared_state(
    lt_matrix: Optional[SharedTriangularSparseMatrix],
    ut_matrix: Optional[SharedTriangularSparseMatrix],
):
    """
    Register the given matrices in the global namespace.

    Parameters
    ----------
    lt_matrix: Optional[SharedTriangularSparseMatrix]
        a lower-triangular sparse matrix

    ut_matrix: Optional[SharedTriangularSparseMatrix]
        a upper-triangular sparse matrix
    """
    if lt_matrix is not None:
        global _lower_triangular_matrix  # noqa
        _lower_triangular_matrix = lt_matrix
    if ut_matrix is not None:
        global _upper_triangular_matrix  # noqa
        _upper_triangular_matrix = ut_matrix


def unset_shared_state():  # noqa
    """
    Remove the matrices registered by set_shared_state() from the global namespace.
    """
    call_gc = False
    if shared_state_avail("lower"):
        global _lower_triangular_matrix  # noqa
        del _lower_triangular_matrix
        call_gc = True

    if shared_state_avail("upper"):
        global _upper_triangular_matrix  # noqa
        del _upper_triangular_matrix
        call_gc = True

    if call_gc:
        gc.collect()


def shared_state_avail(key: str) -> bool:
    """
    Check whether the lower- or upper-triangular sparse matrices are available in the global namespace.

    Parameters
    ----------
    key: str
        matrix location. Should be one of "lower", "LT", "upper", or "UT".

    Returns
    -------
    bool
        True if the requested matrix is available, False otherwise.
    """
    if key in {"lower", "LT"}:
        return "_lower_triangular_matrix" in globals()

    if key in {"upper", "UT"}:
        return "_upper_triangular_matrix" in globals()

    raise ValueError("Invalid key")


def get_shared_state(key: str) -> SharedTriangularSparseMatrix:
    """
    Fetch the matrix corresponding to the given key from the global namespace.
    Fails if matrices have not been registered by calling set_shared_state()

    Parameters
    ----------
    key: str
        matrix location. Should be one of "lower", "LT", "upper", or "UT".

    Returns
    -------
    SharedTriangularSparseMatrix
        the requested sparse matrix.

    """
    if key in {"lower", "LT"}:
        global _lower_triangular_matrix  # noqa
        return _lower_triangular_matrix

    if key in {"upper", "UT"}:
        global _upper_triangular_matrix  # noqa
        return _upper_triangular_matrix

    raise ValueError("Invalid key")
