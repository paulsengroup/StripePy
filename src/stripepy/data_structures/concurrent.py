# Copyright (C) 2025 Roberto Rossini <roberros@uio.no>
#
# SPDX-License-Identifier: MIT

import concurrent.futures
import functools
import multiprocessing as mp
import pathlib
import time
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import hictkpy
import structlog

from stripepy.algorithms import step1
from stripepy.data_structures import (
    Result,
    ResultFile,
    SharedTriangularSparseMatrix,
    SparseMatrix,
    set_shared_state,
    unset_shared_state,
)
from stripepy.io import ProcessSafeLogger
from stripepy.utils import (
    define_region_of_interest,
    pretty_format_elapsed_time,
)


def _init_shared_state(
    lower_triangular_matrix: Optional[SharedTriangularSparseMatrix],
    upper_triangular_matrix: Optional[SharedTriangularSparseMatrix],
    log_queue: mp.Queue,
    init_mpl: bool,
):
    """
    Function to initialize newly created worker processes.
    """
    ProcessSafeLogger.setup_logger(log_queue)

    if lower_triangular_matrix is not None:
        assert upper_triangular_matrix is not None
        set_shared_state(lower_triangular_matrix, upper_triangular_matrix)

    if not init_mpl:
        return

    try:
        import matplotlib

        matplotlib.use("Agg")
    except ImportError:
        structlog.get_logger().warning("failed to initialize matplotlib backend")
        pass


class ProcessPoolWrapper(object):
    """
    A wrapper around concurrent.futures.ProcessPoolExecutor that hides most of the complexity
    introduced by inter-process communication

    IMPORTANT:
    When nproc=1, no pool of processes is created, instead all methods block when called and
    are executed in the calling process.
    The ProcessPoolWrapper should always be used with a context manager (e.g. with:).
    """

    def __init__(
        self,
        nproc: int,
        main_logger: ProcessSafeLogger,
        init_mpl: bool = False,
        lazy_pool_initialization: bool = False,
        logger=None,
    ):
        """
        Parameters
        ----------
        nproc: int
            number of worker processes
        main_logger: ProcessSafeLogger
            the logger instance to which worker processes will be connected to
        init_mpl: bool
            whether worker processes should initialize matplotlib upon starting
        lazy_pool_initialization: bool
            controls whether the pool initialization can be deferred until the process pool is actually needed.
            Setting this to True can avoid some work if rebind_shared_matrices() will be called before submitting
            tasks to the process pool
        logger:
            logger used to log information regarding the pool initialization
        """
        self._nproc = nproc
        self._pool = None
        self._lt_matrix = None
        self._ut_matrix = None
        self._init_mpl = init_mpl
        self._log_queue = None

        if nproc > 1:
            self._lock = mp.Lock()
            self._log_queue = main_logger.log_queue
            if not lazy_pool_initialization:
                self._initialize_pool(logger)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._pool is None:
            return False

        if exc_type is not None:
            structlog.get_logger().debug("shutting down process pool due to the following exception: %s", exc_val)

        self._pool.shutdown(wait=True, cancel_futures=exc_type is not None)
        if self._lt_matrix is not None:
            unset_shared_state()
            self._lt_matrix = None
            self._ut_matrix = None

        self._pool = None
        return False

    def rebind_shared_matrices(
        self,
        chrom: str,
        ut_matrix: SparseMatrix,
        logger=None,
        max_nnz: Optional[int] = None,
    ):
        """
        Register and share the upper-triangular sparse matrix for the given chromosomes with the worker processes.
        This function takes care of re-allocating shared memory only when strictly necessary.
        This is important, as in order to make the newly allocated shared memory visible to the worker processes,
        the pool of processes has to be re-created (which is a relatively expensive operation on certain platforms).

        Parameters
        ----------
        chrom: str
            name of the chromosome to be registered
        ut_matrix: SparseMatrix
            the upper triangular sparse matrix to be registered
        logger:
            an object suitable for logging
        max_nnz: Optional[int]
            when provided, the maximum number of non-zero elements in the matrix that are expected to
            be bound to the process pool.
            This is just an estimate. If the estimate is incorrect, applications will be slower due to
            a larger number of shared memory allocations.
            When not provided, the number of non-zero elements in ut_matrix will be used.
        """
        if self._nproc < 2:
            return

        if self._can_rebind_shared_matrix(ut_matrix):
            self._ut_matrix.assign(chrom, ut_matrix, logger)
            self._lt_matrix = self._ut_matrix.T

            set_shared_state(self._lt_matrix, self._ut_matrix)
            return

        if max_nnz is not None:
            max_nnz = max(max_nnz, ut_matrix.nnz)

        unset_shared_state()
        self._ut_matrix = (
            SharedTriangularSparseMatrix(chrom, ut_matrix, logger, max_nnz) if ut_matrix is not None else None
        )
        self._lt_matrix = self._ut_matrix.T if self._ut_matrix is not None else None
        set_shared_state(self._lt_matrix, self._ut_matrix)
        self._initialize_pool(logger)

    @property
    def map(self) -> Callable:
        """
        Return the map object associated with the process pool (or the built-in map if nproc=1)

        Returns
        -------
        Callable
        """
        if self._pool is None:
            return map
        return self._pool.map

    def get_mapper(self, chunksize: int = 1) -> Callable:
        """
        Same as map, but using a custom chunksize.

        Parameters
        ----------
        chunksize: int
            the chunksize to bind to the map method associated with the process pool

        Returns
        -------
        Callable
        """
        if self._pool is None:
            return map

        assert chunksize > 0
        return functools.partial(self._pool.map, chunksize=chunksize)

    def submit(self, fx: Callable, *args, **kwargs) -> concurrent.futures.Future:
        """
        Attempts to asynchronously run the given function with the given positional and keyword arguments.
        If nproc=1, blocks and run the function in the calling process.

        Parameters
        ----------
        fx: Callable
            the function to be submitted
        args:
            zero or more positional arguments
        kwargs:
            zero or more keyword arguments

        Returns
        -------
        concurrent.futures.Future
            a Future that will contain the result and raised exception (if any) once the
            function fx has returned.
        """
        if self._pool is None:
            fut = concurrent.futures.Future()
            try:
                fut.set_result(fx(*args, **kwargs))
            except Exception as e:
                fut.set_exception(e)
            return fut

        return self._pool.submit(fx, *args, **kwargs)

    @property
    def ready(self):
        """
        Check whether the process pool has been initialized.
        """
        return self._pool is not None

    def _can_rebind_shared_matrix(self, ut_matrix: SparseMatrix) -> bool:
        return self._ut_matrix is not None and self._ut_matrix.can_assign(ut_matrix)

    def _initialize_pool(
        self,
        logger=None,
    ):
        if self._nproc < 2:
            return

        t0 = time.time()
        if logger is not None:
            prefix = "re-" if self._pool is not None else ""
            logger.debug("%sinitializing a process pool of size %d", prefix, self._nproc)

        if self._pool is not None:
            self._pool.shutdown(wait=True, cancel_futures=False)

        self._pool = concurrent.futures.ProcessPoolExecutor(  # noqa
            max_workers=self._nproc,
            initializer=_init_shared_state,
            initargs=(self._lt_matrix, self._ut_matrix, self._log_queue, self._init_mpl),
        )

        if logger is not None:
            logger.debug("pool initialization took %s", pretty_format_elapsed_time(t0))


class IOManager(object):
    """
    A class to manage IO tasks and processes (potentially in an asynchronous manner).
    When nproc=1, all operations are blocking.

    The following is the main functionality provided by this class:
    - fetch interaction matrices from the matrix file passed to the constructor.
      This can be done:
      - sequentially: fetch_interaction_matrix()
      - asynchronously: fetch_interaction_matrix_async() and fetch_next_interaction_matrix_async()

    - create a ResultFile in append mode and write Result objects to it
    - automatically finalize the managed ResultFile

    IMPORTANT: this class should always be used with a context manager (e.g. with:).
    """

    def __init__(
        self,
        matrix_path: pathlib.Path,
        result_path: pathlib.Path,
        resolution: int,
        normalization: Optional[str],
        genomic_belt: int,
        region_of_interest: Optional[str],
        nproc: int,
        metadata: Dict[str, Any],
        main_logger: ProcessSafeLogger,
    ):
        """
        Parameters
        ----------
        matrix_path: pathlib.Path
            the path to the matrix file in one of the formats supported by hictkpy
        result_path: pathlib.Path
            path where to create the result file in HDF5 format
        resolution: int
            the resolution of the matrix file
        normalization: Optional[str]
            the normalization method to use when fetching interactions
        genomic_belt: int
            the genomic belt used to fetch interactions
        region_of_interest: Optional[str]
            the region of interest to be fetched (if any).
            When provided, should be either "middle" or "start"
        nproc: int
            the number of processes to use (capped to a maximum of 3)
        metadata: Dict[str, Any]
            metadata to be written to the result file
        main_logger: ProcessSafeLogger
            the logger instance to which IO processes will be connected to
        """
        self._path = matrix_path
        self._resolution = resolution
        self._normalization = "NONE" if normalization is None else normalization
        self._genomic_belt = genomic_belt
        self._roi = region_of_interest
        self._tasks = {}

        logger = structlog.get_logger().bind(step="IO")

        self._pool = ProcessPoolWrapper(
            min(nproc, 3),
            main_logger=main_logger,
            init_mpl=False,
            lazy_pool_initialization=False,
            logger=logger,
        )

        logger.info('initializing result file "%s"', result_path)
        with ResultFile.create_from_file(
            result_path,
            mode="a",
            matrix_file=hictkpy.File(matrix_path, resolution),
            normalization=normalization,
            metadata=metadata,
        ) as h5:
            pass
        self._h5_path = result_path
        self._h5_pending_io_task = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._pool.__exit__(exc_type, exc_val, exc_tb)

        structlog.get_logger().bind(step="IO").info('finalizing file "%s"', self._h5_path)
        self._wait_on_io_on_results_file()
        with ResultFile.append(self._h5_path) as h5:
            h5.finalize()

    def fetch_interaction_matrix(
        self,
        chrom_name: str,
        chrom_size: int,
    ) -> Tuple[SparseMatrix, Optional[SparseMatrix], Optional[SparseMatrix]]:
        """
        Fetch interactions for the given chromosome.

        Parameters
        ----------
        chrom_name: str
            name of the chromosome whose interactions should be fetched
        chrom_size: int
            size of the chromosome whose interactions should be fetched

        Returns
        -------
        Tuple[SparseMatrix, Optional[SparseMatrix], Optional[SparseMatrix]]
        A three-element tuple with:
        1) An upper-triangular sparse matrix with the genome-wide interactions for the given chromosome
        2) A symmetric sparse matrix with the raw interactions spanning the region of interest
        3) A symmetric sparse matrix with the processed interactions spanning the region of interest

        2), and 3) will be None if IOManager was constructed with region_of_interest=None.
        All three matrices have shape NxN, where N is the number of bins in the given chromosome.
        """
        data = self._get_interaction_matrix(chrom_name)
        if data is not None:
            structlog.get_logger().bind(chrom=chrom_name, step="IO").info("returning pre-fetched interactions")
            return data

        roi = define_region_of_interest(self._roi, chrom_size, self._resolution)
        return IOManager._fetch(  # noqa
            self._path,
            self._resolution,
            self._normalization,
            self._genomic_belt,
            chrom_name,
            roi,
        )

    def fetch_interaction_matrix_async(
        self,
        chrom_name: str,
        chrom_size: int,
    ):
        """
        Same as fetch_interaction_matrix, but asynchronous when nproc>1.
        """
        if not self._pool.ready:
            return

        assert chrom_name not in self._tasks

        roi = define_region_of_interest(self._roi, chrom_size, self._resolution)
        self._tasks[chrom_name] = self._pool.submit(
            IOManager._fetch,
            self._path,
            self._resolution,
            self._normalization,
            self._genomic_belt,
            chrom_name,
            roi,
        )

    def fetch_next_interaction_matrix_async(
        self,
        tasks: Sequence[Tuple[str, int, bool]],
    ):
        """
        Loops over the given tasks and attempts to asynchronously fetch interaction for the first task where skip=False.
        Does nothing when nproc<2.

        Parameters
        ----------
        tasks: Sequence[Tuple[str, int, bool]]
            A sequence of the tasks that remains to be processed.

            Each task should consist of a triplet where:
            1) is the chromosome name
            2) is the chromosome size
            3) is a boolean indicating whether the task should be skipped
        """
        if not self._pool.ready:
            return

        if len(tasks) == 0 or tasks[0][0] in self._tasks:
            return

        for chrom, size, skip in tasks:
            if not skip:
                self.fetch_interaction_matrix_async(chrom, size)
                return

    def write_results(self, result: Result):
        """
        Write the given result object to the ResultFile managed by the current IOManager instance.

        Parameters
        ----------
        result: Result
            the result object to be written to the managed ResultFile
        """
        self._wait_on_io_on_results_file()
        self._h5_pending_io_task = self._pool.submit(IOManager._write_results, self._h5_path, result)

    @staticmethod
    def _fetch(
        path: pathlib.Path,
        resolution: int,
        normalization: str,
        genomic_belt: int,
        chrom_name: str,
        roi: Optional[Dict],
    ):
        t0 = time.time()
        logger = structlog.get_logger().bind(chrom=chrom_name, step="IO")

        logger.info("fetching interactions using normalization=%s", normalization)
        f = hictkpy.File(path, resolution=resolution)
        chrom_size = f.chromosomes()[chrom_name]
        diagonal_band_width = (genomic_belt // resolution) + 1 if genomic_belt < chrom_size else None
        matrix = f.fetch(chrom_name, normalization=normalization, diagonal_band_width=diagonal_band_width).to_csr()

        logger.info("fetched %d pixels in %s", matrix.count_nonzero(), pretty_format_elapsed_time(t0))

        logger = structlog.get_logger().bind(chrom=chrom_name, step=(1,))
        logger.info("data pre-processing")
        t0 = time.time()
        ut_matrix, roi_matrix_raw, roi_matrix_proc = step1.run(
            matrix,
            genomic_belt,
            resolution,
            roi=roi,
            logger=logger,
        )
        logger.info("preprocessing took %s", pretty_format_elapsed_time(t0))

        return ut_matrix, roi_matrix_raw, roi_matrix_proc

    def _get_interaction_matrix(
        self, chrom: str
    ) -> Optional[Tuple[SparseMatrix, Optional[SparseMatrix], Optional[SparseMatrix]]]:
        res = self._tasks.pop(chrom, None)
        if res is not None:
            res = res.result()  # noqa
        return res

    @staticmethod
    def _write_results(path: pathlib.Path, result: Result):
        logger = structlog.get_logger().bind(chrom=result.chrom[0], step="IO")
        logger.info('writing results to file "%s"', path)
        start_time = time.time()
        with ResultFile.append(path) as h5:
            h5.write_descriptors(result)
        logger.info('successfully written results to "%s" in %s', path, pretty_format_elapsed_time(start_time))

    def _wait_on_io_on_results_file(self):
        """
        This checks for exceptions and ensures that we are not trying to perform concurrent writes on the same result file
        (e.g. starting to write results for chr2 before we're done writing results for chr1, or finalizing the result file
        before results for the last chromosome has been completely written to file)
        """

        if self._h5_pending_io_task is not None:
            self._h5_pending_io_task.result()
            self._h5_pending_io_task = None
