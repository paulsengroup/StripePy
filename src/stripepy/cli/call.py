# Copyright (C) 2024 Andrea Raffo <andrea.raffo@ibv.uio.no>
#
# SPDX-License-Identifier: MIT

import concurrent.futures
import contextlib
import functools
import json
import multiprocessing as mp
import pathlib
import shutil
import sys
import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import hictkpy
import numpy as np
import pandas as pd
import structlog

from stripepy import IO, others, stripepy
from stripepy.cli import logging
from stripepy.utils import stripe
from stripepy.utils.common import _import_matplotlib, pretty_format_elapsed_time  # noqa
from stripepy.utils.progress_bar import initialize_progress_bar
from stripepy.utils.shared_sparse_matrix import (
    SharedTriangularSparseMatrix,
    SparseMatrix,
    set_shared_state,
    unset_shared_state,
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
    logging.ProcessSafeLogger.setup_logger(log_queue)

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
        main_logger: logging.ProcessSafeLogger,
        init_mpl: bool = False,
        lazy_pool_initialization: bool = False,
        logger=None,
    ):
        """
        Parameters
        ----------
        nproc: int
            number of worker processes
        main_logger: logging.ProcessSafeLogger
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
        main_logger: logging.ProcessSafeLogger,
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
        main_logger: logging.ProcessSafeLogger
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
        with IO.ResultFile.create_from_file(
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
        with IO.ResultFile.append(self._h5_path) as h5:
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

        roi = others.define_region_of_interest(self._roi, chrom_size, self._resolution)
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

        roi = others.define_region_of_interest(self._roi, chrom_size, self._resolution)
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

    def write_results(self, result: IO.Result):
        """
        Write the given result object to the ResultFile managed by the current IOManager instance.

        Parameters
        ----------
        result: IO.Result
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
        matrix = hictkpy.File(path, resolution=resolution).fetch(chrom_name, normalization=normalization).to_csr()
        logger.info("fetched %d pixels in %s", matrix.count_nonzero(), pretty_format_elapsed_time(t0))

        logger = structlog.get_logger().bind(chrom=chrom_name, step=(1,))
        logger.info("data pre-processing")
        t0 = time.time()
        ut_matrix, roi_matrix_raw, roi_matrix_proc = stripepy.step_1(
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
    def _write_results(path: pathlib.Path, result: IO.Result):
        logger = structlog.get_logger().bind(chrom=result.chrom[0], step="IO")
        logger.info('writing results to file "%s"', path)
        start_time = time.time()
        with IO.ResultFile.append(path) as h5:
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


def _generate_metadata_attribute(
    constrain_heights: bool,
    genomic_belt: int,
    glob_pers_min: float,
    loc_pers_min: float,
    loc_trend_min: float,
    max_width: int,
    min_chrom_size: int,
) -> Dict[str, Any]:
    """
    Generate a dictionary with the metadata to be written to a ResultFile.
    """
    return {
        "constrain-heights": constrain_heights,
        "genomic-belt": genomic_belt,
        "global-persistence-minimum": glob_pers_min,
        "local-persistence-minimum": loc_pers_min,
        "local-trend-minimum": loc_trend_min,
        "max-width": max_width,
        "min-chromosome-size": min_chrom_size,
    }


def _plan_tasks(
    chromosomes: Dict[str, int],
    min_size: int,
    logger,
) -> List[Tuple[str, int, bool]]:
    """
    Generate the list of tasks to be processed.
    Chromosomes whose size is <= min_size will be skipped.
    """
    plan = []
    small_chromosomes = []
    for chrom, length in chromosomes.items():
        skip = length <= min_size
        plan.append((chrom, length, skip))
        if skip:
            small_chromosomes.append(chrom)

    if len(small_chromosomes) != 0:
        logger.warning(
            "the following chromosomes are discarded because shorter than --min-chrom-size=%d bp: %s",
            min_size,
            ", ".join(small_chromosomes),
        )

    return plan


def _generate_empty_result(
    chrom: str,
    chrom_size: int,
    resolution: int,
) -> IO.Result:
    """
    Shortcut to generate an empty Result object for the given chromosome.
    """
    result = IO.Result(chrom, chrom_size)
    result.set_min_persistence(0)

    num_bins = (chrom_size + resolution - 1) // resolution
    for location in ("LT", "UT"):
        result.set("all_minimum_points", [], location)
        result.set("all_maximum_points", [], location)
        result.set("persistence_of_all_minimum_points", [], location)
        result.set("persistence_of_all_maximum_points", [], location)
        result.set("persistent_minimum_points", [], location)
        result.set("persistent_maximum_points", [], location)
        result.set("persistence_of_minimum_points", [], location)
        result.set("persistence_of_maximum_points", [], location)
        result.set("pseudodistribution", np.full(num_bins, np.nan, dtype=float), location)
        result.set("stripes", [], location)

    return result


class _JSONEncoder(json.JSONEncoder):
    """
    An encoder class that can serialize pathlib.Path objects as JSON.
    """

    def default(self, o: Any):
        if isinstance(o, pathlib.Path):
            return str(o)
        return super().default(o)


def _write_param_summary(
    config: Dict[str, Any],
    logger=None,
):
    """
    Log the parameters used by the current invocation of stripepy call.
    """
    if logger is None:
        logger = structlog.get_logger().bind(step="main")

    config_str = json.dumps(config, indent=2, sort_keys=True, cls=_JSONEncoder)
    logger.info(f"CONFIG:\n{config_str}")


def _compute_progress_bar_weights(
    chrom_sizes: Dict[str, int],
    include_plotting: bool,
    nproc: int,
) -> pd.DataFrame:
    """
    Compute the weights used to update the progress bar.

    Returns
    -------
    pd.DataFrame
        A DataFrame with the following columns:
        index) chrom
        0) input
        1) step_1
        2) step_2
        3) step_3
        4) step_4
        5) output
        6) step_5
    """
    # These weights have been computed on a Linux machine (Ryzen 9 7950X3D) using 1 core to process
    # 4DNFI9GMP2J8.mcool at 10kbp
    step_weights = {
        "input": 0.035557,
        "step_1": 0.018159,
        "step_2": 0.015722,
        "step_3": 0.301879,
        "step_4": 0.051055,
        "output": 0.054285,
        "step_5": 0.523344 / nproc,
    }

    if not include_plotting:
        step_weights["step_5"] = 0.0

    tot = sum(step_weights.values())
    step_weights = {k: v / tot for k, v in step_weights.items()}

    weights = []
    for size in chrom_sizes.values():
        weights.extend((size * w for w in step_weights.values()))

    weights = np.array(weights)
    weights /= weights.sum()
    weights = np.minimum(weights.cumsum(), 1.0)

    shape = (len(chrom_sizes), len(step_weights))
    df = pd.DataFrame(weights.reshape(shape), columns=list(step_weights.keys()))
    df["chrom"] = list(chrom_sizes.keys())
    return df.set_index(["chrom"])


def _remove_existing_output_files(
    output_file: pathlib.Path,
    plot_dir: Optional[pathlib.Path],
    chromosomes: Dict[str, int],
):
    """
    Preemptively remove existing output files.
    """
    logger = structlog.get_logger().bind(step="main")
    logger.debug("removing %s", output_file)
    output_file.unlink(missing_ok=True)
    if plot_dir is not None:
        for path in plot_dir.glob("*"):
            if path.stem in chromosomes:
                logger.debug("removing %s", path)
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()


def _merge_results(
    results: Iterable[Union[Tuple[str, IO.Result], concurrent.futures.Future]],
) -> IO.Result:
    """
    Merge two result file objects into one.

    Parameters
    ----------
    results : Iterable[Union[Tuple[str, IO.Result], concurrent.futures.Future]]
        An iterable returning two result objects to be merged.

        Each result can be a:
        - two-element tuple, where the first entry is the location and the second entry is the Result object
        - concurrent.futures.Future object holding two-element tuples like described above

    Returns
    -------
    IO.Result
        The resulting merged Result object.
    """
    keys = (
        "all_minimum_points",
        "all_maximum_points",
        "persistence_of_all_minimum_points",
        "persistence_of_all_maximum_points",
        "persistent_minimum_points",
        "persistent_maximum_points",
        "persistence_of_minimum_points",
        "persistence_of_maximum_points",
        "pseudodistribution",
        "stripes",
    )

    data = {}
    i = 0
    for i, res in enumerate(results, 1):
        if isinstance(res, concurrent.futures.Future):
            res = res.result()

        location, result = res
        data[location] = result

    assert len(data) == i

    result1, result2 = data["lower"], data["upper"]

    for key in keys:
        result1.set(key, result2.get(key, "upper"), "upper", force=True)

    return result1


def _run_step_2_helper(args) -> Tuple[str, IO.Result]:
    return stripepy.step_2(*args)


def _run_step_3_helper(args) -> Tuple[str, IO.Result]:
    return stripepy.step_3(*args)


def _run_step_4_helper(args) -> Tuple[str, List[stripe.Stripe]]:
    return stripepy.step_4(*args)


def _run_step_2(
    chrom_name: str,
    chrom_size: int,
    lt_matrix: Optional[SparseMatrix],
    ut_matrix: Optional[SparseMatrix],
    min_persistence: float,
    pool: ProcessPoolWrapper,
    logger,
) -> IO.Result:
    """
    Helper function to simplify running step_2().

    IMPORTANT: lt_matrix and ut_matrix should be None when nproc>1, and the matrices should be fetched
    from the shared state.
    """
    logger = logger.bind(step=(2,))
    logger.info("topological data analysis")
    t0 = time.time()

    params = (
        (chrom_name, chrom_size, lt_matrix, min_persistence, "lower"),
        (chrom_name, chrom_size, ut_matrix, min_persistence, "upper"),
    )

    tasks = pool.map(_run_step_2_helper, params)
    result = _merge_results(tasks)

    logger.info("topological data analysis took %s", pretty_format_elapsed_time(t0))
    return result


def _run_step_3(
    result: IO.Result,
    lt_matrix: Optional[SparseMatrix],
    ut_matrix: Optional[SparseMatrix],
    resolution: int,
    genomic_belt: int,
    max_width: int,
    loc_pers_min: float,
    loc_trend_min: float,
    tpool: concurrent.futures.ThreadPoolExecutor,
    pool: ProcessPoolWrapper,
    logger,
) -> IO.Result:
    """
    Helper function to simplify running step_3().

    See notes for _run_step_2().
    """
    logger = logger.bind(step=(3,))
    logger.info("shape analysis")
    t0 = time.time()

    if pool.ready:
        executor = pool.get_mapper(chunksize=50)
    else:
        executor = pool.map

    params = (
        (
            result,
            lt_matrix,
            resolution,
            genomic_belt,
            max_width,
            loc_pers_min,
            loc_trend_min,
            "lower",
            executor,
            logger,
        ),
        (
            result,
            ut_matrix,
            resolution,
            genomic_belt,
            max_width,
            loc_pers_min,
            loc_trend_min,
            "upper",
            executor,
            logger,
        ),
    )

    tasks = tpool.map(_run_step_3_helper, params)
    result = _merge_results(tasks)

    logger.info("shape analysis took %s", pretty_format_elapsed_time(t0))

    return result


def _run_step_4(
    result: IO.Result,
    lt_matrix: Optional[SparseMatrix],
    ut_matrix: Optional[SparseMatrix],
    tpool: Union[ProcessPoolWrapper, concurrent.futures.ThreadPoolExecutor],
    pool: ProcessPoolWrapper,
    logger,
) -> IO.Result:
    """
    Helper function to simplify running step_4().

    See notes for _run_step_2().
    """
    t0 = time.time()

    logger = logger.bind(step=(4,))
    logger.info("statistical analysis and post-processing")

    if pool.ready:
        executor = pool.get_mapper(chunksize=50)
    else:
        executor = pool.map

    params = (
        (result.get("stripes", "lower"), lt_matrix, "lower", executor, logger),
        (result.get("stripes", "upper"), ut_matrix, "upper", executor, logger),
    )

    (_, lt_stripes), (_, ut_stripes) = list(tpool.map(_run_step_4_helper, params))
    result.set("stripes", lt_stripes, "LT", force=True)
    result.set("stripes", ut_stripes, "UT", force=True)

    logger.info("statistical analysis and post-processing took %s", pretty_format_elapsed_time(t0))

    return result


def _estimate_max_nnz(
    chrom: str,
    matrix: SparseMatrix,
    chroms: Dict[str, int],
    multiplier: float = 2.0,
) -> int:
    """
    Estimate the maximum number of non-zero elements that are likely to be observed during the current run.
    The logic in this function is fairly simple.
    Given a sparse matrix with interactions for chromosome chrom:
    1) if chrom is the largest chromosome in the current reference genome, simply record the number
       of non-zero elements in its matrix.
    2) else, try to guess the number of non-zero entries in the matrix for the largest chromosome.
       The estimate is based on the number of non-zero elements for the current matrix.

    Parameters
    ----------
    chrom: str
        name of the chromosome to which the given matrix refers to
    matrix: SparseMatrix
        sparse matrix with interactions for the given chromosome
    chroms: Dict[str, int]
        a dictionary mapping chromosome names to their respective size
    multiplier: float
        a coefficient used to adjust the estimated number of non-zero entries

    Returns
    -------
    int
        The estimated number of non-zero entries.
        The returned value is the observed or estimated number of non-zero elements multiplied by the
        given multiplier.
    """
    assert multiplier >= 1
    longest_chrom = max(chroms, key=chroms.get)  # noqa
    if longest_chrom == chrom:
        return int(multiplier * matrix.nnz)

    max_size = chroms[longest_chrom]
    current_size = chroms[chrom]

    ratio = max_size / current_size

    return int(multiplier * ratio * matrix.nnz)


def _fetch_interactions(
    i: int,
    io_manager: IOManager,
    tasks: Sequence[Tuple[str, int, bool]],
    pool: ProcessPoolWrapper,
    chroms: Dict[str, int],
    logger,
) -> Tuple[SparseMatrix, Optional[SparseMatrix], Optional[SparseMatrix]]:
    chrom_name, chrom_size, _ = tasks[i]

    ut_matrix, matrix_roi_raw, matrix_roi_proc = io_manager.fetch_interaction_matrix(chrom_name, chrom_size)
    io_manager.fetch_next_interaction_matrix_async(tasks[i + 1 :])
    if i == 0:
        max_nnz = _estimate_max_nnz(chrom_name, ut_matrix, chroms)
        pool.rebind_shared_matrices(chrom_name, ut_matrix, logger, max_nnz)
    else:
        pool.rebind_shared_matrices(chrom_name, ut_matrix, logger)

    return ut_matrix, matrix_roi_raw, matrix_roi_proc


def run(
    contact_map: pathlib.Path,
    resolution: int,
    output_file: pathlib.Path,
    genomic_belt: int,
    max_width: int,
    glob_pers_min: float,
    constrain_heights: bool,
    loc_pers_min: float,
    loc_trend_min: float,
    force: bool,
    nproc: int,
    min_chrom_size: int,
    verbosity: str,
    main_logger: logging.ProcessSafeLogger,
    roi: Optional[str] = None,
    log_file: Optional[pathlib.Path] = None,
    plot_dir: Optional[pathlib.Path] = None,
    normalization: Optional[str] = None,
) -> int:
    """
    Entrypoint for stripepy call
    """
    args = locals()
    args.pop("main_logger")

    start_global_time = time.time()

    parent_logger = main_logger
    main_logger = structlog.get_logger().bind(step="main")

    _write_param_summary(args, logger=main_logger)

    if roi is not None:
        # Raise an error immediately if --roi was passed and matplotlib is not available
        _import_matplotlib()

    # This takes care of fetching the list of chromosomes after ensuring that the given matrix file
    # satisfies all of StripePy requirements
    chroms = others.open_matrix_file_checked(
        contact_map,
        resolution,
        logger=main_logger,
    ).chromosomes(include_ALL=False)

    if force:
        _remove_existing_output_files(output_file, plot_dir, chroms)

    with contextlib.ExitStack() as ctx:
        # Set up the manager in charge of orchestrating IO operations
        io_manager = ctx.enter_context(
            IOManager(
                matrix_path=contact_map,
                result_path=output_file,
                resolution=resolution,
                normalization=normalization,
                genomic_belt=genomic_belt,
                region_of_interest=roi,
                metadata=_generate_metadata_attribute(
                    constrain_heights=constrain_heights,
                    genomic_belt=genomic_belt,
                    glob_pers_min=glob_pers_min,
                    loc_pers_min=loc_pers_min,
                    loc_trend_min=loc_trend_min,
                    max_width=max_width,
                    min_chrom_size=min_chrom_size,
                ),
                nproc=nproc,
                main_logger=parent_logger,
            )
        )

        # Set up the pool of worker processes
        pool = ctx.enter_context(
            ProcessPoolWrapper(
                nproc=nproc,
                main_logger=parent_logger,
                init_mpl=roi is not None,
                lazy_pool_initialization=True,
                logger=None,
            )
        )

        # Set up the pool of worker threads
        tpool = ctx.enter_context(
            concurrent.futures.ThreadPoolExecutor(max_workers=min(nproc, 2)),
        )

        # Set up the progress bar when appropriate
        disable_bar = not sys.stderr.isatty()
        progress_weights_df = _compute_progress_bar_weights(chroms, include_plotting=roi is not None, nproc=nproc)
        progress_bar = ctx.enter_context(
            initialize_progress_bar(
                total=sum(chroms.values()),
                manual=True,
                disable=disable_bar,
                enrich_print=False,
                file=sys.stderr,
                receipt=False,
                refresh_secs=0.05,
                monitor="{percent:.2%}",
                unit="bp",
                scale="SI",
            )
        )

        if normalization is None:
            normalization = "NONE"

        # Generate a plan for the work to be done
        tasks = _plan_tasks(chroms, min_chrom_size, main_logger)

        # Loop over the planned tasks
        for i, (chrom_name, chrom_size, skip) in enumerate(tasks):
            progress_weights = progress_weights_df.loc[chrom_name, :].to_dict()

            start_local_time = time.time()

            logger = main_logger.bind(chrom=chrom_name)
            logger.info("begin processing")

            if skip:
                # Nothing to do here: write an empty entry for the current chromosome and continue
                logger.warning("writing an empty entry for chromosome %s", chrom_name)
                io_manager.write_results(_generate_empty_result(chrom_name, chrom_size, resolution))
                progress_bar(max(progress_weights.values()))
                continue

            ut_matrix, matrix_roi_raw, matrix_roi_proc = _fetch_interactions(
                i,
                io_manager,
                tasks,
                pool,
                chroms,
                logger,
            )

            if pool.ready:
                # Signal that matrices should be fetched from the shared global state
                lt_matrix = None
                ut_matrix = None
            else:
                lt_matrix = ut_matrix.T

            progress_bar(progress_weights["step_1"])

            result = _run_step_2(
                chrom_name=chrom_name,
                chrom_size=chrom_size,
                lt_matrix=lt_matrix,
                ut_matrix=ut_matrix,
                min_persistence=glob_pers_min,
                pool=pool,
                logger=logger,
            )
            progress_bar(progress_weights["step_2"])

            result = _run_step_3(
                result=result,
                lt_matrix=lt_matrix,
                ut_matrix=ut_matrix,
                resolution=resolution,
                genomic_belt=genomic_belt,
                max_width=max_width,
                loc_pers_min=loc_pers_min,
                loc_trend_min=loc_trend_min,
                tpool=tpool,
                pool=pool,
                logger=logger,
            )
            progress_bar(progress_weights["step_3"])

            result = _run_step_4(
                result=result,
                lt_matrix=lt_matrix,
                ut_matrix=ut_matrix,
                tpool=tpool,
                pool=pool,
                logger=logger,
            )
            progress_bar(progress_weights["step_4"])

            logger.info("processing took %s", pretty_format_elapsed_time(start_local_time))

            io_manager.write_results(result)
            progress_bar(progress_weights["output"])

            if matrix_roi_raw is not None:
                assert matrix_roi_proc is not None
                chrom_roi = others.define_region_of_interest(roi, chrom_size, resolution)

                logger.info("region of interest to be used for plotting: %s:%d-%d", chrom_name, *chrom_roi["genomic"])
                result.set_roi(chrom_roi)
                start_time = time.time()
                logger = logger.bind(step=(5,))
                logger.info("generating plots")
                stripepy.step_5(
                    result,
                    resolution,
                    matrix_roi_raw,
                    matrix_roi_proc,
                    genomic_belt,
                    loc_pers_min,
                    loc_trend_min,
                    plot_dir,
                    pool=pool,
                )

                progress_bar(progress_weights["step_5"])
                logger.info("plotting took %s", pretty_format_elapsed_time(start_time))

    main_logger.info("DONE!")
    main_logger.info("processed %d chromosomes in %s", len(chroms), pretty_format_elapsed_time(start_global_time))

    return 0
