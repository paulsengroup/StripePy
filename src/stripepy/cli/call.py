# Copyright (C) 2024 Andrea Raffo <andrea.raffo@ibv.uio.no>
#
# SPDX-License-Identifier: MIT

import concurrent.futures
import contextlib
import json
import pathlib
import shutil
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import structlog

from stripepy.algorithms import step2, step3, step4, step5
from stripepy.data_structures import (
    IOManager,
    ProcessPoolWrapper,
    Result,
    SparseMatrix,
    Stripe,
)
from stripepy.io import (
    ProcessSafeLogger,
    get_stripepy_call_progress_bar_weights,
    open_matrix_file_checked,
)
from stripepy.utils import (
    define_region_of_interest,
    import_matplotlib,
    pretty_format_elapsed_time,
    pretty_format_genomic_distance,
)


def run(
    contact_map: pathlib.Path,
    resolution: int,
    output_file: pathlib.Path,
    genomic_belt: int,
    max_width: int,
    glob_pers_min: float,
    constrain_heights: bool,
    k: int,
    loc_pers_min: float,
    loc_trend_min: float,
    force: bool,
    nproc: int,
    min_chrom_size: int,
    verbosity: str,
    main_logger: ProcessSafeLogger,
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
        import_matplotlib()

    # This takes care of fetching the list of chromosomes after ensuring that the given matrix file
    # satisfies all of StripePy requirements
    chroms = open_matrix_file_checked(
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
                    k=k,
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

        if normalization is None:
            normalization = "NONE"

        # Generate a plan for the work to be done
        tasks = _plan_tasks(chroms, min_chrom_size, main_logger)

        # Set up the progress bar
        progress_weights_df = get_stripepy_call_progress_bar_weights(
            tasks,
            include_plotting=roi is not None,
            nproc=nproc,
        )
        progress_bar = parent_logger.progress_bar
        progress_bar.add_task(
            task_id="total",
            chrom="unknown",
            step="unknown",
            name=f"{contact_map.stem} • {pretty_format_genomic_distance(resolution)}",
            description="",
            start=True,
            total=progress_weights_df.sum().sum(),
            visible=True,
        )

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
                progress_bar.update(
                    task_id="total",
                    advance=sum(progress_weights.values()),
                    chrom=chrom_name,
                    step="IO",
                )
                continue

            progress_bar.update(
                task_id="total",
                advance=0,
                chrom=chrom_name,
                step="step 1",
            )
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

            progress_bar.update(
                task_id="total",
                advance=progress_weights["step_1"],
                chrom=chrom_name,
                step="step 2",
            )

            result = _run_step_2(
                chrom_name=chrom_name,
                chrom_size=chrom_size,
                lt_matrix=lt_matrix,
                ut_matrix=ut_matrix,
                min_persistence=glob_pers_min,
                pool=pool,
                logger=logger,
            )
            progress_bar.update(
                task_id="total",
                advance=progress_weights["step_2"],
                chrom=chrom_name,
                step="step 3",
            )

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
            progress_bar.update(
                task_id="total",
                advance=progress_weights["step_3"],
                chrom=chrom_name,
                step="step 4",
            )

            result = _run_step_4(
                result=result,
                lt_matrix=lt_matrix,
                ut_matrix=ut_matrix,
                k=k,
                tpool=tpool,
                pool=pool,
                logger=logger,
            )
            progress_bar.update(
                task_id="total",
                advance=progress_weights["step_4"],
                chrom=chrom_name,
                step="step 4" if matrix_roi_raw is None else "step 5",
            )

            logger.info("processing took %s", pretty_format_elapsed_time(start_local_time))

            io_manager.write_results(result)

            if matrix_roi_raw is not None:
                assert matrix_roi_proc is not None
                chrom_roi = define_region_of_interest(roi, chrom_size, resolution)

                logger.info("region of interest to be used for plotting: %s:%d-%d", chrom_name, *chrom_roi["genomic"])
                result.set_roi(chrom_roi)  # noqa
                start_time = time.time()
                logger = logger.bind(step=(5,))
                logger.info("generating plots")
                step5.run(
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

                progress_bar.update(
                    task_id="total",
                    advance=progress_weights["step_5"],
                    chrom=chrom_name,
                    step="step 5",
                )
                logger.info("plotting took %s", pretty_format_elapsed_time(start_time))

    main_logger.info("DONE!")
    main_logger.info("processed %d chromosomes in %s", len(chroms), pretty_format_elapsed_time(start_global_time))

    return 0


def _generate_metadata_attribute(
    constrain_heights: bool,
    genomic_belt: int,
    glob_pers_min: float,
    k: int,
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
        "k-neighbour": k,
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
) -> Result:
    """
    Shortcut to generate an empty Result object for the given chromosome.
    """
    result = Result(chrom, chrom_size)
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
    results: Iterable[Union[Tuple[str, Result], concurrent.futures.Future]],
) -> Result:
    """
    Merge two result file objects into one.

    Parameters
    ----------
    results : Iterable[Union[Tuple[str, Result, concurrent.futures.Future]]
        An iterable returning two result objects to be merged.

        Each result can be a:
        - two-element tuple, where the first entry is the location and the second entry is the Result object
        - concurrent.futures.Future object holding two-element tuples like described above

    Returns
    -------
    Result
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


def _run_step_2_helper(args) -> Tuple[str, Result]:
    return step2.run(*args)


def _run_step_3_helper(args) -> Tuple[str, Result]:
    return step3.run(*args)


def _run_step_4_helper(args) -> Tuple[str, List[Stripe]]:
    return step4.run(*args)


def _run_step_2(
    chrom_name: str,
    chrom_size: int,
    lt_matrix: Optional[SparseMatrix],
    ut_matrix: Optional[SparseMatrix],
    min_persistence: float,
    pool: ProcessPoolWrapper,
    logger,
) -> Result:
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
    result: Result,
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
) -> Result:
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
    result: Result,
    lt_matrix: Optional[SparseMatrix],
    ut_matrix: Optional[SparseMatrix],
    k: int,
    tpool: Union[ProcessPoolWrapper, concurrent.futures.ThreadPoolExecutor],
    pool: ProcessPoolWrapper,
    logger,
) -> Result:
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
        (result.get("stripes", "lower"), lt_matrix, "lower", k, executor, logger),
        (result.get("stripes", "upper"), ut_matrix, "upper", k, executor, logger),
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
