# Copyright (C) 2024 Roberto Rossini <roberros@uio.no>
#
# SPDX-License-Identifier: MIT

import pathlib
import random
import time
from typing import Dict, Optional, Tuple

import hictkpy
import numpy as np
import structlog
from numpy.typing import NDArray

import stripepy.plot
from stripepy.IO import Result, ResultFile
from stripepy.utils.common import _import_matplotlib, pretty_format_elapsed_time

try:
    import matplotlib.pyplot as plt
except ImportError:
    from stripepy.utils.common import _DummyPyplot

    plt = _DummyPyplot()


def _generate_random_region(
    chroms: Dict[str, int],
    resolution: int,
    region_size: int = 2_500_000,
    logger=None,
) -> Tuple[str, int, int]:
    if logger is None:
        logger = structlog.get_logger()

    chrom_names = list(chroms.keys())
    random.shuffle(chrom_names)

    for chrom in chrom_names:
        size = chroms[chrom]
        if size < region_size:
            logger.debug("%s is too small (%dbp): skipping!", chrom, size)
            continue

        if size == region_size:
            return (
                chrom,
                0,
                size,
            )

        offset = region_size // 2
        pos = random.randint(offset, chroms[chrom] - offset)
        start_pos = ((pos - offset) // resolution) * resolution
        end_pos = start_pos + region_size
        return (
            chrom,
            start_pos,
            end_pos,
        )

    raise RuntimeError(
        "Unable to randomly select a region to be plotted. Please manually select the desired region by passing the --region option."
    )


def _fetch_random_region(
    f: hictkpy.File,
    normalization: Optional[str],
    region_size: int = 2_500_000,
    logger=None,
) -> Tuple[str, int, int, NDArray]:
    if logger is None:
        logger = structlog.get_logger()

    for _ in range(10):
        chrom, start_pos, end_pos = _generate_random_region(
            f.chromosomes(include_ALL=False), f.resolution(), region_size
        )
        logger.debug("fetching interactions for %s:%d-%d", chrom, start_pos, end_pos)
        m = f.fetch(f"{chrom}:{start_pos}-{end_pos}", normalization=normalization).to_numpy()
        nnz = (np.isfinite(m) & (m != 0)).sum()

        if nnz / m.size >= 0.75:
            logger.debug("found suitable region: %s:%d-%d", chrom, start_pos, end_pos)
            return chrom, start_pos, end_pos, m
        logger.debug("region is too sparse: discarding region")

    logger.warning(
        "Failed to randomly select a genomic region with appropriate density for plotting.\n"
        "Continuing anyway.\n"
        "For best results, please manually provide the coordinates for a region to be plotted using parameter --region."
    )

    return chrom, start_pos, end_pos, m  # noqa


def _fetch_matrix(
    path: pathlib.Path,
    resolution: int,
    normalization: Optional[str],
    region: Optional[str],
    logger=None,
) -> Tuple[str, int, int, NDArray]:
    if logger is None:
        logger = structlog.get_logger()

    f = hictkpy.MultiResFile(path)[resolution]

    if normalization is None:
        normalization = "NONE"

    if region is None:
        return _fetch_random_region(f, normalization)

    logger.debug("fetching interactions for %s", region)
    matrix = f.fetch(region, normalization=normalization).to_numpy()
    chrom, start, end = _parse_ucsc_region(region, f.chromosomes(include_ALL=False))

    return chrom, start, end, matrix


def _parse_ucsc_region(
    region: str,
    chromosomes: Dict[str, int],
) -> Tuple[str, int, int]:
    try:
        chrom, sep, pos = region.partition(":")
        if len(sep) == 0 and len(pos) == 0:
            return chrom, 0, chromosomes[chrom]

        start_pos, _, end_pos = pos.partition("-")
        return chrom, int(start_pos.replace(",", "")), int(end_pos.replace(",", ""))
    except Exception as e:
        raise RuntimeError(f'Unable to parse region "{region}". Is the given region in UCSC format?') from e


def _validate_hdf5_result(
    path_to_result_file: pathlib.Path,
    resolution: int,
):
    with ResultFile(path_to_result_file) as rf:
        if rf.resolution != resolution:
            raise RuntimeError(
                f'File "{rf.path}" has an unexpected resolution: expected {resolution}, found {rf.resolution}'
            )


def _plot_hic_matrix(
    contact_map: pathlib.Path,
    resolution: int,
    cmap: str,
    log_scale: bool,
    region: Optional[str] = None,
    normalization: Optional[str] = None,
    logger=None,
    **kwargs,
) -> plt.Figure:
    if logger is None:
        logger = structlog.get_logger()

    if normalization is None:
        normalization = "NONE"

    chrom, start, end, matrix = _fetch_matrix(contact_map, resolution, normalization, region)

    logger.info("plotting interactions for %s:%d-%d as a heatmap", chrom, start, end)
    dummy_result = Result(chrom, end)
    fig, _ = stripepy.plot.plot(
        dummy_result,
        resolution=resolution,
        plot_type="matrix",
        start=start,
        end=end,
        matrix=matrix,
        log_scale=log_scale,
        cmap=cmap,
    )

    return fig


def _plot_hic_matrix_with_seeds(
    contact_map: pathlib.Path,
    stripepy_hdf5: pathlib.Path,
    resolution: int,
    cmap: str,
    log_scale: bool,
    region: Optional[str] = None,
    normalization: Optional[str] = None,
    logger=None,
    **kwargs,
) -> plt.Figure:
    if logger is None:
        logger = structlog.get_logger()

    _validate_hdf5_result(stripepy_hdf5, resolution)

    chrom, start, end, matrix = _fetch_matrix(contact_map, resolution, normalization, region)

    with ResultFile(stripepy_hdf5) as h5:
        logger.info("fetching candidate stripes overlapping %s:%d-%d...", chrom, start, end)
        result = h5[chrom]

    logger.info("plotting interactions and seeds for %s:%d-%d", chrom, start, end)
    fig, _ = stripepy.plot.plot(
        result,
        resolution=resolution,
        plot_type="matrix_with_seeds",
        start=start,
        end=end,
        matrix=matrix,
        cmap=cmap,
        log_scale=log_scale,
    )
    return fig


def _plot_hic_matrix_with_stripes(
    contact_map: pathlib.Path,
    stripepy_hdf5: pathlib.Path,
    resolution: int,
    cmap: str,
    override_height: bool,
    mask_regions: bool,
    log_scale: bool,
    region: Optional[str] = None,
    relative_change_threshold: Optional[float] = None,
    normalization: Optional[str] = None,
    logger=None,
    **kwargs,
) -> plt.Figure:
    if logger is None:
        logger = structlog.get_logger()

    _validate_hdf5_result(stripepy_hdf5, resolution)

    if relative_change_threshold is None:
        relative_change_threshold = 0.0

    chrom, start, end, matrix = _fetch_matrix(contact_map, resolution, normalization, region)

    with ResultFile(stripepy_hdf5) as h5:
        logger.info("fetching candidate stripes overlapping %s:%d-%d...", chrom, start, end)
        result = h5[chrom]

    logger.info("plotting interactions and candidate stripes for %s:%d-%d", chrom, start, end)
    plot_type = "matrix_with_stripes_masked" if mask_regions else "matrix_with_stripes"
    fig, _ = stripepy.plot.plot(
        result,
        resolution=resolution,
        plot_type=plot_type,
        start=start,
        end=end,
        matrix=matrix,
        cmap=cmap,
        log_scale=log_scale,
        relative_change_threshold=relative_change_threshold,
        override_height=override_height,
    )

    return fig


def _plot_pseudodistribution(
    stripepy_hdf5: pathlib.Path,
    region: Optional[str],
    logger=None,
    **kwargs,
) -> plt.Figure:
    if logger is None:
        logger = structlog.get_logger()

    with ResultFile(stripepy_hdf5) as h5:
        if region is None:
            chrom, start, end = _generate_random_region(h5.chromosomes, h5.resolution)
        else:
            chrom, start, end = _parse_ucsc_region(region, h5.chromosomes)

        logger.info("fetching pseudo-distribution for interval %s:%d-%d...", chrom, start, end)

        result = h5[chrom]
        resolution = h5.resolution

    logger.info("plotting the pseudo-distribution for interval %s:%d-%d...", chrom, start, end)
    fig, axs = stripepy.plot.plot(
        result,
        resolution=resolution,
        plot_type="pseudodistribution",
        start=start,
        end=end,
        title=False,
    )
    axs[-1].set(xlabel=f"Genomic coordinates ({chrom}; bp)")  # noqa
    return fig


def _plot_stripe_dimension_distribution(
    stripepy_hdf5: pathlib.Path,
    region: Optional[str] = None,
    logger=None,
    **kwargs,
) -> plt.Figure:
    if logger is None:
        logger = structlog.get_logger()

    with ResultFile(stripepy_hdf5) as h5:
        if region is None:
            logger.info("fetching and plotting geometric descriptors for the entire genome...")
            dummy_result = Result("gw", sum(h5.chromosomes.values()))
            fig, _ = stripepy.plot.plot(
                dummy_result,
                resolution=h5.resolution,
                plot_type="geo_descriptors",
                start=0,
                end=dummy_result.chrom[1],
                stripes_lt=h5.get(None, "stripes", "LT"),
                stripes_ut=h5.get(None, "stripes", "UT"),
            )
            fig.suptitle("Stripe shape stats: genome-wide")
        else:
            chrom, start, end = _parse_ucsc_region(region, h5.chromosomes)
            logger.info("fetching and plotting geometric descriptors for %s:%d-%d...", chrom, start, end)
            fig, _ = stripepy.plot.plot(
                h5[chrom],
                resolution=h5.resolution,
                plot_type="geo_descriptors",
                start=start,
                end=end,
            )
            fig.suptitle(f"Stripe shape stats: {chrom}:{start}-{end}")

    fig.tight_layout()
    return fig


def run(
    plot_type: str,
    output_name: pathlib.Path,
    dpi: int,
    force: bool,
    **kwargs,
) -> int:
    logger = structlog.get_logger()
    t0 = time.time()

    # Raise an error immediately if matplotlib is not available
    _import_matplotlib()

    if output_name.exists():
        if force:
            logger.debug('removing existing output file "%s"', output_name)
            output_name.unlink()
        else:
            raise RuntimeError(
                f'Refusing to overwrite file "{output_name}". Pass --force to overwrite existing file(s).'
            )

    logger.info('generating "%s" plot', plot_type)
    kwargs["logger"] = logger

    if "seed" in kwargs:
        logger.debug("setting seed to %d", kwargs["seed"])
        random.seed(kwargs["seed"])

    if plot_type in {"contact-map", "cm"}:
        plot_seeds = kwargs.pop("highlight_seeds")
        plot_stripes = kwargs.pop("highlight_stripes")
        ignore_stripe_heights = kwargs.pop("ignore_stripe_heights")

        if (plot_seeds or plot_stripes) and kwargs["stripepy_hdf5"] is None:
            raise RuntimeError("--stripepy-hdf5 is required when highlighting stripes or seeds.")

        if not plot_seeds and not plot_stripes:
            fig = _plot_hic_matrix(**kwargs)
        elif plot_seeds:
            fig = _plot_hic_matrix_with_seeds(**kwargs)
        else:
            kwargs["override_height"] = ignore_stripe_heights
            kwargs["mask_regions"] = ignore_stripe_heights
            fig = _plot_hic_matrix_with_stripes(**kwargs)
    elif plot_type in {"pseudodistribution", "pd"}:
        fig = _plot_pseudodistribution(**kwargs)
    elif plot_type in {"stripe-hist", "hist"}:
        fig = _plot_stripe_dimension_distribution(**kwargs)
    else:
        raise NotImplementedError

    fig.savefig(output_name, dpi=dpi)

    logger.info("DONE!")
    logger.info("plotting took %s seconds", pretty_format_elapsed_time(t0))

    return 0
