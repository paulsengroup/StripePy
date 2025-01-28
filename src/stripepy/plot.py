# Copyright (C) 2024 Roberto Rossini <roberros@uio.no>
#
# SPDX-License-Identifier: MIT

import functools
import itertools
import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd

from stripepy.IO import Result
from stripepy.utils.common import _DummyPyplot, _import_matplotlib, _import_pyplot
from stripepy.utils.persistence1d import Persistence1DTable

# Dummy values to not break type annotations when matplotlib is not available
plt = _DummyPyplot()
AxesImage = None


@functools.cache
def _get_custom_palettes() -> Dict[str, npt.NDArray]:
    # Source: https://github.com/open2c/cooltools/blob/master/cooltools/lib/plotting.py
    return {
        "fall": np.array(
            (
                (255, 255, 255),
                (255, 255, 204),
                (255, 237, 160),
                (254, 217, 118),
                (254, 178, 76),
                (253, 141, 60),
                (252, 78, 42),
                (227, 26, 28),
                (189, 0, 38),
                (128, 0, 38),
                (0, 0, 0),
            )
        )
        / 255,
        "fruit_punch": np.array(
            (
                (255, 255, 255),
                (255, 204, 204),
                (255, 153, 153),
                (255, 102, 102),
                (255, 50, 50),
                (255, 0, 0),
            )
        )
        / 255,
        "blues": np.array(
            (
                (255, 255, 255),
                (180, 204, 225),
                (116, 169, 207),
                (54, 144, 192),
                (5, 112, 176),
                (4, 87, 135),
                (3, 65, 100),
                (2, 40, 66),
                (1, 20, 30),
                (0, 0, 0),
            )
        )
        / 255,
        "acidblues": np.array(
            (
                (255, 255, 255),
                (162, 192, 222),
                (140, 137, 187),
                (140, 87, 167),
                (140, 45, 143),
                (120, 20, 120),
                (90, 15, 90),
                (60, 10, 60),
                (30, 5, 30),
                (0, 0, 0),
            )
        )
        / 255,
        "nmeth": np.array(
            (
                (236, 250, 255),
                (148, 189, 217),
                (118, 169, 68),
                (131, 111, 43),
                (122, 47, 25),
                (41, 0, 20),
            )
        )
        / 255,
    }


def _list_to_colormap(color_list, name=None):
    mpl = _import_matplotlib()
    color_list = np.array(color_list)
    if color_list.min() < 0:
        raise ValueError("Colors should be 0 to 1, or 0 to 255")
    if color_list.max() > 1.0:
        if color_list.max() > 255:
            raise ValueError("Colors should be 0 to 1 or 0 to 255")
        else:
            color_list = color_list / 255.0
    return mpl.colors.LinearSegmentedColormap.from_list(name, color_list, 256)


def _register_cmaps():
    mpl = _import_matplotlib()
    # make sure we are not trying to register color maps multiple times
    if hasattr(_register_cmaps, "called"):
        return
    for name, pal in _get_custom_palettes().items():
        mpl.colormaps.register(_list_to_colormap(pal, name))
        mpl.colormaps.register(_list_to_colormap(pal[::-1], f"{name}_r"))

    _register_cmaps.called = True


def _format_ticks(ax: plt.Axes, xaxis: bool = True, yaxis: bool = True, rotation: int = 30):
    """
    Function taken from https://cooltools.readthedocs.io/en/latest/notebooks/viz.html
    """
    _import_matplotlib()
    from matplotlib.ticker import EngFormatter, ScalarFormatter

    if xaxis:
        ax.xaxis.set_major_formatter(EngFormatter("b"))
    else:
        ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.tick_bottom()

    if yaxis:
        ax.yaxis.set_major_formatter(EngFormatter("b"))
    else:
        ax.yaxis.set_major_formatter(ScalarFormatter())

    if rotation != 0:
        ax.tick_params(axis="x", rotation=rotation)


def hic_matrix(
    matrix: npt.NDArray,
    region: Tuple[int, int],
    cmap="fruit_punch",
    log_scale: bool = True,
    with_colorbar: bool = False,
    fig: Optional[plt.Figure] = None,
    ax: Optional[plt.Axes] = None,
) -> Tuple[plt.Figure, plt.Axes, AxesImage]:
    """
    Plot the given matrix as a heatmap

    Parameters
    ----------
    matrix: npt.NDArray
        the 2D matrix to be plotted
    region: Tuple[int, int]
        a 2 int tuple containing the first and last genomic coordinates (bp) to be plotted
    cmap
        color map to be used for plotting
    log_scale: bool
        if True, plot the heatmap in log scale
    with_colorbar: bool
        if True, add a color bar for the plotted heatmap
    fig: Optional[plt.Figure]
        figure to use for plotting
    ax: Optional[plt.Axes]
        axis to use for plotting

    Returns
    -------
    fig: plt.Figure
        figure used for plotting
    ax: plt.Axes
        axis used for plotting
    img:
        image returned by ax.imshow()
    """
    mpl = _import_matplotlib()
    plt = _import_pyplot()
    _register_cmaps()

    multiplier = 1.15 if with_colorbar else 1.0
    if fig is None:
        if ax is not None:
            raise RuntimeError("ax should be None when fig is None")
        fig, ax = plt.subplots(1, 1, figsize=(6.4 * multiplier, 6.4))
    elif ax is None:
        raise RuntimeError("ax cannot be None when fig is not None")

    kwargs = {
        "extent": (region[0], region[1], region[1], region[0]),
        "cmap": cmap,
    }
    if log_scale:
        kwargs["norm"] = mpl.colors.LogNorm()
    else:
        kwargs["vmax"] = np.amax(matrix)

    img = ax.imshow(matrix, **kwargs)
    _format_ticks(ax)

    if with_colorbar:
        assert multiplier > 1
        fig.colorbar(img, ax=ax, fraction=multiplier - 1)

    return fig, ax, img


def pseudodistribution(
    pseudo_distrib_lt: Sequence[float],
    pseudo_distrib_ut: Sequence[float],
    region: Tuple[int, int],
    resolution: int,
    highlighted_points_lt: Optional[Sequence[int]] = None,
    highlighted_points_ut: Optional[Sequence[int]] = None,
    colors: Optional[Any] = None,
    fig: Optional[plt.Figure] = None,
    axs: Optional[Tuple[plt.Axes, plt.Axes]] = None,
) -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes]]:
    """
    Plot the given pseudo-distributions as a two line plots

    Parameters
    ----------
    pseudo_distrib_lt: Sequence[float]
        pseudo-distribution values computed on the lower-triangular matrix
    pseudo_distrib_ut: Sequence[float]
        pseudo-distribution values computed on the upper-triangular matrix
    region: Tuple[int, int]
        a 2 int tuple containing the first and last genomic coordinates (bp) to be plotted
    resolution: int
        resolution (bp) of the Hi-C matrix from which the pseudo-distribution were computed
    highlighted_points_lt: Optional[Sequence[int]]
        list of genomic coordinates (lower-triangular matrix) to be highlighted.
        Each pair of genomic coordinates determines a point cloud as follows:
        - for genomic coordinate x, we sample the value of pseudo_distrib_lt at x
        - each point cloud is scatterplotted with the color specified through the colors parameter.
    highlighted_points_ut: Optional[Sequence[int]]
        same as coords2scatter_lt but for the upper-triangular matrix
    colors: Optional[Any]
        one or more colors to be used when generating the scatter plot.
        When provided it should be either:
        - a single color: all points are plotted using the same color
        - a list colors: the list should contain at least
          max(len(color2scatter_lt), len(color2scatter_ut)) colors
    fig: Optional[plt.Figure]
        figure to use for plotting
    axs: Optional[Tuple[plt.Axes, plt.Axes]]
        axes to use for plotting

    Returns
    -------
    fig: plt.Figure
        figure used for plotting
    ax1: plt.Axes
        axis with the plot for upper-triangular pseudo-distribution
    ax2: plt.Axes
        axis with the plot for lower-triangular pseudo-distribution
    """
    plt = _import_pyplot()
    if fig is None:
        if axs is not None:
            raise RuntimeError("axs should be None when fig is None")
        fig, axs = plt.subplots(2, 1, sharex=True)
    elif axs is None:
        raise RuntimeError("axs cannot be None when fig is not None")

    if len(axs) != 2:
        raise RuntimeError("axs should be a tuple with exactly two plt.Axes")

    i1 = region[0] // resolution
    i2 = region[1] // resolution

    if colors is None:
        num_colors = 0
        if highlighted_points_lt is not None:
            num_colors = len(highlighted_points_lt)
        if highlighted_points_ut is not None:
            num_colors = max(num_colors, len(highlighted_points_ut))

        if num_colors > 0:
            colors = ["blue"] * num_colors

    def _plot(data, points, ax, title):
        ax.plot(
            np.arange(i1, i2) * resolution,
            data[i1:i2],
            color="red",
            linewidth=0.5,
            linestyle="solid",
        )
        if points is not None:
            for n, x in enumerate(points):
                if not region[0] <= x <= region[1]:
                    raise RuntimeError("point outside of bounds")
                ax.plot(
                    x,
                    data[x // resolution],
                    marker=".",
                    linestyle="",
                    markersize=5,
                    color=colors[n],
                )
        ax.set(title=title, ylim=(0, 1), ylabel="Pseudo-distribution")
        ax.grid(True)

    _plot(pseudo_distrib_ut, highlighted_points_ut, axs[0], "Upper Triangular")
    _plot(pseudo_distrib_lt, highlighted_points_lt, axs[1], "Lower Triangular")

    _format_ticks(axs[1], yaxis=False, xaxis=True)

    axs[1].set(xlabel="Genomic coordinates (bp)")

    return fig, axs


def plot_sites(
    sites: Sequence[int],
    region: Tuple[int, int],
    location: Optional[str],
    color="blue",
    fig: Optional[plt.Figure] = None,
    ax: Optional[plt.Axes] = None,
):
    """
    Plot one or more sites of interest (e.g. stipe seeds).

    Parameters
    ----------
    sites: Sequence[int]
        list of genomic coordinates (bp) corresponding to the x-coordinates of the sites to be plotted
    region: Tuple[int, int]
        a 2 int tuple containing the first and last genomic coordinates (bp) to be plotted
    location: Optional[str]
        if "lower" (resp. "upper"), then plot sites only on the lower (resp. upper) part of
        the Hi-C matrix. Otherwise, plot sites spanning the whole Hi-C matrix
    color:
        color to be used for plotting
    fig: Optional[plt.Figure]
        figure to use for plotting
    ax: Optional[Tuple[plt.Axes, plt.Axes]]
        axis to use for plotting

    Returns
    -------
    fig: plt.Figure
        figure used for plotting
    ax: plt.Axes
        axis used for plotting
    """
    plt = _import_pyplot()

    if location is not None and location not in {"lower", "upper"}:
        raise ValueError('location should be "lower" or "upper"')

    if fig is None:
        if ax is not None:
            raise RuntimeError("ax should be None when fig is None")
        fig, ax = plt.subplots(1, 1)
    elif ax is None:
        raise RuntimeError("ax cannot be None when fig is not None")

    if len(sites) == 0:
        warnings.warn("stripepy.plot.plot_sites: no sites to plot!")

    for site in sites:
        if not region[0] <= site <= region[1]:
            raise RuntimeError("site outside of bounds")

        x = [site] * 2
        if location == "lower":
            y = [site, region[1] - 1]
        elif location == "upper":
            y = [region[0] + 1, site]
        else:
            y = [region[0] + 1, region[1] - 1]

        ax.plot(x, y, color=color, linestyle="dashed", linewidth=1)

    _format_ticks(ax)

    return fig, ax


def mask_regions_1d(
    matrix: npt.NDArray,
    resolution: int,
    location: Optional[str],
    whitelist: Optional[List[Tuple[int, int]]] = None,
    blacklist: Optional[List[Tuple[int, int]]] = None,
) -> npt.NDArray:
    """
    Mask rows or columns of the given matrix based on whitelisted or blacklisted regions.

    Parameters
    ----------
    matrix: npt.NDArray
        matrix to be masked. The matrix is expected to be a 2D matrix and be symmetric.
    resolution: int
        matrix resolution in bp.
    location: str
        location where the selective masking should be applied.
        When "lower", values in the upper triangular matrix are all set to 0 and values
        in the lower triangular matrix are masked based on whitelisted or blacklisted regions.
        When "upper", values in the lower triangular matrix are all set to 0 and values
        in the upper triangular matrix are masked based on whitelisted or blacklisted regions.
    whitelist: Optional[List[Tuple[int, int]]]
        list of regions to NOT be masked. Regions should be expressed in matrix coordinates.
    blacklist: Optional[List[Tuple[int, int]]]
        list of regions to be masked. Regions should be expressed in matrix coordinates.

    Returns
    -------
    matrix: npt.NDArray
        the masked matrix
    """
    if location is not None and location not in {"lower", "upper"}:
        raise ValueError('location should be "lower" or "upper"')

    if (whitelist is None and blacklist is None) or (whitelist is not None and blacklist is not None):
        raise ValueError("please specify either whitelist or blacklist")

    m = matrix.copy()
    mask = None
    if whitelist is not None:
        idx = []
        if len(whitelist) > 0:
            for i1, i2 in whitelist:
                i1 //= resolution
                i2 //= resolution
                idx.extend(list(range(i1, i2 + 1)))

        mask = np.setdiff1d(np.arange(m.shape[0]), np.unique(idx))

    if blacklist is not None:
        mask = []
        if len(blacklist) > 0:
            for i1, i2 in blacklist:
                i1 //= resolution
                i2 //= resolution
                mask.extend(list(range(i1, i2 + 1)))

        mask = np.unique(mask)

    if mask is not None:
        m[:, mask] = 0

    if location == "upper":
        return np.triu(m)
    if location == "lower":
        return np.tril(m)
    return np.tril(m.T) + np.triu(m)


def draw_boxes(
    regions: List[Tuple[int, int, int, int]],
    bound_box: Tuple[int, int],
    fig: Optional[plt.Figure] = None,
    ax: Optional[plt.Axes] = None,
    **kwargs,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Draw bounding boxes around a list of regions of interest.

    Parameters
    ----------
    regions:
        a list of genomic regions, each consisting of a 4-element tuple (x, y, width, height).
        All coordinates are in genomic units (i.e. bp).
    bound_box:
        first and last genomic position that should be plotted.
        Used to ensure that boxes do not extend outside the plotted area.
    fig:
        figure to use for plotting
    ax:
        axis to use for plotting
    **kwargs:
        additional keyword arguments to pass to plt.plot()

    Returns
    -------
    figure, axes:
        the plt.Figure and plt.Axes used for plotting
    """
    plt = _import_pyplot()

    if fig is None:
        if ax is not None:
            raise RuntimeError("ax should be None when fig is None")
        fig, ax = plt.subplots(1, 1)
    elif ax is None:
        raise RuntimeError("ax cannot be None when fig is not None")

    if "color" not in kwargs:
        kwargs["color"] = "blue"
    if "linestyle" not in kwargs:
        kwargs["linestyle"] = "dashed"

    for x, y, width, height in regions:
        x = np.clip([x, x, x + width, x + width, x], bound_box[0], bound_box[1])
        y = np.clip([y, y + height, y + height, y + width, y], bound_box[0], bound_box[1])
        ax.plot(x, y, **kwargs)
    return fig, ax


def _fetch_persistence_maximum_points(result: Result, resolution: int, start: int, end: int) -> Dict[str, npt.NDArray]:
    def fetch(v: npt.NDArray[int], left_bound: int, right_bound: int) -> Tuple[npt.NDArray[int], npt.NDArray[int]]:
        assert left_bound >= 0
        assert right_bound >= left_bound

        idx = np.where((v >= left_bound) & (v < right_bound))[0]
        return idx.astype(int), v[idx].astype(int)

    pd_lt = result.get("pseudodistribution", "LT")
    pd_ut = result.get("pseudodistribution", "UT")

    min_persistence = result.min_persistence
    lt_persistence = Persistence1DTable.calculate_persistence(pd_lt, min_persistence=min_persistence, sort_by="max")
    ut_persistence = Persistence1DTable.calculate_persistence(pd_ut, min_persistence=min_persistence, sort_by="max")

    lt_idx, lt_seeds = fetch(
        lt_persistence.max.index.to_numpy(),
        start // resolution,
        end // resolution,
    )
    ut_idx, ut_seeds = fetch(
        ut_persistence.max.index.to_numpy(),
        start // resolution,
        end // resolution,
    )

    return {
        "pseudodistribution_lt": pd_lt,
        "pseudodistribution_ut": pd_ut,
        "seeds_lt": lt_seeds,
        "seeds_ut": ut_seeds,
        "seed_indices_lt": lt_idx,
        "seed_indices_ut": ut_idx,
    }


def _plot_pseudodistribution(
    result: Result, resolution: int, start: int, end: int, title: Optional[str]
) -> Tuple[plt.Figure, npt.NDArray[plt.Axes]]:
    data = _fetch_persistence_maximum_points(result, resolution, start, end)

    fig, axs = pseudodistribution(
        data["pseudodistribution_lt"],
        data["pseudodistribution_ut"],
        region=(start, end),
        resolution=resolution,
        highlighted_points_lt=data["seeds_lt"] * resolution,
        highlighted_points_ut=data["seeds_ut"] * resolution,
    )

    if title is not None:
        fig.suptitle(title)
    fig.tight_layout()
    return fig, axs


def _plot_hic_matrix(
    matrix: npt.NDArray,
    start: int,
    end: int,
    cmap: str = "fruit_punch",
    log_scale: bool = True,
    title: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    fig, ax, _ = hic_matrix(
        matrix,
        (start, end),
        cmap=cmap,
        log_scale=log_scale,
        with_colorbar=True,
    )

    if title is not None:
        fig.suptitle(title)
    fig.tight_layout()

    return fig, ax


def _plot_hic_matrix_with_seeds(
    matrix: npt.NDArray,
    result: Result,
    resolution: int,
    start: int,
    end: int,
    cmap: str = "fruit_punch",
    log_scale: bool = True,
    title: Optional[str] = None,
) -> Tuple[plt.Figure, npt.NDArray[plt.Axes]]:
    plt = _import_pyplot()

    data = _fetch_persistence_maximum_points(result, resolution, start, end)

    fig, axs = plt.subplots(1, 2, figsize=(13.5, 6.4), sharey=True)

    for ax in axs:
        _, _, img = hic_matrix(
            matrix,
            (start, end),
            cmap=cmap,
            log_scale=log_scale,
            with_colorbar=False,
            fig=fig,
            ax=ax,
        )

    plot_sites(
        data["seeds_lt"] * resolution,
        (start, end),
        location="lower",
        fig=fig,
        ax=axs[0],
    )

    plot_sites(
        data["seeds_ut"] * resolution,
        (start, end),
        location="upper",
        fig=fig,
        ax=axs[1],
    )

    axs[0].set(title="Lower Triangular")
    axs[1].set(title="Upper Triangular")

    if title is not None:
        fig.suptitle(title)
    fig.tight_layout()

    fig.subplots_adjust(right=0.94)
    cbar_ax = fig.add_axes((0.95, 0.05, 0.015, 0.9))
    fig.colorbar(img, cax=cbar_ax)  # noqa

    return fig, axs


def _plot_hic_matrix_with_stripes(
    matrix: npt.NDArray,
    result: Result,
    resolution: int,
    start: int,
    end: int,
    relative_change_threshold: float = 0.0,
    cmap: str = "fruit_punch",
    override_height: Optional[int] = None,
    mask_regions: bool = False,
    log_scale: bool = False,
    title: Optional[str] = None,
) -> Tuple[plt.Figure, npt.NDArray[plt.Axes]]:
    plt = _import_pyplot()

    geo_descriptors_lt = result.get_stripe_geo_descriptors("LT")
    geo_descriptors_ut = result.get_stripe_geo_descriptors("UT")

    for col in ("seed", "left_bound", "right_bound", "top_bound", "bottom_bound"):
        geo_descriptors_lt[col] *= resolution
        geo_descriptors_ut[col] *= resolution

    chrom_size = result.chrom[1]

    if relative_change_threshold > 0:
        mask_lt = (
            result.get_stripe_bio_descriptors("LT")["rel_change"].iloc[geo_descriptors_lt.index]
            >= relative_change_threshold
        )
        mask_ut = (
            result.get_stripe_bio_descriptors("UT")["rel_change"].iloc[geo_descriptors_ut.index]
            >= relative_change_threshold
        )

        geo_descriptors_lt = geo_descriptors_lt[mask_lt]
        geo_descriptors_ut = geo_descriptors_ut[mask_ut]

    geo_descriptors_lt = geo_descriptors_lt[geo_descriptors_lt["right_bound"].between(start, end, inclusive="both")]
    geo_descriptors_ut = geo_descriptors_ut[geo_descriptors_ut["right_bound"].between(start, end, inclusive="both")]

    outlines_lt = [
        (min(lb - start, chrom_size), min(rb - start, chrom_size), min(bb - tb, chrom_size))
        for lb, rb, bb, tb in geo_descriptors_lt[["left_bound", "right_bound", "bottom_bound", "top_bound"]].itertuples(
            index=False
        )
    ]

    outlines_ut = [
        (min(lb - start, chrom_size), min(rb - start, chrom_size), min(tb - bb, chrom_size))
        for lb, rb, bb, tb in geo_descriptors_ut[["left_bound", "right_bound", "bottom_bound", "top_bound"]].itertuples(
            index=False
        )
    ]

    if mask_regions:
        whitelist = [(x, y) for x, y, _ in outlines_lt]
        m1 = np.triu(matrix) + np.tril(
            mask_regions_1d(
                matrix,
                resolution,
                whitelist=whitelist,
                location="lower",
            ),
            k=-1,
        )

        whitelist = [(x, y) for x, y, _ in outlines_ut]
        m2 = np.tril(matrix) + np.triu(
            mask_regions_1d(
                matrix,
                resolution,
                whitelist=whitelist,
                location="upper",
            ),
            k=1,
        )
    else:
        m1 = matrix
        m2 = matrix

    fig, axs = plt.subplots(1, 2, figsize=(13.5, 6.4), sharey=True)

    _, _, img = hic_matrix(
        m1,
        (start, end),
        cmap=cmap,
        log_scale=log_scale,
        with_colorbar=False,
        fig=fig,
        ax=axs[0],
    )
    hic_matrix(
        m2,
        (start, end),
        cmap=cmap,
        log_scale=log_scale,
        with_colorbar=False,
        fig=fig,
        ax=axs[1],
    )

    rectangles = []
    for lb, ub, height in outlines_lt:
        x = min(start + lb, chrom_size)
        y = min(start + lb, chrom_size)
        width = min(ub - lb, chrom_size)
        if override_height is not None:
            height = min(end - x, override_height)
        rectangles.append((x, y, width, height))

    draw_boxes(rectangles, (start, end), color="blue", linestyle="dashed", fig=fig, ax=axs[0])

    rectangles = []
    for lb, ub, height in outlines_ut:
        x = min(start + lb, chrom_size)
        y = min(start + lb, chrom_size)
        width = min(ub - lb, chrom_size)
        if override_height is not None:
            height = min(start - x, override_height)
        rectangles.append((x, y, width, height))

    draw_boxes(rectangles, (start, end), color="blue", linestyle="dashed", fig=fig, ax=axs[1])

    axs[0].set(title="Lower Triangular")
    axs[1].set(title="Upper Triangular")

    if title is not None:
        fig.suptitle(title)
    fig.tight_layout()

    fig.subplots_adjust(right=0.94)
    cbar_ax = fig.add_axes((0.95, 0.05, 0.015, 0.9))
    fig.colorbar(img, cax=cbar_ax)

    return fig, axs


def _plot_stripe_dimension_distribution(
    geo_descriptors_lt: pd.DataFrame,
    geo_descriptors_ut: pd.DataFrame,
    resolution: int,
) -> Tuple[plt.Figure, npt.NDArray[plt.Axes]]:
    plt = _import_pyplot()
    from matplotlib.ticker import EngFormatter

    fig, axs = plt.subplots(2, 2, figsize=(12.8, 8), sharex="col", sharey="col")

    stripe_widths_lt = (geo_descriptors_lt["right_bound"] - geo_descriptors_lt["left_bound"]) * resolution
    stripe_heights_lt = (geo_descriptors_lt["bottom_bound"] - geo_descriptors_lt["top_bound"]) * resolution

    stripe_widths_ut = (geo_descriptors_ut["right_bound"] - geo_descriptors_ut["left_bound"]) * resolution
    stripe_heights_ut = (geo_descriptors_ut["bottom_bound"] - geo_descriptors_ut["top_bound"]) * resolution

    for ax in itertools.chain.from_iterable(axs):
        ax.xaxis.set_major_formatter(EngFormatter("b"))
        ax.xaxis.tick_bottom()

    axs[0][0].hist(stripe_widths_lt, bins=max(1, (stripe_widths_lt.max() - stripe_widths_lt.min()) // resolution))
    axs[0][1].hist(stripe_heights_lt, bins="auto")
    axs[1][0].hist(stripe_widths_ut, bins=max(1, (stripe_widths_ut.max() - stripe_widths_ut.min()) // resolution))
    axs[1][1].hist(stripe_heights_ut, bins="auto")

    axs[0][0].set(title="Stripe width distribution (lower triangle)", ylabel="Count")
    axs[0][1].set(title="Stripe height distribution (lower triangle)")
    axs[1][0].set(title="Stripe width distribution (upper triangle)", xlabel="Width (bp)", ylabel="Count")
    axs[1][1].set(title="Stripe height distribution (upper triangle)", xlabel="Height (bp)")

    fig.tight_layout()
    return fig, axs


def plot(
    result: Result,
    resolution: int,
    plot_type: str,
    start: int,
    end: int,
    matrix: Optional[npt.NDArray] = None,
    **kwargs,
) -> Tuple[plt.Figure, npt.NDArray[plt.Axes]]:
    assert start >= 0
    assert start <= end
    assert resolution > 0

    valid_plot_types = {
        "pseudodistribution",
        "matrix",
        "matrix_with_seeds",
        "matrix_with_stripes_masked",
        "matrix_with_stripes",
        "geo_descriptors",
    }
    if plot_type not in valid_plot_types:
        raise ValueError(f"{plot_type} is not a valid plot type: valid types are {', '.join(valid_plot_types)}")

    if matrix is None and plot_type.startswith("matrix"):
        raise ValueError(f'matrix parameter is required when plot_type is "{plot_type}"')

    title = kwargs.pop("title", f"{result.chrom[0]}:{start}-{end}")
    if not title:
        title = None

    if plot_type == "pseudodistribution":
        return _plot_pseudodistribution(
            result,
            resolution,
            start,
            end,
            title=title,
        )

    if plot_type == "matrix":
        return _plot_hic_matrix(
            matrix,
            start,
            end,
            title=title,
            **kwargs,
        )

    if plot_type == "matrix_with_seeds":
        return _plot_hic_matrix_with_seeds(
            matrix,
            result,
            resolution,
            start,
            end,
            title=title,
            **kwargs,
        )

    if plot_type == "matrix_with_stripes_masked":
        override_height = kwargs.pop("override_height", True)
        return _plot_hic_matrix_with_stripes(
            matrix,
            result,
            resolution,
            start,
            end,
            title=title,
            mask_regions=True,
            override_height=end - start if override_height else None,
            **kwargs,
        )

    if plot_type == "matrix_with_stripes":
        override_height = kwargs.pop("override_height", False)
        return _plot_hic_matrix_with_stripes(
            matrix,
            result,
            resolution,
            start,
            end,
            title=title,
            mask_regions=False,
            override_height=end - start if override_height else None,
            **kwargs,
        )

    if plot_type == "geo_descriptors":
        df1 = kwargs.get("stripes_lt", result.get_stripe_geo_descriptors("LT"))
        df2 = kwargs.get("stripes_ut", result.get_stripe_geo_descriptors("UT"))

        left_bound = start // resolution
        right_bound = (end + resolution - 1) // resolution

        df1 = df1[df1["seed"].between(left_bound, right_bound, inclusive="both")]
        df2 = df2[df2["seed"].between(left_bound, right_bound, inclusive="both")]

        return _plot_stripe_dimension_distribution(
            df1,
            df2,
            resolution,
        )
