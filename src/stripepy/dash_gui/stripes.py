import numpy as np
import plotly.graph_objects as go
from colorscales import contrast

from stripepy.data_structures import ResultFile


def add_stripes_chrom_restriction(f, fig, chromosome_name, result, resolution, layers, color_map, rel_change):
    chromosome_name, _, spans = chromosome_name.partition(":")
    pre_span_length = 0
    for chrom_name, pre_span_lengths in f.chromosomes().items():
        if chrom_name == chromosome_name:
            break
        else:
            pre_span_length += pre_span_lengths
    if spans:
        pre_span_in_chromosome, _, end_limit = spans.partition("-")
        pre_span_in_chromosome = int(pre_span_in_chromosome.replace(",", ""))
        end_limit = int(end_limit.replace(",", ""))
        fig = extract_stripes_part_of_chromosome(
            fig,
            result,
            resolution,
            layers,
            pre_span_length + pre_span_in_chromosome,
            pre_span_in_chromosome,
            int(end_limit),
            color_map,
            rel_change,
        )
    else:
        fig = extract_stripes_whole_chromosome(fig, result, resolution, layers, pre_span_length, color_map, rel_change)
    return fig


def add_stripes_whole_chrom(
    f, fig, result, resolution, layers, chromosome_name, color_map, subtract_from_start, rel_change
):
    pre_chrom_span = 0
    for chrom_name, pre_span_lengths in f.chromosomes().items():
        if chrom_name == chromosome_name:
            fig = extract_stripes_whole_chromosome(
                fig,
                result,
                resolution,
                layers,
                pre_chrom_span,
                color_map,
                subtract_from_start,
                rel_change,
            )
        pre_chrom_span += pre_span_lengths
    return fig


def extract_stripes_whole_chromosome(
    fig, result, resolution, layers, pre_chrom_span, color_map, subtract_from_start, rel_change
):
    geo_frame_LT = result.get_stripe_geo_descriptors("LT")
    bio_frame_LT = result.get_stripe_bio_descriptors("LT")
    geo_frame_LT["relative_change"] = bio_frame_LT["rel_change"]
    geo_frame_LT = geo_frame_LT[geo_frame_LT["relative_change"] > rel_change]
    for rows in geo_frame_LT.iterrows():
        array = _get_correct_cells(rows)
        x_values, y_values = _get_square(array)
        fig.add_trace(
            _add_stripe_whole_chrom(
                x_values, y_values, resolution, pre_chrom_span, layers, color_map, subtract_from_start
            )
        )

    geo_frame_UT = result.get_stripe_geo_descriptors("UT")
    bio_frame_UT = result.get_stripe_bio_descriptors("UT")
    geo_frame_UT["relative_change"] = bio_frame_UT["rel_change"]
    geo_frame_UT = geo_frame_UT[geo_frame_UT["relative_change"] > rel_change]
    for rows in geo_frame_UT.iterrows():
        array = _get_correct_cells(rows)
        x_values, y_values = _get_square(array)
        fig.add_trace(
            _add_stripe_whole_chrom(
                x_values, y_values, resolution, pre_chrom_span, layers, color_map, subtract_from_start
            )
        )
    return fig


def extract_stripes_part_of_chromosome(
    fig, result, resolution, layers, pre_interval_span, margin, end_limit, color_map, rel_change
):
    geo_frame_LT = result.get_stripe_geo_descriptors("LT")
    bio_frame_LT = result.get_stripe_bio_descriptors("LT")
    geo_frame_LT["relative_change"] = bio_frame_LT["rel_change"]
    geo_frame_LT = geo_frame_LT[geo_frame_LT["relative_change"] > rel_change]
    for rows in geo_frame_LT.iterrows():
        array = _get_correct_cells(rows)
        array = _truncate_values(array, resolution, margin, end_limit)
        if array is None:
            continue
        x_values, y_values = _get_square(array)
        if x_values is None or y_values is None:
            continue
        fig.add_trace(_add_stripe_chrom_restriction(x_values, y_values, resolution, margin, layers, color_map))

    geo_frame_UT = result.get_stripe_geo_descriptors("UT")
    bio_frame_UT = result.get_stripe_bio_descriptors("UT")
    geo_frame_UT["relative_change"] = bio_frame_UT["rel_change"]
    geo_frame_UT = geo_frame_UT[geo_frame_UT["relative_change"] > rel_change]
    for rows in geo_frame_UT.iterrows():
        array = _get_correct_cells(rows)
        array = _truncate_values(array, resolution, margin, end_limit)
        if array is None:
            continue
        x_values, y_values = _get_square(array)
        if x_values is None or y_values is None:
            continue
        fig.add_trace(_add_stripe_chrom_restriction(x_values, y_values, resolution, margin, layers, color_map))
    return fig


def _get_square(array):
    x_array = np.array([array[0], array[1], array[1], array[0], array[0]])
    y_array = np.array([array[2], array[2], array[3], array[3], array[2]])
    return x_array, y_array


def _get_correct_cells(df_row):
    series = df_row[1]
    return series[2:]


def _truncate_values(array, resolution, margin, end_limit):
    if array[0] < margin / resolution or array[1] < margin / resolution:
        return None
    if array[0] > end_limit / resolution or array[1] > end_limit / resolution:
        return None
    if array[2] < margin / resolution and array[3] < margin / resolution:
        return None
    if array[2] > end_limit / resolution and array[3] > end_limit / resolution:
        return None
    array[0] = max(array[0], margin / resolution)
    array[2] = max(array[2], margin / resolution)
    array[1] = min(array[1], end_limit / resolution)
    array[3] = min(array[3], end_limit / resolution)
    return array


def _add_stripe_whole_chrom(cols, rows, resolution, margin, layer, color_map, subtract_from_start):
    if subtract_from_start:
        return go.Scatter(
            x=cols,
            y=rows,
            xaxis=layer[0],
            yaxis=layer[1],
            fillcolor=contrast(color_map, "stripe"),
            marker_color=contrast(color_map, "stripe"),
            hoverlabel={
                "bgcolor": contrast(color_map, "stripe"),
            },
        )
    return go.Scatter(
        x=cols + (margin / resolution),
        y=rows + (margin / resolution),
        xaxis=layer[0],
        yaxis=layer[1],
        fillcolor=contrast(color_map, "stripe"),
        marker_color=contrast(color_map, "stripe"),
        hoverlabel={
            "bgcolor": contrast(color_map, "stripe"),
        },
    )


def _add_stripe_chrom_restriction(cols, rows, resolution, margin, layer, color_map):
    return go.Scatter(
        x=cols - (margin / resolution),
        y=rows - (margin / resolution),
        xaxis=layer[0],
        yaxis=layer[1],
        fillcolor=contrast(color_map, "stripe"),
        marker_color=contrast(color_map, "stripe"),
        hoverlabel={
            "bgcolor": contrast(color_map, "stripe"),
        },
    )
