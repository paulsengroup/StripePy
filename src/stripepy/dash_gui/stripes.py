import numpy as np
import plotly.graph_objects as go
from colorscales import contrast


def add_stripes(
    f,
    fig,
    result,
    resolution,
    layers,
    color_map,
    chromosome_name,
    rel_change,
    margin=0,
    end_limit=0,
    is_whole_chromosome=True,
):
    length_of_earlier_chromosomes = 0
    if is_whole_chromosome:
        for chrom_name, chromosome_lengths in f.chromosomes().items():
            if chrom_name == chromosome_name:
                end_limit = length_of_earlier_chromosomes + chromosome_lengths
                break
            else:
                length_of_earlier_chromosomes += chromosome_lengths
                pass

    fig = extract_stripes(
        fig,
        result,
        resolution,
        layers,
        color_map,
        rel_change,
        length_of_earlier_chromosomes + margin,
        end_limit,
        is_whole_chromosome,
    )
    return fig


def extract_stripes(fig, result, resolution, layers, color_map, rel_change, margin, end_limit, is_whole_chromosome):
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
        fig.add_trace(_draw_stripe(x_values, y_values, resolution, margin, layers, color_map, is_whole_chromosome))

    geo_frame_UT = result.get_stripe_geo_descriptors("UT")
    bio_frame_UT = result.get_stripe_bio_descriptors("UT")
    geo_frame_UT["relative_change"] = bio_frame_UT["rel_change"]
    geo_frame_UT = geo_frame_UT[geo_frame_UT["relative_change"] > rel_change]
    for rows in geo_frame_UT.iterrows():
        array = _get_correct_cells(rows)
        if not is_whole_chromosome:
            array = _truncate_values(array, resolution, margin, end_limit)
            if array is None:
                continue
        x_values, y_values = _get_square(array)
        if x_values is None or y_values is None:
            continue
        fig.add_trace(_draw_stripe(x_values, y_values, resolution, margin, layers, color_map, is_whole_chromosome))
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


def _draw_stripe(cols, rows, resolution, margin, layer, color_map, is_whole_chromosome):
    if is_whole_chromosome:
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
    else:
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


def add_stripes_visualisation_change(
    fig,
    stripe_list,
    resolution,
    color_map,
    chromosome_name,
    relative_change_constant,
    layer_x,
    layer_y,
):
    chrom_name, _, spans = chromosome_name.partition(":")
    if spans:
        pre_span_in_chromosome, _, end_limit = spans.partition("-")
        margin = int(pre_span_in_chromosome.replace(",", ""))
        end_limit = int(end_limit.replace(",", ""))
    for stripe in stripe_list:
        seed, top_pers, left_bound, right_bound, top_bound, bottom_bound, rel_change = stripe
        if relative_change_constant > rel_change:
            continue
        array = [left_bound, right_bound, top_bound, bottom_bound]
        array = _truncate_values(array, resolution, margin, end_limit)
        if array is None:
            continue
        x_values, y_values = _get_square(array)
        fig.add_trace(
            _add_stripe_chrom_restriction(x_values, y_values, resolution, margin, (layer_x, layer_y), color_map)
        )
    return fig


def add_stripes_rel_change_filter(
    fig, stripe_list, resolution, color_map, relative_change_threshold, traces, margin, end_limit
):
    for stripe in stripe_list:
        seed, top_pers, left_bound, right_bound, top_bound, bottom_bound, rel_change = stripe
        if rel_change < relative_change_threshold:
            continue
        array = [left_bound, right_bound, top_bound, bottom_bound]
        array = _truncate_values(array, resolution, margin, end_limit)
        if array is None:
            continue
        x_values, y_values = _get_square(array)
        fig.add_trace(_draw_stripe(x_values, y_values, resolution, margin, traces, color_map, False))
    return fig
