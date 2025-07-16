import numpy as np
import plotly.graph_objects as go

from stripepy.data_structures import ResultFile


def add_stripes_chrom_restriction(f, fig, chromosome_name, result, resolution, layers, color_map):
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
            int(end_limit),
            color_map,
        )
    else:
        fig = extract_stripes_whole_chromosome(fig, result, resolution, layers, pre_span_length, color_map)
    return fig


def add_stripes_whole_chrom(f, fig, result, resolution, layers, chromosome_name, color_map):
    pre_chrom_span = 0
    for chrom_name, pre_span_lengths in f.chromosomes().items():
        if chrom_name == chromosome_name:
            fig = extract_stripes_whole_chromosome(fig, result, resolution, layers, pre_chrom_span, color_map)
        pre_chrom_span += pre_span_lengths
    return fig


def extract_stripes_whole_chromosome(fig, result, resolution, layers, margin, color_map):
    geo_frame_LT = result.get_stripe_geo_descriptors("LT")
    bio_frame_LT = result.get_stripe_bio_descriptors("LT")
    geo_frame_LT["relative_change"] = bio_frame_LT["rel_change"]
    for rows in geo_frame_LT.iterrows():
        array = _get_correct_cells(rows)
        x_values, y_values = _get_square(array)
        fig.add_trace(_add_stripe_whole_chrom(x_values, y_values, resolution, margin, layers, color_map))

    geo_frame_UT = result.get_stripe_geo_descriptors("UT")
    bio_frame_UT = result.get_stripe_bio_descriptors("UT")
    geo_frame_UT["relative_change"] = bio_frame_UT["rel_change"]
    for rows in geo_frame_UT.iterrows():
        array = _get_correct_cells(rows)
        x_values, y_values = _get_square(array)
        fig.add_trace(_add_stripe_whole_chrom(x_values, y_values, resolution, margin, layers, color_map))
    return fig


def extract_stripes_part_of_chromosome(fig, result, resolution, layers, margin, end_limit, color_map):
    geo_frame_LT = result.get_stripe_geo_descriptors("LT")
    bio_frame_LT = result.get_stripe_bio_descriptors("LT")
    geo_frame_LT["relative_change"] = bio_frame_LT["rel_change"]
    for rows in geo_frame_LT.iterrows():
        array = _get_correct_cells(rows)
        x_values, y_values = _get_square(array)
        if _is_within(x_values, y_values, (margin, end_limit)):
            fig.add_trace(_add_stripe_chrom_restriction(x_values, y_values, resolution, margin, layers), color_map)
        else:
            continue

    geo_frame_UT = result.get_stripe_geo_descriptors("UT")
    bio_frame_UT = result.get_stripe_bio_descriptors("UT")
    geo_frame_UT["relative_change"] = bio_frame_UT["rel_change"]
    for rows in geo_frame_UT.iterrows():
        array = _get_correct_cells(rows)
        x_values, y_values = _get_square(array)
        if _is_within(x_values, y_values, (margin, end_limit)):
            fig.add_trace(_add_stripe_chrom_restriction(x_values, y_values, resolution, margin, layers), color_map)
        else:
            continue
    return fig


def _get_square(array):
    x_array = np.array([array[0], array[1], array[1], array[0], array[0]])
    y_array = np.array([array[2], array[2], array[3], array[3], array[2]])
    return x_array, y_array


def _get_correct_cells(df_row):
    series = df_row[1]
    return series[2:]


def _is_within(col_values, row_values, borders):
    if col_values[0] < borders[0] and col_values[1] < borders[0]:
        return False
    elif col_values[0] > borders[1] and col_values[1] > borders[1]:
        return False
    elif row_values[0] < borders[0] and row_values[1] < borders[0]:
        return False
    elif row_values[0] > borders[1] and row_values[1] > borders[1]:
        return False
    else:
        return True


def _add_stripe_whole_chrom(cols, rows, resolution, margin, layer, color_map):
    return go.Scatter(
        x=cols + (margin / resolution),
        y=rows + (margin / resolution),
        xaxis=layer[0],
        yaxis=layer[1],
        fillcolor="green",
        marker_color="green",
    )


def _add_stripe_chrom_restriction(cols, rows, resolution, margin, layer, color_map):
    return go.Scatter(
        x=cols - (margin / resolution),
        y=rows - (margin / resolution),
        xaxis=layer[0],
        yaxis=layer[1],
        fillcolor="green",
        marker_color="green",
    )
