import numpy as np
import plotly.graph_objects as go

from stripepy.data_structures import ResultFile


def add_stripes_chrom_restriction(f, fig, chromosome_name, data_string, resolution, layers):
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
            data_string,
            chromosome_name,
            resolution,
            layers,
            pre_span_length + pre_span_in_chromosome,
            int(end_limit),
        )
    else:
        fig = extract_stripes_whole_chromosome(fig, data_string, chromosome_name, resolution, layers, 0)
    return fig


def add_stripes_whole_chrom(f, fig, data_string, resolution, layers):
    pre_chrom_span = 0
    for chrom_name, pre_span_lengths in f.chromosomes().items():
        fig = extract_stripes_whole_chromosome(fig, data_string, chrom_name, resolution, layers, pre_chrom_span)
        pre_chrom_span += pre_span_lengths
    return fig


def extract_stripes_whole_chromosome(fig, data_string, chrom_name, resolution, layers, margin):
    with ResultFile(data_string) as h5:
        geo_frame_LT = h5.get(chrom_name, "geo_descriptors", "LT")
        bio_frame_LT = h5.get(chrom_name, "bio_descriptors", "LT")
        geo_frame_LT["relative_change"] = bio_frame_LT["rel_change"]
        for rows in geo_frame_LT.iterrows():
            array = _get_correct_cols(rows)
            x_values, y_values = _get_square(array)
            fig.add_trace(_add_stripe_whole_chrom(x_values, y_values, resolution, margin, layers))

        geo_frame_UT = h5.get(chrom_name, "geo_descriptors", "UT")
        bio_frame_UT = h5.get(chrom_name, "bio_descriptors", "UT")
        geo_frame_UT["relative_change"] = bio_frame_UT["rel_change"]
        for rows in geo_frame_UT.iterrows():
            array = _get_correct_cols(rows)
            x_values, y_values = _get_square(array)
            fig.add_trace(_add_stripe_whole_chrom(x_values, y_values, resolution, margin, layers))
    return fig


def extract_stripes_part_of_chromosome(fig, data_string, chrom_name, resolution, layers, margin, end_limit):
    with ResultFile(data_string) as h5:
        geo_frame_LT = h5.get(chrom_name, "geo_descriptors", "LT")
        bio_frame_LT = h5.get(chrom_name, "bio_descriptors", "LT")
        geo_frame_LT["relative_change"] = bio_frame_LT["rel_change"]
        for rows in geo_frame_LT.iterrows():
            array = _get_correct_cols(rows)
            x_values, y_values = _get_square(array)
            if _is_within(x_values, y_values, (margin, end_limit), resolution):
                fig.add_trace(_add_stripe_chrom_restriction(x_values, y_values, resolution, margin, layers))
            else:
                continue

        geo_frame_UT = h5.get(chrom_name, "geo_descriptors", "UT")
        bio_frame_UT = h5.get(chrom_name, "bio_descriptors", "UT")
        geo_frame_UT["relative_change"] = bio_frame_UT["rel_change"]
        for rows in geo_frame_UT.iterrows():
            array = _get_correct_cols(rows)
            x_values, y_values = _get_square(array)
            if _is_within(x_values, y_values, (margin, end_limit), resolution):
                fig.add_trace(_add_stripe_chrom_restriction(x_values, y_values, resolution, margin, layers))
            else:
                continue
    return fig


def _get_square(array):
    x_array = np.array([array[0], array[1], array[1], array[0], array[0]])
    y_array = np.array([array[2], array[2], array[3], array[3], array[2]])
    return x_array, y_array


def _get_correct_cols(row):
    return np.array([row[1][2], row[1][3], row[1][4], row[1][5]])


def _is_within(col_values, row_values, borders, resolution):
    if col_values[0] < borders[0] / resolution:
        return False
    elif col_values[1] > borders[1] / resolution:
        return False
    elif row_values[0] < borders[0] / resolution:
        return False
    elif row_values[2] > borders[1] / resolution:
        return False
    else:
        return True


def _add_stripe_whole_chrom(cols, rows, resolution, margin, layer):
    return go.Scatter(
        x=cols + (margin / resolution),
        y=rows + (margin / resolution),
        xaxis=layer[0],
        yaxis=layer[1],
        fillcolor="green",
        marker_color="green",
    )


def _add_stripe_chrom_restriction(cols, rows, resolution, margin, layer):
    return go.Scatter(
        x=cols - (margin / resolution),
        y=rows - (margin / resolution),
        xaxis=layer[0],
        yaxis=layer[1],
        fillcolor="green",
        marker_color="green",
    )
