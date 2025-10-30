# Copyright (C) 2025 Bendik Berg <bendber@ifi.uio.no>
#
# SPDX-License-Identifier: MIT
from pathlib import Path

import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from callbacks import (
    call_stripes_callback,
    filter_stripes_callback,
    look_for_file_callback,
    look_for_normalizations_under_current_resolution_callback,
    open_file_dialog_callback,
    pick_saved_callback,
    update_plot_callback,
)
from components.dbc_warnings import warning_gen_wide, warning_stale_component
from components.layout import layout
from dash import Dash, Input, Output, State, no_update

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME])


app.layout = layout()


@app.callback(
    Output("file-path", "value", allow_duplicate=True),
    Output("warning-window", "children", allow_duplicate=True),
    Input("filepath-dialog", "n_clicks"),
    State("last-used-file-directory", "children"),
    prevent_initial_call=True,
    running=[
        (Output("resolution", "disabled"), True, False),
    ],
)
def open_file_dialog(n_clicks, base_directory):
    return open_file_dialog_callback(base_directory)


@app.callback(
    Output("resolution", "options"),
    Output("resolution", "value", allow_duplicate=True),
    Output("chromosomes", "children"),
    Output("last-used-file-directory", "children"),
    Output("chromosome-name-icon", "hidden"),
    Output("color-map-icon", "hidden"),
    Output("normalization-icon", "hidden"),
    Output("resolution", "disabled"),
    Output("chromosome-name", "disabled"),
    Output("color-map", "disabled"),
    Output("submit-chromosome", "disabled"),
    Output("normalization", "disabled"),
    Output("data", "hidden"),
    Output("warning-window", "children", allow_duplicate=True),
    Input("file-path", "value"),
    prevent_initial_call=True,
)
def look_for_file(file_path):
    return look_for_file_callback(file_path)


@app.callback(
    Output("file-path", "value", allow_duplicate=True),
    Output("resolution", "value", allow_duplicate=True),
    Output("radio-log", "value"),
    Output("chromosome-name", "value"),
    Output("normalization", "value", allow_duplicate=True),
    Output("submit-chromosome", "n_clicks"),
    Output("warning-window", "children", allow_duplicate=True),
    Input("pick-from-saved", "n_clicks"),
    State("files-list", "value"),
    State("submit-chromosome", "n_clicks"),
    prevent_initial_call=True,
)
def pick_saved(n_clicks, saved_string, update_plot_n_clicks):
    return pick_saved_callback(saved_string, update_plot_n_clicks)


@app.callback(
    Output("normalization", "options", allow_duplicate=True),
    Output("normalization", "value", allow_duplicate=True),
    Input("resolution", "value"),
    State("file-path", "value"),
    State("normalization", "value"),
    prevent_initial_call=True,
    running=[
        (Output("normalization", "disabled"), True, False),
    ],
)
def look_for_normalizations_under_current_resolution(resolution, path, current_normalization):
    return look_for_normalizations_under_current_resolution_callback(resolution, path, current_normalization)


def _disable_radio_elements(become_disabled):
    return [
        {"label": "log scale", "value": "log scale", "disabled": become_disabled},
        {"label": "linear scale", "value": "linear scale", "disabled": become_disabled},
    ]


@app.callback(
    Output("HeatMap", "figure", allow_duplicate=True),
    Output("files-list", "options"),
    Output("heat-map", "hidden"),
    Output("last-used-path", "children"),
    Output("last-used-resolution", "children"),
    Output("last-used-scale-type", "children"),
    Output("last-used-region", "children"),
    Output("last-used-color-map", "children"),
    Output("last-used-normalization", "children", allow_duplicate=True),
    Output("warning-window", "children", allow_duplicate=True),
    Input("submit-chromosome", "n_clicks"),
    State("chromosome-name", "value"),
    State("color-map", "value"),
    State("normalization", "value"),
    State("file-path", "value"),
    State("resolution", "value"),
    State("radio-log", "value"),
    State("files-list", "options"),
    State("rel-change-input", "value"),
    State("last-used-path", "children"),
    State("last-used-resolution", "children"),
    State("last-used-scale-type", "children"),
    State("last-used-region", "children"),
    State("last-used-color-map", "children"),
    State("last-used-normalization", "children"),
    State("HeatMap", "figure"),
    State("result-ut-stripes", "children"),
    State("result-lt-stripes", "children"),
    prevent_initial_call=True,
    running=[
        (Output("filepath-dialog", "disabled"), True, False),
        (Output("resolution", "disabled"), True, False),
        (Output("radio-log", "options"), _disable_radio_elements(True), _disable_radio_elements(False)),
        (Output("chromosome-name", "disabled"), True, False),
        (Output("color-map", "disabled"), True, False),
        (Output("normalization", "disabled"), True, False),
        (Output("submit-chromosome", "disabled"), True, False),
    ],
)
def update_plot(
    n_clicks,
    chromosome_region,
    colorMap,
    normalization,
    filepath,
    resolution,
    scale_type,
    files_list,
    rel_change,
    last_used_path,
    last_used_resolution,
    last_used_scale_type,
    last_used_region,
    last_used_color_map,
    last_used_normalization,
    fig,
    ut_stripes,
    lt_stripes,
):
    return update_plot_callback(
        chromosome_region,
        colorMap,
        normalization,
        filepath,
        resolution,
        scale_type,
        files_list,
        _string_to_int(rel_change),
        last_used_path,
        last_used_resolution,
        last_used_scale_type,
        last_used_region,
        last_used_color_map,
        last_used_normalization,
        fig,
        _string_to_stripe(ut_stripes, "After"),
        _string_to_stripe(lt_stripes, "After"),
    )


@app.callback(
    Output("calling-last-used-path", "children"),
    Output("calling-last-used-resolution", "children"),
    Output("calling-last-used-region", "children"),
    Output("calling-last-used-normalization", "children", allow_duplicate=True),
    Output("last-used-gen-belt", "children"),
    Output("last-used-max-width", "children"),
    Output("last-used-glob-pers-min", "children"),
    Output("last-used-constrain-heights", "children"),
    Output("last-used-k", "children"),
    Output("last-used-loc-pers-min", "children"),
    Output("last-used-loc-trend-min", "children"),
    Output("last-used-nproc", "children"),
    Output("last-used-rel-change", "children"),
    Output("HeatMap", "figure", allow_duplicate=True),
    Output("warning-window", "children", allow_duplicate=True),
    Output("result-chrom-name", "children"),
    Output("result-chrom-size", "children"),
    Output("result-min-persistence", "children"),
    Output("result-ut-pseudodistribution", "children"),
    Output("result-lt-pseudodistribution", "children"),
    Output("result-ut-all-minimum-points", "children"),
    Output("result-lt-all-minimum-points", "children"),
    Output("result-ut-all-maximum-points", "children"),
    Output("result-lt-all-maximum-points", "children"),
    Output("result-ut-persistence-of-all-minimum-points", "children"),
    Output("result-lt-persistence-of-all-minimum-points", "children"),
    Output("result-ut-persistence-of-all-maximum-points", "children"),
    Output("result-lt-persistence-of-all-maximum-points", "children"),
    Output("result-ut-persistent-minimum-points", "children"),
    Output("result-lt-persistent-minimum-points", "children"),
    Output("result-ut-persistent-maximum-points", "children"),
    Output("result-lt-persistent-maximum-points", "children"),
    Output("result-ut-persistence-of-minimum-points", "children"),
    Output("result-lt-persistence-of-minimum-points", "children"),
    Output("result-ut-persistence-of-maximum-points", "children"),
    Output("result-lt-persistence-of-maximum-points", "children"),
    Output("result-ut-stripes", "children"),
    Output("result-lt-stripes", "children"),
    Input("start-calling", "n_clicks"),
    State("file-path", "value"),
    State("resolution", "value"),
    State("chromosome-name", "value"),
    State("color-map", "value"),
    State("normalization", "value"),
    State("gen-belt-input", "value"),
    State("max-width-input", "value"),
    State("glob-pers-input", "value"),
    State("constrain-heights-input", "value"),
    State("k-neighbours-input", "value"),
    State("loc-min-pers-input", "value"),
    State("loc-trend-input", "value"),
    State("nproc-input", "value"),
    State("rel-change-input", "value"),
    State("calling-last-used-path", "children"),
    State("calling-last-used-resolution", "children"),
    State("calling-last-used-region", "children"),
    State("calling-last-used-normalization", "children"),
    State("last-used-gen-belt", "children"),
    State("last-used-max-width", "children"),
    State("last-used-glob-pers-min", "children"),
    State("last-used-constrain-heights", "children"),
    State("last-used-k", "children"),
    State("last-used-loc-pers-min", "children"),
    State("last-used-loc-trend-min", "children"),
    State("last-used-nproc", "children"),
    State("last-used-rel-change", "children"),
    State("HeatMap", "figure"),
    State("result-chrom-name", "children"),
    State("result-chrom-size", "children"),
    State("result-min-persistence", "children"),
    State("result-ut-pseudodistribution", "children"),
    State("result-lt-pseudodistribution", "children"),
    State("result-ut-all-minimum-points", "children"),
    State("result-lt-all-minimum-points", "children"),
    State("result-ut-all-maximum-points", "children"),
    State("result-lt-all-maximum-points", "children"),
    State("result-ut-persistence-of-all-minimum-points", "children"),
    State("result-lt-persistence-of-all-minimum-points", "children"),
    State("result-ut-persistence-of-all-maximum-points", "children"),
    State("result-lt-persistence-of-all-maximum-points", "children"),
    State("result-ut-persistent-minimum-points", "children"),
    State("result-lt-persistent-minimum-points", "children"),
    State("result-ut-persistent-maximum-points", "children"),
    State("result-lt-persistent-maximum-points", "children"),
    State("result-ut-persistence-of-minimum-points", "children"),
    State("result-lt-persistence-of-minimum-points", "children"),
    State("result-ut-persistence-of-maximum-points", "children"),
    State("result-lt-persistence-of-maximum-points", "children"),
    State("result-ut-stripes", "children"),
    State("result-lt-stripes", "children"),
    prevent_initial_call=True,
    running=[
        (Output("filepath-dialog", "disabled"), True, False),
        (Output("resolution", "disabled"), True, False),
        (Output("radio-log", "options"), _disable_radio_elements(True), _disable_radio_elements(False)),
        (Output("chromosome-name", "disabled"), True, False),
        (Output("color-map", "disabled"), True, False),
        (Output("normalization", "disabled"), True, False),
        (Output("submit-chromosome", "disabled"), True, False),
        (Output("gen-belt-input", "disabled"), True, False),
        (Output("max-width-input", "disabled"), True, False),
        (Output("glob-pers-input", "disabled"), True, False),
        (Output("constrain-heights-input", "disabled"), True, False),
        (Output("k-neighbours-input", "disabled"), True, False),
        (Output("loc-min-pers-input", "disabled"), True, False),
        (Output("loc-trend-input", "disabled"), True, False),
        (Output("nproc-input", "disabled"), True, False),
        (Output("rel-change-input", "disabled"), True, False),
        (Output("start-calling", "disabled"), True, False),
    ],
)
def call_stripes(
    n_clicks,
    path,
    resolution,
    region,
    color_map,
    normalization,
    gen_belt,
    max_width,
    glob_pers_min,
    constrain_heights,
    k,
    loc_pers_min,
    loc_trend_min,
    nproc,
    rel_change,
    last_used_path,
    last_used_resolution,
    last_used_region,
    last_used_normalization,
    last_used_gen_belt,
    last_used_max_width,
    last_used_glob_pers_min,
    last_used_constrain_heights,
    last_used_k,
    last_used_loc_pers_min,
    last_used_loc_trend_min,
    last_used_nproc,
    last_used_rel_change,
    fig,
    result_region,
    result_chrom_size,
    result_min_persistence,
    result_ut_pseudodistribution,
    result_lt_pseudodistribution,
    result_ut_all_minimum_points,
    result_lt_all_minimum_points,
    result_ut_all_maximum_points,
    result_lt_all_maximum_points,
    result_ut_persistence_of_all_minimum_points,
    result_lt_persistence_of_all_minimum_points,
    result_ut_persistence_of_all_maximum_points,
    result_lt_persistence_of_all_maximum_points,
    result_ut_persistent_minimum_points,
    result_lt_persistent_minimum_points,
    result_ut_persistent_maximum_points,
    result_lt_persistent_maximum_points,
    result_ut_persistence_of_minimum_points,
    result_lt_persistence_of_minimum_points,
    result_ut_persistence_of_maximum_points,
    result_lt_persistence_of_maximum_points,
    result_ut_stripes,
    result_lt_stripes,
):
    path = Path(path)
    if normalization == "No normalization":
        normalization = None
    if not isinstance(fig, go.Figure):
        fig = go.Figure(fig)
    raw_plot = [trace for trace in fig["data"] if type(trace) == go.Heatmap]
    fig["data"] = tuple(raw_plot)

    if region == "":
        return _not_implemented_gen_wide()
    chromosome_name, _, frame = region.partition(":")
    margin, _, end_limit = frame.partition("-")
    margin = _string_to_int(margin)
    end_limit = _string_to_int(end_limit)

    last_used_chromosome_name, _, last_used_frame = last_used_region.partition(":")

    current_settings = (
        path,
        resolution,
        chromosome_name,
        normalization,
        gen_belt.replace(",", ""),
        nproc,
        glob_pers_min,
        max_width.replace(",", ""),
        loc_trend_min,
        k,
        rel_change,
        loc_pers_min,
        constrain_heights,
    )
    past_settings = (
        Path(last_used_path),
        last_used_resolution,
        last_used_chromosome_name,
        last_used_normalization,
        last_used_gen_belt,
        last_used_nproc,
        last_used_glob_pers_min,
        last_used_max_width,
        last_used_loc_trend_min,
        last_used_k,
        last_used_rel_change,
        last_used_loc_pers_min,
        last_used_constrain_heights,
    )
    from_where_to_call = _compare(current_settings, past_settings)

    restriction_scope = _find_restriction_scope(region)
    traces = ("x2", "y2") if restriction_scope == "whole genome" else ("x1", "y1")

    if from_where_to_call == "After":
        return filter_stripes_callback(
            fig,
            resolution,
            color_map,
            _string_to_int(rel_change),
            traces,
            margin,
            end_limit,
            _string_to_stripe(result_ut_stripes, "After"),
            _string_to_stripe(result_lt_stripes, "After"),
        )
    elif from_where_to_call == "No change":
        return _stale_fields()
    else:
        return call_stripes_callback(
            path,
            resolution,
            region,
            color_map,
            normalization,
            _string_to_int(gen_belt),
            _string_to_int(max_width),
            _string_to_int(glob_pers_min),
            _string_to_bool(constrain_heights),
            _string_to_int(k),
            _string_to_int(loc_pers_min),
            _string_to_int(loc_trend_min),
            _string_to_int(nproc),
            _string_to_int(rel_change),
            fig,
            result_region,
            result_chrom_size,
            result_min_persistence,
            _string_to_list(result_ut_pseudodistribution),
            _string_to_list(result_lt_pseudodistribution),
            _string_to_list(result_ut_all_minimum_points),
            _string_to_list(result_lt_all_minimum_points),
            _string_to_list(result_ut_all_maximum_points),
            _string_to_list(result_lt_all_maximum_points),
            _string_to_list(result_ut_persistence_of_all_minimum_points),
            _string_to_list(result_lt_persistence_of_all_minimum_points),
            _string_to_list(result_ut_persistence_of_all_maximum_points),
            _string_to_list(result_lt_persistence_of_all_maximum_points),
            _string_to_list(result_ut_persistent_minimum_points),
            _string_to_list(result_lt_persistent_minimum_points),
            _string_to_list(result_ut_persistent_maximum_points),
            _string_to_list(result_lt_persistent_maximum_points),
            _string_to_list(result_ut_persistence_of_minimum_points),
            _string_to_list(result_lt_persistence_of_minimum_points),
            _string_to_list(result_ut_persistence_of_maximum_points),
            _string_to_list(result_lt_persistence_of_maximum_points),
            _string_to_stripe(result_ut_stripes, from_where_to_call),
            _string_to_stripe(result_lt_stripes, from_where_to_call),
            from_where_to_call,
            traces,
            chromosome_name,
            margin,
            end_limit,
            restriction_scope,
        )


def _string_to_int(string):
    """
    Turn string representation of a number into a number
    """
    assert isinstance(string, str)
    try:
        if isinstance(string, int):
            return string
        if isinstance(string, float):
            return string
        if "," in string:
            string = string.replace(",", "")
        if "." in string:
            return float(string)
        return int(string)
    except ValueError:
        if string == "inf":
            return float("inf")


def _string_to_bool(string):
    assert isinstance(string, str)
    if string == "":
        return False
    return eval(string)


def _string_to_list(string):
    """
    Turn a string representation of numbers into a list containing the numbers
    """
    assert isinstance(string, str)
    if string == "":
        return []
    return_list = []
    for value in string.split(";"):
        return_list.append(_string_to_int(value))
    return return_list


def _string_to_stripe(string, where):
    """
    Similar to _string_to_list. Tailor made for the strings that store stripe info.
    Returns different scopes of data depending on what step of StripePy is run.
    """
    if string == "":
        return []
    if where == "Step 2":
        return []

    stripes = []
    for stripe in string.split(";"):
        new_stripe = stripe.split(":")
        if where == "Step 3":
            relevant_section = new_stripe[:2]
        if where == "Step 4":
            relevant_section = new_stripe[:6]
        if where == "After":
            relevant_section = new_stripe
        add_list = [_string_to_int(element) for element in relevant_section]
        stripes.append(add_list)
    return stripes


def _compare(current_settings, past_settings):
    """
    Calculate the first necessary step of StripePy to run, if any.
    """
    for index, comparison in enumerate(current_settings):
        if comparison != past_settings[index]:
            if index <= 6:  # Matrix plotting, gen_belt, nproc
                print("Go from step 2")
                return "Step 2"
            if index <= 8:  # max width, local trend minimum
                print("Go from step 3")
                return "Step 3"
            if index <= 9:  # k neighbours
                print("Go from step 4")
                return "Step 4"
            if index <= 10:  # relative change
                print("Go from after step 4")
                return "After"
            if index <= 12:  # local minimum persistence, constrain heights
                break
    return "No change"


def _find_restriction_scope(chrom_name):
    """
    Calculates whether to visualise a region of a chromosome, a whole chromosome, or the entire genome
    """
    restriction_scope = ""
    chrom, _, frame = chrom_name.partition(":")
    if frame:
        restriction_scope = "chromosome restriction"
    if chrom:
        "single chromosome"
    else:
        "whole genome"
    return restriction_scope


def _stale_fields():
    """
    No stripes are drawn on the Hi-C matrix, and a new warning pops up in the warning banner.
    """
    return (
        *[no_update] * 14,
        warning_stale_component(
            (
                "file path",
                "resolution",
                "chromosome name",
                "color map",
                "normalization",
                "genomic belt",
                "max width",
                "global minimum persistence",
                "constrain heights",
                "k neighbours",
                "local minimal persistence",
                "local trend minimum",
                "number of processors",
                "relative signal change",
            )
        ),
        *[no_update] * 23,
    )


def _not_implemented_gen_wide():
    return (
        *[no_update] * 14,
        warning_gen_wide(),
        *[no_update] * 23,
    )


@app.callback(
    Output("meta-info", "hidden"),
    Input("show-metadata", "n_clicks"),
    State("meta-info", "hidden"),
    prevent_initial_call=True,
)
def hide_show_metadata(n_clicks, hidden_state):
    return not (hidden_state)


@app.callback(
    Output("show-metadata", "children"),
    Input("meta-info", "hidden"),
)
def change_metadata_button_name(hidden_state):
    if hidden_state:
        new_button_label = "Show metadata"
    else:
        new_button_label = "Hide metadata"
    return new_button_label


if __name__ == "__main__":
    app.run(debug=True)
