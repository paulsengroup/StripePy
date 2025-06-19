# Copyright (C) 2025 Bendik Berg <bendber@ifi.uio.no>
#
# SPDX-License-Identifier: MIT

import dash_bootstrap_components as dbc
from callbacks import (
    call_stripes_callback,
    look_for_file_callback,
    look_for_normalizations_under_current_resolution_callback,
    open_file_dialog_callback,
    open_hdf5_file_dialog_callback,
    pick_saved_callback,
    populate_empty_normalization_list_callback,
    update_plot_callback,
)
from components.layout import layout
from dash import Dash, Input, Output, State

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME])


app.layout = layout()


@app.callback(
    Output("file-path", "value", allow_duplicate=True),
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
    Input("file-path", "value"),
    State("last-used-path", "children"),
    prevent_initial_call=True,
)
def look_for_file(file_path, last_used_path):
    return look_for_file_callback(file_path, last_used_path)


@app.callback(
    Output("file-path", "value", allow_duplicate=True),
    Output("resolution", "value", allow_duplicate=True),
    Output("radio-log", "value"),
    Output("chromosome-name", "value"),
    Output("normalization", "value", allow_duplicate=True),
    Output("submit-chromosome", "n_clicks"),
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
    prevent_initial_call=True,
    running=[
        (Output("normalization", "disabled"), True, False),
    ],
)
def look_for_normalizations_under_current_resolution(resolution, path):
    return look_for_normalizations_under_current_resolution_callback(resolution, path)


@app.callback(
    Output("normalization", "options", allow_duplicate=True),
    Output("normalization", "value", allow_duplicate=True),
    Input("normalization", "options"),
    prevent_initial_call=True,
)
def populate_empty_normalization_list(array):
    return populate_empty_normalization_list_callback(array)


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
    Output("last-used-hdf5", "children"),
    Input("submit-chromosome", "n_clicks"),
    State("chromosome-name", "value"),
    State("color-map", "value"),
    State("normalization", "value"),
    State("file-path", "value"),
    State("resolution", "value"),
    State("radio-log", "value"),
    State("files-list", "options"),
    State("stripes-filepath", "value"),
    State("last-used-path", "children"),
    State("last-used-resolution", "children"),
    State("last-used-scale-type", "children"),
    State("last-used-region", "children"),
    State("last-used-color-map", "children"),
    State("last-used-normalization", "children"),
    State("last-used-hdf5", "children"),
    prevent_initial_call=True,
    running=[
        (Output("resolution", "disabled"), True, False),
        (Output("radio-log", "disabled"), True, False),
        (Output("chromosome-name", "disabled"), True, False),
        (Output("color-map", "disabled"), True, False),
        (Output("normalization", "disabled"), True, False),
        (Output("submit-chromosome", "disabled"), True, False),
    ],
)
def update_plot(
    n_clicks,
    chromosome_name,
    colorMap,
    normalization,
    filepath,
    resolution,
    scale_type,
    files_list,
    hdf5,
    last_used_path,
    last_used_resolution,
    last_used_scale_type,
    last_used_region,
    last_used_color_map,
    last_used_normalization,
    last_used_stripes,
):
    return update_plot_callback(
        chromosome_name,
        colorMap,
        normalization,
        filepath,
        resolution,
        scale_type,
        files_list,
        hdf5,
        last_used_path,
        last_used_resolution,
        last_used_scale_type,
        last_used_region,
        last_used_color_map,
        last_used_normalization,
        last_used_stripes,
    )


@app.callback(
    Output("last-used-normalization", "children", allow_duplicate=True),
    Output("last-used-gen-belt", "children"),
    Output("last-used-max-width", "children"),
    Output("last-used-glob-pers-min", "children"),
    Output("last-used-constrain-heights", "children"),
    Output("last-used-k", "children"),
    Output("last-used-loc-pers-min", "children"),
    Output("last-used-loc-trend-min", "children"),
    Output("last-used-nproc", "children"),
    Output("HeatMap", "figure", allow_duplicate=True),
    Input("start-calling", "n_clicks"),
    State("file-path", "value"),
    State("resolution", "value"),
    State("chromosome-name", "value"),
    State("gen-belt-input", "value"),
    State("max-width-input", "value"),
    State("glob-pers-input", "value"),
    State("constrain-heights-input", "value"),
    State("k-neighbours-input", "value"),
    State("loc-min-pers-input", "value"),
    State("loc-trend-input", "value"),
    State("nproc-input", "value"),
    State("min-chrom-size-input", "value"),
    State("normalization", "value"),
    State("last-used-normalization", "children"),
    State("last-used-gen-belt", "children"),
    State("last-used-max-width", "children"),
    State("last-used-glob-pers-min", "children"),
    State("last-used-constrain-heights", "children"),
    State("last-used-k", "children"),
    State("last-used-loc-pers-min", "children"),
    State("last-used-loc-trend-min", "children"),
    State("last-used-nproc", "children"),
    State("HeatMap", "figure"),
    prevent_initial_call=True,
    running=[
        (Output("resolution", "disabled"), True, False),
        (Output("gen-belt-input", "disabled"), True, False),
        (Output("max-width-input", "disabled"), True, False),
        (Output("glob-pers-input", "disabled"), True, False),
        (Output("constrain-heights-input", "disabled"), True, False),
        (Output("k-neighbours-input", "disabled"), True, False),
        (Output("loc-min-pers-input", "disabled"), True, False),
        (Output("loc-trend-input", "disabled"), True, False),
        (Output("nproc-input", "disabled"), True, False),
        (Output("min-chrom-size-input", "disabled"), True, False),
        (Output("normalization", "disabled"), True, False),
        (Output("start-calling", "disabled"), True, False),
    ],
)
def call_stripes(
    n_clicks,
    path,
    resolution,
    chrom_name,
    gen_belt,
    max_width,
    glob_pers_min,
    constrain_heights,
    k,
    loc_pers_min,
    loc_trend_min,
    nproc,
    min_chrom_size,
    normalization,
    last_used_normalization,
    last_used_gen_belt,
    last_used_max_width,
    last_used_glob_pers_min,
    last_used_constrain_heights,
    last_used_k,
    last_used_loc_pers_min,
    last_used_loc_trend_min,
    last_used_nproc,
    fig,
):
    return call_stripes_callback(
        path,
        resolution,
        chrom_name,
        _string_to_int(gen_belt),
        _string_to_int(max_width),
        _string_to_int(glob_pers_min),
        bool(constrain_heights),
        _string_to_int(k),
        _string_to_int(loc_pers_min),
        _string_to_int(loc_trend_min),
        _string_to_int(nproc),
        _string_to_int(min_chrom_size),
        normalization,
        last_used_normalization,
        _string_to_int(last_used_gen_belt),
        _string_to_int(last_used_max_width),
        _string_to_int(last_used_glob_pers_min),
        _string_to_bool(last_used_constrain_heights),
        _string_to_int(last_used_k),
        _string_to_int(last_used_loc_pers_min),
        _string_to_int(last_used_loc_trend_min),
        _string_to_int(last_used_nproc),
        fig,
    )


def _string_to_int(string):
    assert isinstance(string, str)
    if isinstance(string, int):
        return string
    if isinstance(string, float):
        return string
    if "," in string:
        string = string.replace(",", "")
    if "." in string:
        return float(string)
    return int(string)


def _string_to_bool(string):
    assert isinstance(string, str)
    if string == "":
        return ""
    return bool(string)


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


@app.callback(
    Output("stripes-filepath", "value", allow_duplicate=True),
    Input("find-hdf5", "n_clicks"),
    prevent_initial_call=True,
)
def open_hdf5_file_dialog(n_clicks):
    return open_hdf5_file_dialog_callback()


@app.callback(
    Output("stripes-filepath", "value", allow_duplicate=True),
    Input("delete-stripe-file", "n_clicks"),
    prevent_initial_call=True,
)
def delete_stripe_file(n_clicks):
    return ""


if __name__ == "__main__":
    app.run(debug=True)
