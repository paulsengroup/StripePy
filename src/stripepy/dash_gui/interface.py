# Copyright (C) 2025 Bendik Berg <bendber@ifi.uio.no>
#
# SPDX-License-Identifier: MIT

import dash_bootstrap_components as dbc
from callbacks import (
    call_stripes_callback,
    look_for_file_callback,
    look_for_normalizations_under_current_resolution_callback,
    open_file_dialog_callback,
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
    prevent_initial_call=True,
    running=[
        (Output("resolution", "disabled"), True, False),
    ],
)
def open_file_dialog(n_clicks):
    return open_file_dialog_callback()


@app.callback(
    Output("resolution", "options"),
    Output("resolution", "value", allow_duplicate=True),
    Output("chromosomes", "children"),
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
    filepath, resolution, scale_type, chrom_name, normalization = saved_string.split(";")
    return filepath, int(resolution), scale_type, chrom_name, normalization, update_plot_n_clicks + 1


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
    Output("HeatMap", "figure"),
    Output("files-list", "options"),
    Output("heat-map", "hidden"),
    Output("last-used-path", "children"),
    Output("last-used-resolution", "children"),
    Output("last-used-scale-type", "children"),
    Output("last-used-region", "children"),
    Output("last-used-color-map", "children"),
    Output("last-used-normalization", "children"),
    Input("submit-chromosome", "n_clicks"),
    State("chromosome-name", "value"),
    State("color-map", "value"),
    State("normalization", "value"),
    State("file-path", "value"),
    State("resolution", "value"),
    State("radio-log", "value"),
    State("files-list", "options"),
    State("last-used-path", "children"),
    State("last-used-resolution", "children"),
    State("last-used-scale-type", "children"),
    State("last-used-region", "children"),
    State("last-used-color-map", "children"),
    State("last-used-normalization", "children"),
    prevent_initial_call=True,
    running=[
        (Output("resolution", "disabled"), True, False),
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
    last_used_path,
    last_used_resolution,
    last_used_scale_type,
    last_used_region,
    last_used_color_map,
    last_used_normalization,
):
    return update_plot_callback(
        chromosome_name,
        colorMap,
        normalization,
        filepath,
        resolution,
        scale_type,
        files_list,
        last_used_path,
        last_used_resolution,
        last_used_scale_type,
        last_used_region,
        last_used_color_map,
        last_used_normalization,
    )


@app.callback(
    Input("start-calling", "n_clicks"),
    State("chromosome-name", "value"),
    State("resolution", "value"),
    State("gen-belt-input", "value"),
    State("max-width-input", "value"),
    State("glob-pers-input", "value"),
    State("constrain-heights-input", "value"),
    State("loc-min-pers-input", "value"),
    State("loc-trend-input", "value"),
    # State("force-input", "value"),
    State("nproc-input", "value"),
    State("min-chrom-size-input", "value"),
    # State("verbosity-input", "value"),
    # State("main-logger-value", "value"),
    # State("roi-input", "value"),
    # State("log-file-input", "value"),
    # State("plot-dir-input", "value"),
    State("normalization", "value"),
    # State("rel-change-input", "value"),
    # State("stripe-type-input", "value"),
    State("file-path", "value"),
    prevent_initial_call=True,
    running=[
        (Output("start-calling", "disabled"), True, False),
    ],
)
def call_stripes(
    n_clicks,
    chromosome_name,
    resolution,
    gen_belt,
    max_width,
    glob_pers,
    constrain_heights,
    loc_pers_min,
    loc_trend_min,
    # force,
    nproc,
    min_chrom_size,
    # verbosity,
    # main_logger,
    # roi,
    # log_file,
    # plot_dir
    normalization,
    # top_pers,
    # rel_change,
    # loc_trend,
    path,
):
    return call_stripes_callback(
        chromosome_name,
        resolution,
        gen_belt,
        max_width,
        glob_pers,
        constrain_heights,
        loc_pers_min,
        loc_trend_min,
        # force,
        nproc,
        min_chrom_size,
        # verbosity,
        normalization,
        path,
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
