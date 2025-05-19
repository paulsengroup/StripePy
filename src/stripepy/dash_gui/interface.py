# Copyright (C) 2025 Bendik Berg <bendber@ifi.uio.no>
#
# SPDX-License-Identifier: MIT

import pathlib

import dash_bootstrap_components as dbc
from callbacks import look_for_file_callback, update_file_callback, update_plot_callback
from components.layout import layout
from dash import Dash, Input, Output, State

from stripepy.cli import call
from stripepy.io import ProcessSafeLogger

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME])


app.layout = layout()


@app.callback(
    Output("resolution", "options"),
    Output("resolution", "value"),
    Output("resolution", "disabled"),
    Output("submit-file", "disabled"),
    Input("look-for-file", "n_clicks"),
    State("file-path", "value"),
    prevent_initial_call=True,
    running=[
        (Output("file-path", "disabled"), True, False),
        (Output("look-for-file", "disabled"), True, False),
    ],
)
def look_for_file(n_clicks, file_path):
    return look_for_file_callback(file_path)


@app.callback(
    Output("meta-info", "children"),
    Output("normalization", "options"),
    Output("normalization", "value"),
    Output("chromosome-name", "disabled"),
    Output("color-map", "disabled"),
    Output("submit-chromosome", "disabled"),
    Output("normalization", "disabled"),
    Output("chromosome-name-icon", "hidden"),
    Output("color-map-icon", "hidden"),
    Output("normalization-icon", "hidden"),
    Input("submit-file", "n_clicks"),
    State("file-path", "value"),
    State("resolution", "value"),
    prevent_initial_call=True,
    running=[
        (Output("file-path", "disabled"), True, False),
        (Output("look-for-file", "disabled"), True, False),
        (Output("resolution", "disabled"), True, False),
        (Output("submit-file", "disabled"), True, False),
    ],
)
def update_file(n_clicks, filename, resolution):
    return update_file_callback(filename, resolution)


@app.callback(
    Output("HeatMap", "figure"),
    Output("data", "hidden"),
    Input("submit-chromosome", "n_clicks"),
    State("chromosome-name", "value"),
    State("color-map", "value"),
    State("normalization", "value"),
    State("file-path", "value"),
    State("resolution", "value"),
    State("radio-log", "value"),
    prevent_initial_call=True,
    running=[
        (Output("file-path", "disabled"), True, False),
        (Output("look-for-file", "disabled"), True, False),
        (Output("resolution", "disabled"), True, False),
        (Output("submit-file", "disabled"), True, False),
        (Output("chromosome-name", "disabled"), True, False),
        (Output("color-map", "disabled"), True, False),
        (Output("normalization", "disabled"), True, False),
        (Output("submit-chromosome", "disabled"), True, False),
    ],
)
def update_plot(n_clicks, chromosome_name, colorMap, normalization, filepath, resolution, scale_type):
    return update_plot_callback(chromosome_name, colorMap, normalization, filepath, resolution, scale_type)


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
    State("force-input", "value"),
    State("nproc-input", "value"),
    State("min-chrom-size-input", "value"),
    State("verbosity-input", "value"),
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
    force,
    nproc,
    min_chrom_size,
    verbosity,
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
    with ProcessSafeLogger(
        verbosity,
        path=pathlib.Path("./tmp/log_file"),
        force=force,
        matrix_file=path,
        print_welcome_message=True,
        progress_bar_type="call",
    ) as main_logger:
        call.run(
            chromosome_name,
            resolution,
            pathlib.Path("./tmp/called_stripes"),  # output file
            gen_belt,
            max_width,
            glob_pers,  # glob_pers_min, or maybe loc_pers_min?
            constrain_heights,  # constrain heights
            loc_pers_min,  # loc_pers_min
            loc_trend_min,
            force,  # force
            nproc,  # nproc
            min_chrom_size,  # min_chrom_size
            verbosity,
            main_logger,  # main_logger,
            # roi,
            log_file=pathlib.Path("./tmp/log_file"),  # log_file,
            # plot_dir,
            normalization=normalization,
        )


@app.callback(
    Output("meta-info", "hidden"),
    Input("show-metadata", "n_clicks"),
    State("meta-info", "hidden"),
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
