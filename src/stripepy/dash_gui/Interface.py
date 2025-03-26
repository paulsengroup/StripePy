# Copyright (C) 2025 Bendik Berg <bendber@ifi.uio.no>
#
# SPDX-License-Identifier: MIT

import pathlib

import hictkpy as htk
import numpy as np
import plotly.express as px
from ColorScales import color_scale
from components.layout import layout
from dash import Dash, Input, Output, State, html
from dash.exceptions import PreventUpdate

from stripepy.cli import call
from stripepy.io import ProcessSafeLogger

app = Dash(__name__)


app.layout = layout()


@app.callback(
    Output("resolution", "options"),
    Output("resolution", "disabled"),
    Output("resolution", "value"),
    Output("submit-file", "disabled"),
    Input("look-for-file", "n_clicks"),
    State("file-path", "value"),
    prevent_initial_call=True,
)
def look_for_file(n_clicks, file_path):
    global last_used_file
    try:
        if file_path == last_used_file:
            raise PreventUpdate
        else:
            pass
    except NameError:
        pass
    last_used_file = file_path

    mrf = htk.MultiResFile(file_path)
    resolutions = mrf.resolutions()
    return resolutions, False, resolutions[0], False


@app.callback(
    Output("meta-info", "children"),
    Output("chromosome-name", "disabled"),
    Output("color-map", "disabled"),
    Output("submit-chromosome", "disabled"),
    Input("submit-file", "n_clicks"),
    State("file-path", "value"),
    State("resolution", "value"),
    prevent_initial_call=True,
    running=[(Output("submit-file", "disabled"), True, False)],
)
def update_file(n_clicks, filename, resolution):
    global last_used_resolution
    try:
        if filename == last_used_file and resolution == last_used_resolution:
            raise PreventUpdate
        else:
            pass
    except NameError:
        pass
    last_used_file = filename
    last_used_resolution = resolution

    path = filename
    bin_size = resolution

    global f
    f = htk.File(path, bin_size)

    metaInfo_chromosomes = html.Div([html.P((chromosome, ":", name)) for chromosome, name in f.chromosomes().items()])
    metaInfo = html.Div([html.P("Chromosomes", style={"fontSize": 24, "fontWeight": "bold"}), metaInfo_chromosomes])

    return metaInfo, False, False, False


@app.callback(
    Output("HeatMap", "figure"),
    Input("submit-chromosome", "n_clicks"),
    State("chromosome-name", "value"),
    State("color-map", "value"),
    State("normalization", "value"),
    State("file-path", "value"),
    State("resolution", "value"),
    prevent_initial_call=True,
    running=[
        (Output("file-path", "disabled"), True, False),
        (Output("look-for-file", "disabled"), True, False),
        (Output("resolution", "disabled"), True, False),
        (Output("submit-file", "disabled"), True, False),
        (Output("submit-chromosome", "disabled"), True, False),
        (Output("chromosome-name", "disabled"), True, False),
        (Output("color-map", "disabled"), True, False),
        (Output("normalization", "disabled"), True, False),
    ],
)
def update_plot(n_clicks, chromosome_name, colorMap, normalization, filepath, resolution):
    global last_used_chromosome_name
    global last_used_colorMap
    global last_used_normalization
    try:
        if (
            chromosome_name == last_used_chromosome_name
            and colorMap == last_used_colorMap
            and normalization == last_used_normalization
            and last_used_file == filepath
            and last_used_resolution == resolution
        ):
            raise PreventUpdate
        else:
            pass
    except NameError:
        pass
    last_used_chromosome_name = chromosome_name
    last_used_colorMap = colorMap
    last_used_normalization = normalization
    last_used_file = filepath
    last_used_resolution = resolution

    if colorMap not in px.colors.named_colorscales():
        colorMap = color_scale(colorMap)

    sel = f.fetch(chromosome_name, normalization=normalization)
    frame = sel.to_numpy()

    np.log(frame, out=frame, where=frame > 0)
    frame /= np.nanmax(frame)

    fig = px.imshow(frame, color_continuous_scale=colorMap)

    fig.update_layout(coloraxis_showscale=False)
    fig.update_yaxes(autorange="reversed")

    return fig


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
    # stripe_type,
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
            max_width,  # What is max width?
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


# import webbrowser

# webbrowser.open("http://127.0.0.1:8050/")

if __name__ == "__main__":
    app.run(debug=True)
