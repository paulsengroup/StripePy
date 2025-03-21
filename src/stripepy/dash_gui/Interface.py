# Copyright (C) 2025 Bendik Berg <bendber@ifi.uio.no>
#
# SPDX-License-Identifier: MIT

import pathlib

import hictkpy as htk
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from ColorScales import color_scale
from components.calling import (
    render_constrain_heights,
    render_force,
    render_genomic_belt,
    render_global_minimum_persistence,
    render_local_minimum_persistence,
    render_local_trend_minimum,
    render_max_width,
    render_minimum_chromosome_size,
    render_nrpoc,
    render_relative_change,
    render_stripe_calling_button,
    render_stripe_type,
    render_verbosity,
)
from components.plotting import (
    render_chromosome_name,
    render_color_map,
    render_filepath,
    render_normalization,
    render_resolution,
    render_submit_button,
)
from dash import Dash, Input, Output, State, dcc, html
from HiCObject import HiCObject

from stripepy.cli import call
from stripepy.io import ProcessSafeLogger

app = Dash(__name__)

hictk_reader = HiCObject()


app.layout = html.Div(
    [
        html.Div(
            [
                html.Div(
                    [
                        dcc.Graph(
                            style={"width": "90vh", "height": "90vh"},
                            id="HeatMap",
                        ),
                    ],
                    style={"display": "inline-block"},
                ),
                html.Div(
                    [  # Right side of screen
                        render_filepath(),
                        render_resolution(),
                        render_chromosome_name(),
                        render_color_map(),
                        render_normalization(),
                        render_submit_button(),
                        render_genomic_belt(),
                        render_max_width(),
                        render_global_minimum_persistence(),
                        render_constrain_heights(),
                        render_local_minimum_persistence(),
                        render_local_trend_minimum(),
                        render_force(),
                        render_nrpoc(),
                        render_minimum_chromosome_size(),
                        render_verbosity(),
                        render_relative_change(),
                        render_relative_change(),
                        render_stripe_type(),
                        render_stripe_calling_button(),
                    ],
                    style={"marginTop": 95},
                ),
            ],
            style={"display": "flex"},
        ),
        html.Div(id="meta-info"),
        html.Div(id="callbacks-file", style={"display": "none"}),
    ]
)


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
    mrf = htk.MultiResFile(file_path)
    print(mrf.resolutions())
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

    hictk_reader.path = filename
    hictk_reader.resolution = resolution

    metaInfo_chromosomes = html.Div(
        [html.P((chromosome, ":", name)) for chromosome, name in hictk_reader._chromosomes.items()]
    )
    metaInfo = html.Div([html.P("Chromosomes", style={"fontSize": 24, "fontWeight": "bold"}), metaInfo_chromosomes])

    return metaInfo, False, False, False


@app.callback(
    Output("HeatMap", "figure"),
    Input("submit-chromosome", "n_clicks"),
    State("chromosome-name", "value"),
    State("color-map", "value"),
    State("normalization", "value"),
    prevent_initial_call=True,
    running=[
        (Output("submit-chromosome", "disabled"), True, False),
        (Output("chromosome-name", "disabled"), True, False),
        (Output("color-map", "disabled"), True, False),
        (Output("normalization", "disabled"), True, False),
    ],
)
def update_plot(
    n_clicks,
    chromosome_name,
    colorMap,
    normalization,
):
    if colorMap not in px.colors.named_colorscales():
        colorMap = color_scale(colorMap)

    hictk_reader.region_of_interest = chromosome_name
    hictk_reader.normalization = normalization

    frame = hictk_reader.selector
    frame = np.where(frame > 0, np.log10(frame), 0)

    fig = go.Figure(data=go.Heatmap(z=normalize(frame, norm="max"), colorscale=colorMap, showscale=False))
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
):
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
        main_logger=ProcessSafeLogger,  # main_logger,
        # roi,
        # log_file,
        # plot_dir,
        normalization=normalization,
    )


# import webbrowser

# webbrowser.open("http://127.0.0.1:8050/")

if __name__ == "__main__":
    app.run(debug=True)
