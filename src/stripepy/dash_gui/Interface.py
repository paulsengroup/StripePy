# Copyright (C) 2025 Bendik Berg <bendber@ifi.uio.no>
#
# SPDX-License-Identifier: MIT

import pathlib

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from ColorScales import color_scale
from dash import Dash, Input, Output, State, dcc, html
from HiCObject import HiCObject
from sklearn.preprocessing import normalize

from stripepy.cli import call

# from stripepy.io.logging import ProcessSafeLogger
# from stripepy.io import logging


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
                        html.Div(
                            [
                                dcc.Input(
                                    placeholder="File path",
                                    type="text",
                                    value="",
                                    id="file-path",
                                    style={"width": 300, "fontSize": 20},
                                ),
                                dcc.Input(
                                    placeholder="Resolution",
                                    type="number",
                                    value="",
                                    id="resolution",
                                    style={"width": 300, "fontSize": 20},
                                ),
                            ],
                        ),
                        html.Button(
                            id="submit-file",
                            n_clicks=0,
                            children="Submit",
                            style={"marginBottom": 100},
                        ),
                        html.Div(
                            [
                                dcc.Input(
                                    placeholder="[Chromosome name]:start_number-end_number",
                                    type="text",
                                    value="",
                                    id="chromosome-name",
                                    disabled=True,
                                    style={"width": 700, "fontSize": 20},
                                ),
                                dcc.Dropdown(
                                    options=["Hi-C", *px.colors.named_colorscales()],
                                    placeholder="Color map",
                                    value=None,
                                    id="color-map",
                                    disabled=True,
                                    style={"width": 300, "fontSize": 20},
                                ),
                            ],
                            style={"height": 60, "width": 800, "display": "block"},
                        ),
                        dcc.Dropdown(
                            options=["KR", "VC", "VC_SQRT"],
                            placeholder="Normalization",
                            value="KR",
                            id="normalization",
                        ),
                        html.Button(
                            n_clicks=0,
                            children="Submit",
                            id="submit-chromosome",
                            disabled=True,
                            style={"marginBottom": 60},
                        ),
                        html.Div(
                            [
                                "genomic belt ",
                                dcc.Input(
                                    type="number",
                                    value=5_000_000,
                                    id="gen-belt-input",
                                    style={"width": 300},
                                ),
                            ],
                            style={"marginTop": 40},
                        ),
                        html.Div(
                            [
                                "max width ",
                                dcc.Input(
                                    type="number",
                                    value=100_000,
                                    id="max-width-input",
                                    style={"width": 300},
                                ),
                            ],
                            style={"marginTop": 40},
                        ),
                        html.Div(
                            [
                                "global minimum persistence ",
                                dcc.Input(
                                    type="number",
                                    value=0.04,
                                    id="glob-pers-input",
                                    style={"width": 300},
                                ),
                            ],
                            style={"marginTop": 40},
                        ),
                        html.Div(
                            [
                                "constrain heights ",
                                dcc.Dropdown(
                                    options=["True", "False"],
                                    value="False",
                                    id="constrain-heights-input",
                                    style={"width": 300},
                                ),
                            ],
                            style={"marginTop": 40},
                        ),
                        html.Div(
                            [
                                "local minimal persistence ",
                                dcc.Input(
                                    type="number",
                                    value=0.33,
                                    id="loc-min-pers-input",
                                ),
                            ],
                            style={"marginTop": 40},
                        ),
                        html.Div(
                            [
                                "local trend minimum ",
                                dcc.Input(
                                    type="number",
                                    value=0.25,
                                    id="loc-trend-input",
                                    style={"width": 300},
                                ),
                            ],
                            style={"marginTop": 40},
                        ),
                        html.Div(
                            [
                                "force ",
                                dcc.Dropdown(
                                    options=["True", "False"],
                                    value="False",
                                    id="force-input",
                                    style={"width": 300},
                                ),
                            ],
                            style={"marginTop": 40},
                        ),
                        html.Div(
                            [
                                "nproc",
                                dcc.Input(
                                    type="number",
                                    value=1,
                                    id="nproc-input",
                                    style={"width": 300},
                                ),
                            ],
                            style={"marginTop": 40},
                        ),
                        html.Div(
                            [
                                "minimum chromosome size ",
                                dcc.Input(
                                    type="number",
                                    value=2_000_000,
                                    id="min-chrom-size-input",
                                ),
                            ],
                            style={"marginTop": 40},
                        ),
                        html.Div(
                            [
                                "verbosity ",
                                dcc.Dropdown(
                                    options=["debug", "info", "warning", "error", "critical"],
                                    value="info",
                                    id="verbosity-input",
                                    style={"width": 300},
                                ),
                            ],
                            style={"marginTop": 40},
                        ),
                        html.Div(
                            [
                                "relative change ",
                                dcc.Input(
                                    type="number",
                                    value=0.5,
                                    id="rel-change-input",
                                    style={"width": 300},
                                ),
                            ],
                            style={"marginTop": 40},
                        ),
                        html.Div(
                            [
                                "stripe type ",
                                dcc.Input(
                                    type="number",
                                    value=0.5,
                                    id="stripe-type-input",
                                    style={"width": 300},
                                ),
                            ],
                            style={"marginTop": 40},
                        ),
                        html.Button(
                            n_clicks=0,
                            children="call stripes",
                            id="start-calling",
                            disabled=False,
                            style={
                                "marginBottom": 30,
                                "marginTop": 30,
                                "display": "block",
                                "background-color": "mediumSlateBlue",
                            },
                        ),
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
