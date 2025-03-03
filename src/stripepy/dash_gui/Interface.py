# Copyright (C) 2025 Bendik Berg <bendber@ifi.uio.no>
#
# SPDX-License-Identifier: MIT

import hictkpy
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dcc, html
from HiCObject import HiCObject

import stripepy

app = Dash(__name__)

hictk_reader = HiCObject()
hictk_reader.path = "4DNFIOTPSS3L.hic"
hictk_reader.resolution = 5000
hictk_reader.region_of_interest = "2L:10,000,000-20,000,000"
hictk_reader.normalization = "KR"

attributes = hictk_reader.attributes
nnz = hictk_reader.nnz
sum = hictk_reader.sum
frame = hictk_reader.frame


# fig = go.Figure(data=go.Heatmap(z=np.log10(frame)))
fig = go.Figure(data=go.Heatmap(z=np.where(frame > 0, np.log10(frame), 0)))
fig.update_yaxes(autorange="reversed")

# Forms add several text boxes. One form for adding a plot instead of dcc. Upload, tie to update_file. File name and resolution.
# https://dash-bootstrap-components.opensource.faculty.ai/docs/components/form/


app.layout = html.Div(
    [
        html.H1("4DNFIOTPSS3L chromosome 2L"),
        html.Div([html.P((attribute, ": ", str(value))) for attribute, value in attributes.items()]),
        html.P(("Non-zero values: ", str(nnz()))),
        html.P(("Sum: ", str(sum()))),
        html.Div(
            [
                html.Div(
                    [
                        dcc.Graph(
                            # id="HeatMap",
                            figure=fig,
                            style={"width": "90vh", "height": "90vh"},
                        ),
                    ],
                    style={"display": "inline-block"},
                    id="HeatMap",
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                dcc.Input(placeholder="File path", type="text", value="", id="file-path"),
                                dcc.Input(placeholder="resolution", type="number", value="", id="resolution"),
                                html.Button(id="submit-file", n_clicks=0, children="Submit"),
                            ],
                            style={"padding-bottom": 60},
                        ),
                        html.Div(
                            [
                                dcc.Input(
                                    placeholder="Chromosome name",
                                    type="text",
                                    value="",
                                    id="chromosome-name",
                                ),
                                dcc.Input(
                                    placeholder="Color map",
                                    type="text",
                                    value="",
                                    id="color-map",
                                ),
                                html.Button(id="submit-chromosome", n_clicks=0, children="Submit"),
                            ],
                            style={"padding-bottom": 60},
                        ),
                        dcc.Dropdown(
                            options=["KR", "VC", "VC_SQRT"],
                            placeholder="Normalization",
                            value="KR",
                            id="normalization",
                        ),
                    ],
                    style={"margin-top": 95},
                ),
            ],
            style={"display": "flex"},
        ),
        html.Div(id="callbacks-output"),
    ]
)


@app.callback(
    Output("HeatMap", "children"),
    # Output("callbacks-output", "children"),
    Input("submit-file", "n_clicks"),
    State("resolution", "value"),
    State("file-path", "value"),
    prevent_initial_call=True,
    running=[(Output("submit-file", "disabled"), True, False)],
)
def update_file(n_clicks, filename, resolution):
    clr = hictkpy.File(filename, resolution)
    # return clr
    # return (filename, resolution)


# Dropdown menu for which chromosome to choose; interval of that chromosome on a slider. Resolution should also be changeable in hindsight, which should ideally trigger the function above.
@app.callback(
    # Output("HeatMap", "children"),
    # Output(chosenChromosome),
    Input("chromosomes-dropdown", "value"),
    Input("chromosome-interval", "value"),
)
def update_plot(
    chromosome_name,
    chromosome_interval,
):
    chosenChromosome = chromosome_name

    chromosome_start, chromosome_end = chromosome_interval
    chromosome_start = str(chromosome_start)
    chromosome_end = str(chromosome_end)

    sel = clr.fetch((chromosome_name, ":", chromosome_start, "-", chromosome_end), join=True, normalization="KR")

    attributes = clr.attributes()
    m = sel.to_numpy()
    frame = pd.DataFrame(m)

    return frame


import webbrowser

# webbrowser.open("http://127.0.0.1:8050/")

if __name__ == "__main__":
    app.run_server(debug=True)
