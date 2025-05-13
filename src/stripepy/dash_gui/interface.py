# Copyright (C) 2025 Bendik Berg <bendber@ifi.uio.no>
#
# SPDX-License-Identifier: MIT

import pathlib

import dash_bootstrap_components as dbc
import hictkpy as htk
import numpy as np
import plotly.graph_objects as go
from colorscales import color_scale
from components.axes import compute_x_axis_chroms, compute_x_axis_range
from components.colorbar import colorbar
from components.layout import layout
from dash import Dash, Input, Output, State, html
from dash.exceptions import PreventUpdate
from plotly.subplots import make_subplots

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
    global last_used_file
    last_used_file = ""
    if file_path == last_used_file:
        raise PreventUpdate
    else:
        pass
    last_used_file = file_path

    mrf = htk.MultiResFile(file_path)
    resolutions = mrf.resolutions()
    return resolutions, resolutions[0], False, False


@app.callback(
    Output("meta-info", "children"),
    Output("normalization", "options"),
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

    avail_normalizations = f.avail_normalizations()

    return metaInfo, avail_normalizations, False, False, False, False, False, False, False


@app.callback(
    Output("HeatMap", "figure"),
    Input("submit-chromosome", "n_clicks"),
    State("chromosome-name", "value"),
    State("color-map", "value"),
    State("normalization", "value"),
    State("file-path", "value"),
    State("resolution", "value"),
    State("radioitems", "value"),
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
def update_plot(n_clicks, chromosome_name, colorMap, normalization, filepath, resolution, radio_element):
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

    colorMap = color_scale(colorMap)

    sel = f.fetch(chromosome_name, normalization=normalization)
    frame = sel.to_numpy()
    to_string_vector = np.vectorize(str)
    inv_log_frame_string = to_string_vector(frame)

    np.log(frame, out=frame, where=np.isnan(frame) == False)
    under_lowest_real_value = np.min(frame[np.isfinite(frame)]) - abs(np.min(frame[np.isfinite(frame)]))
    # isfinite() dicounts nan, inf and -inf

    frame = np.where(np.isneginf(frame), under_lowest_real_value, frame)

    if chromosome_name:
        fig = go.Figure(
            data=go.Heatmap(
                z=frame,
                colorbar=colorbar(frame),
                colorscale=colorMap,
                customdata=inv_log_frame_string,
                hovertemplate="%{customdata}<extra></extra>",
            )
        )

        tickvals, ticktext = compute_x_axis_range(chromosome_name, f, resolution, radio_element)
        fig.update_xaxes(tickvals=tickvals, ticktext=ticktext, showgrid=False)
        fig.update_yaxes(autorange="reversed", showgrid=False)
        fig.update_layout(plot_bgcolor="mediumslateblue")
        # NaN-values are transparent
    else:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Heatmap(
                z=frame,
                colorbar=colorbar(frame),
                colorscale=colorMap,
            ),
            secondary_y=False,
        )
        fig.add_trace(
            go.Heatmap(
                z=frame,
                colorbar=colorbar(frame),
                colorscale=colorMap,
                customdata=inv_log_frame_string,
                hovertemplate="%{customdata}<extra></extra>",
                hoverlabel={
                    "bgcolor": "green",
                },
            ),
            secondary_y=True,
        )

        tickvals, ticktext = compute_x_axis_range(chromosome_name, f, resolution, radio_element)
        tickvals_chrom, ticktext_chrom = compute_x_axis_chroms(f)
        fig.update_layout(
            xaxis1=dict(tickvals=tickvals, ticktext=ticktext, showgrid=False, side="bottom"),
            xaxis2=dict(tickvals=tickvals_chrom, ticktext=ticktext_chrom, showgrid=False, side="top"),
            yaxis=dict(autorange="reversed", showgrid=False, visible=True),
            yaxis2=dict(autorange="reversed", showgrid=False, visible=False, side="right"),
            plot_bgcolor="mediumslateblue",
        )
        fig.data[1].update(xaxis="x2")

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
