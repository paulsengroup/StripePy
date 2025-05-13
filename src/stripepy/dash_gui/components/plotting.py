import dash_bootstrap_components as dbc
import plotly.express as px
from dash import dcc, html


def render_filepath():
    """Text input for path to local file, pop-up tooltip and activation button"""
    return html.Div(
        [
            "File path",
            html.Div(
                [
                    dcc.Input(
                        placeholder="File path",
                        type="text",
                        value=None,
                        id="file-path",
                        style={"width": 300, "fontSize": 20},
                    ),
                    "  ",
                    html.I(
                        id="file-path-icon",
                        className="fa-solid fa-circle-question",
                    ),
                    dbc.Tooltip(
                        "Input the path to a file on your computer to let hictk-py access it.",
                        target="file-path-icon",
                        placement="right",
                        trigger="hover",
                    ),
                ],
            ),
            html.Button(
                id="look-for-file",
                n_clicks=0,
                children="Find file",
            ),
            html.Br(),
        ],
    )


def render_resolution():
    """Dropdown menu for resolution options and activation button"""
    return html.Div(
        [
            "Resolution",
            dcc.Dropdown(
                value=None,
                disabled=True,
                id="resolution",
                style={"width": 300, "fontSize": 20},
            ),
            html.Button(
                id="submit-file",
                n_clicks=0,
                children="Submit",
                disabled=True,
                style={"marginBottom": 50},
            ),
            html.Br(),
        ],
    )


def render_radio_log_scale():
    """Lets user choose between log scale and normal"""
    return html.Div(
        [
            dcc.RadioItems(
                options=["log scale", "raw scale"],
                value="log scale",
                id="radio-log",
            ),
            html.Br(),
        ],
    )


def render_radio_items():
    """Options list for SI suffix on lower X axis"""
    return html.Div(
        [
            dcc.RadioItems(
                options=["Kb", "Mb", "Gb"],
                value="Mb",
                id="radioitems",
            )
        ],
    )


def render_chromosome_name():
    """Text input for square restriction of matrix and pop-up tooltip"""
    return html.Div(
        [
            dcc.Input(
                placeholder="[Chromosome name]:start_number-end_number",
                type="text",
                # value="2L:10,000,000-20,000,000",
                value="",
                id="chromosome-name",
                disabled=True,
                style={"width": 600, "fontSize": 20},
            ),
            "  ",
            html.I(
                id="chromosome-name-icon",
                className="fa-solid fa-circle-question",
                hidden=True,
            ),
            dbc.Tooltip(
                "Chromosome names and spans can be found in the dropdown menu.",
                target="chromosome-name-icon",
                placement="right",
                trigger="hover",
            ),
            html.Br(),
        ],
    )


def render_color_map():
    """Dropdown menu of color options and pop-up tooltip"""
    return html.Div(
        [
            dcc.Dropdown(
                options=[
                    "fruit_punch ",
                    "fall ",
                    "blues ",
                    "acidblues ",
                    "nmeth ",
                    *px.colors.named_colorscales(),
                ],
                placeholder="Color map",
                value="fruit_punch ",
                id="color-map",
                disabled=True,
                style={"width": 300, "fontSize": 20},
            ),
            "  ",
            html.I(
                id="color-map-icon",
                className="fa-solid fa-circle-question",
                hidden=True,
            ),
            dbc.Tooltip(
                "The color maps are the custom colorscales from cooltools",
                target="color-map-icon",
                placement="right",
                trigger="hover",
            ),
            html.Br(),
            html.Br(),
            html.Br(),
        ],
        style={"display": "flex", "alignItems": "center"},
    )


def render_normalization():
    """Dropdown menu of normalization techniques and pop-up tooltip"""
    return html.Div(
        [
            dcc.Dropdown(
                placeholder="Normalization",
                value="KR",
                id="normalization",
                disabled=True,
                style={"width": 300, "fontSize": 20},
            ),
            "  ",
            html.I(
                id="normalization-icon",
                className="fa-solid fa-circle-question",
                hidden=True,
            ),
            dbc.Tooltip(
                "Choose between Knight-Ruiz, Vanilla coverage and Vanilla coverage (square root)",
                target="normalization-icon",
                placement="right",
                trigger="hover",
            ),
            html.Br(),
        ],
        style={"display": "flex", "alignItems": "center"},
    )


def render_submit_button():
    """Activation button for plotting"""
    return html.Button(
        n_clicks=0,
        children="Submit",
        id="submit-chromosome",
        disabled=True,
        style={"marginBottom": 60},
    )
