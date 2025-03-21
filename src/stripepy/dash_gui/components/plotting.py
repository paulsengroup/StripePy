import plotly.express as px
from dash import dcc, html


def render_filepath():
    return html.Div(
        [
            dcc.Input(
                placeholder="File path",
                type="text",
                value="C:\\Users\\grebk\\OneDrive\\Dokumenter\\Studier\\IFI\\Master\\4DNFIOTPSS3L.hic",
                id="file-path",
                style={"width": 300, "fontSize": 20},
            ),
            html.Br(),
            html.Button(
                id="look-for-file",
                n_clicks=0,
                children="Find file",
            ),
            html.Br(),
        ],
    )


def render_resolution():
    return html.Div(
        [
            dcc.Dropdown(
                value=None,
                disabled=True,
                id="resolution",
                style={"width": 300, "fontSize": 20},
            ),
            html.Br(),
            html.Button(
                id="submit-file",
                n_clicks=0,
                children="Submit",
                disabled=True,
                style={"marginBottom": 100},
            ),
            html.Br(),
        ],
    )


def render_chromosome_name():
    return html.Div(
        [
            dcc.Input(
                placeholder="[Chromosome name]:start_number-end_number",
                type="text",
                value="2L:10,000,000-20,000,000",
                id="chromosome-name",
                disabled=True,
                style={"width": 700, "fontSize": 20},
            ),
            html.Br(),
        ],
    )


def render_color_map():
    return html.Div(
        [
            dcc.Dropdown(
                options=["Hi-C", *px.colors.named_colorscales()],
                placeholder="Color map",
                value=None,
                id="color-map",
                disabled=True,
                style={"width": 300, "fontSize": 20},
            ),
            html.Br(),
        ],
    )


def render_normalization():
    return html.Div(
        [
            dcc.Dropdown(
                options=["KR", "VC", "VC_SQRT"],
                placeholder="Normalization",
                value="KR",
                id="normalization",
            ),
            html.Br(),
        ],
    )


def render_submit_button():
    return html.Button(
        n_clicks=0,
        children="Submit",
        id="submit-chromosome",
        disabled=True,
        style={"marginBottom": 60},
    )
