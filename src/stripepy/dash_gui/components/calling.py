from dash import dcc, html


def render_genomic_belt():
    return html.Div(
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
    )


def render_max_width():
    return html.Div(
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
    )


def render_global_minimum_persistence():
    return html.Div(
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
    )


def render_constrain_heights():
    return html.Div(
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
    )


def render_local_minimum_persistence():
    return html.Div(
        [
            "local minimal persistence ",
            dcc.Input(
                type="number",
                value=0.33,
                id="loc-min-pers-input",
            ),
        ],
        style={"marginTop": 40},
    )


def render_local_trend_minimum():
    return html.Div(
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
    )


def render_force():
    return html.Div(
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
    )


def render_nrpoc():
    return html.Div(
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
    )


def render_minimum_chromosome_size():
    return html.Div(
        [
            "minimum chromosome size ",
            dcc.Input(
                type="number",
                value=2_000_000,
                id="min-chrom-size-input",
            ),
        ],
        style={"marginTop": 40},
    )


def render_verbosity():
    return html.Div(
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
    )


def render_relative_change():
    return html.Div(
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
    )


def render_stripe_type():
    return html.Div(
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
    )


def render_stripe_calling_button():
    return html.Button(
        n_clicks=0,
        children="call stripes",
        id="start-calling",
        disabled=False,
        style={
            "marginBottom": 30,
            "marginTop": 30,
            "display": "block",
            "backgroundColor": "mediumSlateBlue",
        },
    )
