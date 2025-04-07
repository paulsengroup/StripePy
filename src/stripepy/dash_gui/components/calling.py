from dash import dcc, html

DEFAULT_GEN_BELT = "5 000 000"
DEFAULT_MAX_WIDTH = "100 000"
DEFAULT_GLOB_PERS = "0.04"
DEFAULT_CONSTRAIN_HEIGHTS = "False"
DEFAULT_LOC_MIN_PERS = "0.33"
DEFAULT_LOC_TREND = "0.25"
DEFAULT_FORCE = "False"
DEFAULT_NPROC = "1"
DEFAULT_MIN_CHROM_SIZE = "2 000 000"
DEFAULT_VERBOSITY = "info"
DEFAULT_REL_CHANGE = "0.5"


def render_genomic_belt():
    return html.Div(
        [
            "genomic belt ",
            dcc.Input(
                type="text",
                value=DEFAULT_GEN_BELT,
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
                type="text",
                value=DEFAULT_MAX_WIDTH,
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
                type="text",
                value=DEFAULT_GLOB_PERS,
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
                value=DEFAULT_CONSTRAIN_HEIGHTS,
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
                type="text",
                value=DEFAULT_LOC_MIN_PERS,
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
                type="text",
                value=DEFAULT_LOC_TREND,
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
                value=DEFAULT_FORCE,
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
                type="text",
                value=DEFAULT_NPROC,
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
                type="text",
                value=DEFAULT_MIN_CHROM_SIZE,
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
                value=DEFAULT_VERBOSITY,
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
                type="text",
                value=DEFAULT_REL_CHANGE,
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
