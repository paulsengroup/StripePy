import dash_bootstrap_components as dbc
from dash import dcc, html

DEFAULT_GEN_BELT = "5,000,000"
DEFAULT_MAX_WIDTH = "100,000"
DEFAULT_GLOB_PERS = "0.04"
DEFAULT_CONSTRAIN_HEIGHTS = "False"
DEFAULT_K_NEIGHBOURS = "3"
DEFAULT_LOC_MIN_PERS = "0.33"
DEFAULT_LOC_TREND = "0.25"
DEFAULT_NPROC = "16"
DEFAULT_REL_CHANGE = "5.0"


def render_genomic_belt():
    """Genomic belt DCC input field and pop-up tooltip"""
    return html.Div(
        [
            "genomic belt ",
            dcc.Input(
                type="text",
                value=DEFAULT_GEN_BELT,
                id="gen-belt-input",
                style={"width": 300},
            ),
            "  ",
            html.I(
                id="gen-belt-icon",
                className="fa-solid fa-circle-question",
            ),
            dbc.Tooltip(
                f"Radius of the band, centred around the diagonal, where the search is restricted to, in bp.",
                target="gen-belt-icon",
                placement="right",
                trigger="hover",
            ),
        ],
        style={"marginTop": 40},
    )


def render_max_width():
    """Maximum width DCC input field and pop-up tooltip"""
    return html.Div(
        [
            "max width ",
            dcc.Input(
                type="text",
                value=DEFAULT_MAX_WIDTH,
                id="max-width-input",
                style={"width": 300},
            ),
            "  ",
            html.I(
                id="max-width-icon",
                className="fa-solid fa-circle-question",
            ),
            dbc.Tooltip(
                f"Maximum stripe width, in bp.",
                target="max-width-icon",
                placement="right",
                trigger="hover",
            ),
        ],
        style={"marginTop": 40},
    )


def render_global_minimum_persistence():
    """Global minimum persistence DCC input field and pop-up tooltip"""
    return html.Div(
        [
            "global minimum persistence ",
            dcc.Input(
                type="text",
                value=DEFAULT_GLOB_PERS,
                id="glob-pers-input",
                style={"width": 300},
            ),
            "  ",
            html.I(
                id="glob-pers-icon",
                className="fa-solid fa-circle-question",
            ),
            dbc.Tooltip(
                f"Threshold value between 0 and 1 to filter persistence maxima points and identify loci of interest, aka seeds.",
                target="glob-pers-icon",
                placement="right",
                trigger="hover",
            ),
        ],
        style={"marginTop": 40},
    )


def render_constrain_heights():
    """Height restriction DCC dropdown field and pop-up tooltip"""
    return html.Div(
        [
            "constrain heights ",
            dcc.Dropdown(
                options=["True", "False"],
                value=DEFAULT_CONSTRAIN_HEIGHTS,
                id="constrain-heights-input",
                style={"width": 300},
            ),
            "  ",
            html.I(
                id="constrain-heights-icon",
                className="fa-solid fa-circle-question",
            ),
            dbc.Tooltip(
                f"Use peaks in signal to constrain the stripe height.",
                target="constrain-heights-icon",
                placement="right",
                trigger="hover",
            ),
        ],
        style={"marginTop": 40, "display": "flex", "alignItems": "center"},
    )


def render_k_neighbours():
    return html.Div(
        [
            "k neighbours ",
            dcc.Input(type="text", value=DEFAULT_K_NEIGHBOURS, id="k-neighbours-input"),
            "  ",
            html.I(
                id="k-neighbours-icon",
                className="fa-solid fa-circle-question",
            ),
            dbc.Tooltip(
                f"k for the k-neighbour, i.e., number of bins adjacent to the stripe boundaries on both sides.",
                target="k-neighbours-icon",
                placement="right",
                trigger="hover",
            ),
        ],
        style={"marginTop": 40},
    )


def render_local_minimum_persistence():
    """Local minimum persistence DCC input field and pop-up tooltip"""
    return html.Div(
        [
            "local minimal persistence ",
            dcc.Input(
                type="text",
                value=DEFAULT_LOC_MIN_PERS,
                id="loc-min-pers-input",
            ),
            "  ",
            html.I(
                id="loc-min-pers-icon",
                className="fa-solid fa-circle-question",
            ),
            dbc.Tooltip(
                f"Threshold value between 0 and 1 to find peaks in signal in a horizontal domain while estimating the height of a stripe; when --constrain-heights is set to 'False', it is not used.",
                target="loc-min-pers-icon",
                placement="right",
                trigger="hover",
            ),
        ],
        style={"marginTop": 40},
    )


def render_local_trend_minimum():
    """Stripe height threshold DCC input field and pop-up tooltip"""
    return html.Div(
        [
            "local trend minimum ",
            dcc.Input(
                type="text",
                value=DEFAULT_LOC_TREND,
                id="loc-trend-input",
                style={"width": 300},
            ),
            "  ",
            html.I(
                id="loc-trend-icon",
                className="fa-solid fa-circle-question",
            ),
            dbc.Tooltip(
                f"Threshold value between 0 and 1 to estimate the height of a stripe; the higher this value, the shorter the stripe; it is always used when --constrain-heights is set to 'False', but could be necessary also when --constrain-heights is 'True' and no persistent maximum other than the global maximum is found.",
                target="loc-trend-icon",
                placement="right",
                trigger="hover",
            ),
        ],
        style={"marginTop": 40},
    )


def render_nproc():
    """Parallelization count DCC input field and pop-up tooltip"""
    return html.Div(
        [
            "nproc",
            dcc.Input(
                type="text",
                value=DEFAULT_NPROC,
                id="nproc-input",
                style={"width": 300},
            ),
            "  ",
            html.I(
                id="nproc-icon",
                className="fa-solid fa-circle-question",
            ),
            dbc.Tooltip(
                f"Maximum number of parallel processes to use.",
                target="nproc-icon",
                placement="right",
                trigger="hover",
            ),
        ],
        style={"marginTop": 40},
    )


def render_relative_change():
    """Relative change DCC input field and pop-up tooltip"""
    return html.Div(
        [
            "relative change ",
            dcc.Input(
                type="text",
                value=DEFAULT_REL_CHANGE,
                id="rel-change-input",
                style={"width": 300},
            ),
            "  ",
            html.I(
                id="rel-change-icon",
                className="fa-solid fa-circle-question",
            ),
            dbc.Tooltip(
                f"Cutoff for the relative change.\n"
                "Only used when highlighting architectural stripes.\n"
                "The relative change is computed as the ratio between the average number of interactions found inside a stripe and the number of interactions in a neighborhood outside of the stripe.",
                target="rel-change-icon",
                placement="right",
                trigger="hover",
            ),
        ],
        style={"marginTop": 40},
    )


def render_stripe_calling_button():
    """Button that activates the stripe calling process"""
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
