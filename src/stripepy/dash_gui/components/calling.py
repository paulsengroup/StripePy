import dash_bootstrap_components as dbc
from dash import dcc, html

DEFAULT_GEN_BELT = "5,000,000"
DEFAULT_MAX_WIDTH = "100,000"
DEFAULT_GLOB_PERS = "0.04"
DEFAULT_CONSTRAIN_HEIGHTS = "False"
DEFAULT_LOC_MIN_PERS = "0.33"
DEFAULT_LOC_TREND = "0.25"
DEFAULT_FORCE = "False"
DEFAULT_NPROC = "1"
DEFAULT_MIN_CHROM_SIZE = "2,000,000"
DEFAULT_VERBOSITY = "info"
DEFAULT_REL_CHANGE = "0.5"


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
                f"Radius of the band, centred around the diagonal, where the search is restricted to (in bp, default: {DEFAULT_GEN_BELT}.",
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
                f"Maximum stripe width, in bp (default: {DEFAULT_MAX_WIDTH}).",
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
                f"Threshold value between 0 and 1 to filter persistence maxima points and identify loci of interest, aka seeds (default: {DEFAULT_GLOB_PERS}).",
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
                f"Use peaks in signal to constrain the stripe height (default: {DEFAULT_CONSTRAIN_HEIGHTS}).",
                target="constrain-heights-icon",
                placement="right",
                trigger="hover",
            ),
        ],
        style={"marginTop": 40, "display": "flex", "alignItems": "center"},
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
                f"Threshold value between 0 and 1 to find peaks in signal in a horizontal domain while estimating the height of a stripe; when --constrain-heights is set to 'False', it is not used (default: {DEFAULT_LOC_MIN_PERS}).",
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
                f"Threshold value between 0 and 1 to estimate the height of a stripe (default: {DEFAULT_LOC_TREND}); the higher this value, the shorter the stripe; it is always used when --constrain-heights is set to 'False', but could be necessary also when --constrain-heights is 'True' and no persistent maximum other than the global maximum is found.",
                target="loc-trend-icon",
                placement="right",
                trigger="hover",
            ),
        ],
        style={"marginTop": 40},
    )


def render_force():
    """Force overwrite DCC dropdown field and pop-up tooltip"""
    return html.Div(
        [
            "force ",
            dcc.Dropdown(
                options=["True", "False"],
                value=DEFAULT_FORCE,
                id="force-input",
                style={"width": 300},
            ),
            "  ",
            html.I(
                id="force-icon",
                className="fa-solid fa-circle-question",
            ),
            dbc.Tooltip(
                f"Overwrite existing file(s) (default: {DEFAULT_FORCE}).",
                target="force-icon",
                placement="right",
                trigger="hover",
            ),
        ],
        style={"marginTop": 40, "display": "flex", "alignItems": "center"},
    )


def render_nrpoc():
    """Parallellization count DCC input field and pop-up tooltip"""
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
                f"Maximum number of parallel processes to use (default: {DEFAULT_NPROC}).",
                target="nproc-icon",
                placement="right",
                trigger="hover",
            ),
        ],
        style={"marginTop": 40},
    )


def render_minimum_chromosome_size():
    """Minimum chromosome size DCC input field and pop-up tooltip"""
    return html.Div(
        [
            "minimum chromosome size ",
            dcc.Input(
                type="text",
                value=DEFAULT_MIN_CHROM_SIZE,
                id="min-chrom-size-input",
            ),
            "  ",
            html.I(
                id="min-chrom-size-icon",
                className="fa-solid fa-circle-question",
            ),
            dbc.Tooltip(
                f"Minimum size, in bp, for a chromosome to be analysed (default: {DEFAULT_MIN_CHROM_SIZE}).",
                target="min-chrom-size-icon",
                placement="right",
                trigger="hover",
            ),
        ],
        style={"marginTop": 40},
    )


def render_verbosity():
    """Verbosity DCC dropdown field and pop-up tooltip"""
    return html.Div(
        [
            "verbosity ",
            dcc.Dropdown(
                options=["debug", "info", "warning", "error", "critical"],
                value=DEFAULT_VERBOSITY,
                id="verbosity-input",
                style={"width": 300},
            ),
            "  ",
            html.I(
                id="verbosity-icon",
                className="fa-solid fa-circle-question",
            ),
            dbc.Tooltip(
                f"Set verbosity of output to the console (default: {DEFAULT_VERBOSITY}).",
                target="verbosity-icon",
                placement="right",
                trigger="hover",
            ),
        ],
        style={"marginTop": 40, "display": "flex", "alignItems": "center"},
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
                f"Cutoff for the relative change (default: {DEFAULT_REL_CHANGE}).\n"
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
