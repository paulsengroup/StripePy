from dash import dcc, html

from .calling import (
    render_constrain_heights,
    render_genomic_belt,
    render_global_minimum_persistence,
    render_k_neighbours,
    render_local_minimum_persistence,
    render_local_trend_minimum,
    render_max_width,
    render_minimum_chromosome_size,
    render_nrpoc,
    render_relative_change,
    render_stripe_calling_button,
)
from .metadata import render_hidden, render_metadata
from .plotting import (
    render_chromosome_name,
    render_color_map,
    render_filepath,
    render_normalization,
    render_radio_log_scale,
    render_resolution,
    render_submit_button,
    render_warning,
)


def layout():
    """
    Contains the web page content in HTML form.

    Returns
    -------
    An HTML div containing every other component of the web page.
    """
    return html.Div(
        [
            html.Div(
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
                                hidden=True,
                                id="heat-map",
                            ),
                            render_metadata(),
                        ],
                        style={"display": "inline-block"},
                        hidden=True,
                        id="data",
                    ),
                    html.Div(
                        [  # Right side of screen
                            render_warning(),
                            render_filepath(),
                            render_resolution(),
                            render_radio_log_scale(),
                            render_chromosome_name(),
                            render_color_map(),
                            render_normalization(),
                            render_submit_button(),
                            render_genomic_belt(),
                            render_max_width(),
                            render_global_minimum_persistence(),
                            render_constrain_heights(),
                            render_k_neighbours(),
                            render_local_minimum_persistence(),
                            render_local_trend_minimum(),
                            render_nrpoc(),
                            render_minimum_chromosome_size(),
                            render_relative_change(),
                            render_stripe_calling_button(),
                        ],
                        style={"marginTop": 95},
                    ),
                ],
                style={"display": "flex"},
            ),
            html.Div(id="callbacks-file", style={"display": "none"}),
            render_hidden(),
        ]
    )
