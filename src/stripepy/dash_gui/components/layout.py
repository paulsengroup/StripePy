from dash import dcc, html

from .calling import (
    render_constrain_heights,
    render_force,
    render_genomic_belt,
    render_global_minimum_persistence,
    render_local_minimum_persistence,
    render_local_trend_minimum,
    render_max_width,
    render_minimum_chromosome_size,
    render_nrpoc,
    render_relative_change,
    render_stripe_calling_button,
    render_verbosity,
)
from .plotting import (
    render_chromosome_name,
    render_color_map,
    render_filepath,
    render_normalization,
    render_radio_items,
    render_resolution,
    render_submit_button,
)


def layout():
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            dcc.Graph(
                                style={"width": "90vh", "height": "90vh"},
                                id="HeatMap",
                            ),
                            html.Div(
                                id="meta-info",
                            ),
                        ],
                        style={"display": "inline-block"},
                    ),
                    html.Div(
                        [  # Right side of screen
                            render_filepath(),
                            render_resolution(),
                            render_radio_items(),
                            render_chromosome_name(),
                            render_color_map(),
                            render_normalization(),
                            render_submit_button(),
                            render_genomic_belt(),
                            render_max_width(),
                            render_global_minimum_persistence(),
                            render_constrain_heights(),
                            render_local_minimum_persistence(),
                            render_local_trend_minimum(),
                            render_force(),
                            render_nrpoc(),
                            render_minimum_chromosome_size(),
                            render_verbosity(),
                            render_relative_change(),
                            render_stripe_calling_button(),
                        ],
                        style={"marginTop": 95},
                    ),
                ],
                style={"display": "flex"},
            ),
            html.Div(id="callbacks-file", style={"display": "none"}),
        ]
    )
