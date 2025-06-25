from dash import dcc, html


def render_metadata():
    return html.Div(
        [
            html.Button(
                id="show-metadata",
                n_clicks=0,
                children="Hide metadata",
                type="button",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            dcc.RadioItems(id="files-list"),
                            html.Button(n_clicks=0, children="Choose saved file", id="pick-from-saved"),
                        ],
                        id="picked-files",
                    ),
                    html.Div(
                        [],
                        id="chromosomes",
                    ),
                ],
                id="meta-info",
                style={"overflow": "scroll", "max-height": 500},
            ),
        ],
    )


def render_hidden():
    return html.Div(
        [
            html.P("", id="last-used-path"),
            html.P("", id="last-used-resolution"),
            html.P("", id="last-used-scale-type"),
            html.P("", id="last-used-region"),
            html.P("", id="last-used-color-map"),
            html.P("", id="last-used-normalization"),
            html.P("", id="calling-last-used-path"),
            html.P("", id="calling-last-used-resolution"),
            html.P("", id="calling-last-used-scale-type"),
            html.P("", id="calling-last-used-region"),
            html.P("", id="calling-last-used-color-map"),
            html.P("", id="calling-last-used-normalization"),
            html.P("-1", id="last-used-gen-belt"),
            html.P("-1", id="last-used-max-width"),
            html.P("-1", id="last-used-glob-pers-min"),
            html.P("", id="last-used-constrain-heights"),
            html.P("-1", id="last-used-k"),
            html.P("-1", id="last-used-loc-pers-min"),
            html.P("-1", id="last-used-loc-trend-min"),
            html.P("-1", id="last-used-nproc"),
            html.Button(id="created-stripes-map", n_clicks=0),
            html.P("", id="last-used-file-directory"),
        ],
        id="last-used-values",
        hidden=True,
    )
