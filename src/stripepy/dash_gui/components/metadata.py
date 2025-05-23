from dash import dcc, html


def render_metadata():
    return html.Div(
        [
            html.Button(
                id="show-metadata",
                n_clicks=0,
                children="Show metadata",
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
            ),
        ],
    )


def render_hidden():
    return html.Div(
        [
            html.P("", id="last-used-path"),
            html.P("", id="last-used-resolution"),
            html.P("", id="last-used-region"),
            html.P("", id="last-used-color-map"),
            html.P("", id="last-used-normalization"),
        ],
        id="last-used-values",
        hidden=True,
    )
