from dash import dcc, html


def render_metadata():
    """
    Render function for data including:
    - Name and length of chromosomes in the file
    - Earlier runs, for quick reruns
    - Buttons relating to rerunning and toggling hide/show metadata
    """
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
    """
    Stores all data related to the last runs of matrix plotting and stripe calling.
    """
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
            html.P("-1", id="last-used-rel-change"),
            html.Button(id="created-stripes-map", n_clicks=0),
            html.P("", id="last-used-file-directory"),
            html.P("", id="result-chrom-name"),  # chrom_name
            html.P("", id="result-chrom-size"),  # chrom_size
            html.P("", id="result-min-persistence"),  # min_persistence
            html.P("", id="result-ut-pseudodistribution"),  # ut_pseudodistribution
            html.P("", id="result-lt-pseudodistribution"),  # lt_pseudodistribution
            html.P("", id="result-ut-all-minimum-points"),  # ut_all_minimum_points
            html.P("", id="result-lt-all-minimum-points"),  # lt_all_minimum_points
            html.P("", id="result-ut-all-maximum-points"),  # ut_all_maximum_points
            html.P("", id="result-lt-all-maximum-points"),  # lt_all_maximum_points
            html.P("", id="result-ut-persistence-of-all-minimum-points"),  # ut_persistence_of_all_minimum_points
            html.P("", id="result-lt-persistence-of-all-minimum-points"),  # lt_persistence_of_all_minimum_points
            html.P("", id="result-ut-persistence-of-all-maximum-points"),  # ut_persistence_of_all_maximum_points
            html.P("", id="result-lt-persistence-of-all-maximum-points"),  # lt_persistence_of_all_maximum_points
            html.P("", id="result-ut-persistent-minimum-points"),  # ut_persistent_minimum_points
            html.P("", id="result-lt-persistent-minimum-points"),  # lt_persistent_minimum_points
            html.P("", id="result-ut-persistent-maximum-points"),  # ut_persistent_maximum_points
            html.P("", id="result-lt-persistent-maximum-points"),  # lt_persistent_maximum_points
            html.P("", id="result-ut-persistence-of-minimum-points"),  # ut_persistence_of_minimum_points
            html.P("", id="result-lt-persistence-of-minimum-points"),  # lt_persistence_of_minimum_points
            html.P("", id="result-ut-persistence-of-maximum-points"),  # ut_persistence_of_maximum_points
            html.P("", id="result-lt-persistence-of-maximum-points"),  # lt_persistence_of_maximum_points
            html.P("", id="result-ut-stripes"),  # ut_stripes
            html.P("", id="result-lt-stripes"),  # lt_stripes
        ],
        id="last-used-values",
        hidden=True,
    )
