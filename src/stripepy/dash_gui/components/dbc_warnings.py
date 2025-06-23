import dash_bootstrap_components as dbc
from dash import html
from dash_bootstrap_components import Alert


def compose_stale_component_warning(comp_names):
    comp_naming_string = ""
    for name in comp_names:
        comp_naming_string += "\n•" + name
    return comp_naming_string


def warning_stale_component(comp_names):
    comp_naming_string = ""
    for index, name in enumerate(comp_names):
        comp_naming_string += " " + name
        if not index == len(comp_names) - 1:
            comp_naming_string += ","
    return html.Div(
        [
            Alert(f"Matrix did not update: change one of:{comp_naming_string}", color="info"),
        ],
        id="alert-message",
    )


def warning_no_stripes():
    return html.Div(
        [
            Alert("No stripes were found", color="info"),
        ],
        id="alert-message",
    )


def warning_null():
    return html.Div([], id="alert-message", hidden=True)


def warning_pick_save_file():
    return html.Div(
        [
            Alert(
                'Please make sure one of the save strings are highlighted when pressing the "Choose save file" button.',
                color="warning",
            )
        ],
        id="alert-message",
    )
