import pathlib
from pathlib import Path
from tkinter import *
from tkinter import filedialog

import hictkpy as htk
import numpy as np
import plotly.graph_objects as go
from colorscales import color_scale
from components.axes import compute_x_axis_chroms, compute_x_axis_range
from components.colorbar import colorbar
from dash import dcc, html
from dash.exceptions import PreventUpdate
from plotly.subplots import make_subplots

from stripepy.cli import call
from stripepy.io import ProcessSafeLogger


def open_file_dialog_callback(n_clicks, look_for_file_n_clicks):
    root = Tk()
    root.filename = filedialog.askopenfilename(
        initialdir="C:\\", title="Select file", filetypes=(("Hi-C files", "*.hic *.cool *.mcool"), ("all files", "*.*"))
    )
    return root.filename, look_for_file_n_clicks + 1


def look_for_file_callback(file_path):
    file_path = Path(file_path)
    global last_used_file
    last_used_file = ""
    if file_path == last_used_file:
        raise PreventUpdate
    else:
        pass
    last_used_file = file_path

    mrf = htk.MultiResFile(file_path)
    resolutions = mrf.resolutions().tolist()

    # Pick the resolution closest to 25kb
    resolution_value = _pick_closest(resolutions, 25000)

    return resolutions, resolution_value, False, False


def _pick_closest(array, target_res):
    if target_res in array:
        return target_res

    last = array[0]
    for head in array:
        if last < target_res and head > target_res:
            if abs(int(last) - target_res) < abs(int(head) - target_res):
                # last value checked is closer to target value than the current checked value
                return last
            else:
                return head
        else:
            last = head


def update_file_callback(filename, resolution):
    global last_used_resolution
    try:
        if filename == last_used_file and resolution == last_used_resolution:
            raise PreventUpdate
        else:
            pass
    except NameError:
        pass
    last_used_file = filename
    last_used_resolution = resolution

    path = filename
    bin_size = resolution

    global f
    f = htk.File(path, bin_size)

    metaInfo_chromosomes = html.Div([html.P((chromosome, ":", name)) for chromosome, name in f.chromosomes().items()])
    metaInfo = html.Div(
        [html.P("Chromosomes", style={"fontSize": 24, "fontWeight": "bold"}), metaInfo_chromosomes], id="chromosomes"
    )

    avail_normalizations = f.avail_normalizations()
    avail_normalizations.append("No normalization")

    return metaInfo, avail_normalizations, avail_normalizations[0], False, False, False, False, False, False, False


def update_plot_callback(chromosome_name, colorMap, normalization, filepath, resolution, scale_type, files_list):
    filepath = Path(filepath)
    global last_used_chromosome_name
    global last_used_colorMap
    global last_used_normalization
    try:
        if (
            chromosome_name == last_used_chromosome_name
            and colorMap == last_used_colorMap
            and normalization == last_used_normalization
            and last_used_file == filepath
            and last_used_resolution == resolution
        ):
            raise PreventUpdate
        else:
            pass
    except NameError:
        pass
    last_used_chromosome_name = chromosome_name
    last_used_colorMap = colorMap
    last_used_normalization = normalization
    last_used_file = filepath
    last_used_resolution = resolution

    colorMap = color_scale(colorMap)
    if normalization == "No normalization":
        normalization = None

    sel = f.fetch(chromosome_name, normalization=normalization)
    frame = sel.to_numpy()
    frame = frame.astype(np.float32)
    to_string_vector = np.vectorize(str)
    inv_log_frame_string = to_string_vector(frame)

    if scale_type == "log scale":
        np.log(frame, out=frame, where=np.isnan(frame) == False)
    under_lowest_real_value = np.min(frame[np.isfinite(frame)]) - abs(np.min(frame[np.isfinite(frame)]))
    # isfinite() dicounts nan, inf and -inf

    frame = np.where(np.isneginf(frame), under_lowest_real_value, frame)

    if chromosome_name:
        fig = go.Figure(
            data=go.Heatmap(
                z=frame,
                colorbar=colorbar(frame, scale_type),
                colorscale=colorMap,
                customdata=inv_log_frame_string,
                hovertemplate="%{customdata}<extra></extra>",
            )
        )

        tickvals, ticktext = compute_x_axis_range(chromosome_name, f, resolution)
        fig.update_xaxes(tickvals=tickvals, ticktext=ticktext, showgrid=False)
        fig.update_yaxes(autorange="reversed", showgrid=False)
        fig.update_layout(plot_bgcolor="mediumslateblue")
        # NaN-values are transparent
    else:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Heatmap(
                z=frame,
                colorbar=colorbar(frame, scale_type),
                colorscale=colorMap,
            ),
            secondary_y=False,
        )
        fig.add_trace(
            go.Heatmap(
                z=frame,
                colorbar=colorbar(frame, scale_type),
                colorscale=colorMap,
                customdata=inv_log_frame_string,
                hovertemplate="%{customdata}<extra></extra>",
                hoverlabel={
                    "bgcolor": "green",
                },
            ),
            secondary_y=True,
        )

        tickvals, ticktext = compute_x_axis_range(chromosome_name, f, resolution)
        tickvals_chrom, ticktext_chrom = compute_x_axis_chroms(f)
        fig.update_layout(
            xaxis1=dict(tickvals=tickvals, ticktext=ticktext, showgrid=False, side="bottom"),
            xaxis2=dict(tickvals=tickvals_chrom, ticktext=ticktext_chrom, showgrid=False, side="top"),
            yaxis=dict(autorange="reversed", showgrid=False, visible=True),
            yaxis2=dict(autorange="reversed", showgrid=False, visible=False, side="right"),
            plot_bgcolor="mediumslateblue",
        )
        fig.data[1].update(xaxis="x2")

    # file_path = str(file_path)
    filepath_assembled_string = f"{filepath};{resolution};{chromosome_name};{normalization}"
    try:
        if filepath_assembled_string not in [values for dicts in files_list for values in dicts.values()]:
            files_list.append(
                {
                    "label": f"res={resolution}, norm={normalization}, region={chromosome_name if chromosome_name else "entire"}: {filepath.name}",
                    "value": f"{filepath};{resolution};{chromosome_name};{normalization}",
                }
            )
            # files_list.append(filepath)
    except TypeError:
        files_list = [
            {
                "label": f"res={resolution}, norm={normalization}, region={chromosome_name if chromosome_name else "entire"}: {filepath.name}",
                "value": f"{filepath};{resolution};{chromosome_name};{normalization}",
            }
        ]
        # files_list = [filepath]
    saved_paths = [file for file in files_list]

    return fig, files_list, False


def call_stripes_callback(
    chromosome_name,
    resolution,
    gen_belt,
    max_width,
    glob_pers,
    constrain_heights,
    loc_pers_min,
    loc_trend_min,
    force,
    nproc,
    min_chrom_size,
    verbosity,
    normalization,
    path,
):
    with ProcessSafeLogger(
        verbosity,
        path=pathlib.Path("./tmp/log_file"),
        force=force,
        matrix_file=path,
        print_welcome_message=True,
        progress_bar_type="call",
    ) as main_logger:
        call.run(
            chromosome_name,
            resolution,
            pathlib.Path("./tmp/called_stripes"),  # output file
            gen_belt,
            max_width,
            glob_pers,  # glob_pers_min, or maybe loc_pers_min?
            constrain_heights,  # constrain heights
            loc_pers_min,  # loc_pers_min
            loc_trend_min,
            force,  # force
            nproc,  # nproc
            min_chrom_size,  # min_chrom_size
            verbosity,
            main_logger,  # main_logger,
            # roi,
            log_file=pathlib.Path("./tmp/log_file"),  # log_file,
            # plot_dir,
            normalization=normalization,
        )
