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
from stripes import add_stripes_chrom_restriction, add_stripes_whole_chrom

from stripepy.cli import call
from stripepy.io import ProcessSafeLogger, open_matrix_file_checked


def open_file_dialog_callback(base_directory):
    if base_directory == "":
        go_to_directory = "."
    else:
        go_to_directory = base_directory
    root = Tk()
    root.filename = filedialog.askopenfilename(
        initialdir=go_to_directory,
        title="Select file",
        filetypes=(("Hi-C files", "*.hic *.cool *.mcool"), ("all files", "*.*")),
    )
    root.destroy()
    return root.filename


def look_for_file_callback(file_path, last_used_path):
    if file_path == last_used_path:
        raise PreventUpdate
    else:
        pass
    file_path = Path(file_path)

    f, resolutions, resolution_value = _pick_resolution_and_array(file_path)

    metaInfo_chromosomes = html.Div([html.P((chromosome, ":", name)) for chromosome, name in f.chromosomes().items()])
    metaInfo = html.Div(
        [html.P("Chromosomes", style={"fontSize": 24, "fontWeight": "bold"}), metaInfo_chromosomes], id="chromosomes"
    )

    return (
        resolutions,
        resolution_value,
        metaInfo,
        str(file_path.parent),
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
    )


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


def _is_multi_res(path):
    if htk.is_cooler(path):
        return False
    else:
        return True


def _pick_resolution_and_array(path):
    file_is_multi_res = _is_multi_res(path)
    if file_is_multi_res:
        temp_f = htk.MultiResFile(path)
        resolutions = temp_f.resolutions().tolist()

        # Pick the resolution closest to 25kb
        resolution_value = _pick_closest(resolutions, 25000)
        f = htk.File(path, resolution_value)
    else:
        f = htk.File(path)
        resolutions = [f.resolution()]
        resolution_value = resolutions
    return f, resolutions, resolution_value


def pick_saved_callback(saved_string, update_plot_n_clicks):
    if saved_string is None:
        raise PreventUpdate
    filepath, resolution, scale_type, chrom_name, normalization = saved_string.split(";")
    return filepath, int(resolution), scale_type, chrom_name, normalization, update_plot_n_clicks + 1


def look_for_normalizations_under_current_resolution_callback(resolution, path):
    f = open_matrix_file_checked(path, resolution)
    avail_normalizations = f.avail_normalizations()
    avail_normalizations.append("No normalization")
    return avail_normalizations, avail_normalizations[0]


def populate_empty_normalization_list_callback(array):
    if not isinstance(array, list):
        return ["No normalization options available"], "Error"
    elif len(array) == 0:
        return ["No normalization options available"], "Error"
    else:
        return array, array[0]


def update_plot_callback(
    chromosome_name,
    colorMap,
    normalization,
    filepath,
    resolution,
    scale_type,
    files_list,
    stripes_filepath,
    last_used_path,
    last_used_resolution,
    last_used_scale_type,
    last_used_region,
    last_used_color_map,
    last_used_normalization,
    last_used_stripes,
):
    filepath = Path(filepath)
    try:
        if (
            filepath == Path(last_used_path)
            and resolution == last_used_resolution
            and last_used_region == chromosome_name
            and last_used_scale_type == scale_type
            and last_used_color_map == colorMap
            and last_used_normalization == normalization
            and last_used_stripes == stripes_filepath
        ):
            raise PreventUpdate
        else:
            pass
    except NameError:
        pass

    colorMap_code = color_scale(colorMap)
    # "No normalization" is stored in dropdown menu; "None" is stored in saved files.
    if normalization == "No normalization" or normalization == "None":
        normalization = None

    f = open_matrix_file_checked(filepath, resolution)

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

    access_data_string = stripes_filepath

    if chromosome_name:
        fig = go.Figure()
        fig.add_trace(
            go.Heatmap(
                z=frame,
                colorbar=colorbar(frame, scale_type),
                colorscale=colorMap_code,
                customdata=inv_log_frame_string,
                hovertemplate="%{customdata}<extra></extra>",
                hoverlabel={
                    "bgcolor": "green",
                },
                name="First matrix",
                xaxis="x1",
                yaxis="y1",
            )
        )

        tickvals, ticktext = compute_x_axis_range(chromosome_name, f, resolution)
        fig.update_xaxes(tickvals=tickvals, ticktext=ticktext, showgrid=False)
        fig.update_yaxes(autorange="reversed", showgrid=False)
        fig.update_layout(plot_bgcolor="mediumslateblue")
        # NaN-values are transparent
        traces_x_axis, traces_y_axis = "x1", "y1"
        if access_data_string:
            fig = add_stripes_chrom_restriction(
                f, fig, chromosome_name, access_data_string, resolution, (traces_x_axis, traces_y_axis)
            )
    else:
        fig = go.Figure()
        fig.add_trace(
            go.Heatmap(
                z=frame,
                colorbar=colorbar(frame, scale_type),
                colorscale=colorMap_code,
                name="First matrix",
                xaxis="x1",
                yaxis="y1",
            )
        )
        fig.add_trace(
            go.Heatmap(
                z=frame,
                colorbar=colorbar(frame, scale_type),
                colorscale=colorMap_code,
                customdata=inv_log_frame_string,
                hovertemplate="%{customdata}<extra></extra>",
                hoverlabel={
                    "bgcolor": "green",
                },
                name="Second matrix",
                xaxis="x2",
                yaxis="y2",
            )
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
        traces_x_axis, traces_y_axis = "x2", "y2"
        if access_data_string:
            fig = add_stripes_whole_chrom(f, fig, access_data_string, resolution, layers=(traces_x_axis, traces_y_axis))

    fig.layout.update(showlegend=False)

    filepath_assembled_string = f"{filepath};{resolution};{scale_type};{chromosome_name};{normalization}"
    try:
        if filepath_assembled_string not in [values for dicts in files_list for values in dicts.values()]:
            files_list.append(
                {
                    "label": f"res={resolution}, scaletype={scale_type}, norm={normalization}, region={chromosome_name if chromosome_name else "entire"}: {filepath.name}",
                    "value": f"{filepath};{resolution};{scale_type};{chromosome_name};{normalization}",
                }
            )
    except TypeError:
        files_list = [
            {
                "label": f"res={resolution}, scaletype={scale_type}, norm={normalization}, region={chromosome_name if chromosome_name else "entire"}: {filepath.name}",
                "value": f"{filepath};{resolution};{scale_type};{chromosome_name};{normalization}",
            }
        ]

    return (
        fig,
        files_list,
        False,
        str(filepath),
        resolution,
        scale_type,
        chromosome_name,
        colorMap,
        normalization,
        stripes_filepath,
    )


def open_hdf5_file_dialog_callback():
    root = Tk()
    root.filename = filedialog.askopenfilename(
        initialdir=".", title="Select file", filetypes=(("HDF5-files", "*.hdf5"), ("all files", "*.*"))
    )
    root.destroy()
    return root.filename


def call_stripes_callback(
    path,
    resolution,
    gen_belt,
    max_width,
    glob_pers_min,
    constrain_heights,
    k,
    loc_pers_min,
    loc_trend_min,
    # force,
    nproc,
    min_chrom_size,
    # verbosity,
    normalization,
    press_hidden_button,
):
    path = Path(path)
    filename = path.stem
    output_file = f"./tmp/{filename}/{resolution}/stripes.hdf5"
    try:
        with ProcessSafeLogger(
            level="info",  # verbosity
            path=Path(f"./tmp/{filename}/{resolution}/log_file"),
            force=True,
            matrix_file=Path(path),
            print_welcome_message=True,
            progress_bar_type="call",
        ) as main_logger:
            call.run(
                Path(path),
                resolution,
                Path(output_file),  # output file
                gen_belt,
                max_width,
                glob_pers_min,
                constrain_heights,
                k,  # k
                loc_pers_min,
                loc_trend_min,
                True,  # force
                nproc,
                min_chrom_size,
                "info",  # verbosity
                main_logger,  # main_logger,
                None,  # roi,
                Path(f"./tmp/{filename}/{resolution}/log_file"),  # log_file,
                Path(f"./tmp/{filename}/{resolution}/plot_dir"),  # plot_dir,
                normalization,
            )
    except FileExistsError as e:
        pass
    return str(output_file), press_hidden_button + 1
