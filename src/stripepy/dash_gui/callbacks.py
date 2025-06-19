import concurrent.futures
import contextlib
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

from stripepy.algorithms import step1, step2, step3, step4
from stripepy.cli import call
from stripepy.cli.call import *
from stripepy.data_structures import IOManager, ProcessPoolWrapper
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
    chrom_name,
    gen_belt,
    max_width,
    glob_pers_min,
    constrain_heights,
    k,
    loc_pers_min,
    loc_trend_min,
    nproc,
    min_chrom_size,
    normalization,
    last_used_normalization,
    last_used_gen_belt,
    last_used_max_width,
    last_used_glob_pers_min,
    last_used_constrain_heights,
    last_used_k,
    last_used_loc_pers_min,
    last_used_loc_trend_min,
    last_used_nproc,
    fig,
):
    if not isinstance(fig, go.Figure):
        fig = go.Figure(fig)
    f = open_matrix_file_checked(path, resolution)
    chroms = f.chromosomes(include_ALL=False)
    path = Path(path)
    filename = path.stem
    output_file = f"./tmp/{filename}/{resolution}/stripes.hdf5"

    functions_sequence = _where_to_start_calling_sequence(
        (normalization, gen_belt, glob_pers_min, max_width, loc_trend_min, k),
        (
            last_used_normalization,
            last_used_gen_belt,
            last_used_glob_pers_min,
            last_used_max_width,
            last_used_loc_trend_min,
            last_used_k,
        ),
    )
    chrom, _, region = chrom_name.partition(":")
    start_segment, _, end_segment = region.partition("-")
    function_scope = "NONE"
    if start_segment and end_segment:
        function_scope = "START_AND_END_SEGMENT"
        traces_x_axis, traces_y_axis = "x1", "y1"
    elif start_segment:
        function_scope = "END_SEGMENT_ONLY"
        traces_x_axis, traces_y_axis = "x1", "y1"
    elif chrom:
        function_scope = "SINGLE_CHROM"
        traces_x_axis, traces_y_axis = "x1", "y1"
    elif not chrom and not region:
        function_scope = "WHOLE_GENOME"
        traces_x_axis, traces_y_axis = "x2", "y2"
    if functions_sequence[-1] and function_scope != "WHOLE_GENOME":
        function_scope = "SINGLE_CHROM"
        traces_x_axis, traces_y_axis = "x1", "y1"
    with contextlib.ExitStack() as ctx:
        # Set up the pool of worker processes
        pool = ctx.enter_context(
            ProcessPoolWrapper(
                nproc=nproc,
                main_logger=None,
                init_mpl=False,  # roi is not None,
                lazy_pool_initialization=True,
                logger=None,
            )
        )

        # Set up the pool of worker threads
        tpool = ctx.enter_context(
            concurrent.futures.ThreadPoolExecutor(max_workers=min(nproc, 2)),
        )
        if (
            function_scope == "START_AND_END_SEGMENT"
            or function_scope == "END_SEGMENT_ONLY"
            or function_scope == "SINGLE_CHROM"
        ):
            tasks = call._plan_tasks({chrom: chroms[chrom]}, min_chrom_size, None)
        else:
            tasks = call._plan_tasks(chroms, min_chrom_size, None)  # logger set to None for the time being
        for i, (chromosome_name, chrom_size, skip) in enumerate(tasks):
            if skip:
                continue
            for function in functions_sequence:
                if isinstance(function, bool):
                    break
                if function == step1.run:
                    print("Running step 1")
                    if pool.ready:
                        # Signal that matrices should be fetched from the shared global state
                        lt_matrix = None
                        ut_matrix = None
                    else:
                        ut_matrix = _fetch_interactions(
                            i,
                            tasks,
                            pool,
                            path,
                            normalization,
                            chroms,
                            resolution,
                            gen_belt,
                        )
                        lt_matrix = ut_matrix.T
                if function == call._run_step_2_helper:
                    print("Running step 2")
                    params = (
                        (chromosome_name, chrom_size, lt_matrix, glob_pers_min, "lower"),
                        (chromosome_name, chrom_size, ut_matrix, glob_pers_min, "upper"),
                    )
                    tasks_ = pool.map(function, params)
                    result = call._merge_results(tasks_)
                if function == call._run_step_3_helper:
                    print("Running step 3")
                    if pool.ready:
                        executor = pool.get_mapper(chunksize=50)
                    else:
                        executor = pool.map
                    params = (
                        (
                            result,
                            lt_matrix,
                            resolution,
                            gen_belt,
                            max_width,
                            loc_pers_min,
                            loc_trend_min,
                            "lower",
                            executor,
                            None,
                        ),
                        (
                            result,
                            ut_matrix,
                            resolution,
                            gen_belt,
                            max_width,
                            loc_pers_min,
                            loc_trend_min,
                            "upper",
                            executor,
                            None,
                        ),
                    )
                    tasks_ = pool.map(function, params)
                    result = call._merge_results(tasks_)
                if function == call._run_step_4_helper:
                    print("Running step 4")
                    if pool.ready:
                        executor = pool.get_mapper(chunksize=50)
                    else:
                        executor = pool.map
                    params = (
                        (result.get("stripes", "lower"), lt_matrix, "lower", k, executor, None),
                        (result.get("stripes", "upper"), ut_matrix, "upper", k, executor, None),
                    )
                    (_, lt_stripes), (_, ut_stripes) = list(tpool.map(function, params))
                    result.set("stripes", lt_stripes, "LT", force=True)
                    result.set("stripes", ut_stripes, "UT", force=True)
    ####
    #### Add stripes as traces
    ####
    if result.empty:
        raise PreventUpdate  # TODO: add message telling user no stripes were found
    if function_scope == "START_AND_END_SEGMENT":
        fig = add_stripes_chrom_restriction(f, fig, chrom_name, result, resolution, (traces_x_axis, traces_y_axis))
    elif function_scope == "END_SEGMENT_ONLY":
        fig = add_stripes_chrom_restriction_at_end(
            f, fig, chrom_name, result, resolution, (traces_x_axis, traces_y_axis)
        )
    elif function_scope == "SINGLE_CHROM":
        fig = add_stripes_whole_chrom(f, fig, result, resolution, (traces_x_axis, traces_y_axis), chrom)
    elif function_scope == "WHOLE_GENOME":
        for chrom in chroms:
            fig = add_stripes_whole_chrom(f, fig, result, resolution, (traces_x_axis, traces_y_axis), chrom)

    return (
        normalization,
        str(gen_belt),
        str(max_width),
        str(glob_pers_min),
        str(constrain_heights),
        str(k),
        str(loc_pers_min),
        str(loc_trend_min),
        str(nproc),
        fig,
    )


def _fetch_interactions(
    i,
    tasks,
    pool,
    path,
    normalization,
    chroms,
    resolution,
    gen_belt,
):
    chrom_name, _, _ = tasks[i]
    ut_matrix, roi_matrix_raw, roi_matrix_proc = IOManager._fetch(
        path, resolution, normalization, gen_belt, chrom_name, None
    )
    if i == 0:
        max_nnz = call._estimate_max_nnz(chrom_name, ut_matrix, chroms)
        pool.rebind_shared_matrices(chrom_name, ut_matrix, None, max_nnz)
    else:
        pool.rebind_shared_matrices(chrom_name, ut_matrix, None)
    return ut_matrix


def _where_to_start_calling_sequence(input_params, state_params):
    functions_list = [step1.run, call._run_step_2_helper, call._run_step_3_helper, call._run_step_4_helper]
    for index, input_ in enumerate(input_params):
        if input_ != state_params[index]:
            if index == 0:  # normalization
                return (*functions_list, True)
            if index == 1:  # genomic belt
                return (*functions_list, True)
            if index == 2:  # global persistence minimum
                return (*functions_list[1:], True)
            if index == 3:  # max width
                return (*functions_list[2:], False)
            if index == 4:  # local trend minimum
                return (*functions_list[2:], False)
            if index == 5:  # k neighbours
                return (*functions_list[3:], False)
    return False
