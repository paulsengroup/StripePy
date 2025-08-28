import concurrent.futures
import contextlib
import pathlib
from pathlib import Path
from tkinter import *
from tkinter import filedialog

import hictkpy as htk
import numpy as np
import plotly.graph_objects as go
from colorscales import color_scale, contrast
from components.axes import compute_x_axis_chroms, compute_x_axis_range
from components.colorbar import colorbar
from components.dbc_warnings import (
    compose_stale_component_warning,
    warning_cancel,
    warning_no_stripes,
    warning_null,
    warning_pick_save_file,
    warning_stale_component,
)
from dash import dcc, html, no_update
from dash.exceptions import PreventUpdate
from plotly.subplots import make_subplots
from stripes import add_stripes_chrom_restriction, add_stripes_whole_chrom

from stripepy.algorithms import step1, step2, step3, step4
from stripepy.cli import call
from stripepy.cli.call import *
from stripepy.data_structures import IOManager, ProcessPoolWrapper, Result, Stripe
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
    if root.filename == "":
        return no_update, warning_cancel()
    return root.filename, warning_null()


def look_for_file_callback(file_path, metaInfo):
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
        warning_null(),
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
        return no_update, no_update, no_update, no_update, no_update, no_update, warning_pick_save_file()
    filepath, resolution, scale_type, chrom_name, normalization = saved_string.split(";")
    return filepath, int(resolution), scale_type, chrom_name, normalization, update_plot_n_clicks + 1, warning_null()


def look_for_normalizations_under_current_resolution_callback(resolution, path, current_normalization):
    f = open_matrix_file_checked(path, resolution)
    avail_normalizations = f.avail_normalizations()
    avail_normalizations.append("No normalization")
    if current_normalization in avail_normalizations:
        return avail_normalizations, no_update
    return avail_normalizations, "No normalization"


def update_plot_callback(
    chromosome_name,
    colorMap,
    normalization,
    filepath,
    resolution,
    scale_type,
    files_list,
    last_used_path,
    last_used_resolution,
    last_used_scale_type,
    last_used_region,
    last_used_color_map,
    last_used_normalization,
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
        ):
            return (
                no_update,
                no_update,
                no_update,
                no_update,
                no_update,
                no_update,
                no_update,
                no_update,
                no_update,
                warning_stale_component(
                    (
                        "file path",
                        "resolution",
                        "chromosome name",
                        "scale type",
                        "color map",
                        "normalization",
                        "stripes filepath",
                    )
                ),
            )
        else:
            pass
    except NameError:
        pass

    colorMap_code = color_scale(colorMap)
    # "No normalization" is stored in dropdown menu; "None" is stored in saved files.
    if normalization == "No normalization":
        normalization_parameter = None

    f = open_matrix_file_checked(filepath, resolution)

    sel = f.fetch(chromosome_name, normalization=normalization_parameter)
    frame = sel.to_numpy()
    frame = frame.astype(np.float32)
    to_string_vector = np.vectorize(str)
    inv_log_frame_string = to_string_vector(frame)

    filter_for_finite_and_positive = np.isfinite(frame) & (frame > 0)
    if scale_type == "log scale":
        if frame[filter_for_finite_and_positive].mean() < 1:  # Scale matrix if the mean value is less than 1
            scaling_product = 1 / np.min(frame[filter_for_finite_and_positive])
            frame[filter_for_finite_and_positive] *= scaling_product
        np.log(frame, out=frame, where=np.isnan(frame) == False)
    lowest_real_value = np.min(frame[filter_for_finite_and_positive])  # isfinite() dicounts nan, inf and -inf
    frame = np.where(np.isneginf(frame), lowest_real_value, frame)

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
                    "bgcolor": contrast(colorMap, "map"),
                },
                name="First matrix",
                xaxis="x1",
                yaxis="y1",
            )
        )

        tickvals, ticktext = compute_x_axis_range(chromosome_name, f, resolution)
        fig.update_xaxes(tickvals=tickvals, ticktext=ticktext, showgrid=False)
        fig.update_yaxes(tickvals=tickvals, ticktext=ticktext, autorange="reversed", showgrid=False)
        fig.update_layout(plot_bgcolor=contrast(colorMap, "background"))
        # NaN-values are transparent
        traces_x_axis, traces_y_axis = "x1", "y1"
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
                    "bgcolor": contrast(colorMap, "map"),
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
            yaxis=dict(tickvals=tickvals, ticktext=ticktext, autorange="reversed", showgrid=False, visible=True),
            yaxis2=dict(autorange="reversed", showgrid=False, visible=False, side="right"),
            plot_bgcolor=contrast(colorMap, "background"),
        )
        traces_x_axis, traces_y_axis = "x2", "y2"

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
        warning_null(),
    )


def call_stripes_callback(
    path,
    resolution,
    scale_type,
    chrom_name,
    color_map,
    normalization,
    gen_belt,
    max_width,
    glob_pers_min,
    constrain_heights,
    k,
    loc_pers_min,
    loc_trend_min,
    nproc,
    rel_change,
    last_used_path,
    last_used_resolution,
    last_used_scale_type,
    last_used_region,
    last_used_color_map,
    last_used_normalization,
    last_used_gen_belt,
    last_used_max_width,
    last_used_glob_pers_min,
    last_used_constrain_heights,
    last_used_k,
    last_used_loc_pers_min,
    last_used_loc_trend_min,
    last_used_nproc,
    last_used_rel_change,
    fig,
    result_chrom_name,
    result_chrom_size,
    result_min_persistence,
    result_ut_pseudodistribution,
    result_lt_pseudodistribution,
    result_ut_all_minimum_points,
    result_lt_all_minimum_points,
    result_ut_all_maximum_points,
    result_lt_all_maximum_points,
    result_ut_persistence_of_all_minimum_points,
    result_lt_persistence_of_all_minimum_points,
    result_ut_persistence_of_all_maximum_points,
    result_lt_persistence_of_all_maximum_points,
    result_ut_persistent_minimum_points,
    result_lt_persistent_minimum_points,
    result_ut_persistent_maximum_points,
    result_lt_persistent_maximum_points,
    result_ut_persistence_of_minimum_points,
    result_lt_persistence_of_minimum_points,
    result_ut_persistence_of_maximum_points,
    result_lt_persistence_of_maximum_points,
    result_ut_stripes,
    result_lt_stripes,
):
    if normalization == "No normalization" or normalization == "None":
        normalization = None
    min_chrom_size = 1
    path = Path(path)
    if not isinstance(fig, go.Figure):
        fig = go.Figure(fig)
    raw_plot = [trace for trace in fig["data"] if type(trace) == go.Heatmap]
    fig["data"] = tuple(raw_plot)
    f = open_matrix_file_checked(path, resolution)
    chroms = f.chromosomes(include_ALL=False)
    functions_sequence = _where_to_start_calling_sequence(
        (
            str(path),
            resolution,
            scale_type,
            chrom_name,
            color_map,
            normalization,
            gen_belt,
            nproc,
            glob_pers_min,
            max_width,
            loc_trend_min,
            k,
            rel_change,
            loc_pers_min,
            constrain_heights,
        ),
        (
            last_used_path,
            last_used_resolution,
            last_used_scale_type,
            last_used_region,
            last_used_color_map,
            last_used_normalization,
            last_used_gen_belt,
            last_used_nproc,
            last_used_glob_pers_min,
            last_used_max_width,
            last_used_loc_trend_min,
            last_used_k,
            last_used_rel_change,
            last_used_loc_pers_min,
            last_used_constrain_heights,
        ),
    )
    if not functions_sequence:
        return (
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            warning_stale_component(
                (
                    "file path",
                    "resolution",
                    "scale type",
                    "chromosome name",
                    "color map",
                    "normalization",
                    "genomic belt",
                    "max width",
                    "global minimum persistence",
                    "constrain heights",
                    "k neighbours",
                    "local minimal persistence",
                    "local trend minimum",
                    "number of processors",
                    "relative signal change",
                )
            ),
            *[no_update] * 23,
        )
    result_package = [
        result_chrom_name,
        result_chrom_size,
        result_min_persistence,
        result_ut_pseudodistribution,
        result_lt_pseudodistribution,
        result_ut_all_minimum_points,
        result_lt_all_minimum_points,
        result_ut_all_maximum_points,
        result_lt_all_maximum_points,
        result_ut_persistence_of_all_minimum_points,
        result_lt_persistence_of_all_minimum_points,
        result_ut_persistence_of_all_maximum_points,
        result_lt_persistence_of_all_maximum_points,
        result_ut_persistent_minimum_points,
        result_lt_persistent_minimum_points,
        result_ut_persistent_maximum_points,
        result_lt_persistent_maximum_points,
        result_ut_persistence_of_minimum_points,
        result_lt_persistence_of_minimum_points,
        result_ut_persistence_of_maximum_points,
        result_lt_persistence_of_maximum_points,
        result_ut_stripes,
        result_lt_stripes,
    ]
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
    with contextlib.ExitStack() as ctx:
        # Set up logger for the process pool
        main_logger = ctx.enter_context(ProcessSafeLogger("warning", path=None, progress_bar_type="call"))
        logger = structlog.get_logger().bind(step="main")
        # Set up the pool of worker processes
        pool = ctx.enter_context(
            ProcessPoolWrapper(
                nproc=nproc,
                main_logger=main_logger,
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
        FOUND_STRIPES = False
        for i, (chromosome_name, chrom_size, skip) in enumerate(tasks):
            if function_scope == "SINGLE_CHROM":
                subtract_from_start = True
            else:
                subtract_from_start = False
            if skip:
                continue
            for j, function in enumerate(functions_sequence):
                if isinstance(function, bool):
                    break
                if j == 0:
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
                if function == call._run_step_2:
                    print("Running step 2")
                    result = function(
                        chromosome_name,
                        chrom_size,
                        lt_matrix,
                        ut_matrix,
                        glob_pers_min,
                        pool,
                        logger.bind(chrom=chromosome_name),
                    )
                if function == call._run_step_3:
                    print("Running step 3")
                    if j == 0:
                        print("Not first call, composing result")
                        result = _compose_result(result_package, function)
                    result = function(
                        result,
                        lt_matrix,
                        ut_matrix,
                        resolution,
                        gen_belt,
                        max_width,
                        loc_pers_min,
                        loc_trend_min,
                        tpool,
                        pool,
                        logger.bind(chrom=chromosome_name),
                    )
                if function == call._run_step_4:
                    print("Running step 4")
                    if j == 0:
                        print("Not first call, composing result")
                        result = _compose_result(result_package, function)
                    result = function(
                        result,
                        lt_matrix,
                        ut_matrix,
                        k,
                        tpool,
                        pool,
                        logger.bind(chrom=chromosome_name),
                    )

                    #####
                    ### Add stripes
                    #####
                    if not result.empty:
                        FOUND_STRIPES = True
                    if function_scope == "START_AND_END_SEGMENT":
                        fig = add_stripes_chrom_restriction(
                            f,
                            fig,
                            chrom_name,
                            result,
                            resolution,
                            (traces_x_axis, traces_y_axis),
                            color_map,
                            rel_change,
                        )
                    elif function_scope == "END_SEGMENT_ONLY":
                        fig = add_stripes_chrom_restriction_at_end(
                            f, fig, chrom_name, result, resolution, (traces_x_axis, traces_y_axis), color_map
                        )
                    elif function_scope == "SINGLE_CHROM" or function_scope == "WHOLE_GENOME":
                        fig = add_stripes_whole_chrom(
                            f,
                            fig,
                            result,
                            resolution,
                            (traces_x_axis, traces_y_axis),
                            chromosome_name,
                            color_map,
                            subtract_from_start,
                            rel_change,
                        )
    ####
    #### Add stripes as traces
    ####
    if not FOUND_STRIPES:
        return (
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            warning_no_stripes(),
            *_unpack_result(result),
        )
    else:
        return (
            str(path),
            resolution,
            scale_type,
            chrom_name,
            color_map,
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
            warning_null(),
            *_unpack_result(result),
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
    functions_list = [step1.run, call._run_step_2, call._run_step_3, call._run_step_4]
    for index, input_ in enumerate(input_params):
        if input_ != state_params[index]:
            if (
                index <= 7
            ):  # path, resolution, log/lin scale, chromosome region, color mapping, normalization, genomic belt, nproc
                return (*functions_list, True)
            if index == 8:  # global persistence minimum
                return (*functions_list[1:], True)
            if index <= 10:  # max width, local trend minimum
                return (*functions_list[2:], False)
            if index <= 12:  # k neighbours, relative change
                return (*functions_list[3:], False)
            if index <= 14:  # local minimum persistence, constrain heights
                return False
    return False


def _compose_result(result_package, starting_point):
    attributes_list = [
        "pseudodistribution",
        "all_minimum_points",
        "all_maximum_points",
        "persistence_of_all_minimum_points",
        "persistence_of_all_maximum_points",
        "persistent_minimum_points",
        "persistent_maximum_points",
        "persistence_of_minimum_points",
        "persistence_of_maximum_points",
    ]
    result = Result(result_package.pop(0), result_package.pop(0))
    result.set_min_persistence(result_package.pop(0))
    for attribute in attributes_list:
        result.set(attribute, np.array(result_package.pop(0)), "upper")
        result.set(attribute, np.array(result_package.pop(0)), "lower")
    upper_stripes_list = result_package.pop(0)
    lower_stripes_list = result_package.pop(0)
    if starting_point == call._run_step_3:  # Propagate the data collected in step 2
        upper_stripes = [
            Stripe(seed, top_pers=pers, where="upper_triangular") for seed, pers, _, _, _, _ in upper_stripes_list
        ]
        lower_stripes = [
            Stripe(seed, top_pers=pers, where="lower_triangular") for seed, pers, _, _, _, _ in lower_stripes_list
        ]
    elif starting_point == call._run_step_4:  # Propagate the data collected in step 3
        upper_stripes = [
            Stripe(seed, top_pers=pers, horizontal_bounds=(lb, rb), vertical_bounds=(tb, bb), where="upper_triangular")
            for seed, pers, lb, rb, tb, bb in upper_stripes_list
        ]
        lower_stripes = [
            Stripe(seed, top_pers=pers, horizontal_bounds=(lb, rb), vertical_bounds=(tb, bb), where="lower_triangular")
            for seed, pers, lb, rb, tb, bb in lower_stripes_list
        ]
    result.set("stripes", upper_stripes, "upper")
    result.set("stripes", lower_stripes, "lower")
    return result


def _unpack_result(result):
    chrom_name, chrom_size = result.chrom
    up_pse = result.get("pseudodistribution", "upper")
    lt_pse = result.get("pseudodistribution", "lower")
    up_all_min = result.get("all_minimum_points", "upper")
    lt_all_min = result.get("all_minimum_points", "lower")
    up_all_max = result.get("all_maximum_points", "upper")
    lt_all_max = result.get("all_maximum_points", "lower")
    up_pers_all_min = result.get("persistence_of_all_minimum_points", "upper")
    lt_pers_all_min = result.get("persistence_of_all_minimum_points", "lower")
    up_pers_all_max = result.get("persistence_of_all_maximum_points", "upper")
    lt_pers_all_max = result.get("persistence_of_all_maximum_points", "lower")
    up_pers_min = result.get("persistent_minimum_points", "upper")
    lt_pers_min = result.get("persistent_minimum_points", "lower")
    up_pers_max = result.get("persistent_maximum_points", "upper")
    lt_pers_max = result.get("persistent_maximum_points", "lower")
    up_pers_of_min = result.get("persistence_of_minimum_points", "upper")
    lt_pers_of_min = result.get("persistence_of_minimum_points", "lower")
    up_pers_of_max = result.get("persistence_of_maximum_points", "upper")
    lt_pers_of_max = result.get("persistence_of_maximum_points", "lower")
    up_stripes = result.get("stripes", "upper")
    lt_stripes = result.get("stripes", "lower")
    return (
        chrom_name,
        chrom_size,
        result.min_persistence,
        _make_into_string(up_pse.tolist()),
        _make_into_string(lt_pse.tolist()),
        _make_into_string(up_all_min.tolist()),
        _make_into_string(lt_all_min.tolist()),
        _make_into_string(up_all_max.tolist()),
        _make_into_string(lt_all_max.tolist()),
        _make_into_string(up_pers_all_min.tolist()),
        _make_into_string(lt_pers_all_min.tolist()),
        _make_into_string(up_pers_all_max.tolist()),
        _make_into_string(lt_pers_all_max.tolist()),
        _make_into_string(up_pers_min.tolist()),
        _make_into_string(lt_pers_min.tolist()),
        _make_into_string(up_pers_max.tolist()),
        _make_into_string(lt_pers_max.tolist()),
        _make_into_string(up_pers_of_min.tolist()),
        _make_into_string(lt_pers_of_min.tolist()),
        _make_into_string(up_pers_of_max.tolist()),
        _make_into_string(lt_pers_of_max.tolist()),
        _make_stripes_into_string(up_stripes.tolist()),
        _make_stripes_into_string(lt_stripes.tolist()),
    )


def _make_into_string(array):
    """
    Convert a numpy array into a string representation
    """
    list_stored_string = ""
    while array:
        list_stored_string += str(array.pop(0)) + ";"
    return list_stored_string[:-1]  # Remove the last semicolon


def _make_stripes_into_string(array):
    """
    Convert a numpy array of stripes into a string representation
    """
    list_stored_string = ""
    for stripe in array:
        list_stored_string += f"{stripe.seed}:{stripe.top_persistence}:{stripe.left_bound}:{stripe.right_bound}:{stripe.top_bound}:{stripe.bottom_bound};"
    return list_stored_string[:-1]  # Remove the last semicolon
