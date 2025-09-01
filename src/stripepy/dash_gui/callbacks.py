import contextlib
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
    warning_cancel,
    warning_no_stripes,
    warning_null,
    warning_pick_save_file,
    warning_stale_component,
)
from dash import html, no_update
from stripes import (
    add_stripes,
    add_stripes_rel_change_filter,
    add_stripes_visualisation_change,
)

from stripepy.algorithms import step1
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


def look_for_file_callback(file_path):
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
        *[False] * 9,
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
    chromosome_region,
    colorMap,
    normalization,
    filepath,
    resolution,
    scale_type,
    files_list,
    rel_change,
    last_used_path,
    last_used_resolution,
    last_used_scale_type,
    last_used_region,
    last_used_color_map,
    last_used_normalization,
    fig,
    existing_ut_stripes,
    existing_lt_stripes,
):
    KEEP_STRIPES = False
    DRAW_STRIPES = False
    filepath = Path(filepath)
    if (
        filepath == Path(last_used_path)
        and resolution == last_used_resolution
        and last_used_normalization == normalization
    ):
        if (
            last_used_region == chromosome_region
            and last_used_scale_type == scale_type
            and last_used_color_map == colorMap
        ):  # Give warning that nothing changed
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
                    )
                ),
            )
        elif colorMap != last_used_color_map:
            DRAW_STRIPES = True
        elif chromosome_region.partition(":")[0] == last_used_region.partition(":")[0]:
            if (
                chromosome_region.partition(":")[2] != last_used_region.partition(":")[2]
            ):  # Same chromosome, but different region; redraw stripes
                DRAW_STRIPES = True
            else:  # Only scale type or color map changed; keep stripes
                KEEP_STRIPES = True
        else:  # Different chromosome; delete stripes
            KEEP_STRIPES = False
    else:  # Significant change; redraw heatmap
        KEEP_STRIPES = False

    colorMap_code = color_scale(colorMap)
    f = open_matrix_file_checked(filepath, resolution)
    sel = f.fetch(chromosome_region, normalization=None if normalization == "No normalization" else normalization)
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

    if chromosome_region:  # The plot is either a chromosome or a part of one
        if KEEP_STRIPES:  # fig["data"] is an immutable data structure, so the object is edited destructively
            new_trace = [
                go.Heatmap(
                    z=frame,
                    colorbar=colorbar(frame, scale_type),
                    colorscale=colorMap_code,
                    customdata=inv_log_frame_string,
                    hovertemplate="%{customdata}<extra></extra>",
                    hoverlabel={
                        "bgcolor": contrast(colorMap, "label"),
                    },
                    name="First matrix",
                    xaxis="x1",
                    yaxis="y1",
                )
            ]
            new_trace += list(fig["data"][1:])
            fig = go.Figure(data=tuple(new_trace))
        else:
            fig = go.Figure()
            fig.add_trace(
                go.Heatmap(
                    z=frame,
                    colorbar=colorbar(frame, scale_type),
                    colorscale=colorMap_code,
                    customdata=inv_log_frame_string,
                    hovertemplate="%{customdata}<extra></extra>",
                    hoverlabel={
                        "bgcolor": contrast(colorMap, "label"),
                    },
                    name="First matrix",
                    xaxis="x1",
                    yaxis="y1",
                )
            )

        tickvals, ticktext = compute_x_axis_range(chromosome_region, f, resolution)
        fig.update_xaxes(tickvals=tickvals, ticktext=ticktext, showgrid=False)
        fig.update_yaxes(tickvals=tickvals, ticktext=ticktext, autorange="reversed", showgrid=False)
        fig.update_layout(plot_bgcolor=contrast(colorMap, "background"))
        # NaN-values are transparent
        traces_x_axis, traces_y_axis = "x1", "y1"
    else:
        if KEEP_STRIPES:
            # fig["data"] is an immutable data structure, so the object is re-created
            new_trace = [
                go.Heatmap(
                    z=frame,
                    colorbar=colorbar(frame, scale_type),
                    colorscale=colorMap_code,
                    name="First matrix",
                    xaxis="x1",
                    yaxis="y1",
                ),
                go.Heatmap(
                    z=frame,
                    colorbar=colorbar(frame, scale_type),
                    colorscale=colorMap_code,
                    customdata=inv_log_frame_string,
                    hovertemplate="%{customdata}<extra></extra>",
                    hoverlabel={
                        "bgcolor": contrast(colorMap, "label"),
                    },
                    name="Second matrix",
                    xaxis="x2",
                    yaxis="y2",
                ),
            ]
            new_trace += list(fig["data"][2:])
            fig = go.Figure(data=tuple(new_trace))
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
                        "bgcolor": contrast(colorMap, "label"),
                    },
                    name="Second matrix",
                    xaxis="x2",
                    yaxis="y2",
                )
            )

        tickvals, ticktext = compute_x_axis_range(chromosome_region, f, resolution)
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

    if DRAW_STRIPES and existing_lt_stripes and existing_ut_stripes:
        fig = add_stripes_visualisation_change(
            fig, existing_lt_stripes, resolution, colorMap, chromosome_region, rel_change, traces_x_axis, traces_y_axis
        )
        fig = add_stripes_visualisation_change(
            fig, existing_ut_stripes, resolution, colorMap, chromosome_region, rel_change, traces_x_axis, traces_y_axis
        )

    filepath_assembled_string = f"{filepath};{resolution};{scale_type};{chromosome_region};{normalization}"
    try:
        if filepath_assembled_string not in [values for dicts in files_list for values in dicts.values()]:
            files_list.append(
                {
                    "label": f"res={resolution}, scaletype={scale_type}, norm={normalization}, region={chromosome_region if chromosome_region else "entire"}: {filepath.name}",
                    "value": f"{filepath};{resolution};{scale_type};{chromosome_region};{normalization}",
                }
            )
    except TypeError:
        files_list = [
            {
                "label": f"res={resolution}, scaletype={scale_type}, norm={normalization}, region={chromosome_region if chromosome_region else "entire"}: {filepath.name}",
                "value": f"{filepath};{resolution};{scale_type};{chromosome_region};{normalization}",
            }
        ]

    return (
        fig,
        files_list,
        False,
        str(filepath),
        resolution,
        scale_type,
        chromosome_region,
        colorMap,
        normalization,
        warning_null(),
    )


def call_stripes_callback(
    path,
    resolution,
    region,
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
    fig,
    result_region,
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
    from_where_to_call,
    traces,
    chromosome_name,
    margin,
    end_limit,
    restriction_scope,
):
    min_chrom_size = 1
    f = open_matrix_file_checked(path, resolution)
    chroms = f.chromosomes(include_ALL=False)
    functions_sequence = _where_to_start_calling_sequence(from_where_to_call)
    result_package = [
        result_region,
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
    traces_x_axis, traces_y_axis = traces
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
        if restriction_scope == "whole genome":
            tasks = call._plan_tasks(chroms, min_chrom_size, None)  # logger set to None for the time being
        else:
            tasks = call._plan_tasks({chromosome_name: chroms[chromosome_name]}, min_chrom_size, None)
        FOUND_STRIPES = False
        for i, (chromosome_name, chrom_size, skip) in enumerate(tasks):
            if skip:
                continue
            for j, function in enumerate(functions_sequence):
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
            if restriction_scope == "chromosome restriction":
                fig = add_stripes(
                    f, fig, result, resolution, traces, color_map, chromosome_name, rel_change, margin, end_limit, False
                )
            else:
                fig = add_stripes(f, fig, result, resolution, traces, color_map, chromosome_name, rel_change)
    ####
    #### Add stripes as traces
    ####
    if not FOUND_STRIPES:
        return (
            *[no_update] * 14,
            warning_no_stripes(),
            *_unpack_result(result),
        )
    else:
        return (
            str(path),
            resolution,
            region,
            normalization,
            str(gen_belt),
            str(max_width),
            str(glob_pers_min),
            str(constrain_heights),
            str(k),
            str(loc_pers_min),
            str(loc_trend_min),
            str(nproc),
            str(rel_change),
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


def _where_to_start_calling_sequence(from_where_to_call):
    functions_list = [step1.run, call._run_step_2, call._run_step_3, call._run_step_4]
    if from_where_to_call == "Step 2":
        return (*functions_list[1:],)
    if from_where_to_call == "Step 3":
        return (*functions_list[2:],)
    if from_where_to_call == "Step 4":
        return (*functions_list[3:],)


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
    if starting_point == call._run_step_3:  # Propagate the data collected in step 2
        upper_stripes_list = result_package.pop(0)[:2]
        lower_stripes_list = result_package.pop(0)[:2]
        upper_stripes = [Stripe(seed, top_pers=pers, where="upper_triangular") for seed, pers in upper_stripes_list]
        lower_stripes = [Stripe(seed, top_pers=pers, where="lower_triangular") for seed, pers in lower_stripes_list]
    elif starting_point == call._run_step_4:  # Propagate the data collected in step 3
        upper_stripes_list = result_package.pop(0)[:6]
        lower_stripes_list = result_package.pop(0)[:6]
        upper_stripes = []
        for stripe_string in upper_stripes_list:
            seed, pers, lb, rb, tb, bb = stripe_string
            new_stripe = Stripe(seed, top_pers=pers, where="upper_triangular")
            new_stripe.set_horizontal_bounds(lb, rb)
            new_stripe.set_vertical_bounds(tb, bb)
            upper_stripes.append(new_stripe)

        lower_stripes = []
        for stripe_string in lower_stripes_list:
            seed, pers, lb, rb, tb, bb = stripe_string
            new_stripe = Stripe(seed, top_pers=pers, where="lower_triangular")
            new_stripe.set_horizontal_bounds(lb, rb)
            new_stripe.set_vertical_bounds(tb, bb)
            lower_stripes.append(new_stripe)
    elif starting_point == "After":
        upper_stripes_list = result_package.pop(0)
        lower_stripes_list = result_package.pop(0)
        return upper_stripes_list, lower_stripes_list
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
        list_stored_string += f"{stripe.seed}:{stripe.top_persistence}:{stripe.left_bound}:{stripe.right_bound}:{stripe.top_bound}:{stripe.bottom_bound}:{stripe.rel_change};"
    return list_stored_string[:-1]  # Remove the last semicolon


def filter_stripes_callback(fig, resolution, colorMap, rel_change, traces, margin, end_limit, ut_stripes, lt_stripes):
    fig = add_stripes_rel_change_filter(fig, ut_stripes, resolution, colorMap, rel_change, traces, margin, end_limit)
    fig = add_stripes_rel_change_filter(fig, lt_stripes, resolution, colorMap, rel_change, traces, margin, end_limit)
    return (*[no_update] * 12, rel_change, fig, warning_null(), *[no_update] * 23)
