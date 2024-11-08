import numpy as np

from .persistence1d import (
    diversify_extremum_points_and_persistence,
    filter_extremum_points_by_persistence,
    run_persistence,
)


def TDA(marginal_pd, min_persistence=0):

    # Compute the extremum points (i.e., minimum and maximum points) and their persistence:
    extremum_points_and_persistence = run_persistence(marginal_pd, level_sets="upper")

    # Filter extremum points by persistence threshold:
    filtered_extremum_points_and_persistence = filter_extremum_points_by_persistence(
        extremum_points_and_persistence, threshold=min_persistence
    )

    # Split extremum points into minimum and maximum points:
    (filtered_min_points_and_persistence, filtered_max_points_and_persistence) = (
        diversify_extremum_points_and_persistence(filtered_extremum_points_and_persistence, level_set="upper")
    )

    # Sorting maximum points (and, as a consequence, the corresponding minimum points) w.r.t. persistence:
    argsorting = np.argsort(list(zip(*filtered_max_points_and_persistence))[1]).tolist()

    if len(filtered_min_points_and_persistence) == 0:
        filtered_min_points = []
        persistence_of_filtered_min_points = []
    else:
        filtered_min_points = np.array(list(zip(*filtered_min_points_and_persistence))[0])[argsorting[:-1]].tolist()
        persistence_of_filtered_min_points = np.array(list(zip(*filtered_min_points_and_persistence))[1])[
            argsorting[:-1]
        ].tolist()

    # Indices of maximum points and their persistence:
    filtered_max_points = np.array(list(zip(*filtered_max_points_and_persistence))[0])[argsorting].tolist()
    persistence_of_filtered_max_points = np.array(list(zip(*filtered_max_points_and_persistence))[1])[
        argsorting
    ].tolist()

    return (
        filtered_min_points,
        persistence_of_filtered_min_points,
        filtered_max_points,
        persistence_of_filtered_max_points,
    )
