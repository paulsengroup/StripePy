import numpy as np

from .unionfind import UnionFind

# Implementation adapted from the library by Tino Weinkauf downloadable at:
# https://www.csc.kth.se/~weinkauf/notes/persistence1d.html

# ATT: in its original implementation, the terms minima/maxima/extrema were sometimes used in place of
# minimum/maximum/extremum points. This notation is here fixed.
# For more details, consult: https://en.wikipedia.org/wiki/Maximum_and_minimum


def run_persistence(data, level_sets="lower"):
    """
    This function finds local extrema and their persistence in one-dimensional data w.r.t. lower (default,
    level_sets="lower") or upper (level_sets="upper") level sets.

    Local minima and local maxima are extracted, paired, and returned together with their persistence.
    For level_sets="lower", the global minimum is extracted as well.
    For level_sets="upper", the global maximum is extracted as well.

    We assume a connected one-dimensional domain.

    Short explanation for the case of level_sets=="lower" (the case of level_sets="upper" is analogous).

    This function returns a list of extrema together with their persistence. The list is NOT sorted, but the paired
    extrema can be identified, i.e., which minimum and maximum were removed together at a particular persistence
    level. As follows:
    (*)  Odd entries are minima, even entries are maxima.
    (*)  The minimum at 2*i is paired with the maximum at 2*i+1.
    (*)  The last entry of the list is the global minimum (resp. maximum) when level_sets="lower" (resp.
         level_sets="upper"). It is not paired with a maximum (resp. minimum).
    Hence, the list has an odd number of entries.

    Authors: Tino Weinkauf (original implementation) and Andrea Raffo (modified implementation)
    """

    # Number of data to break ties (leftmost index comes first):
    num_elements = len(data)
    sorted_idx = np.argsort(data, kind="stable")[::-1] if level_sets == "upper" else np.argsort(data, kind="stable")

    # Get a union find data structure:
    uf = UnionFind(num_elements)

    # Extrema paired with topological persistence:
    extremum_points_and_persistence = []

    # Watershed:
    for idx in sorted_idx:

        # Get neighborhood indices:
        left_idx = max(idx - 1, 0)
        right_idx = min(idx + 1, num_elements - 1)

        # Count number of components in neighborhood:
        neighbor_components = [
            uf.Find(neighbor) for neighbor in [left_idx, right_idx] if uf.Find(neighbor) != UnionFind.NOSET
        ]
        num_neighbor_components = len(neighbor_components)

        if num_neighbor_components == 0:
            # Create a new component:
            uf.MakeSet(idx)
        elif num_neighbor_components == 1:
            # Extend the one and only component in the neighborhood
            # Note that NeighborComponents[0] holds the root of a component, since we called Find() earlier to retrieve
            # it!
            uf.ExtendSetByID(neighbor_components[0], idx)
        else:

            if level_sets == "lower":
                # Merge the two components on either side of the current point:
                idx_lowest_minimum = neighbor_components[np.argmin(data[neighbor_components])]
                idx_highest_minimum = [comp for comp in neighbor_components if comp != idx_lowest_minimum][0]
                uf.ExtendSetByID(idx_lowest_minimum, idx)
                uf.Union(idx_highest_minimum, idx_lowest_minimum)

                # Record the two paired extrema: index of minimum, index of maximum, persistence value:
                persistence = data[idx] - data[idx_highest_minimum]
                extremum_points_and_persistence.append((idx_highest_minimum, persistence))
                extremum_points_and_persistence.append((idx, persistence))

            elif level_sets == "upper":

                # Merge the two components on either side of the current point:
                idx_highest_maximum = neighbor_components[np.argmax(data[neighbor_components])]
                idx_lowest_maximum = [comp for comp in neighbor_components if comp != idx_highest_maximum][0]
                uf.ExtendSetByID(idx_highest_maximum, idx)
                uf.Union(idx_lowest_maximum, idx_highest_maximum)

                # Record the two paired extrema: index of minimum, index of maximum, persistence value:
                persistence = data[idx_lowest_maximum] - data[idx]
                extremum_points_and_persistence.append((idx, persistence))
                extremum_points_and_persistence.append((idx_lowest_maximum, persistence))

    # Global minimum (or maximum):
    if level_sets == "lower":
        extremum_points_and_persistence.append((uf.Find(0), np.inf))
    elif level_sets == "upper":
        extremum_points_and_persistence.append((uf.Find(0), np.inf))

    return extremum_points_and_persistence


def diversify_extremum_points_and_persistence(extremum_points_and_persistence, level_set):
    min_points_and_persistence = [t for t in extremum_points_and_persistence[::2]]
    max_points_and_persistence = [t for t in extremum_points_and_persistence[1::2]]

    if level_set == "upper":
        max_points_and_persistence = max_points_and_persistence + [min_points_and_persistence[-1]]
        min_points_and_persistence = min_points_and_persistence[:-1]

    return min_points_and_persistence, max_points_and_persistence


def filter_extremum_points_by_persistence(extremum_points_and_persistence, threshold):

    filtered_extremum_points_and_persistence = [
        (point, persistence) for point, persistence in extremum_points_and_persistence if persistence > threshold
    ]
    return filtered_extremum_points_and_persistence
