import numpy as np

from . import stripe

# TODO All functions defined in this file appear to be unused.
#      Review!


def filter_stripes(candida_stripes, threshold):
    rel_changes = np.array([candida_stripe.rel_change for candida_stripe in candida_stripes])
    keep_candidate = rel_changes > threshold

    return keep_candidate


def filter_L_stripes(I, sites, HIoIs, VIoIs, threshold, output_folder=None):
    """
    :param I:               Hi-C matrix
    :param HIoIs:           Horizontal Intervals of Interest
    :param VIoIs:           Vertical Intervals of Interest
    :param threshold:       threshold to filter out candidates
    :return:                   -
    """

    # Candidate stripes:
    candida_stripes = []

    # Indices stripes to keep or discard:
    candida2keep = []
    candida2disc = []

    # Regions 4 check:
    L_HIoI_regions4check = []
    L_VIoI_regions4check = []

    # Statistical analysis of the avg pixel intensity in a neighborhood:

    inner_means = [0] * len(sites)

    for num_cand, (site, HIoI, VIoI) in enumerate(zip(sites, HIoIs, VIoIs)):

        # Update list of candidate stripes:
        candida_stripe = stripe.Stripe(site, HIoI[0], HIoI[1], VIoI[0], VIoI[1])

        # Restricting I to the rectangular domain defined by current HIoI and VIoI:
        convex_comb = int(round(0.99 * VIoI[0] + 0.01 * VIoI[1]))
        # restrI = I[HIoI[1]:VIoI[1], HIoI[0]:HIoI[1]]
        restrI = I[convex_comb : VIoI[1], HIoI[0] : HIoI[1]]

        # Horizontal enlargements:
        enl_HIoI_1 = max(0, HIoI[0] - 3)
        enl_HIoI_2 = min(I.shape[1], HIoI[1] + 3)

        # Five-number statistics inside the candidate stripe:
        candida_stripe.inner_descriptors["five-number"] = [
            np.min(restrI),
            np.percentile(restrI, 25),
            np.percentile(restrI, 50),
            np.percentile(restrI, 75),
            np.max(restrI),
        ]
        candida_stripe.inner_descriptors["mean"] = np.mean(restrI)
        candida_stripe.inner_descriptors["std"] = np.std(restrI)

        inner_means[num_cand] = candida_stripe.inner_descriptors["mean"]

        # Mean intensity - left neighborhood:
        candida_stripe.outer_descriptors["l-mean"] = np.mean(I[convex_comb : VIoI[1], enl_HIoI_1 : HIoI[0]])

        # Mean intensity - right neighborhood:
        candida_stripe.outer_descriptors["r-mean"] = np.mean(I[convex_comb : VIoI[1], HIoI[1] : enl_HIoI_2])

        # Mean intensity:
        candida_stripe.outer_descriptors["mean"] = (
            candida_stripe.outer_descriptors["l-mean"] + candida_stripe.outer_descriptors["r-mean"]
        ) / 2

        # Relative change in mean intensity:
        candida_stripe.rel_change = (
            (candida_stripe.inner_descriptors["mean"] - candida_stripe.outer_descriptors["mean"])
            / (candida_stripe.outer_descriptors["mean"])
            * 100
        )

        candida_stripes.append(candida_stripe)

        # Filtration:
        if candida_stripe.rel_change < threshold:
            candida2disc += [num_cand]
        else:
            candida2keep += [num_cand]

        L_HIoI_regions4check += [[HIoI[1], enl_HIoI_2]]
        L_VIoI_regions4check += [[convex_comb, VIoI[1]]]

    if output_folder is not None:
        np.savetxt(f"{output_folder}/LT_inner_means_{threshold:.2f}.txt", inner_means, delimiter=",")

    return candida_stripes, candida2keep, candida2disc, L_HIoI_regions4check, L_VIoI_regions4check


def filter_U_stripes(I, sites, HIoIs, VIoIs, threshold, output_folder=None):
    """
    :param I:               Hi-C matrix
    :param HIoIs:           Horizontal Intervals of Interest
    :param VIoIs:           Vertical Intervals of Interest
    :param threshold:       threshold to filter out candidates
    :return:                   -
    """

    # Candidate stripes:
    candida_stripes = []

    # Indices stripes to keep or discard:
    candida2keep = []
    candida2disc = []

    # Regions 4 check:
    U_HIoI_regions4check = []
    U_VIoI_regions4check = []

    inner_means = [0] * len(sites)

    # Statistical analysis of the avg pixel intensity in a neighborhood:
    for num_cand, (site, HIoI, VIoI) in enumerate(zip(sites, HIoIs, VIoIs)):

        # Update list of candidate stripes:
        candida_stripe = stripe.Stripe(HIoI[0], HIoI[1], VIoI[0], VIoI[1], seed=site)

        # Restricting Iproc1 to the rectangular domain defined by current HIoI and VIoI:
        convex_comb = int(round(0.01 * VIoI[0] + 0.99 * VIoI[1]))
        restrI = I[VIoI[0] : convex_comb, HIoI[0] : HIoI[1]]

        # Horizontal enlargements:
        enl_HIoI_1 = max(0, HIoI[0] - 3)
        enl_HIoI_2 = min(I.shape[1], HIoI[1] + 3)

        print(f"{num_cand}) {site} {HIoI} {VIoI} {enl_HIoI_1} {enl_HIoI_2}")

        # Five-number statistics inside the candidate stripe:
        candida_stripe.inner_descriptors["five-number"] = [
            np.min(restrI),
            np.percentile(restrI, 25),
            np.percentile(restrI, 50),
            np.percentile(restrI, 75),
            np.max(restrI),
        ]
        candida_stripe.inner_descriptors["mean"] = np.mean(restrI)
        candida_stripe.inner_descriptors["std"] = np.std(restrI)

        inner_means[num_cand] = candida_stripe.inner_descriptors["mean"]

        # Mean intensity - left neighborhood:
        # l_mean = np.mean(I[HIoI[0]:VIoI[1], enl_HIoI_1:HIoI[0]])
        candida_stripe.outer_descriptors["l-mean"] = np.mean(I[VIoI[0] : convex_comb, enl_HIoI_1 : HIoI[0]])

        # Mean intensity - right neighborhood:
        # r_mean = np.mean(I[enl_HIoI_2:VIoI[1], HIoI[1]:enl_HIoI_2])
        candida_stripe.outer_descriptors["r-mean"] = np.mean(I[VIoI[0] : convex_comb, HIoI[1] : enl_HIoI_2])

        # Mean intensity:
        candida_stripe.outer_descriptors["mean"] = (
            candida_stripe.outer_descriptors["l-mean"] + candida_stripe.outer_descriptors["r-mean"]
        ) / 2

        # Relative change in mean intensity:
        candida_stripe.rel_change = (
            (candida_stripe.inner_descriptors["mean"] - candida_stripe.outer_descriptors["mean"])
            / (candida_stripe.outer_descriptors["mean"])
            * 100
        )

        print(
            f"{num_cand}) "
            f"{candida_stripe.inner_descriptors["mean"]}\t"
            f"{candida_stripe.outer_descriptors["mean"]}\t"
            f"{candida_stripe.outer_descriptors["l-mean"]}\t"
            f"{candida_stripe.outer_descriptors["r-mean"]}\t{candida_stripe.rel_change}"
        )

        candida_stripes.append(candida_stripe)

        # Filtration:
        if candida_stripe.rel_change < threshold:
            candida2disc += [num_cand]
        else:
            candida2keep += [num_cand]

        U_HIoI_regions4check += [[HIoI[1], enl_HIoI_2]]
        U_VIoI_regions4check += [[convex_comb, VIoI[1]]]

    if output_folder is not None:
        np.savetxt(f"{output_folder}/UT_inner_means_{threshold:.2f}.txt", inner_means, delimiter=",")

    return candida_stripes, candida2keep, candida2disc, U_HIoI_regions4check, U_VIoI_regions4check
