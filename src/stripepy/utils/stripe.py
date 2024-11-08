import numpy as np


class Stripe:

    def __init__(self, seed=None, left_bw=None, right_bw=None, upp_bw=None, low_bw=None, top_pers=None, where=None):

        # TODO: robomics
        # -) make left_bw, right_bw, upp_bw, and low_bw positional and mandatory
        # -) where should be optional keyword: if not specified, it can be determined by the four mandatory values
        # -) top_pers is in [0, +inf]
        # -) where is "lower_triangular", "upper_triangular", or None (if not set)

        self.seed = seed
        self.persistence = top_pers
        self.L_bound = left_bw
        self.R_bound = right_bw
        self.U_bound = upp_bw
        self.D_bound = low_bw
        self.where = where
        self.inner_descriptors = {"five-number": [], "mean": None, "std": None}
        self.outer_descriptors = {"l-mean": None, "r-mean": None, "mean": None}
        self.rel_change = None
        self.RoI = None

    def compute_biodescriptors(self, I):

        # TODO: robomics
        # Check coordinates

        if self.where == "lower_triangular":
            convex_comb = int(round(0.99 * self.U_bound + 0.01 * self.D_bound))
            rows = slice(convex_comb, self.D_bound)
            cols = slice(self.L_bound, self.R_bound)
            restrI = I[rows, :].tocsc()[:, cols].toarray()
        elif self.where == "upper_triangular":
            convex_comb = int(round(0.01 * self.U_bound + 0.99 * self.D_bound))
            rows = slice(self.U_bound, convex_comb)
            cols = slice(self.L_bound, self.R_bound)
            restrI = I[rows, :].tocsc()[:, cols].toarray()
        else:
            print("Set one between lower_triangular and upper_triangular")
            exit(1)

        # ATT This can avoid empty stripes, which can occur e.g. when the column has (approximately) constant entries
        if np.prod(restrI.shape) > 0:

            # Horizontal enlargements:
            enl_HIoI_1 = max(0, self.L_bound - 3)
            enl_HIoI_2 = min(I.shape[1], self.R_bound + 3)

            # Five-number statistics inside the candidate stripe:
            self.inner_descriptors["five-number"] = [
                np.min(restrI),
                np.percentile(restrI, 25),
                np.percentile(restrI, 50),
                np.percentile(restrI, 75),
                np.max(restrI),
            ]
            self.inner_descriptors["mean"] = np.mean(restrI)
            self.inner_descriptors["std"] = np.std(restrI)

            # Mean intensity - left and right neighborhoods:
            if self.where == "lower_triangular":

                # Mean intensity - left neighborhood:
                self.outer_descriptors["l-mean"] = (
                    np.mean(I[convex_comb : self.D_bound, enl_HIoI_1 : self.L_bound])
                    if enl_HIoI_1 != self.L_bound
                    else np.nan
                )

                # Mean intensity - right neighborhood:
                self.outer_descriptors["r-mean"] = (
                    np.mean(I[convex_comb : self.D_bound, self.R_bound : enl_HIoI_2])
                    if enl_HIoI_2 != self.R_bound
                    else np.nan
                )

            elif self.where == "upper_triangular":

                # Mean intensity - left neighborhood:
                self.outer_descriptors["l-mean"] = (
                    np.mean(I[self.U_bound : convex_comb, enl_HIoI_1 : self.L_bound])
                    if enl_HIoI_1 != self.L_bound
                    else np.nan
                )

                # Mean intensity - right neighborhood:
                self.outer_descriptors["r-mean"] = (
                    np.mean(I[self.U_bound : convex_comb, self.R_bound : enl_HIoI_2])
                    if enl_HIoI_2 != self.R_bound
                    else np.nan
                )

            # Mean intensity:
            if np.isnan(self.outer_descriptors["r-mean"]):
                self.outer_descriptors["mean"] = self.outer_descriptors["l-mean"]
            elif np.isnan(self.outer_descriptors["l-mean"]):
                self.outer_descriptors["mean"] = self.outer_descriptors["r-mean"]
            else:
                self.outer_descriptors["mean"] = (
                    self.outer_descriptors["l-mean"] + self.outer_descriptors["r-mean"]
                ) / 2

            # Relative change in mean intensity:
            self.rel_change = (
                (
                    abs(self.inner_descriptors["mean"] - self.outer_descriptors["mean"])
                    / (self.outer_descriptors["mean"])
                    * 100
                )
                if self.outer_descriptors["mean"] != 0
                else -1.0  # TODO rea1991: check if nan is more suitable
            )

        else:
            # TODO rea1991 Check if still a problem, otherwise remove!
            self.inner_descriptors["five-number"] = [-1, -1, -1, -1, -1]
            self.inner_descriptors["mean"] = -1
            self.inner_descriptors["std"] = -1
            self.outer_descriptors["r-mean"] = -1
            self.outer_descriptors["l-mean"] = -1
            self.outer_descriptors["mean"] = -1
            self.rel_change = -1
