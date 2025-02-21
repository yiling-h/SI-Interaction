import numpy as np
from scipy.interpolate import BSpline

class cubic_spline():
    def __init__(self, data_nl, nknots=10, knots=None, intercept=False,
                 order=3, data_l=None):
        """
        Input:
            Data (np.array) : n x p_1 matrix, assume no intercept term
            nknots (np.array) : p - dim, each entry corresponds to number of knots
                                for each covariate
            nknots (int) : a constant integer, indicating each covariate has
                           the same amount of knots
            knots (dict) : p - elements, indexed by integers in [p],
                           each key corresponds to knows for that covariate,
                           each value is an np.array
            order (int) : a constant integer indicating the order of polynomial terms to include
            data_l (np.array): an n x p_2 array that contains column indices
                               for features to be included as linear effects
        """
        self.data_nl = data_nl
        self.data_l = data_l
        self.n = data_nl.shape[0]
        self.p_1 = data_nl.shape[1]
        self.intercept = intercept
        self.groups = []
        self.order = order
        if intercept:
            self.groups.append(0)

        # Set knots if knots were given
        if knots is not None:
            self.knots = knots
            self.nknots = np.zeros((self.p_1,))
            for i in range(self.p_1):
                self.nknots[i] = int(knots[i].shape[0])
        else:
            self.knots = None
            self.nknots = int(nknots)

        # if an integer nknots is given
        if np.array(nknots).shape == ():
            self.nknots = np.zeros((self.p_1,))
            for i in range(self.p_1):
                self.nknots[i] = int(nknots)

        # set group indices for derived covariates based on the original covariate
        for i in range(self.p_1):
            if self.intercept:
                self.groups.extend(list(np.array(i+1).repeat(self.nknots[i]+self.order)))
            else:
                self.groups.extend(list(np.array(i).repeat(self.nknots[i]+self.order)))

        # self.nknots should be assigned value at this point anyways
        ncol = np.sum(self.nknots) + (self.order * self.p_1)
        if intercept:
            ncol = ncol + 1

        # If linear parts are given, set linear parts parameters
        if data_l is not None:
            prev_groups = np.unique(self.groups).shape[0]
            self.p_2 = data_l.shape[1]
            for j in range(self.p_2):
                self.groups.append(prev_groups+j)
            ncol = ncol + self.p_2

        self.ncol = int(ncol)

    def construct_splines(self, equally_spaced=True, use_quantiles=None):
        """
        Constructing splines basis from the original data
        """
        self.spline_data = np.zeros((self.n,self.ncol))
        if self.knots is None and equally_spaced:
            self.knots = {}
            for i in range(self.p_1):
                min_i = np.min(self.data_nl[:,i])
                max_i = np.min(self.data_nl[:, i])
                self.knots[i] = np.linspace(min_i, max_i, int(self.nknots[i]))
            print("Equally spaced knots used.")

        if self.knots is None and use_quantiles:
            self.quantiles = {}
            self.knots = {}
            for i in range(self.p_1):
                self.quantiles[i] = np.linspace(0.1, 0.9, int(self.nknots[i]))
                self.knots[i] = np.quantile(self.data_nl[:, i],
                                            q=self.quantiles[i])
            print("Equally spaced quantile knots used.")

        def h(vec):
            positive_flag = (vec > 0)
            new_vec = np.zeros((vec.shape[0],))
            new_vec[positive_flag] = vec[positive_flag]
            return new_vec

        # Setting the spline basis
        col_idx = 0
        if self.intercept:
            self.spline_data[:, 0] = np.ones((self.n,))
            col_idx = col_idx + 1
        for i in range(self.p_1):
            for j in range(self.order):
                self.spline_data[:, col_idx + j] = self.data_nl[:,i]**(j+1)
            col_idx = col_idx + self.order


            for j in range(int(self.nknots[i])):
                self.spline_data[:, col_idx] = h((self.data_nl[:, i] - self.knots[i][j])**3)
                col_idx = col_idx + 1

        # Attach linear part
        if self.data_l is not None:
            for j in range(self.p_2):
                self.spline_data[:,j+col_idx] = self.data_l[:,j]

    def get_spline_data(self):
        return self.spline_data

    def get_groups(self):
        return np.array(self.groups)

    def get_knots(self):
        return self.knots

class b_spline():
    def __init__(self, data_nl, nknots=6, knots=None, intercept=False,
                 degree=2, data_l=None):
        """
        Input:
            Data (np.array) : n x p_1 matrix, assume no intercept term
            nknots (np.array) : p - dim, each entry corresponds to number of knots
                                for each covariate
            nknots (int) : a constant integer, indicating each covariate has
                           the same amount of knots
            knots (dict) : p - elements, indexed by integers in [p],
                           each key corresponds to knows for that covariate,
                           each value is an np.array
            degree (int) : a constant integer indicating the B-spline degree
            data_l (np.array): an n x p_2 array that contains column indices
                               for features to be included as linear effects
        """
        self.data_nl = data_nl
        self.data_l = data_l
        self.n = data_nl.shape[0]
        self.p_1 = data_nl.shape[1]
        self.intercept = intercept
        self.groups = []
        self.group_sizes = []
        self.degree = degree
        if intercept:
            self.groups.append(0)
            self.group_sizes.append(1)

        # Set knots if knots were given
        if knots is not None:
            self.knots = knots
            self.nknots = np.zeros((self.p_1,))
            for i in range(self.p_1):
                self.nknots[i] = int(knots[i].shape[0])
        else:
            self.knots = None
            self.nknots = int(nknots)

        # if an integer nknots is given
        if np.array(nknots).shape == ():
            self.nknots = np.zeros((self.p_1,))
            for i in range(self.p_1):
                self.nknots[i] = int(nknots)

    def construct_splines(self, equally_spaced=True, use_quantiles=None,
                          center=True, scale=True, orthogonal=True):
        """
        Constructing splines basis from the original data
        """
        self.spline_data = None
        if self.knots is None and equally_spaced:
            self.knots = {}
            for i in range(self.p_1):
                min_i = np.min(self.data_nl[:, i])
                max_i = np.max(self.data_nl[:, i])
                range_i = max_i - min_i
                self.knots[i] = np.linspace(min_i - 1 * range_i, max_i + 1 * range_i, int(self.nknots[i]))
            print("Equally spaced knots used.")

        if self.knots is None and use_quantiles:
            self.quantiles = {}
            self.knots = {}
            for i in range(self.p_1):
                self.quantiles[i] = np.linspace(0, 1, int(self.nknots[i]))
                self.knots[i] = np.quantile(self.data_nl[:, i],
                                            q=self.quantiles[i])
            print("Equally spaced quantile knots used.")

        # Setting the spline basis
        col_idx = 0
        if self.intercept:
            self.spline_data = np.ones((self.n,1))
            col_idx = col_idx + 1
        for i in range(self.p_1):
            basis_dim = self.nknots[i] - self.degree - 1
            assert basis_dim > 0
            #print(basis_dim)
            #print(i)
            def check_rank_deficient(design, tol=1e-5):
                ones = np.ones((design.shape[0],))
                x, residuals, rank, s = np.linalg.lstsq(design, ones, rcond=None)
                rank_deficient = (np.linalg.norm(residuals) < tol)
                return rank_deficient

            c = np.repeat(0, int(basis_dim))
            #print(self.knots[i])
            #print(self.degree)
            spl = BSpline(self.knots[i], c=c, k=self.degree, extrapolate=True)
            design = (spl.design_matrix(self.data_nl[:,i], t=self.knots[i], k=self.degree, extrapolate=True)).toarray()
            if center:
                design -= design.mean(0)[None, :]

            if scale:
                scaling = design.std(0)
                design /= scaling[None, :]

            if orthogonal:
                Q, R = np.linalg.qr(design)
                design = Q
                assert (np.allclose(np.dot(Q.T, Q), np.identity(Q.shape[1])))

            if check_rank_deficient(design):
                #if self.intercept:
                design = design[:, :-1]
                basis_dim -= 1
                assert not check_rank_deficient(design)
                #elif i != 0:
                #    design = design[:, :-1]
                #    basis_dim -= 1
                #    assert not check_rank_deficient(design)

            if self.spline_data is None:
                self.spline_data = design
            else:
                self.spline_data = np.hstack([self.spline_data, design])

            self.spline_data[:, int(col_idx):int(col_idx+basis_dim)] = design
            col_idx = col_idx + basis_dim
            self.group_sizes.append(basis_dim)

        # Attach linear part
        if self.data_l is not None:
            self.p_2 = self.data_l.shape[1]
            self.spline_data = np.hstack((self.spline_data,self.data_l))

        # set group indices for derived covariates based on the original covariate
        for i in range(self.p_1):
            if self.intercept:
                self.groups.extend(list(np.array(i + 1).repeat(self.group_sizes[i + 1])))
            else:
                self.groups.extend(list(np.array(i).repeat(self.group_sizes[i])))

        # self.nknots should be assigned value at this point anyways
        ncol = len(self.groups)

        # If linear parts are given, set linear parts parameters
        if self.data_l is not None:
            prev_groups = np.unique(self.groups).shape[0]
            for j in range(self.p_2):
                self.groups.append(prev_groups + j)
            ncol = ncol + self.p_2

        self.ncol = int(ncol)

    def get_spline_data(self):
        return self.spline_data

    def get_groups(self):
        return np.array(self.groups)

    def get_knots(self):
        return self.knots








