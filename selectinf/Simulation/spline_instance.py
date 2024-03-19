import numpy as np
from selectinf.Simulation.spline import cubic_spline, b_spline
from scipy.stats import t as tdist
from scipy.stats import norm

def generate_gaussian_instance(intercept=False,
                               linear_mean=0,
                               linear_sd=1,
                               nonlinear_mean=0,
                               nonlinear_sd=1,
                               n=2000, p_nl=10, p_l=90,
                               s_nl = 3, s_l = 10,
                               nknots = 10, order = 3,
                               signal=1., signal_fac=1., random_signs=True,
                               center=True, scale=True):
    if p_l > 0:
        data_linear = np.random.normal(loc=linear_mean,
                                       scale=linear_sd,
                                       size=(n, p_l))
    else:
        data_linear = None

    data_nonlinear = np.random.normal(loc = nonlinear_mean,
                                      scale = nonlinear_sd,
                                      size = (n, p_nl))
    """data_nonlinear = np.random.uniform(low=-2.5,
                                      high=2.5,
                                      size=(n, p_nl))"""

    cs = cubic_spline(data_nl=data_nonlinear, data_l=data_linear, nknots=nknots, order=order)
    cs.construct_splines(use_quantiles=True, equally_spaced=False)
    design = cs.get_spline_data()
    groups = cs.get_groups()

    group_labels = np.unique(groups)

    # Assigning active groups
    # Assuming no intercept
    group_active = np.random.choice(np.arange(p_nl), s_nl, replace=False)
    if p_l > 0 and s_l > 0:
        linear_active = np.random.choice(np.arange(p_nl, p_nl + p_l), s_l, replace = False)
        group_active = np.append(group_active,linear_active)

    beta = np.zeros(design.shape[1])
    if signal_fac is not None:
        signal = np.sqrt(signal_fac * 2 * np.log(design.shape[1]))
        print(signal)

    signal = np.atleast_1d(signal)

    active = np.isin(groups, group_active)

    if signal.shape == (1,):
        beta[active] = signal[0]
    else:
        beta[active] = np.linspace(signal[0], signal[1], active.sum())
    if random_signs:
        beta[active] *= (2 * np.random.binomial(1, 0.5, size=(active.sum(),)) - 1.)
    beta /= np.sqrt(n)

    if center:
        design -= design.mean(0)[None, :]

    if scale:
        # ----SCALE----
        # scales X by sqrt(n) and sd
        # if we need original X, uncomment the following line
        # X_raw = X
        # ----SCALE----
        scaling = design.std(0) * np.sqrt(n)
        design /= scaling[None, :]
        beta *= np.sqrt(n)

    Y = (design.dot(beta)) + np.random.normal(size = (n,))

    return design, Y, beta, groups

def generate_gaussian_instance_nonlinear(n=2000, p_nl=10, p_l=90,
                                         nknots = 6, degree = 2, SNR=1.,
                                         center=False, scale=True, intercept=True):
    # y = 2 sin(2x1) + x2^2 + exp(−x3) + x4 − 3x5 + 2.5x6 + 10x7 + 2x8 − 7x9 + 5x10
    if p_l > 0:
        data_linear = np.random.uniform(low = 0, high = 1, size = (n, p_l))
    else:
        data_linear = None

    data_nonlinear = np.zeros((n,p_nl))#np.random.uniform(low = 0, high = 1, size = (n, p_nl))
    data_nonlinear[:,0:3] = np.random.uniform(low=-2.5, high=2.5, size=(n, 3))
    data_nonlinear[:,3:] = np.random.uniform(low = 0, high = 1, size = (n, p_nl-3))
    """data_nonlinear = np.random.uniform(low=-2.5,
                                      high=2.5,
                                      size=(n, p_nl))"""

    bs = b_spline(data_nl=data_nonlinear, data_l=data_linear, nknots=nknots, degree=degree,
                  intercept=intercept)
    bs.construct_splines(use_quantiles=True, equally_spaced=False)
    design = bs.get_spline_data()
    groups = bs.get_groups()

    if center:
        design -= design.mean(0)[None, :]

    if scale:
        # ----SCALE----
        # scales X by sqrt(n) and sd
        # if we need original X, uncomment the following line
        # X_raw = X
        # ----SCALE----
        scaling = design.std(0) * np.sqrt(n)
        if intercept:
            scaling[0] = np.sqrt(n)
        design /= scaling[None, :]
        #beta *= np.sqrt(n)

    #Y = (design.dot(beta)) + np.random.normal(size = (n,))
    Y_mean = (2*np.sin(2*data_nonlinear[:,0]) + data_nonlinear[:,1]**2 + np.exp(-data_nonlinear[:,2])
              + data_nonlinear[:,3] - 3*data_nonlinear[:,4] + 2.5*data_nonlinear[:,5] + 10*data_nonlinear[:,6]
              + 2*data_nonlinear[:,7] - 7*data_nonlinear[:,8] + 5*data_nonlinear[:,9])
    Y_mean = Y_mean
    noise_sd = (np.linalg.norm(Y_mean) / np.sqrt(n * SNR))
    Y = Y_mean + np.random.normal(size=(n,), scale=noise_sd)

    active_flag = np.zeros(np.unique(groups).shape[0])
    active_flag[0:10+intercept] = 1.

    return design, Y, Y_mean, groups, active_flag

def generate_gaussian_instance_nonlinear_interaction(n=2000, p_nl=10, p_l=90,
                                                     nknots = 6, degree = 2, SNR=1.,
                                                     center=False, scale=True, intercept=True,
                                                     structure='allpairs'):

    # y = 2 sin(2x1) + x2^2 + exp(−x3) + x4 − 3x5 + 2.5x6 + 10x7 + 2x8 − 7x9 + 5x10
    if p_l > 0:
        data_linear = np.random.uniform(low = 0, high = 1, size = (n, p_l))
    else:
        data_linear = None

    data_nonlinear = np.zeros((n,p_nl))#np.random.uniform(low = 0, high = 1, size = (n, p_nl))
    data_nonlinear[:,0:3] = np.random.uniform(low=-2.5, high=2.5, size=(n, 3))
    data_nonlinear[:,3:] = np.random.uniform(low = 0, high = 1, size = (n, p_nl-3))
    """data_nonlinear = np.random.uniform(low=-2.5,
                                      high=2.5,
                                      size=(n, p_nl))"""

    bs = b_spline(data_nl=data_nonlinear, data_l=data_linear, nknots=nknots, degree=degree,
                  intercept=intercept)
    bs.construct_splines(use_quantiles=True, equally_spaced=False)
    design = bs.get_spline_data()
    groups = bs.get_groups()

    if center:
        design -= design.mean(0)[None, :]

    if scale:
        # ----SCALE----
        # scales X by sqrt(n) and sd
        # if we need original X, uncomment the following line
        # X_raw = X
        # ----SCALE----
        scaling = design.std(0) * np.sqrt(n)
        if intercept:
            scaling[0] = np.sqrt(n)
        design /= scaling[None, :]
        #beta *= np.sqrt(n)

    #Y = (design.dot(beta)) + np.random.normal(size = (n,))
    Y_mean = (2*np.sin(2*data_nonlinear[:,0]) + data_nonlinear[:,1]**2 + np.exp(-data_nonlinear[:,2])
              + data_nonlinear[:,3] - 3*data_nonlinear[:,4] + 2.5*data_nonlinear[:,5] + 10*data_nonlinear[:,6]
              + 2*data_nonlinear[:,7] - 7*data_nonlinear[:,8] + 5*data_nonlinear[:,9])
    Y_mean = Y_mean
    noise_sd = (np.linalg.norm(Y_mean) / np.sqrt(n * SNR))
    Y = Y_mean + np.random.normal(size=(n,), scale=noise_sd)

    active_flag = np.zeros(np.unique(groups).shape[0])
    active_flag[0:10+intercept] = 1.

    return design, Y, Y_mean, groups, active_flag


def generate_gaussian_instance_from_bspline(n=2000, p_nl=10, p_l=90,
                                            s_nl = 3, s_l = 10,
                                            nknots = 6, degree = 2,
                                            SNR = 1, signal_fac=1.,
                                            center=False, scale=True, random_signs=True,
                                            intercept=True):
    #y = 2 sin(2x1) + x2^2 + exp(−x3) + x4 − 3x5 + 2.5x6 + 10x7 + 2x8 − 7x9 + 5x10
    if p_l > 0:
        data_linear = np.random.uniform(low = 0, high = 1, size = (n, p_l))
    else:
        data_linear = None

    data_nonlinear = np.zeros((n,p_nl))#np.random.uniform(low = 0, high = 1, size = (n, p_nl))
    data_nonlinear[:,0:3] = np.random.uniform(low=-2.5, high=2.5, size=(n, 3))
    data_nonlinear[:,3:] = np.random.uniform(low = 0, high = 1, size = (n, p_nl-3))
    """data_nonlinear = np.random.uniform(low=-2.5,
                                      high=2.5,
                                      size=(n, p_nl))"""

    bs = b_spline(data_nl=data_nonlinear, data_l=data_linear, nknots=nknots, degree=degree,
                  intercept=intercept)
    bs.construct_splines(use_quantiles=True, equally_spaced=False)
    design = bs.get_spline_data()
    groups = bs.get_groups()

    group_labels = np.unique(groups)

    # Assigning active groups
    # Assuming no intercept
    group_active = np.random.choice(np.arange(p_nl), s_nl, replace=False)
    if p_l > 0 and s_l > 0:
        linear_active = np.random.choice(np.arange(p_nl, p_nl + p_l), s_l, replace=False)
        group_active = np.append(group_active, linear_active)

    beta = np.zeros(design.shape[1])
    if signal_fac is not None:
        signal = np.sqrt(signal_fac * 2 * np.log(design.shape[1]))
        print(signal)

    signal = np.atleast_1d(signal)

    active = np.isin(groups, group_active)

    if signal.shape == (1,):
        beta[active] = signal[0]
    else:
        beta[active] = np.linspace(signal[0], signal[1], active.sum())
    if random_signs:
        beta[active] *= (2 * np.random.binomial(1, 0.5, size=(active.sum(),)) - 1.)
    beta /= np.sqrt(n)

    if center:
        design -= design.mean(0)[None, :]

    if scale:
        # ----SCALE----
        # scales X by sqrt(n) and sd
        # if we need original X, uncomment the following line
        # X_raw = X
        # ----SCALE----
        scaling = design.std(0) * np.sqrt(n)
        design /= scaling[None, :]
        beta *= np.sqrt(n)

    Y_mean = design.dot(beta)
    noise_sd = (np.linalg.norm(Y_mean) / np.sqrt(n * SNR))
    print("noise_sd:", noise_sd)

    Y = Y_mean + np.random.normal(size=(n,), scale=noise_sd)

    return design, Y, beta, groups

def sample_interaction_pairs(p, s_inter):
    # Construct collection of candidate pairs
    pairs = []
    for i in range(p):
        for j in range(i + 1, p):
            pairs.append((i, j))
    pairs = np.array(pairs)

    # Sample some interaction pairs
    pairs_smpl_idx = np.random.choice(np.arange(len(pairs)), s_inter, replace=False)
    pairs_smpl = pairs[pairs_smpl_idx]

    active_inter = np.zeros((p,p))
    for i in range(pairs_smpl.shape[0]):
        pair = pairs_smpl[i]
        active_inter[pair[0], pair[1]] = 1
        #print(pair[0], pair[1])

    return active_inter, pairs_smpl


def sample_interaction_pairs_hierarchy(p, s_inter, active_idx, mode='weakhierarchy', intercept=False):
    if intercept:
        active_idx -= 1

    # Construct collection of candidate pairs
    pairs = []
    if mode == 'weakhierarchy':
        for i in range(p):
            for j in range(i + 1, p):
                if i in active_idx or j in active_idx:
                    pairs.append((i, j))
    elif mode == 'stronghierarchy':
        for i in range(p):
            for j in range(i + 1, p):
                if i in active_idx and j in active_idx:
                    pairs.append((i, j))
    pairs = np.array(pairs)

    # Sample some interaction pairs
    pairs_smpl_idx = np.random.choice(np.arange(len(pairs)), s_inter, replace=False)
    pairs_smpl = pairs[pairs_smpl_idx]

    active_inter = np.zeros((p, p))
    for i in range(pairs_smpl.shape[0]):
        pair = pairs_smpl[i]
        active_inter[pair[0], pair[1]] = 1
        #print(pair[0], pair[1])

    return active_inter, pairs_smpl
def generate_gaussian_instance_from_bspline_interaction(n=2000, p_nl=10, p_l=90,
                                                        s_nl = 3, s_l = 10,
                                                        nknots = 6, degree = 2,
                                                        SNR = 1,
                                                        center=False, scale=True,
                                                        random_signs=True, intercept=True,
                                                        structure='allpairs', s_interaction=10):

    """
    Returns:

    """

    #y = 2 sin(2x1) + x2^2 + exp(−x3) + x4 − 3x5 + 2.5x6 + 10x7 + 2x8 − 7x9 + 5x10
    if p_l > 0:
        data_linear = np.random.uniform(low = 0, high = 1, size = (n, p_l))
    else:
        data_linear = None

    data_nonlinear = np.zeros((n,p_nl))#np.random.uniform(low = 0, high = 1, size = (n, p_nl))
    data_nonlinear[:,0:3] = np.random.uniform(low=-2.5, high=2.5, size=(n, 3))
    data_nonlinear[:,3:] = np.random.uniform(low = 0, high = 1, size = (n, p_nl-3))
    """data_nonlinear = np.random.uniform(low=-2.5,
                                      high=2.5,
                                      size=(n, p_nl))"""

    bs = b_spline(data_nl=data_nonlinear, data_l=data_linear, nknots=nknots, degree=degree,
                  intercept=intercept)
    bs.construct_splines(use_quantiles=True, equally_spaced=False, center=center)
    design = bs.get_spline_data()
    # Returning group labels with 0 meaning the intercept (if applicable)
    groups = bs.get_groups()

    # Assigning active groups
    group_active = np.random.choice(np.arange(p_nl + intercept), s_nl + 1, replace=False)
    if p_l > 0 and s_l > 0:
        linear_active = np.random.choice(np.arange(p_nl + intercept, p_nl + p_l + intercept),
                                         s_l, replace=False)
        group_active = np.append(group_active, linear_active)

    # Generating sparse main effects
    beta = np.zeros(design.shape[1])
    active = np.isin(groups, group_active)
    beta[active] = 1.

    # Generating sparse interactions
    if p_l > 0:
        data_combined = np.concatenate((data_nonlinear, data_linear), axis=1)
    else:
        data_combined = data_nonlinear
    p = p_l + p_nl

    if structure == 'allpairs':
        # sample interactions without replacement
        active_inter_adj, active_inter_list  = sample_interaction_pairs(p, s_interaction)
    elif structure == 'weakhierarchy':
        # sample interactions without replacement
        active_inter_adj, active_inter_list\
            = sample_interaction_pairs_hierarchy(p, s_interaction,
                                                 active_idx=group_active,
                                                 mode=structure, intercept=intercept)
    elif structure == 'stronghierarchy':
        # sample interactions without replacement
        active_inter_adj, active_inter_list \
            = sample_interaction_pairs_hierarchy(p, s_interaction,
                                                 active_idx=group_active,
                                                 mode=structure, intercept=intercept)

    data_interaction = {}
    for i in range(p):
        for j in range(i+1, p):
            data_interaction[(i,j)] = data_combined[:,i] * data_combined[:,j]
            if center:
                data_interaction[(i,j)] = data_interaction[(i,j)] - np.mean(data_interaction[(i,j)])
            if scale:
                scaling = np.std(data_interaction[(i,j)]) * np.sqrt(n)
                data_interaction[(i, j)] = data_interaction[(i,j)] / scaling

    gamma = np.ones((s_interaction,))

    if random_signs:
        beta[active] *= (2 * np.random.binomial(1, 0.5, size=(active.sum(),)) - 1.)
        gamma *= (2 * np.random.binomial(1, 0.5, size=(s_interaction,)) - 1.)
    beta /= np.sqrt(n)
    gamma /= np.sqrt(n)

    if center:
        design -= design.mean(0)[None, :]

    if scale:
        # ----SCALE----
        # scales X by sqrt(n) and sd
        # if we need original X, uncomment the following line
        # X_raw = X
        # ----SCALE----
        scaling = design.std(0) * np.sqrt(n)
        design /= scaling[None, :]
        beta *= np.sqrt(n)
        gamma *= np.sqrt(n)

    # interaction * gamma
    interaction_proj = np.zeros((n,))
    # Construct (dense) design of interactions:
    for i in range(active_inter_list.shape[0]):
        pair = active_inter_list[i]
        interaction_proj += data_interaction[(pair[0], pair[1])] * gamma[i]

    Y_mean = design.dot(beta) + interaction_proj

    noise_sd = (np.sqrt(np.linalg.norm(design.dot(beta))**2 +
                        np.linalg.norm(interaction_proj)**2) /
                np.sqrt(n * SNR))
    print("noise_sd:", noise_sd)

    Y = Y_mean + np.random.normal(size=(n,), scale=noise_sd)

    return (design, data_interaction, Y, Y_mean, data_combined,
            groups, active, active_inter_adj, active_inter_list)

def generate_gaussian_instance_nonlinear_interaction(n=2000, p_nl=10, p_l=90,
                                                    s_l = 10, rho=0.3, rho_noise=0.,
                                                    nknots = 6, degree = 2,
                                                    SNR = 1,
                                                    center=False, scale=True,
                                                    random_signs=True, intercept=True,
                                                    structure='allpairs', s_interaction=10,
                                                     interaction_signal=1.):

    """
    Returns:

    """

    def sample_correlated_uniforms(n, k, rho):
        """
        Sample n instances of k correlated uniform variables.

        Parameters:
        - n: int, number of samples.
        - k: int, number of variables.
        - R: array-like, k x k correlation matrix.

        Returns:
        - samples: n x k array of samples from correlated uniform distributions.
        """
        R = (1-rho)*np.eye(k)
        R += rho

        # Mean vector for the multivariate normal distribution
        mean = np.zeros(k)

        # Generate samples from the multivariate normal distribution
        normal_samples = np.random.multivariate_normal(mean, R, size=n)

        # Transform to uniform samples using the CDF of the normal distribution
        uniform_samples = norm.cdf(normal_samples)

        return uniform_samples

    #y = 2 sin(2x1) + x2^2 + exp(−x3) + x4 − 3x5 + 2.5x6 + 10x7 + 2x8 − 7x9 + 5x10
    if p_l > 0:
        data_linear = np.random.uniform(low = 0, high = 1, size = (n, p_l))
    else:
        data_linear = None

    data_nonlinear = np.zeros((n,p_nl))#np.random.uniform(low = 0, high = 1, size = (n, p_nl))
    # data_nonlinear[:,0:3] = np.random.uniform(low=-2.5, high=2.5, size=(n, 3))
    """
    data_nonlinear[:, 0:3] = sample_correlated_uniforms(n=n, k=3, rho=rho) * 2.5
    print(np.corrcoef(data_nonlinear[:, 0:3].T))
    # data_nonlinear[:,3:] = np.random.uniform(low = 0, high = 1, size = (n, p_nl-3))
    data_nonlinear[:, 3:10] = sample_correlated_uniforms(n=n, k=7, rho=rho)
    data_nonlinear[:, 10:] = sample_correlated_uniforms(n=n, k=p_nl - 10, rho=rho_noise)
    """
    data_nonlinear = sample_correlated_uniforms(n=n, k=p_nl, rho=rho_noise)

    bs = b_spline(data_nl=data_nonlinear, data_l=data_linear, nknots=nknots, degree=degree,
                  intercept=intercept)
    bs.construct_splines(use_quantiles=True, equally_spaced=False, center=center)
    design = bs.get_spline_data()
    # Returning group labels with 0 meaning the intercept (if applicable)
    groups = bs.get_groups()

    # Assigning active groups
    # Assuming no active linear covariates
    group_active = np.array(range(10 + intercept))
    active = np.isin(groups, group_active)

    # Generating sparse interactions
    if p_l > 0:
        data_combined = np.concatenate((data_nonlinear, data_linear), axis=1)
    else:
        data_combined = data_nonlinear
    p = p_l + p_nl

    if structure == 'allpairs':
        # sample interactions without replacement
        active_inter_adj, active_inter_list  = sample_interaction_pairs(p, s_interaction)
    elif structure == 'weakhierarchy':
        # sample interactions without replacement
        active_inter_adj, active_inter_list\
            = sample_interaction_pairs_hierarchy(p, s_interaction,
                                                 active_idx=group_active,
                                                 mode=structure, intercept=intercept)
    elif structure == 'stronghierarchy':
        # sample interactions without replacement
        active_inter_adj, active_inter_list \
            = sample_interaction_pairs_hierarchy(p, s_interaction,
                                                 active_idx=group_active,
                                                 mode=structure, intercept=intercept)

    data_interaction = {}
    for i in range(p):
        for j in range(i+1, p):
            data_interaction[(i,j)] = data_combined[:,i] * data_combined[:,j]
            if center:
                data_interaction[(i,j)] = data_interaction[(i,j)] - np.mean(data_interaction[(i,j)])
            if scale:
                scaling = np.std(data_interaction[(i,j)]) * np.sqrt(n)
                data_interaction[(i, j)] = data_interaction[(i,j)] / scaling

    gamma = np.ones((s_interaction,)) * interaction_signal
    # gamma /= np.sqrt(n) # Uncomment after debugging

    if center:
        design -= design.mean(0)[None, :]

    if scale:
        # ----SCALE----
        # scales X by sqrt(n) and sd
        # if we need original X, uncomment the following line
        # X_raw = X
        # ----SCALE----
        scaling = design.std(0) * np.sqrt(n)
        if intercept:
            scaling[0] = np.sqrt(n)
        design /= scaling[None, :]
        gamma *= np.sqrt(n)

    # interaction * gamma
    interaction_proj = np.zeros((n,))
    # Construct (dense) design of interactions:
    for i in range(active_inter_list.shape[0]):
        pair = active_inter_list[i]
        interaction_proj += data_interaction[(pair[0], pair[1])] * gamma[i]

    Y_mean_main = (2*np.sin(2*data_nonlinear[:,0]) + data_nonlinear[:,1]**2 + np.exp(-data_nonlinear[:,2])
              + data_nonlinear[:,3] - 3*data_nonlinear[:,4] + 2.5*data_nonlinear[:,5] + 10*data_nonlinear[:,6]
              + 2*data_nonlinear[:,7] - 7*data_nonlinear[:,8] + 5*data_nonlinear[:,9])

    if scale:
        Y_mean_main /= np.sqrt(n)

    noise_sd = (np.std(Y_mean_main + interaction_proj) /
                np.sqrt(SNR))
    #print("noise_sd:", noise_sd)

    Y_mean = Y_mean_main + interaction_proj
    Y = Y_mean + np.random.normal(size=(n,), scale=noise_sd)

    #print("main effects:", np.linalg.norm(Y_mean_main))
    #print("interaction:", np.linalg.norm(interaction_proj))

    return (design, data_interaction, Y, Y_mean, data_combined,
            groups, active, active_inter_adj, active_inter_list)

def generate_gaussian_instance_nonlinear_interaction_simple(n=2000, p_nl=10, p_l=90,
                                                    s_l = 10, rho=0.3, rho_noise=0.,
                                                            full_corr=True,
                                                    nknots = 6, degree = 2,
                                                    SNR = 1,
                                                    center=False, scale=True,
                                                    random_signs=True, intercept=True,
                                                    structure='allpairs', s_interaction=10,
                                                     interaction_signal=1.):

    """
    Returns:

    """

    def sample_correlated_uniforms(n, k, rho):
        """
        Sample n instances of k correlated uniform variables.

        Parameters:
        - n: int, number of samples.
        - k: int, number of variables.
        - R: array-like, k x k correlation matrix.

        Returns:
        - samples: n x k array of samples from correlated uniform distributions.
        """
        R = (1-rho)*np.eye(k)
        R += rho

        # Mean vector for the multivariate normal distribution
        mean = np.zeros(k)

        # Generate samples from the multivariate normal distribution
        normal_samples = np.random.multivariate_normal(mean, R, size=n)

        # Transform to uniform samples using the CDF of the normal distribution
        uniform_samples = norm.cdf(normal_samples)

        return uniform_samples

    #y = 2 sin(2x1) + x2^2 + exp(−x3) + x4 − 3x5 + 2.5x6 + 10x7 + 2x8 − 7x9 + 5x10
    if p_l > 0:
        data_linear = np.random.uniform(low = 0, high = 1, size = (n, p_l))
    else:
        data_linear = None

    #np.random.uniform(low = 0, high = 1, size = (n, p_nl))
    # data_nonlinear[:,0:3] = np.random.uniform(low=-2.5, high=2.5, size=(n, 3))
    # print(np.corrcoef(data_nonlinear[:, 0:3].T))
    # data_nonlinear[:,3:] = np.random.uniform(low = 0, high = 1, size = (n, p_nl-3))
    if full_corr:
        data_nonlinear = sample_correlated_uniforms(n=n, k=p_nl, rho=rho)
        data_nonlinear[:, 0:3] *= 2.5
    else:
        data_nonlinear = np.zeros((n, p_nl))
        data_nonlinear[:, 0:3] = sample_correlated_uniforms(n=n, k=3, rho=rho) * 2.5
        data_nonlinear[:, 3:] = sample_correlated_uniforms(n=n, k=p_nl - 3, rho=rho_noise)

    """data_nonlinear = np.random.uniform(low=-2.5,
                                      high=2.5,
                                      size=(n, p_nl))"""

    bs = b_spline(data_nl=data_nonlinear, data_l=data_linear, nknots=nknots, degree=degree,
                  intercept=intercept)
    bs.construct_splines(use_quantiles=True, equally_spaced=False, center=center)
    design = bs.get_spline_data()
    # Returning group labels with 0 meaning the intercept (if applicable)
    groups = bs.get_groups()

    # Assigning active groups
    # Assuming no active linear covariates
    group_active = np.array(range(3 + intercept))
    active = np.isin(groups, group_active)

    # Generating sparse interactions
    if p_l > 0:
        data_combined = np.concatenate((data_nonlinear, data_linear), axis=1)
    else:
        data_combined = data_nonlinear
    p = p_l + p_nl

    if structure == 'allpairs':
        # sample interactions without replacement
        active_inter_adj, active_inter_list  = sample_interaction_pairs(p, s_interaction)
    elif structure == 'weakhierarchy':
        # sample interactions without replacement
        active_inter_adj, active_inter_list\
            = sample_interaction_pairs_hierarchy(p, s_interaction,
                                                 active_idx=group_active,
                                                 mode=structure, intercept=intercept)
    elif structure == 'stronghierarchy':
        # sample interactions without replacement
        active_inter_adj, active_inter_list \
            = sample_interaction_pairs_hierarchy(p, s_interaction,
                                                 active_idx=group_active,
                                                 mode=structure, intercept=intercept)

    data_interaction = {}
    for i in range(p):
        for j in range(i+1, p):
            data_interaction[(i,j)] = data_combined[:,i] * data_combined[:,j]
            if center:
                data_interaction[(i,j)] = data_interaction[(i,j)] - np.mean(data_interaction[(i,j)])
            if scale:
                scaling = np.std(data_interaction[(i,j)]) * np.sqrt(n)
                data_interaction[(i, j)] = data_interaction[(i,j)] / scaling

    gamma = np.ones((s_interaction,)) * interaction_signal
    # gamma /= np.sqrt(n) # Uncomment after debugging

    if center:
        design -= design.mean(0)[None, :]

    if scale:
        # ----SCALE----
        # scales X by sqrt(n) and sd
        # if we need original X, uncomment the following line
        # X_raw = X
        # ----SCALE----
        scaling = design.std(0) * np.sqrt(n)
        if intercept:
            scaling[0] = np.sqrt(n)
        design /= scaling[None, :]
        gamma *= np.sqrt(n)

    # interaction * gamma
    interaction_proj = np.zeros((n,))
    # Construct (dense) design of interactions:
    for i in range(active_inter_list.shape[0]):
        pair = active_inter_list[i]
        interaction_proj += data_interaction[(pair[0], pair[1])] * gamma[i]

    Y_mean_main = (2*np.sin(2*data_nonlinear[:,0]) + data_nonlinear[:,1]**2 + np.exp(-data_nonlinear[:,2])
              + data_nonlinear[:,3] - 3*data_nonlinear[:,4] + 2.5*data_nonlinear[:,5] + 10*data_nonlinear[:,6]
              + 2*data_nonlinear[:,7] - 7*data_nonlinear[:,8] + 5*data_nonlinear[:,9])

    if scale:
        Y_mean_main /= np.sqrt(n)

    noise_sd = (np.std(Y_mean_main + interaction_proj) /
                np.sqrt(SNR))
    #print("noise_sd:", noise_sd)

    Y_mean = Y_mean_main + interaction_proj
    Y = Y_mean + np.random.normal(size=(n,), scale=noise_sd)

    #print("main effects:", np.linalg.norm(Y_mean_main))
    #print("interaction:", np.linalg.norm(interaction_proj))

    return (design, data_interaction, Y, Y_mean, data_combined,
            groups, active, active_inter_adj, active_inter_list)


_cov_cache = {}
def _design(n, p, rho, equicorrelated):
    """
    Create an equicorrelated or AR(1) design.
    """
    if equicorrelated:
        X = (np.sqrt(1 - rho) * np.random.standard_normal((n, p)) +
             np.sqrt(rho) * np.random.standard_normal(n)[:, None])

        def equi(rho, p):
            if ('equi', p, rho) not in _cov_cache:
                sigmaX = (1 - rho) * np.identity(p) + rho * np.ones((p, p))
                cholX = np.linalg.cholesky(sigmaX)
                _cov_cache[('equi', p, rho)] = sigmaX, cholX
            return _cov_cache[('equi', p, rho)]

        sigmaX, cholX = equi(rho=rho, p=p)
    else:
        def AR1(rho, p):
            if ('AR1', p, rho) not in _cov_cache:
                idx = np.arange(p)
                cov = rho ** np.abs(np.subtract.outer(idx, idx))
                _cov_cache[('AR1', p, rho)] = cov, np.linalg.cholesky(cov)
            cov, chol = _cov_cache[('AR1', p, rho)]
            return cov, chol

        sigmaX, cholX = AR1(rho=rho, p=p)
        X = np.random.standard_normal((n, p)).dot(cholX.T)
    return X, sigmaX, cholX


def gaussian_polynomial_interaction_instance(n=500, p=200, s=7, rho=0., order=1,
                                             covar_mean=0, partial_cor=True,
                                            SNR = 1, interaction_signal=1,
                                            random_signs=False,
                                            scale=True, center=True,
                                            equicorrelated=True,
                                            structure=None, s_interaction=0,
                                            intercept=False):
    """
    A testing instance for the LASSO.
    If equicorrelated is True design is equi-correlated in the population,
    normalized to have columns of norm 1.
    If equicorrelated is False design is auto-regressive.
    For the default settings, a $\lambda$ of around 13.5
    corresponds to the theoretical $E(\|X^T\epsilon\|_{\infty})$
    with $\epsilon \sim N(0, \sigma^2 I)$.

    Parameters
    ----------

    n : int
        Sample size

    p : int
        Number of features

    s : int
        True sparsity

    sigma : float
        Noise level

    rho : float
        Correlation parameter. Must be in interval [0,1] for
        equicorrelated, [-1,1] for AR(1).

    signal : float or (float, float)
        Sizes for the coefficients. If a tuple -- then coefficients
        are equally spaced between these values using np.linspace.

    random_signs : bool
        If true, assign random signs to coefficients.
        Else they are all positive.

    df : int
        Degrees of freedom for noise (from T distribution).

    scale : bool
        Scale columns of design matrix?

    center : bool
        Center columns of design matrix?

    equicorrelated : bool
        Should columns of design be equi-correlated
        or AR(1)?

    Returns
    -------

    X : np.float((n,p))
        Design matrix.

    y : np.float(n)
        Response vector.

    beta : np.float(p)
        True coefficients.

    active : np.int(s)
        Non-zero pattern.

    sigma : float
        Noise level.

    sigmaX : np.ndarray((p,p))
        Row covariance.

    Notes
    -----

    The size of signal is for a "normalized" design, where np.diag(X.T.dot(X)) == np.ones(p).
    If scale=False, this signal is divided by np.sqrt(n), otherwise it is unchanged.

    """

    X, sigmaX = _design(n, p, rho, equicorrelated)[:2]
    X = X + covar_mean
    design = X**order

    if intercept:
        ones = np.ones((X.shape[0], 1))
        # Append the column of ones to the left of X
        design = np.hstack((ones, design))

    # Sample a support of size s for beta
    beta = np.zeros(p+intercept)
    signal = np.atleast_1d(1)
    if signal.shape == (1,):
        beta[intercept:s+intercept] = signal[0]
    else:
        beta[intercept:s+intercept] = np.linspace(signal[0], signal[1], s)
    np.random.shuffle(beta)

    active = np.zeros(p+intercept, np.bool)
    active[beta != 0] = True
    active_idx = np.array(range(p+intercept))[active]

    # Returned pairs are in original indices (starting from 0, no intercept)
    if structure == 'allpairs':
        # sample interactions without replacement
        active_inter_adj, active_inter_list = sample_interaction_pairs(p, s_interaction)
    elif structure == 'weakhierarchy':
        # sample interactions without replacement
        active_inter_adj, active_inter_list \
            = sample_interaction_pairs_hierarchy(p, s_interaction,
                                                 active_idx=active_idx,
                                                 mode=structure, intercept=False)
    elif structure == 'stronghierarchy':
        # sample interactions without replacement
        active_inter_adj, active_inter_list \
            = sample_interaction_pairs_hierarchy(p, s_interaction,
                                                 active_idx=active_idx,
                                                 mode=structure, intercept=False)
    #print(active_inter_list)

    data_interaction = {}
    for i in range(p):
        for j in range(i + 1, p):
            data_interaction[(i, j)] = X[:, i] * X[:, j]
            if center:
                data_interaction[(i, j)] = data_interaction[(i, j)] - np.mean(data_interaction[(i, j)])
            if scale:
                scaling = np.std(data_interaction[(i, j)]) * np.sqrt(n)
                data_interaction[(i, j)] = data_interaction[(i, j)] / scaling

    gamma = np.ones((s_interaction,)) * interaction_signal

    if random_signs:
        beta[active] *= (2 * np.random.binomial(1, 0.5, size=(active.sum(),)) - 1.)
        gamma *= (2 * np.random.binomial(1, 0.5, size=(s_interaction,)) - 1.)
    beta /= np.sqrt(n)
    gamma /= np.sqrt(n)

    if center:
        centering = design.mean(0)[None, :]
        if intercept:
            centering[0] = 0
        design -= centering
        X -= X.mean(0)[None, :]

    if scale:
        # ----SCALE----
        # scales X by sqrt(n) and sd
        # if we need original X, uncomment the following line
        # X_raw = X
        # ----SCALE----
        scaling = design.std(0) * np.sqrt(n)
        if intercept:
            scaling[0] = np.sqrt(n)

        design /= scaling[None, :]
        X /= X.std(0) * np.sqrt(n)
        beta *= np.sqrt(n)
        gamma *= np.sqrt(n)

    # interaction * gamma
    interaction_proj = np.zeros((n,))
    # Construct (dense) design of interactions:
    for i in range(active_inter_list.shape[0]):
        pair = active_inter_list[i]
        interaction_proj += data_interaction[(pair[0], pair[1])] * gamma[i]

    Y_mean = design.dot(beta) + interaction_proj

    noise_sd = (np.std(design.dot(beta) + interaction_proj) /
                np.sqrt(SNR))
    print("noise_sd:", noise_sd)

    Y = Y_mean + np.random.normal(size=(n,), scale=noise_sd)

    return (design, data_interaction, Y, Y_mean, X,
            active, active_inter_adj, active_inter_list)

def gaussian_group_polynomial_interaction_instance(n=500, p=200, s=7, rho=0., order=1,
                                                   covar_mean=0, rho_noise=0,
                                                   full_corr=False,
                                                   SNR = 1, interaction_signal=1,
                                                   random_signs=False,
                                                   scale=True, center=True,
                                                   equicorrelated=True,
                                                   structure=None, s_interaction=0,
                                                   intercept=False):
    """
    A testing instance for the LASSO.
    If equicorrelated is True design is equi-correlated in the population,
    normalized to have columns of norm 1.
    If equicorrelated is False design is auto-regressive.
    For the default settings, a $\lambda$ of around 13.5
    corresponds to the theoretical $E(\|X^T\epsilon\|_{\infty})$
    with $\epsilon \sim N(0, \sigma^2 I)$.

    Parameters
    ----------

    n : int
        Sample size

    p : int
        Number of features

    s : int
        True sparsity

    sigma : float
        Noise level

    rho : float
        Correlation parameter. Must be in interval [0,1] for
        equicorrelated, [-1,1] for AR(1).

    signal : float or (float, float)
        Sizes for the coefficients. If a tuple -- then coefficients
        are equally spaced between these values using np.linspace.

    random_signs : bool
        If true, assign random signs to coefficients.
        Else they are all positive.

    df : int
        Degrees of freedom for noise (from T distribution).

    scale : bool
        Scale columns of design matrix?

    center : bool
        Center columns of design matrix?

    equicorrelated : bool
        Should columns of design be equi-correlated
        or AR(1)?

    Returns
    -------

    X : np.float((n,p))
        Design matrix.

    y : np.float(n)
        Response vector.

    beta : np.float(p)
        True coefficients.

    active : np.int(s)
        Non-zero pattern.

    sigma : float
        Noise level.

    sigmaX : np.ndarray((p,p))
        Row covariance.

    Notes
    -----

    The size of signal is for a "normalized" design, where np.diag(X.T.dot(X)) == np.ones(p).
    If scale=False, this signal is divided by np.sqrt(n), otherwise it is unchanged.

    """

    if intercept:
        groups = [0] + (list(np.arange(p).repeat(order) + 1))
    else:
        groups = list(np.arange(p).repeat(order))
    groups = np.array(groups)
    # Assigning active groups
    group_active = np.random.choice(np.arange(intercept, p + intercept), s, replace=False)
    active = np.isin(groups, group_active)

    if full_corr:
        X, sigmaX = _design(n, p, rho, equicorrelated)[:2]
    else:
        X_signal, sigmaX_signal = _design(n, s, rho, equicorrelated)[:2]
        X_noise, sigmaX_noise = _design(n, p - s, rho_noise, equicorrelated)[:2]

        X = np.zeros((n,p))
        X[:,group_active] = X_signal
        X[:,np.setdiff1d(range(p), group_active).tolist()] = X_noise

    X = X + covar_mean
    design = np.zeros((n, p * order))
    for i in range(p):
        for j in range(order):
            design[:,order*i + j] = X[:,i]**(j+1)

    if intercept:
        ones = np.ones((X.shape[0], 1))
        # Append the column of ones to the left of X
        design = np.hstack((ones, design))

    # Sample a support of size s for beta
    beta = np.zeros(p*order+intercept)
    signal = np.atleast_1d(1)
    if signal.shape == (1,):
        beta[active] = signal[0]
    else:
        beta[active] = np.linspace(signal[0], signal[1], s)

    # Returned pairs are in original indices (starting from 0, no intercept)
    if structure == 'allpairs':
        # sample interactions without replacement
        active_inter_adj, active_inter_list = sample_interaction_pairs(p, s_interaction)
    elif structure == 'weakhierarchy':
        # sample interactions without replacement
        active_inter_adj, active_inter_list \
            = sample_interaction_pairs_hierarchy(p, s_interaction,
                                                 active_idx=group_active,
                                                 mode=structure, intercept=intercept)
    elif structure == 'stronghierarchy':
        # sample interactions without replacement
        active_inter_adj, active_inter_list \
            = sample_interaction_pairs_hierarchy(p, s_interaction,
                                                 active_idx=group_active,
                                                 mode=structure, intercept=intercept)
    #print(active_inter_list)

    data_interaction = {}
    for i in range(p):
        for j in range(i + 1, p):
            data_interaction[(i, j)] = X[:, i] * X[:, j]
            if center:
                data_interaction[(i, j)] = data_interaction[(i, j)] - np.mean(data_interaction[(i, j)])
            if scale:
                scaling = np.std(data_interaction[(i, j)]) * np.sqrt(n)
                data_interaction[(i, j)] = data_interaction[(i, j)] / scaling

    gamma = np.ones((s_interaction,)) * interaction_signal

    if random_signs:
        beta[active] *= (2 * np.random.binomial(1, 0.5, size=(active.sum(),)) - 1.)
        gamma *= (2 * np.random.binomial(1, 0.5, size=(s_interaction,)) - 1.)
    beta /= np.sqrt(n)
    gamma /= np.sqrt(n)

    if center:
        centering = design.mean(0)[None, :]
        if intercept:
            centering[0] = 0
        design -= centering
        X -= X.mean(0)[None, :]

    if scale:
        # ----SCALE----
        # scales X by sqrt(n) and sd
        # if we need original X, uncomment the following line
        # X_raw = X
        # ----SCALE----
        scaling = design.std(0) * np.sqrt(n)
        if intercept:
            scaling[0] = np.sqrt(n)

        design /= scaling[None, :]
        X /= X.std(0) * np.sqrt(n)
        beta *= np.sqrt(n)
        gamma *= np.sqrt(n)

    # interaction * gamma
    interaction_proj = np.zeros((n,))
    # Construct (dense) design of interactions:
    for i in range(active_inter_list.shape[0]):
        pair = active_inter_list[i]
        interaction_proj += data_interaction[(pair[0], pair[1])] * gamma[i]

    Y_mean = design.dot(beta) + interaction_proj

    noise_sd = (np.std(design.dot(beta) + interaction_proj) /
                np.sqrt(SNR))
    print("noise_sd:", noise_sd)

    Y = Y_mean + np.random.normal(size=(n,), scale=noise_sd)

    return (design, data_interaction, Y, Y_mean, X,
            active, active_inter_adj, active_inter_list, groups)