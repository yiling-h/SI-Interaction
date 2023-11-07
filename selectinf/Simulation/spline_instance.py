import numpy as np
from selectinf.Simulation.spline import cubic_spline, b_spline

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
                                         nknots = 6, degree = 2,
                                         center=False, scale=True, intercept=True):
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
    Y = Y_mean + np.random.normal(size=(n,))

    active_flag = np.zeros(np.unique(groups).shape[0])
    active_flag[0:10+intercept] = 1.

    return design, Y, Y_mean, groups, active_flag

