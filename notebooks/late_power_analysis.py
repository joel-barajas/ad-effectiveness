import numpy as np
from effectEstimators import late_estimators as late


# Power estimation analysis evaluation to find lower confidence interval bound to be over zero
# It returns 1 if the lower confidence interval in LATE or LATE lift is greater than zero
def f_sig_eval(pd_in, hold_out, p0_d0, p0_d1, lift_d1, n_total, test_sig):
    # priors
    alpha_0 = 0.50
    beta_0 = 0.50
    # Iterations
    n_burnin = 5000
    n_samples = 100000

    user_counts = late.count_point_est(1 - hold_out,
                                       pd_in,
                                       p0_d0,
                                       p0_d1,
                                       (1 + lift_d1) * p0_d1,
                                       n_total)
    [theta_d1_samples, theta_d0_samples, theta_n_samples, p_sel_samples] = late.effect_mcmcEst(user_counts, n_burnin,
                                                                                               n_samples, alpha_0,
                                                                                               beta_0)
    late_samp = (theta_d1_samples - theta_d0_samples)
    lift_samp = late_samp / theta_d0_samples
    lift_low_ci = np.percentile(lift_samp, 100 * test_sig / 2)
    late_low_ci = np.percentile(late_samp, 100 * test_sig / 2)

    if lift_low_ci > 0 or late_low_ci > 0:
        return 1
    else:
        return 0


# Performs binary search to find the minimum p_sel (probability of targeting)
def search_psel(array, hold_out, p0_d0, p0_d1, lift_d1, n_total, test_sig):
    lower = 0
    upper = len(array) - 1
    upper_val = f_sig_eval(array[upper], hold_out, p0_d0, p0_d1, lift_d1, n_total, test_sig)
    lower_val = f_sig_eval(array[lower], hold_out, p0_d0, p0_d1, lift_d1, n_total, test_sig)

    if upper_val == 0:
        return array[upper], 0.0
    elif lower_val == 1:
        return array[lower], 1.0

    while (upper - lower) > 1:
        x = lower + (upper - lower) // 2
        val = f_sig_eval(array[x], hold_out, p0_d0, p0_d1, lift_d1, n_total, test_sig)
        if val == 0:
            lower = x
        elif val == 1:
            upper = x
        #print([array[lower], array[upper], val])
    return array[upper], 1.0
