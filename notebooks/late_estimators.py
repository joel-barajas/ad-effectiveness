import numpy as np


# Data input format (all user counts):
# input = [N_{{0,1}C}^0,    - Control non-converters
#          N_{{0,1}C}^1,    - Control converter
#          N_{0S}^0,        - Treatment, no ad exposure, non-converter
#          N_{0S}^1,        - Treatment, no ad exposure, converter
#          N_{1S}^0,        - Treatment, ad exposure, non-converter
#          N_{1S}^1,        - Treatment, ad exposure, converter
#          ]

# Method of moments for probabilities that we know
def prob_point_init(in_counts):
    p0_d0 = in_counts[3] / sum(in_counts[2:4])
    p1_d1 = in_counts[5] / sum(in_counts[4:6])
    pd = sum(in_counts[4:6]) / sum(in_counts[2:6])

    p0_camp = in_counts[1] / sum(in_counts[0:2])
    p1_camp = (in_counts[3] + in_counts[5]) / sum(in_counts[2:6])
    p0_d1 = (p0_camp - p0_d0 * (1 - pd)) / pd
    return p1_d1, p0_d1, p0_d0, pd, p0_camp, p1_camp


# Counter estimation given a total population, conversion probabilities and probability of treatment
def count_point_est(p_treat, p_sel, p_d0, p0_d1, p1_d1, n_total):
    n_treat = int(n_total * p_treat)
    n_control = n_total - n_treat

    user_counts = np.zeros(6)
    user_counts[1] = np.ceil(n_control * (p0_d1 * p_sel + p_d0 * (1 - p_sel)))
    user_counts[0] = n_control - user_counts[1]

    n_treat_d1 = int(n_treat * p_sel)
    n_treat_d0 = n_treat - n_treat_d1

    user_counts[3] = np.ceil(n_treat_d0 * p_d0)
    user_counts[2] = n_treat_d0 - user_counts[3]

    user_counts[5] = np.ceil(n_treat_d1 * p1_d1)
    user_counts[4] = n_treat_d1 - user_counts[5]
    return user_counts


# Sampling counters given a total population, conversion probabilities and probability of treatment
def count_sample(p_treat, p_sel, p_d0, p0_d1, p1_d1, n_total):
    n_treat = np.random.binomial(n=n_total,p=p_treat)
    n_control = n_total - n_treat

    user_counts = np.zeros(6)
    user_counts[1] = np.random.binomial(n=n_control, p=(p0_d1 * p_sel + p_d0 * (1 - p_sel)))
    user_counts[0] = n_control - user_counts[1]

    n_treat_d1 = np.random.binomial(n=n_treat, p=p_sel)
    n_treat_d0 = n_treat - n_treat_d1

    user_counts[3] = np.random.binomial(n=n_treat_d0, p=p_d0)
    user_counts[2] = n_treat_d0 - user_counts[3]

    user_counts[5] = np.random.binomial(n=n_treat_d1, p=p1_d1)
    user_counts[4] = n_treat_d1 - user_counts[5]
    return user_counts


# Late Method of moments assuming the mixture of potentially exposed users (exposed if assigned) or compliers
# and never-exposed users or defiers in CACE notation
# See Bayesian Data Analysis Gelman page 224
# or Appendix C http://alumni.soe.ucsc.edu/~jbarajas/publications/paper_MarketingScience.pdf
def effect_pointEst(in_counts):
    [p1_d1, p0_d1, p0_d0, pd, p0_camp, p1_camp] = prob_point_init(in_counts)
    late = p1_d1 - p0_d1
    late_lift = (p1_d1 - p0_d1) / p0_d1
    itt = p1_camp - p0_camp
    return late, late_lift, pd, itt


def effect_mcmcEst(in_counts, N_burnin, N_samples, alpha_0, beta_0):
    # Initialize with point estimate
    [theta_d1, theta_d0, theta_n, p_sel, p0_camp, p1_camp] = prob_point_init(in_counts)

    # Initialization of exposed label for control
    N_d1 = sum(in_counts[4:6])
    N_n1 = sum(in_counts[2:4])
    N_n1_Conv = in_counts[3]
    N_n1_nConv = in_counts[2]

    # Probability of p_sel in control (exposed-if-assigned users)
    Pd_d0_Conv = p_sel * theta_d0 / (p_sel * theta_d0 + (1 - p_sel) * theta_n)
    Pd_d0_nConv = p_sel * (1 - theta_d0) / (p_sel * (1 - theta_d0) + (1 - p_sel) * (1 - theta_n))

    # Sampling counts in control (# of those who would have seen exposed in treatment but are in control)
    N_d0_Conv = np.random.binomial(n=in_counts[1], p=Pd_d0_Conv)
    N_d0_nConv = np.random.binomial(n=in_counts[0], p=Pd_d0_nConv)
    N_n0_Conv = in_counts[1] - N_d0_Conv
    N_n0_nConv = in_counts[0] - N_d0_nConv

    # Just screen printing purposes
    prop = 1.5
    k = 1

    # ----------Burn in iterations ----------------------------
    for s in range(N_burnin):
        if s >= round(prop * k * N_burnin):
            k += 1
            print(s)

        # Beta posterior distribution and sampling for p_sel
        # p_sel = prob of qualified P(D=1)
        alpha_p_sel = alpha_0 + N_d1 + N_d0_Conv + N_d0_nConv
        beta_p_sel = beta_0 + N_n1 + N_n0_Conv + N_n0_nConv
        p_sel = np.random.beta(a=alpha_p_sel, b=beta_p_sel)

        # Beta posterior distribution and sampling for theta_d1
        # theta_d1 = P(Y=1|D=1,Z=1) -- Fixed during the iterations
        alpha_d1 = alpha_0 + in_counts[5]
        beta_d1 = beta_0 + in_counts[4]
        theta_d1 = np.random.beta(a=alpha_d1, b=beta_d1)

        # Beta posterior distribution and sampling for theta_d0
        # theta_d0 = P(Y=1 | D = 1, Z = 0)
        alpha_d0 = alpha_0 + N_d0_Conv
        beta_d0 = beta_0 + N_d0_nConv
        theta_d0 = np.random.beta(a=alpha_d0, b=beta_d0)

        # Beta posterior distribution and sampling for theta_n
        # theta_n = P(Y=1|D=0,Z=0) = P(Y=1|D=0,Z=1) = P(Y=1|D=0)
        alpha_n = alpha_0 + N_n0_Conv + N_n1_Conv
        beta_n = beta_0 + N_n0_nConv + N_n1_nConv
        theta_n = np.random.beta(a=alpha_n, b=beta_n)

        # Finding probabilities of converter/non-converters and exposed-if-assigned users in control
        Pd_d0_Conv = p_sel * theta_d0 / (p_sel * theta_d0 + (1 - p_sel) * theta_n)
        Pd_d0_nConv = p_sel * (1 - theta_d0) / (p_sel * (1 - theta_d0) + (1 - p_sel) * (1 - theta_n))

        # Sampling by the unobserved user counts in control
        N_d0_Conv = np.random.binomial(n=in_counts[1], p=Pd_d0_Conv)
        N_d0_nConv = np.random.binomial(n=in_counts[0], p=Pd_d0_nConv)
        N_n0_Conv = in_counts[1] - N_d0_Conv
        N_n0_nConv = in_counts[0] - N_d0_nConv

    # Actual samples after burnin
    p_sel_samples = np.zeros(N_samples)
    theta_d1_samples = np.zeros(N_samples)
    theta_d0_samples = np.zeros(N_samples)
    theta_n_samples = np.zeros(N_samples)

    prop = 1.5
    k = 1

    for s in range(N_samples):
        if s >= round(prop * k * N_samples):
            k += 1
            print(s)

        # Beta posterior distribution and sampling for p_sel
        # p_sel = prob of qualified P(D=1)
        alpha_p_sel = alpha_0 + N_d1 + N_d0_Conv + N_d0_nConv
        beta_p_sel = beta_0 + N_n1 + N_n0_Conv + N_n0_nConv
        p_sel = np.random.beta(a=alpha_p_sel, b=beta_p_sel)

        # Beta posterior distribution and sampling for theta_d1
        # theta_d1 = P(Y=1|D=1,Z=1) -- Fixed during the iterations
        alpha_d1 = alpha_0 + in_counts[5]
        beta_d1 = beta_0 + in_counts[4]
        theta_d1 = np.random.beta(a=alpha_d1, b=beta_d1)

        # Beta posterior distribution and sampling for theta_d0
        # theta_d0 = P(Y=1 | D = 1, Z = 0)
        alpha_d0 = alpha_0 + N_d0_Conv
        beta_d0 = beta_0 + N_d0_nConv
        theta_d0 = np.random.beta(a=alpha_d0, b=beta_d0)

        # Beta posterior distribution and sampling for theta_n
        # theta_n = P(Y=1|D=0,Z=0) = P(Y=1|D=0,Z=1) = P(Y=1|D=0)
        alpha_n = alpha_0 + N_n0_Conv + N_n1_Conv
        beta_n = beta_0 + N_n0_nConv + N_n1_nConv
        theta_n = np.random.beta(a=alpha_n, b=beta_n)

        # Finding probabilities of converter/non-converters and exposed-if-assigned users in control
        Pd_d0_Conv = p_sel * theta_d0 / (p_sel * theta_d0 + (1 - p_sel) * theta_n)
        Pd_d0_nConv = p_sel * (1 - theta_d0) / (p_sel * (1 - theta_d0) + (1 - p_sel) * (1 - theta_n))

        # Sampling by the unobserved user counts in control
        N_d0_Conv = np.random.binomial(n=in_counts[1], p=Pd_d0_Conv)
        N_d0_nConv = np.random.binomial(n=in_counts[0], p=Pd_d0_nConv)
        N_n0_Conv = in_counts[1] - N_d0_Conv
        N_n0_nConv = in_counts[0] - N_d0_nConv

        p_sel_samples[s] = p_sel
        theta_d1_samples[s] = theta_d1
        theta_d0_samples[s] = theta_d0
        theta_n_samples[s] = theta_n

    return theta_d1_samples, theta_d0_samples, theta_n_samples, p_sel_samples


# Function to compute basic metrics with confidence intervals
# Default percentiles are [5, 50, 95]
def find_metrics(theta_d1_samples, theta_d0_samples, theta_n_samples, p_sel_samples, in_counts, perc_vect=[5, 50, 95]):
    # Lift estimation
    lift_perc = 100 * (theta_d1_samples - theta_d0_samples) / theta_d0_samples
    CI_lift = np.percentile(lift_perc, perc_vect)

    # % of conversions attribution in study group
    ATTR_samp = 100 * (theta_d1_samples - theta_d0_samples) * sum(in_counts[4:6]) / (in_counts[5])
    CI_ATTR = np.percentile(ATTR_samp, perc_vect)
    return CI_lift, CI_ATTR

