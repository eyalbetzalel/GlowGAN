import numpy as np
def kld(p, q, revFlag=False):

    # Original Calculation : (1.0 / len(p)) * sum((np.log(p[i] / q[i]) for i in range(len(p)))

    # Variance calculation using Jack-Knife Resampling method :

    # Step 1 - Calculate xi :

    p_np = np.asarray(p)
    q_np = np.asarray(q)
    if revFlag:
        # Since we sample from p and want to estimate the reverse KL over q , we need to multiply by factor due to
        # importance sampling.
        log_vec = (1.0 / (len(p) - 1)) * (q_np/p_np)
        # log_vec = (1.0 / (len(p) - 1)) * np.log(q_np / p_np) * (q_np / p_np)
    else:
        log_vec = (1.0 / (len(p) - 1)) * np.log(p_np / q_np)

    xi = [0] * len(p)

    for i in range(len(p)):
        xi[i] = np.sum(np.delete(log_vec, i))

    # Step 2 - Calculate Mean and VAR :

    mu = (1.0 / len(p)) * sum((xi[i]) for i in range(len(p)))
    var = ((len(p)-1) / len(p)) * sum(((xi[i]-mu) ** 2) for i in range(len(p)))

    return mu, var
def tvd(p, q):

    # Total Variation Distance Calculation
    # Original Calculation : (1.0 / len(p)) * sum(0.5 * np.abs(p[i]/q[i] - 1) for i in range(len(p)))
    # Variance calculation using Jack-Knife Resampling method.

    # Step 1 - Calculate xi :

    p_np = np.asarray(p)
    q_np = np.asarray(q)
    tvd_vec = (1.0 / (len(p) - 1)) * (0.5 * np.abs((p_np/q_np) - 1))
    xi = [0] * len(p)

    for i in range(len(p)):
        xi[i] = np.sum(np.delete(tvd_vec, i))

    # Step 2 - Calculate Mean and VAR :

    mu = (1.0 / len(p)) * sum((xi[i]) for i in range(len(p)))
    var = ((len(p) - 1) / len(p)) * sum(((xi[i] - mu) ** 2) for i in range(len(p)))

    return mu, var
def chi2_pearson(p,q):
    # chi square divergence
    # Estimate is 1/m \sum_i dQn(zi) / dP(zi) - 1 with zi ~ Qn.
    # squared Hellinger distance
    # Estimate is 2 - 2 / m \sum_i exp(0.5 log (dP(zi) / dQn(zi))), zi ~ Qn.
    p_np = np.asarray(p)
    q_np = np.asarray(q)
    log_vec = (1.0 / (len(p) - 1)) * ((p_np / q_np)**2-1)
    xi = [0] * len(p)

    for i in range(len(p)):
        xi[i] = np.sum(np.delete(log_vec, i))

    # Step 2 - Calculate Mean and VAR :

    mu = (1.0 / len(p)) * sum((xi[i]) for i in range(len(p)))
    var = ((len(p) - 1) / len(p)) * sum(((xi[i] - mu) ** 2) for i in range(len(p)))

    return mu, var
def alphadiv(p,q,alpha):

    p_np = np.asarray(p)
    q_np = np.asarray(q)
    t = (p_np / q_np)
    log_vec = (1.0 / (len(p) - 1)) * (np.power(t, alpha) - t) / (alpha * (alpha - 1))
    xi = [0] * len(p)

    for i in range(len(p)):
        xi[i] = np.sum(np.delete(log_vec, i))

    # Step 2 - Calculate Mean and VAR :

    mu = (1.0 / len(p)) * sum((xi[i]) for i in range(len(p)))
    var = ((len(p) - 1) / len(p)) * sum(((xi[i] - mu) ** 2) for i in range(len(p)))

    return mu, var
def hsq(p,q):
    # squared Hellinger distance
    # Estimate is 2 - 2 / m \sum_i exp(0.5 log (dP(zi) / dQn(zi))), zi ~ Qn.
    p_np = np.asarray(p)
    q_np = np.asarray(q)
    log_vec = 2 * (1 - (1.0 / (len(p) - 1)) * np.exp(0.5 * np.log(q_np / p_np)))
    xi = [0] * len(p)

    for i in range(len(p)):
        xi[i] = np.sum(np.delete(log_vec, i))

    # Step 2 - Calculate Mean and VAR :

    mu = (1.0 / len(p)) * sum((xi[i]) for i in range(len(p)))
    var = ((len(p) - 1) / len(p)) * sum(((xi[i] - mu) ** 2) for i in range(len(p)))

    return mu, var
