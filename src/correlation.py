
from variance import covariance

def correlation(vi, vj):
    # v_i standard deviation = sqrt(vi co-variance)
    vi_std_div = (covariance(vi) ** (1/2))
    # v_j standard deviation
    vj_std_div = (covariance(vj) ** (1/2))
    # co-variance of vi and vj (v_ij)
    covar_vij = covariance(vi, vj)
    # calculate correlation of vi and vj
    return (covar_vij / (vi_std_div * vj_std_div))