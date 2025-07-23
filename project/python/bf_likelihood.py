#to be run on a GPU

import numpy as np
import cupy as cp  # New: GPU array lib
from cosmosis.datablock import names

def Boost_Factor_Model_GPU(R, rs, b0):
    # Move inputs to GPU
    R_gpu = cp.asarray(R)
    rs_gpu = cp.asarray(rs)
    b0_gpu = cp.asarray(b0)

    x = R_gpu / rs_gpu
    fx = cp.zeros_like(x)

    fx = cp.where(x > 1, cp.arctan(cp.sqrt(x**2 - 1)) / cp.sqrt(x**2 - 1), fx)
    fx = cp.where(x == 1, 1, fx)
    fx = cp.where(x < 1, cp.arctanh(cp.sqrt(1 - x**2)) / cp.sqrt(1 - x**2), fx)

    denominator = x**2 - 1
    denominator = cp.where(denominator == 0, 1e-10, denominator)

    B = 1 + b0_gpu * (1 - fx) / denominator
    B = cp.where(cp.isnan(B), (b0_gpu + 3) / 3, B)

    return cp.asnumpy(B)  # Return to CPU (NumPy array) for rest of code

def setup(options):
    path = "/global/cfs/cdirs/des/jesteves/data/boost_factor/y1/profiles"
    R, data_vector, sigma_B = np.genfromtxt(path + "/full-unblind-v2-mcal-zmix_y1clust_l{l}_z{z}_zpdf_boost.dat".format(l=3, z=0), unpack=True)
    covariance = np.genfromtxt(path + "/full-unblind-v2-mcal-zmix_y1clust_l{l}_z{z}_zpdf_boost_cov.dat".format(l=3, z=0), unpack=True)

    R = R[:8]
    data_vector = data_vector[:8]
    sigma_B = sigma_B[:8]
    covariance = covariance[:8, :8]

    inv_cov = np.linalg.inv(covariance)

    config = {
        'R': R,
        'data_vector': data_vector,
        'sigma_B': sigma_B,
        'covariance': covariance,
        'inv_cov': inv_cov
    }

    return config

def execute(block, config):
    R = config['R']
    data_vector = config['data_vector']
    inv_cov = config['inv_cov']

    logrs = block["Boost_Factor_Model_Values", "logrs_30"]
    logb0 = block["Boost_Factor_Model_Values", "logb0_30"]

    rs = 10**logrs
    b0 = 10**logb0

    # GPU-accelerated model prediction
    model_prediction = Boost_Factor_Model_GPU(R, rs, b0)

    diff = model_prediction - data_vector
    chisq = np.dot(diff, np.dot(inv_cov, diff))
    log_L = -0.5 * chisq

    block["likelihoods", "boost_factor_likelihood_like"] = log_L

    return 0
