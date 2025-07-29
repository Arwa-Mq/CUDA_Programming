# I write this project as a part of the CUDA course application. 
'''
this code is considring the redshifts and richenecs bins
I didn't test this yet...
'''

import numpy as np
import cupy as cp
import os

from cosmosis.datablock import names

def Boost_Factor_Model_GPU(R, rs, b0):
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

    return cp.asnumpy(B)

def setup(options):
    path = "/global/cfs/cdirs/des/jesteves/data/boost_factor/y1/profiles"

    # Parse all desired l/z bins from options (or hardcode a list for now)
    lz_bins = [(3, 0), (3, 1), (4, 0)]  # <-- update this list as needed

    all_data = {}
    for l, z in lz_bins:
        R, data_vector, sigma_B = np.genfromtxt(
            f"{path}/full-unblind-v2-mcal-zmix_y1clust_l{l}_z{z}_zpdf_boost.dat", unpack=True)
        covariance = np.genfromtxt(
            f"{path}/full-unblind-v2-mcal-zmix_y1clust_l{l}_z{z}_zpdf_boost_cov.dat", unpack=True)

        R = R[:8]
        data_vector = data_vector[:8]
        sigma_B = sigma_B[:8]
        covariance = covariance[:8, :8]
        inv_cov = np.linalg.inv(covariance)

        all_data[(l, z)] = {
            'R': R,
            'data_vector': data_vector,
            'sigma_B': sigma_B,
            'covariance': covariance,
            'inv_cov': inv_cov
        }

    return all_data

def execute(block, config):
    total_logL = 0.0

    for (l, z), data in config.items():
        try:
            logrs = block["Boost_Factor_Model_Values", f"logrs_{l}{z}"]
            logb0 = block["Boost_Factor_Model_Values", f"logb0_{l}{z}"]
        except:
            continue  # skip if this bin isn't included in the parameter file

        rs = 10**logrs
        b0 = 10**logb0

        model_prediction = Boost_Factor_Model_GPU(data['R'], rs, b0)
        diff = model_prediction - data['data_vector']
        chisq = np.dot(diff, np.dot(data['inv_cov'], diff))
        logL = -0.5 * chisq
        total_logL += logL

    block["likelihoods", "boost_factor_likelihood_like"] = total_logL
    return 0
