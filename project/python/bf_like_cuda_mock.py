# this like can be run on GPU using CUDA..
######## it is using a mock data ######## 
######## the code been tested and giving a good results. 
import cupy as cp
from cosmosis.datablock import names

def Boost_Factor_Model(R, rs, b0):
    x = R / rs
    fx = cp.zeros_like(x)

    # Avoid dividing by 0 by masking regions
    gt1 = x > 1
    eq1 = cp.isclose(x, 1)
    lt1 = x < 1

    fx[gt1] = cp.arctan(cp.sqrt(x[gt1]**2 - 1)) / cp.sqrt(x[gt1]**2 - 1)
    fx[eq1] = 1.0
    fx[lt1] = cp.arctanh(cp.sqrt(1 - x[lt1]**2)) / cp.sqrt(1 - x[lt1]**2)

    denominator = x**2 - 1
    denominator = cp.where(cp.isclose(denominator, 0.0), 1e-10, denominator)

    B = 1 + b0 * (1 - fx) / denominator
    B = cp.where(cp.isnan(B), (b0 + 3) / 3.0, B)
    return B

def setup(options):
    R = cp.logspace(-1, 2, 100)
    B0 = 0.3
    Rs = 0.5
    data_B = Boost_Factor_Model(R, Rs, B0)

    variance = cp.ones(data_B.size) * 0.1**2
    sigma_B = data_B * cp.random.normal(loc=0, scale=cp.sqrt(variance))

    config = {
        'R': R,
        'data_vector': data_B,
        'sigma_B': sigma_B,
        'covariance': variance,
    }

    return config

def execute(block, config):
    R = config['R']
    data_vector = config['data_vector']
    sigma_B = config['sigma_B']

    logrs = block["Boost_Factor_Model_Values", "logrs_00"]
    logb0 = block["Boost_Factor_Model_Values", "logb0_00"]

    rs = 10**logrs
    b0 = 10**logb0

    model_prediction = Boost_Factor_Model(R, rs, b0)

    chisq = cp.sum(((model_prediction - data_vector) / sigma_B) ** 2)
    log_L = -0.5 * chisq

    # Move scalar back to CPU for CosmoSIS
    #block["likelihoods", "boost_factor_likelihood_like"] = log_L.get()
    block["likelihoods", "boost_factor_likelihood_like"] = float(log_L.get())

    return 0
