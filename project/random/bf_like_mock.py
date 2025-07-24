# Arwa's likelihood. 
import numpy as np
from cosmosis.datablock import names

def Boost_Factor_Model(R, rs, b0):
    x = R / rs
    fx = np.zeros_like(x)
    fx[x > 1] = np.arctan(np.sqrt(x[x > 1]**2 - 1)) / np.sqrt(x[x > 1]**2 - 1)
    fx[x == 1] = 1
    fx[x < 1] = np.arctanh(np.sqrt(1 - x[x < 1]**2)) / np.sqrt(1 - x[x < 1]**2)
    #fix the warning error
    denominator = x**2 - 1
    denominator[denominator == 0] = 1e-10  # or some small value
    B = 1 + b0 * (1 - fx) / denominator
    B[np.isnan(B)] = (b0 + 3) / 3
    return B
    
def setup(options):
    # mock data files. 
    R = np.logspace(-1, 2, 100)
    B0 = 1.5
    Rs = 0.5 
    data_B = Boost_Factor_Model(R, Rs, B0)
    variance = np.ones(data_B.size)*0.1**2
    sigma_B =  data_B*np.random.normal(loc=0, scale=variance**(1/2), size=data_B.size)
    #data = data_B + sigma_B
    
    #inv_cov = np.linalg.inv(covariance)     # Invert covariance matrix now to save time later

    # Package config data
    config = {
        'R': R,
        'data_vector': data_B,
        'sigma_B': sigma_B,
        'covariance': variance,
        #'inv_cov': inv_cov
    }

    return config

def execute(block, config):
    R = config['R']
    data_vector = config['data_vector']
    inv_cov = config['covariance'] #config['inv_cov']

    # Read parameter values from the block
    # DO: change l and z according to which data file you are using, also do that in the bf_values.ini file. 
    logrs = block["Boost_Factor_Model_Values", "logrs_00"]
    logb0 = block["Boost_Factor_Model_Values", "logb0_00"]

    rs = 10**logrs
    b0 = 10**logb0

    # Compute model prediction at current parameter values
    model_prediction = Boost_Factor_Model(R, rs, b0)

    diff = model_prediction - data_vector
    # Chi-squared using covariance
    chisq = np.dot(diff, np.dot(inv_cov, diff))
    log_L = -0.5 * chisq

    # Store likelihood in datablock
    block["likelihoods", "boost_factor_likelihood_like"] = log_L

    return 0