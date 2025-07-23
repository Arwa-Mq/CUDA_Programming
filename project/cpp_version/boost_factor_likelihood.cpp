#include <cmath>
#include <vector>
#include <iostream>
#include "cosmosis/datablock/datablock.hh"

using namespace std;
using namespace cosmosis;

vector<double> Boost_Factor_Model(const vector<double>& R, double rs, double b0) {
    vector<double> B(R.size());
    for (size_t i = 0; i < R.size(); ++i) {
        double x = R[i] / rs;
        double fx = 0.0;

        if (x > 1.0) {
            fx = atan(sqrt(x * x - 1.0)) / sqrt(x * x - 1.0);
        } else if (fabs(x - 1.0) < 1e-8) {
            fx = 1.0;
        } else {
            fx = atanh(sqrt(1.0 - x * x)) / sqrt(1.0 - x * x);
        }

        double denom = x * x - 1.0;
        if (fabs(denom) < 1e-10) denom = 1e-10;

        B[i] = 1.0 + b0 * (1.0 - fx) / denom;
        if (std::isnan(B[i])) {
            B[i] = (b0 + 3.0) / 3.0;
        }
    }
    return B;
}

int setup(DataBlock* options, DataBlock* config) {
    // Hardcoded path to data (adjust this!)
    string data_path = "/global/cfs/cdirs/des/jesteves/data/boost_factor/y1/profiles";
    string file = data_path + "/full-unblind-v2-mcal-zmix_y1clust_l3_z0_zpdf_boost.dat";
    string covfile = data_path + "/full-unblind-v2-mcal-zmix_y1clust_l3_z0_zpdf_boost_cov.dat";

    FILE* fp = fopen(file.c_str(), "r");
    FILE* fpcov = fopen(covfile.c_str(), "r");

    if (!fp || !fpcov) {
        cerr << "Could not open data files!" << endl;
        return 1;
    }

    vector<double> R, data_vector;
    double r_tmp, b_tmp, sigma_tmp;

    for (int i = 0; i < 8; i++) {
        fscanf(fp, "%lf %lf %lf", &r_tmp, &b_tmp, &sigma_tmp);
        R.push_back(r_tmp);
        data_vector.push_back(b_tmp);
    }

    double cov[8][8];
    for (int i = 0; i < 8; ++i)
        for (int j = 0; j < 8; ++j)
            fscanf(fpcov, "%lf", &cov[i][j]);

    fclose(fp);
    fclose(fpcov);

    config->put_vector("boost", "R", R);
    config->put_vector("boost", "data", data_vector);
    for (int i = 0; i < 8; i++)
        for (int j = 0; j < 8; j++)
            config->put_double("boost_cov", "C_" + to_string(i) + "_" + to_string(j), cov[i][j]);

    return 0;
}

int execute(DataBlock* block, DataBlock* config) {
    vector<double> R, data;
    config->get_array("boost", "R", R);
    config->get_array("boost", "data", data);

    // Get parameters
    double logrs, logb0;
    block->get_val("Boost_Factor_Model_Values", "logrs_30", logrs);
    block->get_val("Boost_Factor_Model_Values", "logb0_30", logb0);

    double rs = pow(10.0, logrs);
    double b0 = pow(10.0, logb0);

    vector<double> model = Boost_Factor_Model(R, rs, b0);

    // Load covariance
    double cov[8][8];
    for (int i = 0; i < 8; i++)
        for (int j = 0; j < 8; j++)
            config->get_val("boost_cov", "C_" + to_string(i) + "_" + to_string(j), cov[i][j]);

    // Compute chi-squared
    double chisq = 0.0;
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            double diff_i = model[i] - data[i];
            double diff_j = model[j] - data[j];
            chisq += diff_i * cov[i][j] * diff_j;
        }
    }

    double logL = -0.5 * chisq;
    block->put_double("likelihoods", "boost_factor_likelihood_like", logL);

    return 0;
}

int cleanup(DataBlock* config) {
    return 0;
}
