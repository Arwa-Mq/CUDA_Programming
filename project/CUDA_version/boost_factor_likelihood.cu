#include <cmath>
#include <cstdio>
#include <vector>
#include <string>
#include <iostream>
#include <cuda_runtime.h>

#include "cosmosis/datablock/datablock.hh"

using namespace std;
using namespace cosmosis;

__global__
void boost_factor_kernel(const double* R, double* B, double rs, double b0, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

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

    double B_i = 1.0 + b0 * (1.0 - fx) / denom;
    if (isnan(B_i)) {
        B_i = (b0 + 3.0) / 3.0;
    }

    B[i] = B_i;
}

void run_boost_factor_gpu(const vector<double>& R_cpu, double rs, double b0, vector<double>& B_out) {
    int N = R_cpu.size();
    double *R_d, *B_d;

    cudaMalloc(&R_d, N * sizeof(double));
    cudaMalloc(&B_d, N * sizeof(double));

    cudaMemcpy(R_d, R_cpu.data(), N * sizeof(double), cudaMemcpyHostToDevice);

    int threads = 128;
    int blocks = (N + threads - 1) / threads;
    boost_factor_kernel<<<blocks, threads>>>(R_d, B_d, rs, b0, N);

    cudaMemcpy(B_out.data(), B_d, N * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(R_d);
    cudaFree(B_d);
}

int setup(DataBlock* options, DataBlock* config) {
    // Read data from disk
    string path = "/global/cfs/cdirs/des/jesteves/data/boost_factor/y1/profiles";
    FILE* fp = fopen((path + "/full-unblind-v2-mcal-zmix_y1clust_l3_z0_zpdf_boost.dat").c_str(), "r");
    FILE* fpcov = fopen((path + "/full-unblind-v2-mcal-zmix_y1clust_l3_z0_zpdf_boost_cov.dat").c_str(), "r");

    if (!fp || !fpcov) {
        cerr << "ERROR: Data file not found.\n";
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

    double logrs, logb0;
    block->get_val("Boost_Factor_Model_Values", "logrs_30", logrs);
    block->get_val("Boost_Factor_Model_Values", "logb0_30", logb0);

    double rs = pow(10.0, logrs);
    double b0 = pow(10.0, logb0);

    vector<double> model(R.size());
    run_boost_factor_gpu(R, rs, b0, model);

    // Covariance
    double cov[8][8];
    for (int i = 0; i < 8; i++)
        for (int j = 0; j < 8; j++)
            config->get_val("boost_cov", "C_" + to_string(i) + "_" + to_string(j), cov[i][j]);

    // Chi-squared
    double chisq = 0.0;
    for (int i = 0; i < 8; ++i)
        for (int j = 0; j < 8; ++j)
            chisq += (model[i] - data[i]) * cov[i][j] * (model[j] - data[j]);

    double logL = -0.5 * chisq;
    block->put_double("likelihoods", "boost_factor_likelihood_like", logL);

    return 0;
}

int cleanup(DataBlock* config) {
    return 0;
}
