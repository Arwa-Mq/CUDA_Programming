#include <cmath>
#include <vector>

std::vector<double> Boost_Factor_Model(const std::vector<double>& R, double rs, double b0) {
    std::vector<double> B(R.size());

    for (size_t i = 0; i < R.size(); ++i) {
        double x = R[i] / rs;
        double fx = 0.0;

        if (x > 1.0) {
            fx = std::atan(std::sqrt(x * x - 1.0)) / std::sqrt(x * x - 1.0);
        } else if (x == 1.0) {
            fx = 1.0;
        } else {
            fx = std::atanh(std::sqrt(1.0 - x * x)) / std::sqrt(1.0 - x * x);
        }

        double denom = x * x - 1.0;
        if (std::abs(denom) < 1e-10) denom = 1e-10;

        B[i] = 1.0 + b0 * (1.0 - fx) / denom;

        if (std::isnan(B[i])) {
            B[i] = (b0 + 3.0) / 3.0;
        }
    }

    return B;
}
