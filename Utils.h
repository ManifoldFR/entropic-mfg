#ifndef UTILS_H_
#define UTILS_H_
#include <Eigen/Core>
#include "matplotlibcpp.h"

using Eigen::MatrixXd;
namespace plt = matplotlibcpp;

namespace utils
{

#ifndef WITHOUT_NUMPY

/**
 *  Plot the kernel matrix K.
 * 
 */
void plot_kernel(MatrixXd& K, int nx, int ny) {
    std::vector<float> z(ny * nx);
    for (int j=0; j<ny; j++) {
        for (int i=0; i<nx; i++) {
            z.at(nx*j + i) = (float)K(i, j);
        }
    }

    const float* zptr = &(z[0]);
    std::cout << "Making figure" << std::endl;
    const int colors = 1;
    // Figure is defined
    plt::imshow(zptr, ny, nx, colors);
    plt::title("Euclidean heat kernel $K$");
    plt::tight_layout();

    plt::show();

}

inline void plot_kernel(MatrixXd& K, int nx) {
    plot_kernel(K, nx, nx);
}

#endif

}  // namespace utils

#endif