#pragma once

#include "Kernels.h"
#include "matplotlibcpp.h"

using Eigen::MatrixXd;
using kernels::BaseKernel;
namespace plt = matplotlibcpp;

namespace utils
{

#ifndef WITHOUT_NUMPY

/**
 *  Plot the kernel matrix K.
 * 
 */
void plot_kernel(const double* kernel_data, int nx) {
    std::vector<float> z(kernel_data, kernel_data+nx*nx);
    
    const float* zptr = &(z[0]);
    std::cout << "Making figure" << std::endl;
    const int colors = 1;
    // Figure is defined
    plt::imshow(zptr, nx, nx, colors);
    plt::title("Euclidean heat kernel $K$");
    plt::tight_layout();

    plt::show();
}

#endif

}  // namespace utils
