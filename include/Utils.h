#pragma once

#include "Kernels.h"
#include "matplotlibcpp.h"

using Eigen::MatrixXd;
using kernels::KernelBase;
namespace plt = matplotlibcpp;

namespace utils
{

/**
 *  Plot the kernel matrix K.
 * 
 */
void plot_kernel(const double* kernel_data, int nx) {
    std::vector<float> z(kernel_data, kernel_data+nx*nx);
    
    std::cout << "Making figure" << std::endl;
    const int colors = 1;
    // Figure is defined
    plt::imshow(z.data(), nx, nx, colors);
    plt::title("Euclidean heat kernel $K$");
    plt::tight_layout();

    plt::show();

}

template<typename S = double>
void plot_solution(const Eigen::Matrix<S, -1, -1>& x)
{
    const double *xdata = x.data();
    std::vector<float> z(xdata, xdata+x.rows()*x.cols());
    const int colors = 1;

    plt::figure();
    plt::imshow(z.data(), x.rows(), x.cols(), colors);
    plt::title("Solution");
    plt::show();
}

}  // namespace utils
