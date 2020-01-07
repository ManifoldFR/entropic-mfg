#include <Eigen/Core>
#include <iostream>
#include <vector>
#include "include/Kernels.h"
#include "MessagePassing.h"
#include "Utils.h"

using kernels::EuclideanHeatKernel;
using utils::plot_kernel;
typedef EuclideanHeatKernel<2> Kernel;


int main() {
    size_t nx = 31;  // number of points
    size_t ny = 51;  // number of points
    double variance = 0.1;
    Kernel ker(nx, nx, 0., 1., 0., 2., variance);

    auto x = MatrixXd::Ones(nx, ny);
    auto y = ker(x);
    std::cout << "nrows " << y.rows() << ", " << y.cols() << std::endl;

    std::cout << "Hello from main" << std::endl;
    std::printf("Number of dimensions: %d\n", (int)ker.ndim());

#ifndef WITHOUT_NUMPY
    plot_kernel(ker.K1, (int) nx);
#endif
    
    return 0;
}
