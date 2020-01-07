#include <Eigen/Core>
#include <iostream>
#include <vector>
#include "Kernels.h"
#include "MessagePassing.h"
#include "matplotlibcpp.h"

using kernels::EuclideanHeatKernel;
typedef EuclideanHeatKernel<2> Kernel;

namespace plt = matplotlibcpp;


#ifndef WITHOUT_NUMPY
void plot_kernel(MatrixXd& K, int nx) {
    std::vector<float> z(nx * nx);
    for (int j=0; j<nx; j++) {
        for (int i=0; i<nx; i++) {
            z.at(nx*j + i) = (float)K(i, j);
        }
    }

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
