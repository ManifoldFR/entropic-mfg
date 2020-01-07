#include <Eigen/Core>
#include <iostream>
#include <vector>
#include "Kernels.h"
#include "MessagePassing.h"
#include "matplotlibcpp.h"

using kernels::EuclideanHeatKernel;
typedef EuclideanHeatKernel<2> Kernel;

namespace plt = matplotlibcpp;


int main() {
    size_t nx = 16;  // number of points
    double variance = 0.1;
    Kernel ker(nx, nx, 0., 1., 0., 1., variance);

    std::cout << "Hello from main" << std::endl;
    std::printf("Number of dimensions: %d\n", (int)ker.ndim());
    std::cout << ker.K1 << std::endl;

    std::vector<float> z(nx * nx);
    for (int j=0; j<nx; j++) {
        for (int i=0; i<nx; i++) {
            z.at(nx*j + i) = (float)ker.K1(i, j);
        }
    }

    const float* zptr = &(z[0]);
    std::cout << "Making figure" << std::endl;
    // plt::title("Heat kernel on dimension 1");
    const int color = 1;
    plt::imshow(zptr, nx, nx, color);
    plt::xlim(0., 1.);
    plt::ylim(0., 1.);

    plt::show("heatkernel.png");

    return 0;
}
