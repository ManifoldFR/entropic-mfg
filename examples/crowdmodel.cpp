#include <Eigen/Core>

#include "MultiSinkhorn.h"
#include "CongestionOperator.h"
#include "Kernels.h"
#include "Utils.h"

using namespace Eigen;
using klprox::CongestionPotentialProx;
using utils::plot_kernel;
typedef kernels::EuclideanHeatKernel<2> Kernel;

int main(int argc, char* argv[]) {

    size_t nx = 101, ny = 101;
    
    double variance = 0.1;
    Kernel ker(nx, nx, 0., 1., 0., 2., variance);

    const double* ker1_data = ker.K1.data();
    plot_kernel(ker1_data, nx);

    return 1;
}

