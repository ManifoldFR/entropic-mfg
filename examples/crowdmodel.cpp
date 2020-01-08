#include <Eigen/Core>

#include "MultiSinkhorn.h"
#include "CongestionOperator.h"
#include "Kernels.h"
#include "Utils.h"

using namespace Eigen;
using klprox::CongestionObstacleProx;
using std::shared_ptr;
using kernels::KernelPtr;
typedef kernels::EuclideanHeatKernel<2> Kernel;

int main(int argc, char* argv[]) {
    size_t nx = 101, ny = 101;
    
    double variance = 0.1;
    Kernel ker(nx, nx, 0., 1., 0., 2., variance);

    std::cout << "Kernel ndim: " << ker.ndim() << std::endl;

    const double* ker1_data = ker.K1.data();
    utils::plot_kernel(ker1_data, nx);



    ArrayXXi mask(nx, ny);
    for (size_t j=0; j<ny / 3; j++) {
        for (size_t i=40; i<50; i++) {
            mask(i, j) = 1.;
        }
    }

    MatrixXd rho_0(nx, ny);
    for (size_t j=10; j<20; j++) {
        for (size_t i=10; i<20; i++) {
            mask(i, j) = 1.;
        }
    }
    rho_0 = rho_0.array() / rho_0.sum();

    double congest_max = 1.0 * rho_0.maxCoeff();

    KernelPtr kernel_ptr(&ker);

    MatrixXd psi(nx, ny);

    klprox::BaseProximalOperator* pr = new CongestionObstacleProx(mask, congest_max, psi);

    shared_ptr<CongestionObstacleProx> running(
        new CongestionObstacleProx(mask, congest_max, psi));
    
    shared_ptr<CongestionObstacleProx> terminal(running);

    sinkhorn::MultimarginalSinkhorn sink(running, terminal, kernel_ptr);


    return 1;
}

