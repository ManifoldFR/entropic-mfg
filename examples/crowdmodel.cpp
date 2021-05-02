#include <Eigen/Core>

#include "MultiSinkhorn.h"
#include "operators/CongestionOperator.h"
#include "Utils.h"

using namespace Eigen;
using klprox::CongestionObstacleProx;
using kernels::KernelPtr;
using Kernel = kernels::SeparableEuclideanKernel2D;


int main(int argc, char* argv[]) {
    size_t nx = 51, ny = 51;
    
    const double variance_ = 0.1;
    Kernel ker(nx, nx, 0., 1., 0., 2., variance_);

    std::cout << "Kernel ndim: " << ker.ndim() << std::endl;

    const double* ker1_data = ker.K1_.data();

    {
        utils::plot_kernel(ker1_data, nx);
    }


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

    double congest_max_ = 1.0 * rho_0.maxCoeff();

    KernelPtr kernel_ptr(&ker);

    MatrixXd psi(nx, ny);

    klprox::ProxPtr cong_op(
        new CongestionObstacleProx(mask, congest_max_, psi));

    std::shared_ptr<CongestionObstacleProx> running(
        new CongestionObstacleProx(mask, congest_max_, psi));
    
    std::shared_ptr<CongestionObstacleProx> terminal(running);

    std::cout << "Run Sinkhorn algo" << std::endl;
    sinkhorn::MultimarginalSinkhorn sink(running, terminal, kernel_ptr, rho_0);

    std::cout << "Constructor done." << std::endl;

    size_t nsteps = 20;
    std::vector<MatrixXd> potentials;

    for (size_t n = 0; n < nsteps; n++) {
        potentials.push_back(MatrixXd::Ones(nx, ny));
    }
    bool success = sink.solve(40, potentials);
    if (success) std::cout << "Solver successful." << std::endl;

    auto marginals = sink.get_marginals();
    std::cout << "Marginals computed" << std::endl;


    std::vector<size_t> plot_ids { 0, 5, 10, 15 };
    for (auto k = plot_ids.begin(); k != plot_ids.end(); k++)
    {
        utils::plot_solution(marginals[*k]);
    }

    return 0;
}

