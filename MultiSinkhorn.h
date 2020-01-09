#pragma once
/***
 * Generalized multimarginal Sinkhorn algorithm.
 */
#include <Eigen/Core>
#include <vector>
#include <memory>

#include "Kernels.h"
#include "KLOperator.h"


using namespace Eigen;
using std::shared_ptr;



/// Sinkhorn algorithm.
namespace sinkhorn {

using klprox::BaseProximalOperator;
using klprox::ProxPtr;

class MultimarginalSinkhorn {
    private:
    /// Running cost proximal operator
    ProxPtr running;
    /// Terminal cost proximal operator
    ProxPtr terminal;
    /// Initial marginal
    MatrixXd rho_0;
    /// Kernel
    kernels::KernelPtr kernel;


    public:
    void setInitialDistribution(Ref<const MatrixXd>& rho_) {
        rho_0 = rho_;
    }
    MultimarginalSinkhorn(
        ProxPtr running,
        ProxPtr terminal,
        kernels::KernelPtr kernel,
        MatrixXd& rho):
    running(running), terminal(terminal), kernel(kernel), rho_0(rho) {}

    /// Perform one iterate of the multimarginal Sinkhorn algorithm.
    void iterate(std::vector<Ref<MatrixXd>>& potentials);

    void run(std::vector<Ref<MatrixXd>>& potentials, int num_iterations);

    std::vector<MatrixXd> get_marginals(std::vector<Ref<MatrixXd>>& potentials);
};

}
