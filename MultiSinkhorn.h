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

class MultimarginalSinkhorn {
    private:
    /// Running cost proximal operator
    shared_ptr<BaseProximalOperator> running;
    /// Terminal cost proximal operator
    shared_ptr<BaseProximalOperator> terminal;
    /// Initial marginal
    MatrixXd rho_0;
    /// Kernel
    kernels::KernelPtr kernel;


    public:
    MultimarginalSinkhorn(
        shared_ptr<BaseProximalOperator> running,
        shared_ptr<BaseProximalOperator> terminal,
        kernels::KernelPtr kernel):
    running(running), terminal(terminal), kernel(kernel) {}

    /// Perform one iterate of the multimarginal Sinkhorn algorithm.
    void iterate(std::vector<Ref<MatrixXd>>& potentials);

    void run_sinkhorn(std::vector<Ref<MatrixXd>>& potentials, int num_iterations);

    inline std::vector<MatrixXd> get_marginals(std::vector<Ref<MatrixXd>>& potentials);
};

}
