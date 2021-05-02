#pragma once
/***
 * Generalized multimarginal Sinkhorn algorithm.
 */
#include <Eigen/Core>
#include <Eigen/StdVector>
#include <vector>
#include <memory>

#include "Kernels.h"
#include "operators/KLOperator.h"


using namespace Eigen;


/// Sinkhorn algorithm.
namespace sinkhorn {

using klprox::ProxPtr;
using kernels::KernelPtr;

class MultimarginalSinkhorn {
private:
    /// Initial marginal
    MatrixXd rho_0;
    /// Running cost proximal operator
    ProxPtr running;
    /// Terminal cost proximal operator
    ProxPtr terminal;
    /// Kernel
    KernelPtr kernel;
    /// Dual potentials.
    std::vector<MatrixXd> potentials_;
    /// Values of the Hilbert metric
    std::vector<double> metric_vals_;

public:
    double threshold_ = 1e-8;
    auto getConvMetric() { return metric_vals_; }
    size_t nsteps_;
    void setInitialDistribution(Ref<const MatrixXd>& rho_) {
        rho_0.noalias() = rho_;
    }
    MultimarginalSinkhorn(
        ProxPtr running,
        ProxPtr terminal,
        KernelPtr kernel,
        const MatrixXd &rho
    ) : rho_0(rho), running(running), terminal(terminal), kernel(kernel)
    {
    }

    /// Perform one iterate of the multimarginal Sinkhorn algorithm.
    void iterate();

    /// @brief Run the multi-marginal Sinkhorn solver for the optimal transport problem.
    ///
    /// @param[in] num_iterations maximum number of iterations to perform
    /// @param[out] potentials
    bool solve(size_t num_iterations, std::vector<MatrixXd> &potentials);

    /// Return the marginal densities.
    std::vector<MatrixXd> get_marginals();
};

}
