#include "MessagePassing.h"
#include "Kernels.h"

#include <vector>
#include <Eigen/Core>

using kernels::KernelPtr;
using namespace Eigen;


namespace messages
{


/// Perform contraction of dual potentials against the Gibbs kernel.
/// Computes the quantity \f$ \sum_{i_1,\ldots,i_N} K_{i_1\ldots i_N} \prod_j \psi_{i_j}\f$.
MatrixXd contract(const std::vector<MatrixXd>& potentials,
                  size_t idx,
                  KernelPtr ker) {
    size_t nx = potentials[0].rows();
    size_t ny = potentials[0].cols();

    size_t k;

    ArrayXXd fwd_(ArrayXXd::Ones(nx, ny));
    for (k = 0; k < idx; k++) {
        fwd_ = (*ker).multiply(potentials[k].array() * fwd_);
    }

    ArrayXXd bwd_(ArrayXXd::Ones(nx, ny));
    for (k = potentials.size() - 1; k > idx; k--) {
        bwd_ = (*ker).multiply(potentials[k].array() * bwd_);
    }
    return fwd_ * bwd_;
}

std::vector<ArrayXXd> compute_marginals(const std::vector<MatrixXd> &potentials, KernelPtr ker)
{
    size_t num_marginals = potentials.size();
    std::vector<ArrayXXd> result(num_marginals);

    for (size_t i = 0; i < num_marginals; i++)
    {
        result[i] = potentials[i].array() * contract(potentials, i, ker).array();
    }

    return result;
}
}
