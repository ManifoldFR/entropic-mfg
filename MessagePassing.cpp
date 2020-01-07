#include "MessagePassing.h"
#include "Kernels.h"

#include <vector>
#include <Eigen/Core>

using kernels::BaseKernel;
using namespace Eigen;


namespace algorithms
{


MatrixXd contract(std::vector<MatrixXd>& potentials, size_t idx, const BaseKernel& ker) {
    size_t nx = potentials[0].rows();
    size_t ny = potentials[0].cols();
    MatrixXd A_ = MatrixXd::Ones(nx, ny);
    for (size_t k=0; k < idx - 1; k++) {
        A_ = ker(potentials[k].array() * A_.array());
    }

    MatrixXd B_ = MatrixXd::Ones(nx, ny);
    for (size_t k=potentials.size()-1; k > idx + 1; k--) {
        B_ = ker(potentials[k].array() * A_.array());
    }

    return A_.array() * B_.array();
}


std::vector<MatrixXd> compute_marginals(std::vector<MatrixXd>& potentials, const BaseKernel& ker) {
    size_t num_marginals = potentials.size();
    std::vector<MatrixXd> result(num_marginals);

    for (size_t i=0; i < num_marginals; i++) {
        result[i] = potentials[i].array() * contract(potentials, i, ker).array();
    }

    return result;

}

}
