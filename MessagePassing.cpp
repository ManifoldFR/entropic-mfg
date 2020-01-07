#include "MessagePassing.h"

#include <vector>
#include <Eigen/Core>
#include <Kernels.h>

using kernels::Kernel;
using namespace Eigen;


MatrixXd contract(std::vector<MatrixXd> potentials, size_t idx, Kernel ker) {
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

    MatrixXd result = A_.array() * B_.array();

    return result;
}
