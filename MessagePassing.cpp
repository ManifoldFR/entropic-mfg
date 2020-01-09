#include "MessagePassing.h"
#include "Kernels.h"

#include <omp.h>
#include <vector>
#include <Eigen/Core>

using kernels::KernelPtr;
using namespace Eigen;


namespace messages
{


MatrixXd contract(std::vector<Ref<MatrixXd>>& potentials, const int idx, KernelPtr ker) {
    size_t nx = potentials[0].rows();
    size_t ny = potentials[0].cols();

    MatrixXd A_ = MatrixXd::Ones(nx, ny);
    for (int k=0; k < idx; k++) {
        A_ = ker->operator()(potentials[k].array() * A_.array());
    }


    MatrixXd B_ = MatrixXd::Ones(nx, ny);
    for (int k=potentials.size()-1; k > idx; k--) {
        B_ = ker->operator()(potentials[k].array() * B_.array());
    }
    return A_.array() * B_.array();
}


std::vector<MatrixXd> compute_marginals(std::vector<Ref<MatrixXd>>& potentials,
                                        KernelPtr ker) {
    size_t num_marginals = potentials.size();
    std::vector<MatrixXd> result(num_marginals);

    #pragma omp parallel for
    for (int i=0; i < num_marginals; i++) {
        result[i] = potentials[i].array() * contract(potentials, i, ker).array();
    }

    return result;

}

}
