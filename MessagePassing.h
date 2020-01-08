#pragma once

#include <memory>
#include <vector>
#include <Eigen/Core>
#include "Kernels.h"

using kernels::KernelPtr;
using namespace Eigen;

namespace messages
{

/**
 * Perform contraction of the dual potentials `potentials` with respect to the
 * Wiener measure of kernel `ker`.
 * 
 * @param potentials Vector of dual potentials
 * @param idx Index of the marginal to leave out.
 * @param ker Convolutional kernel
 * @return 
 */
MatrixXd contract(std::vector<Ref<MatrixXd>>& potentials,
                  const size_t idx, KernelPtr ker);


std::vector<MatrixXd> compute_marginals(std::vector<Ref<MatrixXd>>& potentials, KernelPtr ker);
    
}
