#pragma once

#include <vector>
#include <Eigen/Core>
#include "Kernels.h"

using kernels::BaseKernel;
using namespace Eigen;

namespace algorithms
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
MatrixXd contract(std::vector<MatrixXd>& potentials, const size_t idx, const BaseKernel& ker);


std::vector<MatrixXd> compute_marginals(std::vector<MatrixXd>& potentials, const BaseKernel& ker);
    
}
