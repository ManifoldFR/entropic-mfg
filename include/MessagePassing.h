#pragma once

#include "Kernels.h"

#include <vector>
#include <Eigen/Core>

using kernels::KernelPtr;
using namespace Eigen;


namespace messages
{


/**
 * Perform contraction of the dual potentials `potentials` with respect to the
 * measure kernel `ker`.
 * 
 * @param potentials Vector of dual potentials
 * @param idx Index of the marginal to leave out.
 * @param ker Convolutional kernel
 * @return 
 */
MatrixXd contract(const std::vector<MatrixXd>& potentials,
                  size_t idx,
                  KernelPtr ker);


std::vector<ArrayXXd> compute_marginals(const std::vector<MatrixXd>& potentials, KernelPtr ker);

   
}
