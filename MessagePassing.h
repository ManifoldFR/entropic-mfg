#ifndef MESSAGES_H_
#define MESSAGES_H_
#include <vector>
#include <Eigen/Core>
#include "include/Kernels.h"

using kernels::Kernel;
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
MatrixXd contract(std::vector<MatrixXd>& potentials, const size_t idx, const Kernel& ker);


std::vector<MatrixXd> compute_marginals(std::vector<MatrixXd>& potentials, const Kernel& ker);
    
}





#endif