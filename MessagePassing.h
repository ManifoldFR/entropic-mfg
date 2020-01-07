#ifndef MESSAGES_H_
#define MESSAGES_H_

#include <vector>
#include <Eigen/Core>
#include <Kernels.h>

using namespace Eigen;


/// @param potentials - Dual potentials
/// @param ker - Convolutional kernel
MatrixXd contract(std::vector<MatrixXd> potentials, size_t idx, kernels::Kernel ker);



#endif