#pragma once

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <Eigen/SparseCholesky>
/**
 * Define discretized Laplacian operator on a retangular grid.
 * 
 * 
 */

using namespace Eigen;

namespace laplacian {


/// Laplacian operator on a rectangular grid with the given mask
void build_matrix(const ArrayXXi& mask) {
    size_t nx = mask.rows();
    size_t ny = mask.cols();
}

}
