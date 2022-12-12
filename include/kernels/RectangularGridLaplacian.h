#pragma once

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <Eigen/SparseCholesky>


using namespace Eigen;

namespace kernels
{
    namespace laplacian
    {

    /// Define a matrix for the 1D Laplacian using the central
    /// differences stencil.
    auto make_laplacian_matrix1d(size_t nx, double h)
    {
        double h_2 = h * h;
        SparseMatrix<double> out(nx, nx);

        for (size_t i = 0; i < nx; i++) {
            out.coeffRef(i, i) = 2. / h_2;
            if (i > 0) {
                out.coeffRef(i, i-1) = 1. / h_2;
            }
            if (i < nx) {
                out.coeffRef(i, i+1) = 1. / h_2;
            }
        }
        out.makeCompressed();
        return out;
    }

    /// Given a mask of the discretization grid for the domain, define the graph obtained by
    /// removing the masked nodes from the classical grid graph.
    ///
    /// Our convention is that a 1 in the mask indicates an obstacle.
    /// The \a x coord index is \c i.
    struct MaskedStencilGraph
    {
        /// Array type for grid mask.
        typedef Array<bool, -1, -1> Mask_t;
        Mask_t mask_;
        size_t n_nodes;
        std::vector<std::vector<size_t>> neighbors_;
    
        MaskedStencilGraph(const Mask_t& mask
        ) : mask_(mask) {
            size_t nx = mask.cols();
            size_t ny = mask.rows();
            n_nodes = 0;
            int v_id;
            for (size_t i = 0; i < nx; i++)
            {
                for (size_t j = 0; j < ny; j++)
                {
                    if (mask_(i, j) == 0) {
                        std::vector<size_t> nlist { n_nodes };
                        n_nodes++;
                        neighbors_.push_back(nlist);
                    }
                }
            }

            // 
            for (size_t i = 0; i < nx; i++)
            {
                for (size_t j = 0; j < ny; j++)
                {
                    if (mask_(i, j) == 0)
                    {
                        std::vector<size_t> nlist{n_nodes};
                        n_nodes++;
                        neighbors_.push_back(nlist);
                    }
                }
            }
        };

        /// @param i row index
        /// @param j columns index
        size_t grid_idx_to_node_id(size_t i, size_t j) {
            size_t nx = mask_.cols();
            size_t ny = mask_.rows();
            int v = j * nx + i;
            return v;
        }

    };

    }

}
