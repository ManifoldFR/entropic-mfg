#pragma once

#include <memory>
#include <iostream>
#include <Eigen/Core>


using namespace Eigen;


namespace kernels
{

template<typename S = double>
class KernelBase {
public:
    virtual ~KernelBase() = default;
    virtual Matrix<S, -1, -1> multiply(const Matrix<S,-1,-1>& x) const = 0;
};

using KernelPtr = std::shared_ptr<KernelBase<double>>;


/// Gaussian (heat) kernel on Euclidean space.
template <size_t Dim, typename S = double>
class EuclideanHeatKernelTpl : public KernelBase<S> {
    typedef S Scalar;

protected:
    size_t dims_[Dim];
    Scalar variance_;
    const size_t ndim_ = Dim;

public:
    const size_t& ndim() { return ndim_; }
    const size_t* dims() { return &dims_[0]; }
    EuclideanHeatKernelTpl() {};
    EuclideanHeatKernelTpl(S variance) : variance_(variance) {};

};


/// Separable Gaussian kernel on Euclidean space.
/// Assumes a kernel \f$ K = K_1\otimes K_2\f$.
class SeparableEuclideanKernel2D : public EuclideanHeatKernelTpl<2, double> {
public:
    /// Individual sub-kernels.
    MatrixXd K1_, K2_;
    double dx, dy;
    SeparableEuclideanKernel2D(
        size_t nx, size_t ny, double xmin, double xmax,
        double ymin, double ymax, double variance = 1.0
    ) : EuclideanHeatKernelTpl<2>{variance}, K1_(nx, nx), K2_(ny, ny), dx((xmax - xmin) / (nx - 1)), dy((ymax - ymin) / (ny - 1))
    {
        dims_[0] = nx;
        dims_[1] = ny;
        const VectorXd xar = VectorXd::LinSpaced(nx, xmin, xmax);
        const VectorXd yar = VectorXd::LinSpaced(ny, ymin, ymax);

        double dist = 0.;
        double var2 = variance_ * variance_;
        // #pragma omp simd
        size_t i = 0;
        size_t j = 0;
        for (i = 0; i < nx; i++)
        {
            for (j = 0; j < nx; j++)
            {
                dist = xar[i] - xar[j];
                K1_(i, j) = exp(-dist*dist / (2. * var2));
            }
        }

        // #pragma omp simd
        for (i = 0; i < ny; i++)
        {
            for (j = 0; j < ny; j++)
            {
                dist = yar[i] - yar[j];
                K2_(i, j) = exp(-dist*dist / (2. * var2));
            }
        }
    }

    /// @param x: Matrix of size nx, ny
    /// Compute the convolution wrt the kernel using the separability.
    /// See Peyré and Cuturi, "Computational Optimal Transport" (2019).
    MatrixXd multiply(const MatrixXd &x) const override
    {
        return (K1_ * x) * K2_;
    }
};





}
