#pragma once

#include <memory>
#include <iostream>
#include <Eigen/Core>

#include "RectangularGridLaplacian.h"

using namespace Eigen;


namespace kernels
{

class BaseKernel {
    public:
    virtual MatrixXd operator()(const MatrixXd& x) const = 0;
};

// /// Define type for pointers to kernels.
// class KernelPtr : public std::shared_ptr<BaseKernel> {
//     public:
//     MatrixXd operator()(const ArrayXXd& x) const {
//         return this->operator()(x);
//     }
// };

typedef std::shared_ptr<BaseKernel> KernelPtr;


/// Heat kernel on Euclidean space.
template <size_t N>
class EuclideanHeatKernel : public BaseKernel {
    private:
    size_t dims_[N];
    double variance;
    const size_t ndim_ = N;

    public:
    const size_t& ndim() { return ndim_; }
    const size_t* dims() { return &dims_[0]; }
};

template<>
class EuclideanHeatKernel<2> : public BaseKernel {
    private:
    double dx, dy;
    size_t dims_[2];
    double variance;
    const size_t ndim_ = 2;

    public:
    MatrixXd K1, K2;
    const size_t& ndim() { return ndim_; }
    const size_t* dims() { return &dims_[0]; }

    EuclideanHeatKernel(size_t nx, size_t ny, double xmin, double xmax,
                        double ymin, double ymax, double variance=1.0
    ): K1(nx, nx), K2(ny, ny), dx((xmax-xmin)/(nx-1)), dy((ymax-ymin)/(ny-1)), variance(variance)
    {
        auto xar = VectorXd::LinSpaced(nx, xmin, xmax);
        auto yar = VectorXd::LinSpaced(ny, ymin, ymax);

        double delta = 0.;
        #pragma omp simd
        for (size_t i=0; i < nx; i++) {
            for (size_t j=0; j < nx; j++) {
                delta = xar[i]-xar[j];
                K1(i, j) = exp(-pow(delta, 2)/(2.*variance));
            }
        }

        #pragma omp simd
        for (size_t i=0; i < ny; i++) {
            for (size_t j=0; j < ny; j++) {
                delta = yar[i]-yar[j];
                K2(i, j) = exp(-pow(delta, 2)/(2.*variance));
            }
        }
    }

    /// @param x: Matrix of size nx, ny
    MatrixXd operator() (const MatrixXd& x) const override {
        return K2 * (K1 * x);
    }


};

}
