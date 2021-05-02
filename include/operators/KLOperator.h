#pragma once

#include <memory>
#include <Eigen/Core>


using namespace Eigen;

/// Kullback Leibler-proximal operators.
/// Used for computing the conjugates of the cost functions.
namespace klprox
{

template<typename S = double>
class ProximalOpBase {
public:
    typedef S Scalar;
    virtual ~ProximalOpBase() = default;
    virtual Matrix<S, -1, -1> operator()(const Matrix<S, -1, -1>& x) const = 0;
};

typedef std::shared_ptr<ProximalOpBase<>> ProxPtr;

}  // namespace klprox
