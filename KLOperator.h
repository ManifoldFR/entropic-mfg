#pragma once

#include <Eigen/Core>


using namespace Eigen;

/// KL-proximal operators.
namespace klprox
{

class BaseProximalOperator {
    public:
    virtual ~BaseProximalOperator() = default;
    virtual MatrixXd operator()(const MatrixXd& x) const = 0;
};

}  // namespace klprox
