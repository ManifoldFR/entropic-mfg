#pragma once

#include <Eigen/Core>
#include <vector>
#include <memory>


using namespace Eigen;

/// Namespace for KL-proximal operators.
namespace klprox
{

class BaseProximalOperator {
    public:
    virtual MatrixXd operator()(MatrixXd& x) const = 0;
};

class MultimarginalSinkhorn {
    private:
    /// Running cost proximal operator
    std::shared_ptr<BaseProximalOperator> running;
    /// Terminal cost proximal operator
    std::shared_ptr<BaseProximalOperator> terminal;
};

}  // namespace klprox



