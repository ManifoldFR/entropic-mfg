#ifndef SINKHORN_H_
#define SINKHORN_H_
#include <Eigen/Core>
#include <vector>


using namespace Eigen;

/// Namespace for KL-proximal operators.
namespace klprox
{

class BaseProximalOperator {
    public:
    virtual MatrixXd operator()(MatrixXd& x) const = 0;
};

}  // namespace klprox







#endif