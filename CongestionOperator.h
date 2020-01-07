#ifndef CONGESTION_OP_H_
#define CONGESTION_OP_H_
#include <Eigen/Core>
#include "MultiSinkhorn.h"


using namespace Eigen;

namespace klprox
{

/**
 * KL-proximal operator for the hard congestion opertor.
 */
class CongestionPotentialProx: BaseProximalOperator {
    private:
    double congest_max;
    MatrixXd psi;
    
    public:
    CongestionPotentialProx(double congest_max, MatrixXd& psi): congest_max(congest_max), psi(psi) {}
    MatrixXd operator()(MatrixXd& x) const {
        return x.array().min(congest_max);
    }

};

}


#endif 