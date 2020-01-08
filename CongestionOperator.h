#pragma once

#include <Eigen/Dense>
#include "MultiSinkhorn.h"


using namespace Eigen;


namespace klprox
{
/**
 * KL-proximal operator for the hard congestion opertor.
 */
class CongestionPotentialProx: public virtual BaseProximalOperator {
    private:
    double congest_max;
    MatrixXd psi;
    
    public:
    CongestionPotentialProx(double congest_max, const MatrixXd& psi): congest_max(congest_max), psi(psi) {}
    MatrixXd operator()(MatrixXd& x) const override {
        ArrayXXd y = exp(-psi.array()) * x.array();
        return y.min(congest_max);
    }
};

class ObstacleProx: public virtual BaseProximalOperator {
    private:
    /// May be mutable for moving obstacles
    ArrayXXi obstacle_mask;
    const size_t nx, ny;

    public:
    ObstacleProx(const ArrayXXi& mask): obstacle_mask(mask), nx(mask.rows()), ny(mask.cols()) {}

    MatrixXd operator()(MatrixXd& x) const override {
        MatrixXd y(x);  // copy matrix

        for (size_t i=0; i < nx; i++) {
            for (size_t j=0; j < ny; j++) {
                if (obstacle_mask(i, j))
                    y(i, j) = 0.;
            }
        }
        return y;
    }
};

/**
 * Combined KL-proximal operator for both congestion, potential and obstacles.
 * 
 */
class CongestionObstacleProx: public CongestionPotentialProx, public ObstacleProx {
    public:
    CongestionObstacleProx(const ArrayXXi& mask, double congest_max, const MatrixXd& psi
    ): CongestionPotentialProx(congest_max, psi), ObstacleProx(mask) {}


    MatrixXd operator()(MatrixXd& x) const override {
        MatrixXd y = CongestionPotentialProx::operator()(x);
        return ObstacleProx::operator()(y);
    }
};


}
