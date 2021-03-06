#pragma once

#include <omp.h>
#include <Eigen/Dense>
#include "KLOperator.h"


using namespace Eigen;


namespace klprox
{
/**
 * KL-proximal operator for the hard congestion operator
 * with a reference potential function.
 * 
 */
class CongestionPotentialProx: public virtual BaseProximalOperator {
    private:
    double congest_max;
    ArrayXXd psi;
    const size_t nx, ny;
    
    public:
    CongestionPotentialProx(double congest_max, const MatrixXd& psi):
    congest_max(congest_max), psi(psi), nx(psi.rows()), ny(psi.cols()) {}
    MatrixXd operator()(const MatrixXd& x) const override {
        ArrayXXd y = x.array() * exp(-psi);
        return y.min(congest_max);
    }

};

/**
 * KL-proximal operator for the obstacle or "mask" constraint (zero mass on the mask).
 * 
 */
class ObstacleProx: public virtual BaseProximalOperator {
    private:
    /// May be mutable for moving obstacles
    ArrayXXi obstacle_mask;
    const size_t nx, ny;

    public:
    ObstacleProx(const ArrayXXi& mask): obstacle_mask(mask), nx(mask.rows()), ny(mask.cols()) {}

    MatrixXd operator()(const MatrixXd& x) const override {
        auto mask_dbl = 1. - obstacle_mask.cast<double>();
        return x.array() * mask_dbl;
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


    MatrixXd operator()(const MatrixXd& x) const override {
        MatrixXd y = CongestionPotentialProx::operator()(x);
        return ObstacleProx::operator()(y);
    }
};


}
