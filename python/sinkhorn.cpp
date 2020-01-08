#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

#include "MultiSinkhorn.h"
#include "CongestionOperator.h"

using sinkhorn::MultimarginalSinkhorn;

namespace py = pybind11;


PYBIND11_MODULE(pyentropicmfg, m) {
    using namespace klprox;

    py::class_<ObstacleProx>(m, "ObstacleProx")
        .def(py::init<const Eigen::ArrayXi&>());

    py::class_<CongestionPotentialProx>(m, "CongestionPotentialProx")
        .def(py::init<double, const Eigen::MatrixXd&>());

    py::class_<CongestionObstacleProx>(m, "CongestionObstacleProx")
        .def(py::init<const ArrayXXi&, double, const Eigen::MatrixXd&>());

}

