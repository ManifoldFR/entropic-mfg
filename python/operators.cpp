#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/operators.h>

#include "operators/KLOperator.h"
#include "operators/CongestionOperator.h"

namespace py = pybind11;

class PyBaseProximalOperator : klprox::ProximalOpBase<> {
public:
    // Inherit default constructor.
    using klprox::ProximalOpBase<>::ProximalOpBase;

};

void bind_prox(py::module& m) {
    using namespace klprox;

    py::module m2 = m.def_submodule("prox", "Proximal operators");

    py::class_<ProximalOpBase<>, PyBaseProximalOperator, ProxPtr>(m2, "ProximalOpBase")
        .def("__call__", &ProximalOpBase<>::operator());

    py::class_<ObstacleProx, ProximalOpBase<>, std::shared_ptr<ObstacleProx>>(m2, "ObstacleProx")
        .def(py::init<const Eigen::ArrayXXi&>());

    py::class_<CongestionPotentialProx, ProximalOpBase<>, std::shared_ptr<CongestionPotentialProx>>(m2, "CongestionPotentialProx")
        .def(py::init<double, const Eigen::MatrixXd&>());

    py::class_<CongestionObstacleProx, CongestionPotentialProx, ObstacleProx, std::shared_ptr<CongestionObstacleProx>>(m2, "CongestionObstacleProx")
        .def(py::init<const ArrayXXi&, double, const Eigen::MatrixXd&>());

}

