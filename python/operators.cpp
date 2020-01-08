#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/operators.h>

#include "KLOperator.h"
#include "CongestionOperator.h"


namespace py = pybind11;

class PyBaseProximalOperator : klprox::BaseProximalOperator {
public:
    // Inherit default constructor.
    using klprox::BaseProximalOperator::BaseProximalOperator;

    MatrixXd operator()(const MatrixXd& x) const override {
        PYBIND11_OVERLOAD_NAME(
            MatrixXd, /* Return type */
            klprox::BaseProximalOperator, /* Parent class */
            "__call__", /* Name of function in Python */
            operator(), /* Name of function in C++ */
            x      /* Argument(s) */
        );
    }


};

void bind_prox(py::module& m) {
    using namespace klprox;

    py::module m2 = m.def_submodule("prox", "Proximal operators");

    py::class_<BaseProximalOperator, PyBaseProximalOperator>(m2, "BaseProximalOperator")
        .def("__call__", &BaseProximalOperator::operator());

    py::class_<ObstacleProx, BaseProximalOperator>(m2, "ObstacleProx")
        .def(py::init<const Eigen::ArrayXXi&>());

    py::class_<CongestionPotentialProx, BaseProximalOperator>(m2, "CongestionPotentialProx")
        .def(py::init<double, const Eigen::MatrixXd&>());

    py::class_<CongestionObstacleProx, CongestionPotentialProx, ObstacleProx>(m2, "CongestionObstacleProx")
        .def(py::init<const ArrayXXi&, double, const Eigen::MatrixXd&>());

}

