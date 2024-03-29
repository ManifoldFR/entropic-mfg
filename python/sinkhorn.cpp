#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include "MultiSinkhorn.h"

namespace py = pybind11;


void bind_sinkhorn(py::module& m) {

    using namespace sinkhorn;

    py::module m2 = m.def_submodule(
        "sinkhorn", "Generalized Sinkhorn algorithm using message passing "
                    "to update the dual potentials and compute the marginals.");

    
    py::class_<MultimarginalSinkhorn>(m2, "MultiSinkhorn")
        .def(py::init<ProxPtr, ProxPtr,
                      kernels::KernelPtr,
                      Eigen::MatrixXd&>())
        .def("iterate", &MultimarginalSinkhorn::iterate)
        .def("solve", &MultimarginalSinkhorn::solve,
             py::arg("num_iterations"), py::arg("potentials"),
             py::arg("verbose") = true)
        .def("get_marginals", &MultimarginalSinkhorn::get_marginals,
             py::return_value_policy::copy)
        .def_readwrite("threshold", &MultimarginalSinkhorn::threshold_)
        .def_property_readonly("conv_metrics_", &MultimarginalSinkhorn::getConvMetric);
    

}

