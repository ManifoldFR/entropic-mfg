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
        .def("run", &MultimarginalSinkhorn::run)
        .def("get_marginals", &MultimarginalSinkhorn::get_marginals);
    

}

