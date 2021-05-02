#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/operators.h>

#include "Kernels.h"


using namespace kernels;
namespace py = pybind11;

using Kernel_t = KernelBase<>;

class PyBaseKernel : Kernel_t {
public:
    // Inherit default constructor.
    using KernelBase::KernelBase;

};


void bind_kernels(py::module& m) {
    py::module m2 = m.def_submodule(
        "kernels", "Gibbs kernels used in the optimal transport formulation of the MFG. "
                   "These represent the 2-marginal of the Wiener measure.");

    py::class_<Kernel_t, PyBaseKernel, KernelPtr>(m2, "KernelBase")
        .def("__call__", &Kernel_t::multiply);

    py::class_<SeparableEuclideanKernel2D, Kernel_t, std::shared_ptr<SeparableEuclideanKernel2D>>(m2, "EuclideanKernel")
        .def(py::init<size_t, size_t, double, double,
                      double, double, double>())
        .def_readonly("K1", &SeparableEuclideanKernel2D::K1_)
        .def_readonly("K2", &SeparableEuclideanKernel2D::K2_);

}

