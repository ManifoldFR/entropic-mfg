#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/operators.h>

#include "Kernels.h"


using namespace kernels;
namespace py = pybind11;

class PyBaseKernel : BaseKernel {
public:
    // Inherit default constructor.
    using BaseKernel::BaseKernel;

    MatrixXd operator()(const MatrixXd& x) const override {
        PYBIND11_OVERLOAD_NAME(
            MatrixXd,   /* Return type */
            BaseKernel,   /* Parent class */
            "__call__", /* Name of function in Python */
            operator(), /* Name of function in C++ */
            x           /* Argument(s) */
        );
    }


};


void bind_kernels(py::module& m) {
    typedef EuclideanHeatKernel<2> Kernel2D;

    py::module m2 = m.def_submodule(
        "kernels", "Gibbs kernels used in the optimal transport formulation of the MFG. "
                   "These represent the 2-marginal of the Wiener measure.");

    py::class_<BaseKernel, PyBaseKernel, KernelPtr>(m2, "BaseKernel")
        .def("__call__", &BaseKernel::operator());

    py::class_<Kernel2D, BaseKernel, std::shared_ptr<Kernel2D>>(m2, "EuclideanKernel")
        .def(py::init<size_t, size_t, double, double,
                      double, double, double>());

}

