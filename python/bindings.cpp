#include <pybind11/pybind11.h>


namespace py = pybind11;


void bind_prox(py::module &);
void bind_sinkhorn(py::module &);
void bind_kernels(py::module &);


PYBIND11_MODULE(pyentropicmfg, m) {
    m.doc() = "A library to solve variational mean-field games using an entropy "
        "minimization approach.";

    bind_prox(m);
    bind_sinkhorn(m);
    bind_kernels(m);

}



