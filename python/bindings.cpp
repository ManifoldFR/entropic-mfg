#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/operators.h>


namespace py = pybind11;


void bind_prox(py::module &);


PYBIND11_MODULE(pyentropicmfg, m) {
    m.doc() = "A library to solve variational mean-field games using an entropy "
        "minimization approach.";

    bind_prox(m);

    
}



