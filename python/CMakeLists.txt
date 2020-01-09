find_package(pybind11 REQUIRED)

set(SOURCES
    operators.cpp
    kernels.cpp
    sinkhorn.cpp
    bindings.cpp)


pybind11_add_module(pyentropicmfg ${SOURCES})
target_include_directories(pyentropicmfg PRIVATE ../)
target_link_libraries(pyentropicmfg PRIVATE entropicmfg)
target_compile_options(pyentropicmfg PRIVATE -fPIC)