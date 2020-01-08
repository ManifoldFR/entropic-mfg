find_package(pybind11 REQUIRED)


pybind11_add_module(pyentropicmfg sinkhorn.cpp)
target_include_directories(pyentropicmfg PRIVATE ../)
target_link_libraries(pyentropicmfg PRIVATE entropicmfg)
target_compile_options(pyentropicmfg PRIVATE -fPIC)