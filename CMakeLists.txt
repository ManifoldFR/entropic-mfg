cmake_minimum_required(VERSION 3.9.0)
project(EntropicMFG VERSION 0.1.0)

include(CTest)
enable_testing()

find_package(Python COMPONENTS Development NumPy)  # required for matplotlibcpp 
find_package(Eigen3 3.3 REQUIRED)
find_package(OpenMP REQUIRED)

add_definitions(-DEIGEN_DONT_PARALLELIZE)


message(STATUS "Python includes at ${Python_INCLUDE_DIRS} ${Python_NumPy_INCLUDE_DIRS}")

# Include the root directory
set(EntropicMFG_HEADERS
    Kernels.h
    Utils.h
    MultiSinkhorn.h
    KLOperator.h
    CongestionOperator.h
    RectangularGridLaplacian.h
    MessagePassing.h)

set(EntropicMFG_SOURCES
    MultiSinkhorn.cpp
    MessagePassing.cpp)


add_library(entropicmfg SHARED ${EntropicMFG_SOURCES})
target_include_directories(entropicmfg PUBLIC .)
target_link_libraries(entropicmfg Eigen3::Eigen)
target_link_libraries(entropicmfg OpenMP::OpenMP_CXX)
set_target_properties(entropicmfg PROPERTIES
    LINKER_LANGUAGE CXX)
target_compile_options(entropicmfg PRIVATE -fPIC)


if(TARGET Python::NumPy)
    target_sources(entropicmfg PRIVATE matplotlibcpp.h)
    target_link_libraries(entropicmfg Python::Python Python::NumPy)
    target_include_directories(entropicmfg PRIVATE ${Python_INCLUDE_DIRS} ${Python_NumPy_INCLUDE_DIRS})
endif(TARGET Python::NumPy)


# Add examples
add_subdirectory(examples)


# Python bindings for library
option(PythonBindings "Make Python bindings." ON)

if(${PythonBindings})
    message(STATUS "Python bindings enabled.")
    add_subdirectory(python)
endif(${PythonBindings})


set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
