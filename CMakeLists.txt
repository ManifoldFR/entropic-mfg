cmake_minimum_required(VERSION 3.0.0)
project(EntropicMFG VERSION 0.1.0)

include(CTest)
enable_testing()

find_package(Python COMPONENTS Development NumPy)
find_package(Eigen3 REQUIRED)
find_package(OpenMP REQUIRED)

add_definitions(-DEIGEN_DONT_PARALLELIZE)

message(STATUS "Python includes at ${Python_INCLUDE_DIRS} ${Python_NumPy_INCLUDE_DIRS}")

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(EntropicMFG_SOURCES
    Kernels.h
    MultiSinkhorn.h
    Utils.h
    CongestionOperator.h
    RectangularGridLaplacian.h
    MessagePassing.h MessagePassing.cpp)

add_subdirectory(examples)

add_library(entropicmfg ${EntropicMFG_SOURCES})
set_target_properties(entropicmfg PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(entropicmfg Eigen3::Eigen)
target_link_libraries(entropicmfg OpenMP::OpenMP_CXX)


if (${Python_NumPy_FOUND})
    target_sources(entropicmfg PRIVATE matplotlibcpp.h)
    target_link_libraries(entropicmfg Python::Python Python::NumPy)
    target_include_directories(entropicmfg PRIVATE ${Python_INCLUDE_DIRS} ${Python_NumPy_INCLUDE_DIRS})
endif()


set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
