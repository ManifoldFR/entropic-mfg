cmake_minimum_required(VERSION 3.0.0)
project(EntropicMFG VERSION 0.1.0)

include(CTest)
enable_testing()

find_package(Python COMPONENTS Development NumPy)
find_package(Eigen3 REQUIRED)
find_package(OpenMP)

message(STATUS "Python includes at ${Python_INCLUDE_DIRS} ${Python_NumPy_INCLUDE_DIRS}")

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(EntropicMFG_SOURCES
    ./include/Kernels.h
    ./include/MultiSinkhorn.h
    RectangularGridLaplacian.h
    MessagePassing.h MessagePassing.cpp)


add_library(entropicmfg ${EntropicMFG_SOURCES})
target_link_libraries(entropicmfg Eigen3::Eigen)
target_link_libraries(entropicmfg OpenMP::OpenMP_CXX)

set_target_properties(entropicmfg PROPERTIES LINKER_LANGUAGE CXX)

if (${Python_NumPy_FOUND})
    target_sources(entropicmfg PRIVATE matplotlibcpp.h)
    target_link_libraries(entropicmfg Python::Python Python::NumPy)
    target_include_directories(entropicmfg PRIVATE ${Python_INCLUDE_DIRS} ${Python_NumPy_INCLUDE_DIRS})
endif()

file(GLOB EXAMPLE_SOURCES examples/*.cpp)

foreach(sourcefile ${EXAMPLE_SOURCES})
    get_filename_component(testname ${sourcefile} NAME_WE)
    message(STATUS "Found example ${testname}")
    add_executable( ${testname} ${sourcefile} )
    target_link_libraries(${testname} entropicmfg)
    target_link_libraries(${testname} Eigen3::Eigen)
endforeach(sourcefile ${EXAMPLE_SOURCES})



set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
