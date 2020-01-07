cmake_minimum_required(VERSION 3.0.0)
project(EntropicMFG VERSION 0.1.0)

include(CTest)
enable_testing()

find_package(Python COMPONENTS Development NumPy)
find_package(Eigen3 REQUIRED)

message(STATUS "Found Python: ${Python_FOUND}")
message(STATUS "Python includes at ${Python_INCLUDE_DIRS} ${Python_NumPy_INCLUDE_DIRS}")

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(EntropicMFG_SOURCES Kernels.h MessagePassing.h MessagePassing.cpp)


add_library(entropicmfg ${EntropicMFG_SOURCES})
target_link_libraries(entropicmfg Eigen3::Eigen)

set_target_properties(entropicmfg PROPERTIES LINKER_LANGUAGE CXX)

add_executable(main main.cpp)
target_link_libraries(main Eigen3::Eigen)

message(STATUS "Linking Python libraries for Matplotlib")
target_sources(main PRIVATE matplotlibcpp.h)
target_include_directories(main PRIVATE ${Python_INCLUDE_DIRS} ${Python_NumPy_INCLUDE_DIRS})
target_link_libraries(main Python::Python Python::NumPy)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
