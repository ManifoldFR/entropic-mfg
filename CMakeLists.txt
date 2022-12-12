cmake_minimum_required(VERSION 3.9.0)
project(EntropicMFG VERSION 0.1.0)

include(CTest)
enable_testing()

set(CMAKE_CXX_STANDARD 14)

find_package(Python COMPONENTS Development NumPy)  # required for matplotlibcpp 
find_package(Eigen3 3.3 REQUIRED)


message(STATUS "Python includes at ${Python_INCLUDE_DIRS} ${Python_NumPy_INCLUDE_DIRS}")

# Include the root directory
file(GLOB_RECURSE EntropicMFG_HEADERS
     include/*.h*
     )

file(GLOB_RECURSE EntropicMFG_SOURCES
     src/*.cpp
     )


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")

set(DOXYGEN_USE_MATHJAX YES)


if(UNIX)

    add_compile_options(-Wall -Wpedantic -Wextra -march=native)

    add_library(${PROJECT_NAME} SHARED ${EntropicMFG_SOURCES} ${EntropicMFG_HEADERS})
    set_target_properties(${PROJECT_NAME} PROPERTIES LINKER_LANGUAGE CXX)
    target_compile_options(${PROJECT_NAME} PRIVATE "-fPIC")
    target_include_directories(${PROJECT_NAME} PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    )
    target_link_libraries(${PROJECT_NAME} Eigen3::Eigen)

endif()

option(BUILD_WITH_MPL "Build with matplotlibcpp" ON)
option(BUILD_PYTHON "Make Python bindings." ON)



if(BUILD_WITH_MPL)
    target_sources(${PROJECT_NAME} PRIVATE matplotlibcpp.h)
    target_link_libraries(${PROJECT_NAME} Python::Python Python::NumPy)
    target_include_directories(${PROJECT_NAME} PRIVATE ${Python_INCLUDE_DIRS} ${Python_NumPy_INCLUDE_DIRS})
endif()


option(BUILD_EXAMPLES "Build examples" ON)

if(BUILD_EXAMPLES)
    # Add examples
    add_subdirectory(examples)
endif()


if(BUILD_PYTHON)
    message(STATUS "Python bindings enabled.")
    add_subdirectory(python)
endif()


set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
