file(GLOB EXAMPLE_SOURCES *.cpp)


foreach(sourcefile ${EXAMPLE_SOURCES})
    get_filename_component(testname ${sourcefile} NAME_WE)
    add_executable(${testname} ${sourcefile})
    message(STATUS "Found example ${testname}")
    target_link_libraries(${testname} ${PROJECT_NAME})
    target_link_libraries(${testname} Python::Python Python::NumPy)
    target_link_libraries(${testname} Eigen3::Eigen)
    target_include_directories(${testname} PRIVATE ../)
    target_compile_options(${testname} PRIVATE "-fPIC")
endforeach(sourcefile ${EXAMPLE_SOURCES})