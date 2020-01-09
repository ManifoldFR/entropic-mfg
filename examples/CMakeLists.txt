include_directories(PRIVATE ../)


file(GLOB EXAMPLE_SOURCES *.cpp)

foreach(sourcefile ${EXAMPLE_SOURCES})
    get_filename_component(testname ${sourcefile} NAME_WE)
    add_executable(${testname} ${sourcefile})
    message(STATUS "Found example ${testname}")
    target_link_libraries(${testname} entropicmfg)
    target_link_libraries(${testname} OpenMP::OpenMP_CXX)
    target_link_libraries(${testname} Python::Python Python::NumPy)
    target_include_directories(${testname} PRIVATE ${Python_INCLUDE_DIRS} ${Python_NumPy_INCLUDE_DIRS})
endforeach(sourcefile ${EXAMPLE_SOURCES})