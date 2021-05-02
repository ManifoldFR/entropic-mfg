find_package(pybind11 REQUIRED)

file(GLOB SOURCES
     *.cpp)


pybind11_add_module(pyentropicmfg ${SOURCES})
target_include_directories(pyentropicmfg PRIVATE ../)
target_link_libraries(pyentropicmfg PRIVATE ${PROJECT_NAME})
set_target_properties(pyentropicmfg PROPERTIES
                      POSITION_INDEPENDENT_CODE True)

set(python_files
    __init__.py
    )

foreach(pyfile ${python_file})
    install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/${pyfile}")
endforeach()

