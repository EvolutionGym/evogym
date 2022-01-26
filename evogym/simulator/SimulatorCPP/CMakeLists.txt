set(SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR})

file(GLOB_RECURSE HEADER ${SOURCE_DIR}/*.h)
file(GLOB_RECURSE SOURCE ${SOURCE_DIR}/*.cpp)
list(REMOVE_ITEM SOURCE ${SOURCE_DIR}/PythonBindings.cpp)

pybind11_add_module(simulator_cpp PythonBindings.cpp ${HEADER} ${SOURCE})
# set_target_properties(simulator_cpp
#     PROPERTIES
#     ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/evogym"
#     LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/evogym"
#     RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/evogym"
# )

target_include_directories(simulator_cpp PRIVATE ${EXTERNAL_HEADER})
target_include_directories(simulator_cpp PRIVATE ${SOURCE_DIR}/)
target_link_libraries(simulator_cpp PRIVATE glew)
target_link_libraries(simulator_cpp PRIVATE glfw)

if(WIN32)
    target_link_libraries(simulator_cpp PRIVATE kernel32)
    target_link_libraries(simulator_cpp PRIVATE opengl32)
    target_link_libraries(simulator_cpp PRIVATE gdi32)
    target_link_libraries(simulator_cpp PRIVATE user32)
    target_link_libraries(simulator_cpp PRIVATE shell32)
endif()

set(CMAKE_CXX_STANDARD_LIBRARIES -ldl)
set(CMAKE_VERBOSE_MAKEFILE ON)

