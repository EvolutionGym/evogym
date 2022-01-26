# CMake entry point.
cmake_minimum_required(VERSION 3.1)

# Do not want to generate zeor_check because they cause messy folders in
# visual studio 2017.
set(CMAKE_SUPPRESS_REGENERATION true)

set(CMAKE_CXX_STANDARD 11)

set(EXTERNAL_HEADER
  "${CMAKE_CURRENT_LIST_DIR}/eigen"
  "${CMAKE_CURRENT_LIST_DIR}/glew/include"
  "${CMAKE_CURRENT_LIST_DIR}/glfw/include/GLFW"
)

include_directories(${EXTERNAL_HEADER})

# pybind11
add_subdirectory(pybind11)

add_definitions(
  -DTW_STATIC
  -DTW_NO_LIB_PRAGMA
  -DTW_NO_DIRECT3D
  -DGLEW_STATIC
  -D_CRT_SECURE_NO_WARNINGS
)

### GLFW ###
add_subdirectory(glfw)

include_directories(
  glfw/include/GLFW/
  glew/include/
  eigen/
)

### GLEW ###
set(GLEW_SOURCE
  glew/src/glew.c
)
set(GLEW_HEADERS
)
add_library(glew STATIC
  ${GLEW_SOURCE}
  ${GLEW_INCLUDE}
)
target_link_libraries(glew
  ${OPENGL_LIBRARY}
)
