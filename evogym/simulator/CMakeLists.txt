cmake_minimum_required(VERSION 3.1.0)

set(CMAKE_SUPPRESS_REGENERATION true)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

project(simulator_cpp)

# Enable C++ 11
set(CMAKE_CXX_STANDARD 11)

# Suppress warnings.
add_definitions(
  -D_CRT_SECURE_NO_WARNINGS
)
if(UNIX)
  set(GCC_COVERAGE_COMPILE_FLAGS "-Wno-format")
  add_definitions(${GCC_COVERAGE_COMPILE_FLAGS})
endif()

if(UNIX)
  find_package(OpenGL REQUIRED)
  find_package(GLEW REQUIRED)
endif()

set(EXTERNAL_ROOT ${PROJECT_SOURCE_DIR}/externals)
add_subdirectory(${EXTERNAL_ROOT})
get_directory_property(EXTERNAL_HEADER
  DIRECTORY ${EXTERNAL_ROOT}
  DEFINITION EXTERNAL_HEADER)

# Expose PROJECT_DIR to the source code.
add_definitions(-DPROJECT_DIR="${PROJECT_SOURCE_DIR}")
add_definitions(-DGRAPHICS_CODEBASE_SOURCE_DIR="${CMAKE_CURRENT_LIST_DIR}")

include_directories(${EXTERNAL_HEADER})

add_definitions(
  -DTW_STATIC
  -DTW_NO_LIB_PRAGMA
  -DTW_NO_DIRECT3D
  -DGLEW_STATIC
  -D_CRT_SECURE_NO_WARNINGS
)

add_subdirectory(SimulatorCPP/)

set(CMAKE_CXX_STANDARD_LIBRARIES -ldl)
# set(CMAKE_VERBOSE_MAKEFILE ON)

