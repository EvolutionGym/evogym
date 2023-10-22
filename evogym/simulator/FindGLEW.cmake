# FindGLEW.cmake
# Search for GLEW library and include directories

find_path(GLEW_INCLUDE_DIR GL/glew.h)
find_library(GLEW_LIBRARY NAMES GLEW GLEW32)

if (GLEW_INCLUDE_DIR AND GLEW_LIBRARY)
    set(GLEW_FOUND TRUE)
else()
    set(GLEW_FOUND FALSE)
endif()

if (GLEW_FOUND)
    if (NOT GLEW_FIND_QUIETLY)
        message(STATUS "Found GLEW: ${GLEW_LIBRARY}")
    endif ()
else()
    if (GLEW_FIND_REQUIRED)
        message(FATAL_ERROR "Could NOT find GLEW")
    endif ()
endif()

mark_as_advanced(GLEW_INCLUDE_DIR GLEW_LIBRARY)
