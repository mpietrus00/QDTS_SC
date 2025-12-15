cmake_minimum_required(VERSION 3.12)
project(QDTS_SC VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# SuperCollider plugin API
set(SC_PATH "" CACHE PATH "Path to SuperCollider source")

if(NOT SC_PATH)
    message(FATAL_ERROR "SC_PATH not set. Please provide path to SuperCollider source with -DSC_PATH=/path/to/supercollider")
endif()

# Find Eigen (any version 3.x+)
find_package(Eigen3 REQUIRED NO_MODULE)
if(NOT Eigen3_FOUND)
    # Try pkg-config as fallback
    find_package(PkgConfig)
    if(PkgConfig_FOUND)
        pkg_check_modules(EIGEN3 eigen3)
        if(EIGEN3_FOUND)
            include_directories(${EIGEN3_INCLUDE_DIRS})
        endif()
    endif()
endif()

# Include SuperCollider plugin headers
include_directories(
    ${SC_PATH}/include/plugin_interface
    ${SC_PATH}/include/common
    ${SC_PATH}/common
)

# Platform-specific settings
if(APPLE)
    set(CMAKE_SHARED_MODULE_SUFFIX ".scx")
elseif(WIN32)
    set(CMAKE_SHARED_MODULE_SUFFIX ".scx")
else()
    set(CMAKE_SHARED_MODULE_SUFFIX ".so")
endif()

# Build the Solver plugin
add_library(Solver MODULE
    plugins/Solver/Solver.cpp
)

target_link_libraries(Solver PRIVATE Eigen3::Eigen)

# Set output directory
set_target_properties(Solver PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/plugins"
    PREFIX ""
)

# Installation
install(TARGETS Solver
    LIBRARY DESTINATION "lib/SuperCollider/plugins"
)

install(DIRECTORY sc-classes/
    DESTINATION "share/SuperCollider/Extensions/QDTS_SC"
    FILES_MATCHING PATTERN "*.sc"
)

message(STATUS "QDTS_SC - Quadratic Distortion Tone Spectra for SuperCollider")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "SC_PATH: ${SC_PATH}")
