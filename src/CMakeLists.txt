cmake_minimum_required(VERSION 3.22)

# source files for library
add_library(blurtool2 
    image.cpp
    blur.cpp
)

# includes
target_include_directories(blurtool2 PUBLIC ${CMAKE_SOURCE_DIR}/include)
