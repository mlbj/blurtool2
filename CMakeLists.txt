cmake_minimum_required(VERSION 3.22)

# Enable languages C++ and CUDA
project(blurtool2 LANGUAGES CXX CUDA)

# Set C++ and CUDA standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)


# this is outdated
# find_package(CUDA REQUIRED)
# find CUDA
# include_directories(${CUDA_INCLUDE_DIRS})

# find CUDA
find_package(CUDAToolkit REQUIRED)

# Include CUDA directories
include_directories(${CUDAToolkit_INCLUDE_DIRS})

# Enable CUDA separable compilation mode
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# Include directories
include_directories(include)

# Create the executable
add_executable(blur_example main.cpp)

# Define the library
add_library(blurtool2 
    src/image.cu      # CUDA source file
    src/blur.cu      # C++ source file
)

# ensure CUDA compilation
set_source_files_properties(
    src/image.cu
    src/blur.cu
    PROPERTIES LANGUAGE CUDA
)

# Set CUDA separable compilation for the library
set_target_properties(blurtool2 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Link directories and libraries
target_include_directories(blurtool2 PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(blurtool2 PRIVATE cuda cudart cufft)


# Ensure blur_example links with blurtool2
target_include_directories(blur_example PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(blur_example PRIVATE blurtool2)

# GoogleTest setup
include(FetchContent)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)
FetchContent_MakeAvailable(googletest)


# Copy test data to the build directory
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/test/test_image_2x3_f4.npy ${CMAKE_CURRENT_BINARY_DIR}/test_image_2x3_f4.npy COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/test/test_image_2x3_f8.npy ${CMAKE_CURRENT_BINARY_DIR}/test_image_2x3_f8.npy COPYONLY)

# Add the test executable
add_executable(test_image test/test_image.cpp)

# Link GoogleTest and blurtool2
target_link_libraries(test_image PRIVATE gtest_main blurtool2 pthread)

# Register the tests
include(GoogleTest)
gtest_discover_tests(test_image)

enable_testing()

