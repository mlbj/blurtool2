# Download GoogleTest
include(FetchContent)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)
FetchContent_MakeAvailable(googletest)

enable_testing()

# add the test executable
add_executable(test_image test_image.cpp)

# link google test 
target_link_libraries(test_image PRIVATE gtest_main blur_lib)

# discover and register tests
include(GoogleTest)
message(STATUS "Registering tests with gtest_discover_tests()")
gtest_discover_tests(test_image)

