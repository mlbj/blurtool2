cmake_minimum_required(VERSION 3.18)
project(blurtool2)

set(CMAKE_CXX_STANDARD 17)
cmake_minimum_required(VERSION 3.22)


# add subdirectories
add_subdirectory(src) 
add_subdirectory(test)

# enable testing
enable_testing()



