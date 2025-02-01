# source files
add_library(blur_lib 
    image.cpp
    blur.cpp
)

# includes
target_include_directories(blur_lib PUBLIC ${CMAKE_SOURCE_DIR}/include)

