# Define biblioteca com os arquivos de código-fonte
add_library(blur_lib 
    image.cpp
    blur.cpp
)

# Define diretórios de include
target_include_directories(blur_lib PUBLIC ${CMAKE_SOURCE_DIR}/include)
