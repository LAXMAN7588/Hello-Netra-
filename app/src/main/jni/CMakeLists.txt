project(mobilenetssdncnn)

cmake_minimum_required(VERSION 3.10)

set(ncnn_DIR ${CMAKE_SOURCE_DIR}/${ANDROID_ABI}/lib/cmake/ncnn)

find_package(ncnn REQUIRED)

add_library(mobilenetssdncnn SHARED
        mobilenetssdncnn_jni.cpp
)

target_link_libraries(mobilenetssdncnn
        ncnn
        jnigraphics
)