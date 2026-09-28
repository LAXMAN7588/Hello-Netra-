cmake_minimum_required(VERSION 3.22.1)

project(netra_native)

# ============================================================
# Existing Netra / ncnn configuration
# ============================================================

set(ncnn_DIR ${CMAKE_SOURCE_DIR}/arm64-v8a/lib/cmake/ncnn)

find_package(ncnn REQUIRED)

add_library(netra_ncnn SHARED
    netra_ncnn_jni.cpp
)

target_link_libraries(netra_ncnn
    ncnn
    jnigraphics
)

# ============================================================
# Hello-Netra OCR TTS / eSpeak-NG phonemizer
# ============================================================

set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(ENABLE_TESTS OFF CACHE BOOL "" FORCE)
set(USE_ASYNC OFF CACHE BOOL "" FORCE)
set(USE_MBROLA OFF CACHE BOOL "" FORCE)
set(USE_LIBSONIC OFF CACHE BOOL "" FORCE)
set(USE_PCAUDIOLIB OFF CACHE BOOL "" FORCE)
set(USE_KLATT OFF CACHE BOOL "" FORCE)
set(USE_SPEECHPLAYER OFF CACHE BOOL "" FORCE)

set(EXTRA_CFLAGS "-fPIC")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")

add_subdirectory(
    D:/blind/espeak-ng
    espeak_ng_build
)

add_library(hellonetra_phonemizer SHARED
    hellonetra_phonemizer.cpp
)

target_include_directories(hellonetra_phonemizer PRIVATE
    D:/blind/espeak-ng/src/include
)

target_link_libraries(hellonetra_phonemizer
    log
    espeak-ng
)
