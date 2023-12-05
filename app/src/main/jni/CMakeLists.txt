project(segnail)

cmake_minimum_required(VERSION 3.10)

set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/opencv-mobile-4.6.0-android/sdk/native/jni)
#find_package(OpenCV REQUIRED)
find_package(OpenCV REQUIRED core imgproc)

set(ncnn_DIR ${CMAKE_SOURCE_DIR}/ncnn-20221128-android-vulkan/${ANDROID_ABI}/lib/cmake/ncnn)
find_package(ncnn REQUIRED)

add_library(segnail SHARED seg_jni.cpp seg.cpp ndkcamera.cpp  )


target_link_libraries(segnail ncnn ${OpenCV_LIBS} camera2ndk mediandk  )


