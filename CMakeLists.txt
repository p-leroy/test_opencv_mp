cmake_minimum_required(VERSION 3.5)

project(test_opencv_mp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(test_opencv_mp main.cpp)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(test_opencv_mp ${OpenCV_LIBS})
