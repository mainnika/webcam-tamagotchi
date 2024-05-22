cmake_minimum_required(VERSION 3.12)
project(webcam-tamagotchi)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_definitions(-DSPDLOG_USE_STD_FORMAT)

add_compile_options(-Wno-deprecated-enum-enum-conversion)
add_compile_options(-Wno-deprecated-anon-enum-enum-conversion)

find_package(OpenCV REQUIRED)
find_package(spdlog REQUIRED)

include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories( ${spdlog_INCLUDE_DIRS} )

add_executable(webcam-tamagotchi main.cpp)
add_subdirectory(components)

target_link_libraries(webcam-tamagotchi ${OpenCV_LIBS})
target_link_libraries(webcam-tamagotchi ${spdlog_LIBS})
target_link_libraries(webcam-tamagotchi ${fmt_LIBS})
target_link_libraries(webcam-tamagotchi screen-detector-component)
target_link_libraries(webcam-tamagotchi screen-extractor-component)
