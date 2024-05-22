cmake_minimum_required(VERSION 3.12)
project(webcam-tamagotchi)

# Add the source files for the component
add_library(screen-detector-component
    ScreenDetector.cpp
)

add_library(screen-extractor-component
    ScreenExtractor.cpp
)
