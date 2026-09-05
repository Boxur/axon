cmake_minimum_required(VERSION 3.14)

project("axon")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD			23)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS		OFF)
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_subdirectory(external/logger)

add_library(axon STATIC
  src/layer.cpp
  src/network.cpp
)

target_link_libraries(axon PUBLIC logger)

target_include_directories(axon PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
