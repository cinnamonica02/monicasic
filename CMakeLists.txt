# Placeholder for the C++ parts (instruction-set simulator, memory model, host runtime). Nothing to build yet.
# cmake and a C++ compiler are not installed on the dev laptop at the time of writing.
cmake_minimum_required(VERSION 3.20)
project(tiny_moe_chip LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/sim/CMakeLists.txt")
  add_subdirectory(sim)
endif()
