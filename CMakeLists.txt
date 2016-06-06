cmake_minimum_required(VERSION 2.8.6)
project( tensor C CXX )

# CMake module(s) path
set( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR} )
set( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake )
# Find CasADi
find_package(CASADI REQUIRED)
# Include directories
include_directories(
  ${PROJECT_SOURCE_DIR}
)

set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x -O2 -fPIC" )

set( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O2 -fPIC" )


# Simple library that shows how CasADi can be extended
include_directories(${CASADI_INCLUDE_DIR})
add_library(tensor SHARED
            tensor.cpp  tensor.hpp
          )
target_link_libraries(tensor ${CASADI_LIBRARIES})
add_definitions(-DSHARED_LIBRARY)


add_executable(testme
            test.cpp
          )

target_link_libraries(testme tensor)

target_include_directories(testme PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
