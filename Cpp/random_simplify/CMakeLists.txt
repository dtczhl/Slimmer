cmake_minimum_required (VERSION 3.0)
project (random_simplify)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall")

include_directories(include)


set (SOURCES    "src/random_simplify.cpp")

add_library(random_simplify SHARED ${SOURCES})

target_link_libraries(random_simplify gmp CGAL)