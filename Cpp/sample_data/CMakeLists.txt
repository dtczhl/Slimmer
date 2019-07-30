cmake_minimum_required (VERSION 3.14)
project (sample_data)


include_directories(include)


set (SOURCES    "src/main.cpp")

add_executable(sample_data ${SOURCES})

target_link_libraries(sample_data gmp CGAL)