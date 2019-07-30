cmake_minimum_required (VERSION 3.0)
project (sample_data)


include_directories(include)


set (SOURCES    "src/main.cpp"
                "src/grid.cpp"
                "src/hierarchy.cpp")

add_executable(sample_data ${SOURCES})

target_link_libraries(sample_data gmp CGAL)