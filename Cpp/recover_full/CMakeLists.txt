cmake_minimum_required (VERSION 3.0)
project (recover_full)

find_package(PCL 1.8 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS} include)
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

set (SOURCES    "src/main.cpp")

add_executable(recover_full ${SOURCES})

target_link_libraries(recover_full ${PCL_LIBRARIES})
 
