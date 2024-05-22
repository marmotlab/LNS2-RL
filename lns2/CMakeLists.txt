cmake_minimum_required(VERSION 3.17)
project(my_lns2)

set(CMAKE_CXX_STANDARD 17)

IF(NOT CMAKE_BUILD_TYPE)
    SET(CMAKE_BUILD_TYPE "RELEASE")
ENDIF()

include_directories("inc")

find_package(Boost REQUIRED COMPONENTS program_options system filesystem)
include_directories( ${Boost_INCLUDE_DIRS} )

add_subdirectory(pybind11)
pybind11_add_module(my_lns2 main.cpp)
target_link_libraries(my_lns2 PRIVATE ${Boost_LIBRARIES})
