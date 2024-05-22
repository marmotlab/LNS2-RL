cmake_minimum_required(VERSION 3.17)
project(my_env)

set(CMAKE_CXX_STANDARD 17)

IF(NOT CMAKE_BUILD_TYPE)
    SET(CMAKE_BUILD_TYPE "RELEASE")
ENDIF()

include_directories("inc")

find_package(Boost REQUIRED COMPONENTS program_options system filesystem)
include_directories( ${Boost_INCLUDE_DIRS} )

find_package (Eigen3 3.3 REQUIRED NO_MODULE)

add_subdirectory(pybind11)
pybind11_add_module(my_env main.cpp)

target_link_libraries(my_env PRIVATE ${Boost_LIBRARIES} Eigen3::Eigen)




# Find Boost


#
#add_custom_command(
#        TARGET ${PROJECT_NAME}  POST_BUILD
#        COMMAND ${CMAKE_COMMAND} -E copy_directory
#        ${CMAKE_BINARY_DIR}/$<CONFIGURATION> ${CMAKE_CURRENT_LIST_DIR}
#)