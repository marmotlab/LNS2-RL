cmake_minimum_required(VERSION 3.16...3.27)
project(my_lns2 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Set default build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# === Local headers
include_directories("inc")

# === Boost and OpenCV
find_package(Boost REQUIRED COMPONENTS program_options system filesystem)

# Try to find OpenCV, but don't require it
find_package(OpenCV QUIET)
if(OpenCV_FOUND)
    include_directories(${OpenCV_INCLUDE_DIRS})
    add_compile_definitions(OPENCV_ENABLED=1)
    message(STATUS "OpenCV found, enabling visualization")
else()
    message(STATUS "OpenCV not found, visualization will be disabled")
    add_compile_definitions(OPENCV_ENABLED=0)
endif()

include_directories(${Boost_INCLUDE_DIRS})

# Check for filesystem support
include(CheckIncludeFileCXX)
check_include_file_cxx(filesystem HAVE_STD_FILESYSTEM)
if(HAVE_STD_FILESYSTEM)
    add_compile_definitions(HAVE_STD_FILESYSTEM=1)
    # Check if we need to link with the filesystem library
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9.0)
        set(FILESYSTEM_LIBRARIES stdc++fs)
    endif()
else()
    check_include_file_cxx(experimental/filesystem HAVE_STD_EXPERIMENTAL_FILESYSTEM)
    if(HAVE_STD_EXPERIMENTAL_FILESYSTEM)
        add_compile_definitions(HAVE_STD_EXPERIMENTAL_FILESYSTEM=1)
        # For GCC and Clang, we need to link with the filesystem library
        if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
            set(FILESYSTEM_LIBRARIES stdc++fs)
        endif()
    endif()
endif()

# === Rerun (Optional) ===
option(USE_RERUN "Enable Rerun visualization" OFF)

if(USE_RERUN)
    # --- Download and make Rerun SDK available ---
    include(FetchContent)
    FetchContent_Declare(
        rerun_sdk
        URL https://github.com/rerun-io/rerun/releases/latest/download/rerun_cpp_sdk.zip
    )
    FetchContent_MakeAvailable(rerun_sdk)

    # Define a preprocessor macro to enable Rerun in the code
    add_compile_definitions(RERUN_ENABLED=1)
else()
    # Define a preprocessor macro to disable Rerun in the code
    add_compile_definitions(RERUN_ENABLED=0)
endif()

# === Pybind11
add_subdirectory(pybind11)

# === Python Module
pybind11_add_module(my_lns2 main.cpp)
if(USE_RERUN)
    if(OpenCV_FOUND)
        target_link_libraries(my_lns2 PRIVATE
            ${Boost_LIBRARIES}
            ${OpenCV_LIBS}
            ${FILESYSTEM_LIBRARIES}
            rerun_sdk
        )
    else()
        target_link_libraries(my_lns2 PRIVATE
            ${Boost_LIBRARIES}
            ${FILESYSTEM_LIBRARIES}
            rerun_sdk
        )
    endif()
else()
    if(OpenCV_FOUND)
        target_link_libraries(my_lns2 PRIVATE
            ${Boost_LIBRARIES}
            ${OpenCV_LIBS}
            ${FILESYSTEM_LIBRARIES}
        )
    else()
        target_link_libraries(my_lns2 PRIVATE
            ${Boost_LIBRARIES}
            ${FILESYSTEM_LIBRARIES}
        )
    endif()
endif()

# === Native C++ Tester
add_executable(mylns2_tester
    mylns2_tester.cpp
    Instance.cpp
    ConstraintTable.cpp
    PathTable.cpp
    ReservationTable.cpp
    SIPP.cpp
    SingleAgentSolver.cpp
    common.cpp
    mylns2.cpp
)

if(USE_RERUN)
    if(OpenCV_FOUND)
        target_link_libraries(mylns2_tester PRIVATE
            ${Boost_LIBRARIES}
            ${OpenCV_LIBS}
            ${FILESYSTEM_LIBRARIES}
            rerun_sdk
        )
    else()
        target_link_libraries(mylns2_tester PRIVATE
            ${Boost_LIBRARIES}
            ${FILESYSTEM_LIBRARIES}
            rerun_sdk
        )
    endif()
else()
    if(OpenCV_FOUND)
        target_link_libraries(mylns2_tester PRIVATE
            ${Boost_LIBRARIES}
            ${OpenCV_LIBS}
            ${FILESYSTEM_LIBRARIES}
        )
    else()
        target_link_libraries(mylns2_tester PRIVATE
            ${Boost_LIBRARIES}
            ${FILESYSTEM_LIBRARIES}
        )
    endif()
endif()