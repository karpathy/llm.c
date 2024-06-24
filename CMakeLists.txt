cmake_minimum_required(VERSION 3.14)

project(llm.cpp)

set(CMAKE_CXX_STANDARD 17)
enable_testing()

# OpenMP
find_package(OpenMP REQUIRED)
add_compile_definitions(OMP)
add_compile_options(-Ofast -march=native)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")

# Abseil
set(ABSL_PROPAGATE_CXX_STD ON)
add_subdirectory(third_party/abseil-cpp)

# Glog
#add_subdirectory(third_party/glog)
#include_directories(third_party/glog)

# GoogleTest
add_subdirectory(third_party/googletest)

# Eigen
set(EIGEN3_INCLUDE_DIR third_party/eigen)
include_directories(${EIGEN3_INCLUDE_DIR})

# llm.c
add_executable(train_gpt2 train_gpt2.c)
target_link_libraries(train_gpt2 m OpenMP::OpenMP_CXX)

add_executable(test_gpt2 test_gpt2.c)
target_link_libraries(test_gpt2 m OpenMP::OpenMP_CXX)

# llm.cpp
add_executable(test_gpt2_cpp test_gpt2.cpp)
target_link_libraries(test_gpt2_cpp absl::strings absl::log absl::check)

add_executable(nn_test nn_test.cpp)
target_link_libraries(nn_test absl::strings absl::log absl::check GTest::gtest_main)

add_executable(gpt_test gpt_test.cpp)
target_link_libraries(gpt_test absl::strings absl::log absl::check GTest::gtest_main)
