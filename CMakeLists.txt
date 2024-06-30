cmake_minimum_required(VERSION 3.14)

project(llm.cpp)

set(CMAKE_CXX_STANDARD 17)
set(BUILD_SHARED_LIBS OFF)
add_compile_options(-Ofast -march=native)
enable_testing()

# OpenMP
find_package(OpenMP)
if (OpenMP_FOUND)
  add_compile_definitions(OMP)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
endif()

# Abseil
set(ABSL_PROPAGATE_CXX_STD ON)
add_subdirectory(third_party/abseil-cpp)

# GoogleTest
add_subdirectory(third_party/googletest)

# Eigen
set(EIGEN3_INCLUDE_DIR third_party/eigen)
#set(EIGEN_FAST_MATH 0)
include_directories(${EIGEN3_INCLUDE_DIR})

# llm.c
add_executable(train_gpt2 train_gpt2.c)
target_link_libraries(train_gpt2 m ${OpenMP_CXX_LIBRARIES})

add_executable(test_gpt2 test_gpt2.c)
target_link_libraries(test_gpt2 m ${OpenMP_CXX_LIBRARIES})

# llm.cpp
add_executable(test_gpt2_cpp test_gpt2.cpp)
target_link_libraries(test_gpt2_cpp absl::strings absl::log absl::check)

add_executable(nn_test nn_test.cpp)
target_link_libraries(nn_test absl::strings absl::log absl::check GTest::gtest_main)

add_executable(optim_test optim_test.cpp)
target_link_libraries(optim_test absl::strings absl::log absl::check GTest::gtest_main)

add_executable(gpt_test gpt_test.cpp)
target_link_libraries(gpt_test absl::strings absl::log absl::check GTest::gtest_main)
