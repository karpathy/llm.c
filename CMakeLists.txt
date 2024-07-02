cmake_minimum_required(VERSION 3.14)

project(llm.cpp)

set(CMAKE_CXX_STANDARD 17)
set(BUILD_SHARED_LIBS OFF)
add_compile_options(-Ofast -march=native)
enable_testing()
include_directories(.)

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
#add_definitions(-DEIGEN_FAST_MATH=0)
#add_definitions(-DEIGEN_DONT_PARALLELIZE)
#add_definitions(-DEIGEN_DONT_VECTORIZE)
include_directories(${EIGEN3_INCLUDE_DIR})

# llm.c
add_executable(train_gpt2 train_gpt2.c)
target_link_libraries(train_gpt2 m ${OpenMP_CXX_LIBRARIES})

add_executable(test_gpt2 test_gpt2.c)
target_link_libraries(test_gpt2 m ${OpenMP_CXX_LIBRARIES})

# llm.cpp
add_executable(test_gpt2_cpp llmcpp/test_gpt2.cpp)
target_link_libraries(test_gpt2_cpp absl::strings absl::log absl::check)

add_executable(nn_test llmcpp/nn_test.cpp)
target_link_libraries(nn_test absl::strings absl::log absl::check GTest::gtest_main)

add_executable(optim_test llmcpp/optim_test.cpp)
target_link_libraries(optim_test absl::strings absl::log absl::check GTest::gtest_main)

add_executable(gpt_test llmcpp/gpt_test.cpp)
target_link_libraries(gpt_test absl::strings absl::log absl::check GTest::gtest_main)

add_executable(gpt_optim llmcpp/gpt_optim.cpp)
target_link_libraries(gpt_optim absl::strings absl::log absl::check GTest::gtest_main)
