cmake_minimum_required(VERSION 3.14)
project(llm.cpp LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(BUILD_SHARED_LIBS OFF)
#add_compile_options(-Ofast -march=native)
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Ofast -march=native")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast -march=native")

find_program(CCACHE_PROGRAM ccache)
if (CCACHE_PROGRAM)
    message("--- ccache: ${CCACHE_PROGRAM}")
    set(CMAKE_C_COMPILER_LAUNCHER ccache)
    set(CMAKE_CXX_COMPILER_LAUNCHER ccache)
    set(CMAKE_CUDA_COMPILER_LAUNCHER ccache)
endif ()

enable_testing()
include_directories(.)

# OpenMP
find_package(OpenMP)
if (OpenMP_FOUND)
    add_compile_definitions(OMP)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
endif ()

# Abseil
set(ABSL_PROPAGATE_CXX_STD ON)
add_subdirectory(third_party/abseil-cpp)

# GoogleTest
add_subdirectory(third_party/googletest)

# Eigen
set(EIGEN3_INCLUDE_DIR third_party/eigen)
add_definitions(-DEIGEN_DONT_PARALLELIZE)
#add_definitions(-DEIGEN_DONT_VECTORIZE)
add_definitions(-DEIGEN_USE_THREADS)
include_directories(${EIGEN3_INCLUDE_DIR})

# llm.c
add_executable(train_gpt2 train_gpt2.c)
target_link_libraries(train_gpt2 m ${OpenMP_CXX_LIBRARIES})

add_executable(test_gpt2 test_gpt2.c)
target_link_libraries(test_gpt2 m ${OpenMP_CXX_LIBRARIES})

# llm.c GPU
set(CMAKE_CUDA_ARCHITECTURES 60 61 70 75)
find_package(CUDA)
if (CUDA_FOUND)
    set_source_files_properties(llmcpp/pch.hpp PROPERTIES HEADER_FILE_ONLY TRUE)
    add_library(pch_gpu INTERFACE)
    target_sources(pch_gpu INTERFACE llmcpp/pch.hpp)

    add_library(nn_gpu llmcpp/nn.cu)
    target_compile_definitions(nn_gpu PUBLIC EIGEN_USE_GPU)
    target_link_libraries(nn_gpu
            pch_gpu
            absl::strings absl::log absl::check
            ${CUDA_LIBRARIES}
    )
    target_precompile_headers(nn_gpu PUBLIC llmcpp/pch.hpp)

    add_library(gpt_gpu llmcpp/gpt.cu)
    target_link_libraries(gpt_gpu
            nn_gpu
    )
    target_precompile_headers(gpt_gpu PUBLIC llmcpp/pch.hpp)

    #    add_compile_definitions(EIGEN_USE_GPU)
    #SET(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr)
    add_compile_definitions(ENABLE_FP32)
    add_executable(train_gpt2cu train_gpt2.cu)
    set_target_properties(train_gpt2cu PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_ARCHITECTURES "61;70;75"
    )
    target_link_libraries(train_gpt2cu ${CUDA_LIBRARIES} cublas cublasLt)

    add_executable(test_eigen_gpu llmcpp/test_eigen_gpu.cu)
    #    target_compile_definitions(test_eigen_gpu PRIVATE EIGEN_USE_GPU)
    set_target_properties(test_eigen_gpu PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_ARCHITECTURES "61;70;75"
    )
    target_link_libraries(test_eigen_gpu
            absl::strings absl::log absl::check
            #            ${CUDA_LIBRARIES} cublas cublasLt
    )
    #    target_compile_options(test_eigen_gpu PRIVATE -Xcompiler=-Ofast,-march=native)

    # nn_test_gpu
    add_executable(nn_test_gpu llmcpp/nn_test.cu)
    target_link_libraries(nn_test_gpu
            nn_gpu
            GTest::gtest_main
    )

    # gpt_test_gpu
    add_executable(gpt_test_gpu llmcpp/gpt_test.cu)
    target_link_libraries(gpt_test_gpu
            gpt_gpu
            GTest::gtest_main
    )

    # gpt_optim_gpu
    add_executable(gpt_optim_gpu llmcpp/gpt_optim.cu)
    target_link_libraries(gpt_optim_gpu
            gpt_gpu
    )

    # train_gpt2_gpu
    add_executable(train_gpt2_gpu llmcpp/train_gpt2.cu)
    target_link_libraries(train_gpt2_gpu
            gpt_gpu
    )
    target_precompile_headers(train_gpt2_gpu PUBLIC llmcpp/pch.hpp)
endif ()

set_source_files_properties(llmcpp/pch.hpp PROPERTIES HEADER_FILE_ONLY TRUE)
add_library(pch INTERFACE)
target_sources(pch INTERFACE llmcpp/pch.hpp)

# llm.cpp
add_library(nn llmcpp/nn.cpp)
target_link_libraries(nn
        pch
        absl::strings absl::log absl::check)
target_precompile_headers(nn PUBLIC llmcpp/pch.hpp)

add_library(gpt llmcpp/gpt.cpp)
target_link_libraries(gpt nn)

add_library(gpt2 llmcpp/gpt2.cpp)
target_link_libraries(gpt2 gpt)

add_library(optim llmcpp/optim.cpp)
target_link_libraries(optim nn)

add_executable(test_gpt2_cpu llmcpp/test_gpt2.cpp)
target_link_libraries(test_gpt2_cpu gpt2 optim)
target_compile_options(test_gpt2_cpu PRIVATE -Ofast -march=native)

add_executable(train_gpt2_cpu llmcpp/train_gpt2.cpp)
target_link_libraries(train_gpt2_cpu
        gpt2 optim
        profiler
)
target_compile_options(train_gpt2_cpu PRIVATE -Ofast -march=native)

add_executable(nn_test llmcpp/nn_test.cpp)
target_link_libraries(nn_test nn GTest::gtest_main)

add_executable(optim_test llmcpp/optim_test.cpp)
target_link_libraries(optim_test nn GTest::gtest_main)

add_executable(gpt_test llmcpp/gpt_test.cpp)
target_link_libraries(gpt_test gpt GTest::gtest_main)

add_executable(gpt_optim llmcpp/gpt_optim.cpp)
target_link_libraries(gpt_optim gpt)

add_executable(test_eigen_cpu llmcpp/test_eigen_cpu.cpp)
target_link_libraries(test_eigen_cpu absl::strings absl::log absl::check GTest::gtest_main)
target_compile_options(test_eigen_cpu PRIVATE -Ofast -march=native)
