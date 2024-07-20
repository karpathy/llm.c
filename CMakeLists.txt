cmake_minimum_required(VERSION 3.14)
project(llm.cpp LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(BUILD_SHARED_LIBS OFF)
#add_compile_options(-Ofast -march=native)
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Ofast -march=native")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast -march=native")


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
find_package(CUDA)
if (CUDA_FOUND)
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
    target_compile_definitions(test_eigen_gpu PUBLIC EIGEN_USE_GPU)
    set_target_properties(test_eigen_gpu PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_ARCHITECTURES "61;70;75"
    )
    target_link_libraries(test_eigen_gpu
            absl::strings absl::log absl::check
            #            ${CUDA_LIBRARIES} cublas cublasLt
    )
    #    target_compile_options(test_eigen_gpu PRIVATE -Xcompiler=-Ofast,-march=native)

    # nn_test_cu
    add_executable(nn_test_cu llmcpp/nn_test.cu)
    target_compile_definitions(nn_test_cu PRIVATE EIGEN_USE_GPU)
    set_target_properties(nn_test_cu PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_ARCHITECTURES "61;70;75"
    )
    target_link_libraries(nn_test_cu
            absl::strings absl::log absl::check
            GTest::gtest_main
            ${CUDA_LIBRARIES}
    )
endif ()


# llm.cpp
add_executable(test_gpt2_cpp llmcpp/test_gpt2.cpp)
target_link_libraries(test_gpt2_cpp absl::strings absl::log absl::check)
target_compile_options(test_gpt2_cpp PRIVATE -Ofast -march=native)

add_executable(train_gpt2_cpp llmcpp/train_gpt2.cpp)
target_link_libraries(train_gpt2_cpp
        absl::strings absl::log absl::check
        profiler
)
target_compile_options(train_gpt2_cpp PRIVATE -Ofast -march=native)

add_executable(nn_test llmcpp/nn_test.cpp)
target_link_libraries(nn_test absl::strings absl::log absl::check GTest::gtest_main)

add_executable(optim_test llmcpp/optim_test.cpp)
target_link_libraries(optim_test absl::strings absl::log absl::check GTest::gtest_main)

add_executable(gpt_test llmcpp/gpt_test.cpp)
target_link_libraries(gpt_test absl::strings absl::log absl::check GTest::gtest_main)

add_executable(gpt_optim llmcpp/gpt_optim.cpp)
target_link_libraries(gpt_optim absl::strings absl::log absl::check GTest::gtest_main)

add_executable(test_eigen_cpu llmcpp/test_eigen_cpu.cpp)
target_link_libraries(test_eigen_cpu absl::strings absl::log absl::check GTest::gtest_main)
target_compile_options(test_eigen_cpu PRIVATE -Ofast -march=native)