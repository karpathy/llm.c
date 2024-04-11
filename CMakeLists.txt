cmake_minimum_required(VERSION 3.23)

project(llm.c
    LANGUAGES C
    VERSION 0.1.0
)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED True)

set(GPT2_TRAIN_SOURCES platform_utils.h)
set(GPT2_TRAIN_DEFINES)
set(GPT2_TEST_SOURCES platform_utils.h)
set(GPT2_LIB_DEPENDENCIES)
set(GPT2_CUSTOM_DEPENDENCIES)
set(GPT2_COMPILE_FLAGS)

# optional cuda support
if(ENABLE_CUDA)
    find_package(CUDAToolkit)
    if(NOT CUDAToolkit_FOUND AND WIN32)
        message(SEND_ERROR "CUDA is not installed. Go to https://developer.nvidia.com/cuda-toolkit")
    endif()

    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 11)
    set(CMAKE_CUDA_STANDARD_REQUIRED True)

    list(APPEND GPT2_TRAIN_SOURCES train_gpt2.cu)
    list(APPEND GPT2_TEST_SOURCES test_gpt2.cu)

    list(APPEND GPT2_LIB_DEPENDENCIES CUDA::cublas)

    # https://forums.developer.nvidia.com/t/suppress-warning-conversion-from-a-string-literal-to-char-is-deprecated/258050/3
    list(APPEND GPT2_COMPILE_FLAGS "-diag-suppress" "2464")
else()
    list(APPEND GPT2_TRAIN_SOURCES train_gpt2.c)
    list(APPEND GPT2_TEST_SOURCES test_gpt2.c)

    find_package(OpenMP)
    if(OpenMP_C_FOUND)
        list(APPEND GPT2_LIB_DEPENDENCIES OpenMP::OpenMP_C)
        list(APPEND GPT2_TRAIN_DEFINES OMP)
    endif()
endif()

if(NOT WIN32)
list(APPEND GPT2_LIB_DEPENDENCIES m)
endif()

# python requirements
find_package(Python3 REQUIRED COMPONENTS Interpreter)
set(PYTHON_REQUIREMENTS_INSTALLED_MARKER "${CMAKE_BINARY_DIR}/python_requirements_installed.marker")
add_custom_command(
    COMMAND ${Python3_EXECUTABLE} -m pip install -r "${CMAKE_SOURCE_DIR}/requirements.txt" --quiet
    COMMAND ${CMAKE_COMMAND} -E touch ${PYTHON_REQUIREMENTS_INSTALLED_MARKER}
    OUTPUT ${PYTHON_REQUIREMENTS_INSTALLED_MARKER}
    COMMENT "Installing requirements..."
)
add_custom_target(python_requirements ALL
    DEPENDS ${PYTHON_REQUIREMENTS_INSTALLED_MARKER}
)

# tinyshakespeare
set(PREPRO_TINYSHAKESPEARE_MARKER "${CMAKE_BINARY_DIR}/prepro_tinyshakespeare.marker")
add_custom_command(
    COMMAND ${Python3_EXECUTABLE} prepro_tinyshakespeare.py
    COMMAND ${CMAKE_COMMAND} -E touch ${PREPRO_TINYSHAKESPEARE_MARKER}
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT ${PREPRO_TINYSHAKESPEARE_MARKER}
    DEPENDS python_requirements
    COMMENT "Tokenizing TinyShakespeare..."
)
add_custom_target(prepro_tinyshakespeare ALL
    DEPENDS ${PREPRO_TINYSHAKESPEARE_MARKER}
)
list(APPEND GPT2_CUSTOM_DEPENDENCIES prepro_tinyshakespeare)

# TODO: add optional custom target and configuration prefix for tinystories pre-processing

# PyTorch gpt2
set(PYTORCH_GPT2_TRAIN_MARKER "${CMAKE_BINARY_DIR}/pytorch_gpt2.marker")
add_custom_command(
    OUTPUT ${PYTORCH_GPT2_TRAIN_MARKER}
    COMMAND ${Python3_EXECUTABLE} "${CMAKE_SOURCE_DIR}/train_gpt2.py"
    COMMAND ${CMAKE_COMMAND} -E touch ${PYTORCH_GPT2_TRAIN_MARKER}
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Training PyTorch GPT-2 reference implementation..."
)
add_custom_target(pytorch_gpt2_train ALL
    DEPENDS ${PYTORCH_GPT2_TRAIN_MARKER} prepro_tinyshakespeare
)
list(APPEND GPT2_CUSTOM_DEPENDENCIES pytorch_gpt2_train)

# train_gpt2
add_executable(train_gpt2
    ${GPT2_TRAIN_SOURCES}
)
target_link_libraries(train_gpt2
    PRIVATE
        ${GPT2_LIB_DEPENDENCIES}
)
target_compile_definitions(train_gpt2
    PRIVATE
        ${GPT2_TRAIN_DEFINES}
)
add_dependencies(train_gpt2
    ${GPT2_CUSTOM_DEPENDENCIES}
)
target_compile_options(train_gpt2
    PRIVATE
        ${GPT2_COMPILE_FLAGS}
)

# test_gpt2
add_executable(test_gpt2
    ${GPT2_TEST_SOURCES}
)
target_link_libraries(test_gpt2
    PRIVATE
        ${GPT2_LIB_DEPENDENCIES}
)
add_dependencies(test_gpt2
    ${GPT2_CUSTOM_DEPENDENCIES}
)
target_compile_options(test_gpt2
    PRIVATE
        ${GPT2_COMPILE_FLAGS}
)
