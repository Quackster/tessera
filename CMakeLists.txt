# Tessera: C++ LLM inference engine (vulkan / rocm backends).
cmake_minimum_required(VERSION 3.30)
project(tessera LANGUAGES CXX)

# std::expected (mandated by AGENTS.md) is a C++23 feature of libstdc++;
# the code itself stays C++20-style.
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(TESSERA_BACKEND "vulkan" CACHE STRING "Compute backend: vulkan or rocm")
set_property(CACHE TESSERA_BACKEND PROPERTY STRINGS vulkan rocm)

add_library(tessera STATIC
    src/core/decode.cpp
    src/core/decode_device.cpp
    src/core/decode_hybrid.cpp
    src/core/decode_mtp.cpp
    src/core/engine.cpp
    src/core/model.cpp
    src/core/files.cpp
    src/core/tokenizer.cpp
    src/core/buffer.cpp
    src/core/loaders/gguf.cpp
    src/core/loaders/safetensors.cpp
    src/core/json.cpp
    src/core/sampling.cpp
    src/core/numerics/quant.cpp
    src/core/numerics/gemm.cpp
    src/core/numerics/attention.cpp
    src/core/numerics/norm.cpp
    src/core/numerics/conv.cpp
    src/core/numerics/selector.cpp
    src/core/numerics/concat.cpp
    src/serve/http.cpp
    src/serve/server.cpp
    src/serve/jinja/value.cpp
    src/serve/jinja/jinja_parse.cpp
    src/serve/jinja/jinja_eval.cpp
    src/spec/dflash2.cpp
    src/spec/dflash2_config.cpp
    src/spec/dflash2_conv.cpp
    src/spec/dflash2_mlp.cpp
    src/spec/dflash2_attention.cpp
    src/spec/dflash2_layer.cpp
    src/spec/dflash2_stack.cpp
    src/spec/dflash2_selector.cpp
    src/spec/dflash2_fuse.cpp
    src/spec/dflash2_context.cpp
    src/spec/dflash2_block.cpp
    src/spec/dflash2_weights.cpp
    src/spec/dflash2_candidates.cpp
    src/spec/dflash2_drafter.cpp
    src/spec/dflash2_mask.cpp
    src/spec/dflash2_generate.cpp
)
target_include_directories(tessera PUBLIC include PRIVATE src)
target_compile_options(tessera PRIVATE -Wall -Wextra)

if(TESSERA_BACKEND STREQUAL "vulkan")
    target_sources(tessera PRIVATE src/backends/vulkan/vulkan_backend.cpp
                                  src/backends/vulkan/vulkan_compute.cpp)
    target_include_directories(tessera PRIVATE src/backends
                            ${CMAKE_CURRENT_BINARY_DIR}/vulkan_kernels)
    find_package(Vulkan REQUIRED)
    target_link_libraries(tessera PRIVATE Vulkan::Vulkan)
    # Compile the built-in kernels to SPIR-V at configure time and embed
    # the bytes in a generated header (one array per kernel).
    find_program(TESSERA_GLSLANG glslangValidator REQUIRED)
    file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/vulkan_kernels)
    # Parallel lists: kernel file stems and their generated array
    # name prefixes.
    set(TESSERA_KERNEL_STEMS fill gemm_q4k attention rope gemm_fp8 gemm_fp8_block gemm_f32 gemm_bf16
        gemm_mxfp4 gemm_q5k gemm_q6k gemm_q3k gemm_iq4nl gemm_iq4xs
        gemm_iq3s gemm_q80 rmsnorm sigmoid_gate l2norm rmsnorm_gated
        conv1d conv1d_step dflash_conv selector_edge_score delta_step mrope qgate_split add silu_mul repeat_heads ssm_gate delta_step_heads concat_features)
    set(TESSERA_KERNEL_NAMES Fill GemmQ4k Attention Rope GemmFp8 GemmFp8Block GemmF32 GemmBf16
        GemmMxfp4 GemmQ5K GemmQ6K GemmQ3K GemmIq4Nl GemmIq4Xs GemmIq3S
        GemmQ80 Rmsnorm SigmoidGate L2Norm RmsnormGated Conv1d Conv1dStep DflashConv SelectorEdgeScore DeltaStep
        Mrope QgateSplit Add SiluMul RepeatHeads SsmGate DeltaStepHeads ConcatFeatures)
    set(TESSERA_KERNEL_HEADER
"#pragma once
// Generated at configure time from src/backends/vulkan/kernels/*.comp.
// Do not edit.
#include <cstddef>
namespace tessera::backends::vulkan {
")
    foreach(TESSERA_KERNEL IN LISTS TESSERA_KERNEL_STEMS)
        list(FIND TESSERA_KERNEL_STEMS ${TESSERA_KERNEL}
             TESSERA_KERNEL_INDEX)
        list(GET TESSERA_KERNEL_NAMES ${TESSERA_KERNEL_INDEX}
             TESSERA_KERNEL_NAME)
        execute_process(
            # -e is not used: this radv build crashes compiling a
            # non-main entry name, so the built-ins keep the standard
            # entry.
            COMMAND ${TESSERA_GLSLANG} -V -Os
                    ${CMAKE_CURRENT_SOURCE_DIR}/src/backends/vulkan/kernels/${TESSERA_KERNEL}.comp
                    -o ${CMAKE_CURRENT_BINARY_DIR}/vulkan_kernels/${TESSERA_KERNEL}.spirv
            RESULT_VARIABLE TESSERA_GLSLANG_RESULT)
        if(NOT TESSERA_GLSLANG_RESULT EQUAL 0)
            message(FATAL_ERROR "glslangValidator failed to compile ${TESSERA_KERNEL}.comp; install the Vulkan SDK shader tools")
        endif()
        file(READ ${CMAKE_CURRENT_BINARY_DIR}/vulkan_kernels/${TESSERA_KERNEL}.spirv
             TESSERA_SPIRV_HEX HEX)
        string(REPLACE " " "" TESSERA_SPIRV_HEX ${TESSERA_SPIRV_HEX})
        string(REGEX REPLACE "([0-9a-fA-F][0-9a-fA-F])"
               "static_cast<std::byte>(0x\\1)," TESSERA_SPIRV_ARR ${TESSERA_SPIRV_HEX})
        string(APPEND TESSERA_KERNEL_HEADER
"constexpr std::byte k${TESSERA_KERNEL_NAME}SpirV[] = {${TESSERA_SPIRV_ARR}};
")
    endforeach()
    string(APPEND TESSERA_KERNEL_HEADER
"}  // namespace tessera::backends::vulkan
")
    # The quoted expansion keeps the ';' terminators (unquoted expansion
    # splits the header on them).
    file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/vulkan_kernels/kernels_spirv.hpp
         "${TESSERA_KERNEL_HEADER}")
elseif(TESSERA_BACKEND STREQUAL "rocm")
    # The __global__ kernels need the HIP device compiler; the rest of the
    # target stays plain C++ (both use libstdc++, so they link together).
    find_package(ROCmCMakeBuildTools REQUIRED)
    # The ROCm toolchain needs a target GPU ISA. Detect the local device
    # before enabling the HIP language (3+ digit gfx ids; the bare family
    # prefix like "gfx11" is not an ISA).
    if(NOT DEFINED HIP_ARCHITECTURES)
        execute_process(COMMAND rocminfo
                        OUTPUT_VARIABLE TESSERA_ROCMINFO
                        RESULT_VARIABLE TESSERA_ROCMINFO_RC)
        if(TESSERA_ROCMINFO_RC EQUAL 0)
            string(REGEX MATCHALL "gfx[0-9]+" TESSERA_GFX_IDS
                   ${TESSERA_ROCMINFO})
            foreach(TESSERA_GFX_ID IN LISTS TESSERA_GFX_IDS)
                string(LENGTH "${TESSERA_GFX_ID}" TESSERA_GFX_LEN)
                if(TESSERA_GFX_LEN GREATER 4)
                    set(HIP_ARCHITECTURES ${TESSERA_GFX_ID})
                    break()
                endif()
            endforeach()
        endif()
    endif()
    if(NOT DEFINED HIP_ARCHITECTURES)
        message(FATAL_ERROR "cannot detect a ROCm GPU ISA; pass -D HIP_ARCHITECTURES=<gfx...> (e.g. gfx1100)")
    endif()
    enable_language(HIP)
    # The HIP device compiler emits PIC; keep every object in this build
    # position independent so the executables link.
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
    set_target_properties(tessera PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        HIP_ARCHITECTURES ${HIP_ARCHITECTURES})
    target_sources(tessera PRIVATE src/backends/rocm/rocm_backend.cpp
                                   src/backends/rocm/rocm_kernels_basic.cpp
                                   src/backends/rocm/rocm_kernels_block.cpp
                                   src/backends/rocm/rocm_kernels_iq.cpp)
    set_source_files_properties(
        src/backends/rocm/rocm_backend.cpp
        src/backends/rocm/rocm_kernels_basic.cpp
        src/backends/rocm/rocm_kernels_block.cpp
        src/backends/rocm/rocm_kernels_iq.cpp PROPERTIES LANGUAGE HIP)
    target_include_directories(tessera PRIVATE src/backends)
    # The ROCm headers require an explicit platform selection.
    target_compile_definitions(tessera PRIVATE __HIP_PLATFORM_AMD__)
    # System ROCm layout (headers in /usr/include, libs in /usr/lib64).
    find_library(TESSERA_HIP_LIB amdhip64 REQUIRED)
    target_link_libraries(tessera PRIVATE ${TESSERA_HIP_LIB})
else()
    message(FATAL_ERROR "TESSERA_BACKEND must be 'vulkan' or 'rocm' (got '${TESSERA_BACKEND}')")
endif()

add_executable(tessera-cli tools/cli/main.cpp)
target_link_libraries(tessera-cli PRIVATE tessera)

enable_testing()

include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.15.2
)
set(gtest_force_shared_cxx ON)
FetchContent_MakeAvailable(googletest)

# Single test binary: all suites live in tests/*.cpp.
add_executable(tessera-tests
    tests/types_test.cpp
    tests/gguf_test.cpp
    tests/safetensors_test.cpp
    tests/engine_test.cpp
    tests/decode_test.cpp
    tests/tokenizer_test.cpp
    tests/serve_test.cpp
    tests/jinja_test.cpp
    tests/backend_test.cpp
    tests/spec_test.cpp
)
target_link_libraries(tessera-tests PRIVATE tessera GTest::gtest GTest::gtest_main)
target_include_directories(tessera-tests PRIVATE src)

include(GoogleTest)
gtest_discover_tests(tessera-tests)
