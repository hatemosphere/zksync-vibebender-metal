cmake_minimum_required(VERSION 3.28)
# Cmake must be at least 3.28 - the earlier one didn't pass cuda_standard properly.
project(gpu_prover_new_native)
enable_language(CUDA)
if ((NOT DEFINED CMAKE_CUDA_ARCHITECTURES) OR (CMAKE_CUDA_ARCHITECTURES STREQUAL ""))
    set(CMAKE_CUDA_ARCHITECTURES native)
endif ()
add_library(gpu_prover_new_native STATIC
        common.cuh
        context.cu
        context.cuh
        field.cuh
        memory.cuh
        ntt/evals_to_monomials_three_pass.cu
        ntt/evals_to_monomials_two_pass.cu
        ntt/monomials_to_evals_three_pass.cu
        ntt/monomials_to_evals_two_pass.cu
        ntt/ntt.cuh
        ptx.cuh
)
add_subdirectory(bench)
add_subdirectory(ops)
set_target_properties(gpu_prover_new_native PROPERTIES CUDA_STANDARD 20)
set_target_properties(gpu_prover_new_native PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(gpu_prover_new_native PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_compile_options(gpu_prover_new_native PRIVATE --expt-relaxed-constexpr)
target_compile_options(gpu_prover_new_native PRIVATE --ptxas-options=-v)
target_compile_options(gpu_prover_new_native PRIVATE -lineinfo)
target_compile_options(gpu_prover_new_native PRIVATE -g)
target_compile_options(gpu_prover_new_native PRIVATE --keep)
install(TARGETS gpu_prover_new_native DESTINATION .)
