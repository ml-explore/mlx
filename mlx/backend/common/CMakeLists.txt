if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  set(COMPILER ${CMAKE_C_COMPILER})
  set(CLANG TRUE)
else()
  set(COMPILER ${CMAKE_CXX_COMPILER})
endif()

set(COMPILE_DEPS
    ${PROJECT_SOURCE_DIR}/mlx/types/half_types.h
    ${PROJECT_SOURCE_DIR}/mlx/types/fp16.h
    ${PROJECT_SOURCE_DIR}/mlx/types/bf16.h
    ${PROJECT_SOURCE_DIR}/mlx/types/complex.h
    simd/simd.h
    simd/base_simd.h
    simd/math.h
    simd/type.h
    unary_ops.h
    binary_ops.h)

if(MSVC)
  set(SHELL_EXT ps1)
  set(SHELL_CMD powershell -ExecutionPolicy Bypass -File)
else()
  set(SHELL_EXT sh)
  set(SHELL_CMD /bin/bash)
endif()

add_custom_command(
  OUTPUT compiled_preamble.cpp
  COMMAND
    ${SHELL_CMD} ${CMAKE_CURRENT_SOURCE_DIR}/make_compiled_preamble.${SHELL_EXT}
    ${CMAKE_CURRENT_BINARY_DIR}/compiled_preamble.cpp ${COMPILER}
    ${PROJECT_SOURCE_DIR} ${CLANG} ${CMAKE_SYSTEM_PROCESSOR}
  DEPENDS make_compiled_preamble.${SHELL_EXT} compiled_preamble.h
          ${COMPILE_DEPS})

add_custom_target(cpu_compiled_preamble DEPENDS compiled_preamble.cpp)

add_dependencies(mlx cpu_compiled_preamble)

target_sources(
  mlx
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/arg_reduce.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/binary.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/compiled.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/common.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/conv.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/copy.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/eigh.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/erf.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/fft.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/hadamard.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/masked_mm.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/primitives.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/quantized.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/reduce.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/reduce_utils.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/scan.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/select.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/slicing.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/softmax.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/sort.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/threefry.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/indexing.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/load.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/qrf.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/svd.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/inverse.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/cholesky.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/unary.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/utils.cpp
          ${CMAKE_CURRENT_BINARY_DIR}/compiled_preamble.cpp)

if(IOS)
  target_sources(mlx PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/compiled_nocpu.cpp)
else()
  target_sources(mlx PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/compiled_cpu.cpp
                             ${CMAKE_CURRENT_SOURCE_DIR}/jit_compiler.cpp)
endif()
