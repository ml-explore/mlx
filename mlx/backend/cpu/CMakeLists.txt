if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  set(COMPILER ${CMAKE_C_COMPILER})
  set(CLANG TRUE)
else()
  set(COMPILER ${CMAKE_CXX_COMPILER})
  set(CLANG FALSE)
  # On Linux the CPU JIT compiles every kernel with `g++` at run time (see
  # jit_compiler.cpp), so the preamble has to be preprocessed by g++ too.
  # libstdc++ preprocessed by clang++ uses Clang-only builtins that g++ rejects,
  # which breaks mx.compile in Clang builds.
  if(UNIX AND NOT CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    find_program(MLX_CPU_JIT_GXX g++)
    if(MLX_CPU_JIT_GXX)
      set(COMPILER ${MLX_CPU_JIT_GXX})
    else()
      message(
        WARNING "g++ not found: the CPU JIT preamble is preprocessed with "
                "${CMAKE_CXX_COMPILER}, and mx.compile may fail at run time.")
    endif()
  endif()
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

if(WIN32)
  set(SHELL_EXT ps1)
  set(SHELL_CMD powershell -ExecutionPolicy Bypass -File)
else()
  set(SHELL_EXT sh)
  set(SHELL_CMD bash)
endif()

add_custom_command(
  OUTPUT compiled_preamble.cpp
  COMMAND
    ${SHELL_CMD} ${CMAKE_CURRENT_SOURCE_DIR}/make_compiled_preamble.${SHELL_EXT}
    ${CMAKE_CURRENT_BINARY_DIR}/compiled_preamble.cpp ${COMPILER}
    ${PROJECT_SOURCE_DIR} ${CLANG} ${CMAKE_SYSTEM_PROCESSOR}
    "${PREAMBLE_SIMD_FLAGS}"
  DEPENDS make_compiled_preamble.${SHELL_EXT} compiled_preamble.h
          ${COMPILE_DEPS})

add_custom_target(cpu_compiled_preamble DEPENDS compiled_preamble.cpp)

add_dependencies(mlx cpu_compiled_preamble)

target_sources(
  mlx
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/device_info.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/arg_reduce.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/binary.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/conv.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/copy.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/distributed.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/eig.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/eigh.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/encoder.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/fft.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/hadamard.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/matmul.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/gemms/cblas.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/masked_mm.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/primitives.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/quantized.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/reduce.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/scan.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/select.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/softmax.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/logsumexp.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/sort.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/threefry.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/threading.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/indexing.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/luf.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/qrf.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/svd.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/inverse.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/cholesky.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/unary.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/eval.cpp
          ${CMAKE_CURRENT_BINARY_DIR}/compiled_preamble.cpp)

if(MLX_BUILD_ACCELERATE)
  target_sources(mlx PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/gemms/bnns.cpp)
else()
  target_sources(mlx PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/gemms/simd_fp16.cpp
                             ${CMAKE_CURRENT_SOURCE_DIR}/gemms/simd_bf16.cpp)
endif()

if(IOS)
  target_sources(mlx PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../no_cpu/compiled.cpp)
else()
  target_sources(mlx PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/compiled.cpp
                             ${CMAKE_CURRENT_SOURCE_DIR}/jit_compiler.cpp)
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  # Match clang's default floating point exception semantics. MLX does not read
  # FP exception flags, and without this GCC refuses to if-convert (and
  # therefore vectorize) loops with selects, such as the exp approximation used
  # by exp, softmax and sigmoid. Results are unchanged (unlike -ffast-math).
  target_compile_options(mlx
                         PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-fno-trapping-math>)
endif()
