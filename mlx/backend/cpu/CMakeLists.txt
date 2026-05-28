if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  set(COMPILER ${CMAKE_C_COMPILER})
  set(CLANG TRUE)
else()
  set(COMPILER ${CMAKE_CXX_COMPILER})
  set(CLANG FALSE)
endif()

set(MLX_HIGHWAY_INCLUDE_DIR "")

if(MLX_ENABLE_AVX2)
  if(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|x86")
    message(FATAL_ERROR "MLX_ENABLE_AVX2 requires an x86 target")
  endif()

  include(CheckCXXCompilerFlag)
  if(MSVC)
    check_cxx_compiler_flag("/arch:AVX2" MLX_COMPILER_SUPPORTS_AVX2)
    if(NOT MLX_COMPILER_SUPPORTS_AVX2)
      message(FATAL_ERROR "MLX_ENABLE_AVX2 requires /arch:AVX2 support")
    endif()
    set(MLX_AVX2_COMPILE_OPTIONS /arch:AVX2)
    set(PREAMBLE_SIMD_FLAGS "/arch:AVX2")
  else()
    # Highway's AVX2 target assumes the x86 feature set shipped with mainstream
    # Intel/AMD AVX2 CPUs. VMs that mask BMI2/FMA/F16C need a separate, stricter
    # baseline.
    check_cxx_compiler_flag("-mavx2 -mbmi2 -mfma -mf16c"
                            MLX_COMPILER_SUPPORTS_AVX2)
    if(NOT MLX_COMPILER_SUPPORTS_AVX2)
      message(
        FATAL_ERROR
          "MLX_ENABLE_AVX2 requires -mavx2 -mbmi2 -mfma -mf16c support")
    endif()
    set(MLX_AVX2_COMPILE_OPTIONS -mavx2 -mbmi2 -mfma -mf16c)
    set(PREAMBLE_SIMD_FLAGS "-mavx2 -mbmi2 -mfma -mf16c")
  endif()

  # MLX_ENABLE_AVX2 deliberately builds one Highway x86 target: AVX2.
  # HWY_DISABLED_TARGETS masks wider AVX-512-class targets, and
  # HWY_COMPILE_ONLY_STATIC keeps lower fallback targets out of the MLX binary.
  # Future AVX512 support could update this target set explicitly.
  set(MLX_HIGHWAY_AVX2_ONLY_DEFINITIONS
      "HWY_DISABLED_TARGETS=(HWY_AVX2-1)" HWY_DISABLE_PCLMUL_AES
      HWY_COMPILE_ONLY_STATIC)

  set(HWY_ENABLE_CONTRIB
      OFF
      CACHE BOOL "" FORCE)
  set(HWY_ENABLE_EXAMPLES
      OFF
      CACHE BOOL "" FORCE)
  set(HWY_ENABLE_INSTALL
      ON
      CACHE BOOL "" FORCE)
  set(HWY_ENABLE_TESTS
      OFF
      CACHE BOOL "" FORCE)
  set(HWY_FORCE_STATIC_LIBS
      ON
      CACHE BOOL "" FORCE)
  if(MLX_HIGHWAY_SOURCE_DIR)
    set(MLX_HIGHWAY_INCLUDE_DIR ${MLX_HIGHWAY_SOURCE_DIR})
    add_subdirectory(${MLX_HIGHWAY_SOURCE_DIR} ${CMAKE_BINARY_DIR}/highway)
  else()
    FetchContent_Declare(
      highway
      GIT_REPOSITORY https://github.com/google/highway.git
      GIT_TAG ${MLX_HIGHWAY_GIT_TAG}
      GIT_SHALLOW TRUE
      EXCLUDE_FROM_ALL)
    FetchContent_MakeAvailable(highway)
    set(MLX_HIGHWAY_INCLUDE_DIR ${highway_SOURCE_DIR})
  endif()
  if(NOT TARGET hwy)
    message(FATAL_ERROR "MLX_ENABLE_AVX2 requires Highway target hwy")
  endif()
  if(NOT TARGET hwy::hwy)
    add_library(hwy::hwy ALIAS hwy)
  endif()

  target_compile_definitions(hwy PRIVATE ${MLX_HIGHWAY_AVX2_ONLY_DEFINITIONS})
  target_compile_options(hwy PRIVATE ${MLX_AVX2_COMPILE_OPTIONS})
  target_compile_definitions(mlx PRIVATE MLX_USE_HIGHWAY
                                         ${MLX_HIGHWAY_AVX2_ONLY_DEFINITIONS})
  target_compile_options(mlx PRIVATE ${MLX_AVX2_COMPILE_OPTIONS})
  target_link_libraries(mlx PRIVATE hwy::hwy)
  message(STATUS "x86 SIMD: AVX2 enabled (${PREAMBLE_SIMD_FLAGS})")
endif()

set(COMPILE_DEPS
    ${PROJECT_SOURCE_DIR}/mlx/types/half_types.h
    ${PROJECT_SOURCE_DIR}/mlx/types/fp16.h
    ${PROJECT_SOURCE_DIR}/mlx/types/bf16.h
    ${PROJECT_SOURCE_DIR}/mlx/types/complex.h
    simd/simd.h
    simd/base_simd.h
    simd/highway_simd.h
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
    "${PREAMBLE_SIMD_FLAGS}" "${MLX_HIGHWAY_INCLUDE_DIR}"
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
          ${CMAKE_CURRENT_SOURCE_DIR}/indexing.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/luf.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/qrf.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/svd.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/inverse.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/cholesky.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/unary.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/norms.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/rope.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/sdpa.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/eval.cpp
          ${CMAKE_CURRENT_BINARY_DIR}/compiled_preamble.cpp)

if(MLX_ENABLE_AVX2)
  target_sources(mlx PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/norms_highway.cpp)
  target_sources(mlx PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/quantized_highway.cpp)
  target_sources(mlx PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/rope_highway.cpp)
endif()

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

# Include threading backend (platform-specific implementation)
add_subdirectory(threading)
