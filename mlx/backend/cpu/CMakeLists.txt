if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  set(COMPILER ${CMAKE_C_COMPILER})
  set(CLANG TRUE)
else()
  set(COMPILER ${CMAKE_CXX_COMPILER})
  set(CLANG FALSE)
endif()

set(MLX_HIGHWAY_INCLUDE_DIR "")
set(PREAMBLE_SIMD_FLAGS "")
set(PREAMBLE_HWY_AVX2_FLAGS "")
set(PREAMBLE_HWY_AVX3_FLAGS "")

if(MLX_ENABLE_X86_SIMD)
  if(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|x86")
    message(FATAL_ERROR "MLX_ENABLE_X86_SIMD requires an x86 target")
  endif()

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
    message(FATAL_ERROR "MLX_ENABLE_X86_SIMD requires Highway target hwy")
  endif()
  if(NOT TARGET hwy::hwy)
    add_library(hwy::hwy ALIAS hwy)
  endif()

  set(MLX_HIGHWAY_TARGET_DEFINITIONS HWY_DISABLE_PCLMUL_AES)
  if(MSVC AND NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    # Highway marks its AVX3-family targets broken for MSVC. Mirror that policy
    # here by using Highway's target-filtering define.
    set(MLX_HIGHWAY_DISABLED_AVX3_TARGETS
        "HWY_DISABLED_TARGETS=(HWY_AVX3+HWY_AVX3_DL+HWY_AVX3_ZEN4+HWY_AVX3_SPR+HWY_AVX10_2)"
    )
    list(APPEND MLX_HIGHWAY_TARGET_DEFINITIONS
         ${MLX_HIGHWAY_DISABLED_AVX3_TARGETS})
  endif()

  # Keep ISA flags out of the main MLX target. Clang/GCC use Highway's
  # foreach-target attributes; MSVC uses per-target shims below because it needs
  # /arch:AVX2 on AVX2 codegen to avoid slow SSE/AVX transitions.
  target_compile_definitions(hwy PRIVATE ${MLX_HIGHWAY_TARGET_DEFINITIONS})
  target_compile_definitions(mlx PRIVATE MLX_USE_HIGHWAY_KERNELS
                                         ${MLX_HIGHWAY_TARGET_DEFINITIONS})
  target_include_directories(mlx PRIVATE ${MLX_HIGHWAY_INCLUDE_DIR})
  target_link_libraries(mlx PRIVATE hwy::hwy)
  if(MSVC)
    target_compile_definitions(mlx PRIVATE MLX_HAVE_JIT_PREAMBLE_HWY_AVX2)
    set(PREAMBLE_HWY_AVX2_FLAGS
        "/DMLX_USE_HIGHWAY_JIT_SIMD /DHWY_COMPILE_ONLY_STATIC /DHWY_DISABLE_PCLMUL_AES /arch:AVX2"
    )
  elseif(NOT WIN32)
    target_compile_definitions(mlx PRIVATE MLX_HAVE_JIT_PREAMBLE_HWY_AVX2)
    # Highway's AVX2 target includes BMI2, FMA, and F16C in its feature group.
    set(PREAMBLE_HWY_AVX2_FLAGS
        "-DMLX_USE_HIGHWAY_JIT_SIMD -DHWY_COMPILE_ONLY_STATIC -DHWY_DISABLE_PCLMUL_AES -mavx2 -mbmi2 -mfma -mf16c"
    )
    target_compile_definitions(mlx PRIVATE MLX_HAVE_JIT_PREAMBLE_HWY_AVX3)
    # Highway's HWY_AVX3 target maps to AVX-512F/BW/CD/DQ/VL plus AVX2-family
    # prerequisites; use explicit compiler ISA flags for the generated JIT
    # preamble.
    set(PREAMBLE_HWY_AVX3_FLAGS
        "-DMLX_USE_HIGHWAY_JIT_SIMD -DHWY_COMPILE_ONLY_STATIC -DHWY_DISABLE_PCLMUL_AES -mavx512f -mavx512vl -mavx512bw -mavx512dq -mavx512cd -mbmi2 -mfma -mf16c"
    )
  endif()
  if(WIN32 AND NOT MSVC)
    message(
      STATUS
        "x86 SIMD: Highway runtime dispatch enabled, Windows non-MSVC CPU JIT disabled"
    )
  elseif(WIN32)
    message(
      STATUS
        "x86 SIMD: Highway runtime dispatch enabled, Windows CPU JIT capped to HWY_AVX2"
    )
  else()
    message(
      STATUS
        "x86 SIMD: Highway runtime dispatch enabled, toolchain-selected targets"
    )
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

if(MLX_ENABLE_X86_SIMD)
  list(APPEND COMPILE_DEPS highway_utils.h simd/highway_simd.h)
endif()

if(WIN32)
  set(SHELL_EXT ps1)
  set(SHELL_CMD powershell -ExecutionPolicy Bypass -File)
else()
  set(SHELL_EXT sh)
  set(SHELL_CMD bash)
endif()

set(PREAMBLE_OUTPUTS ${CMAKE_CURRENT_BINARY_DIR}/compiled_preamble.cpp)
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/compiled_preamble.cpp
  COMMAND
    ${SHELL_CMD} ${CMAKE_CURRENT_SOURCE_DIR}/make_compiled_preamble.${SHELL_EXT}
    ${CMAKE_CURRENT_BINARY_DIR}/compiled_preamble.cpp ${COMPILER}
    ${PROJECT_SOURCE_DIR} ${CLANG} ${CMAKE_SYSTEM_PROCESSOR}
    "${PREAMBLE_SIMD_FLAGS}" "" get_prebuilt_preamble
  DEPENDS make_compiled_preamble.${SHELL_EXT} compiled_preamble.h
          ${COMPILE_DEPS})

if(PREAMBLE_HWY_AVX2_FLAGS)
  list(APPEND PREAMBLE_OUTPUTS
       ${CMAKE_CURRENT_BINARY_DIR}/compiled_preamble_avx2.cpp)
  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/compiled_preamble_avx2.cpp
    COMMAND
      ${SHELL_CMD}
      ${CMAKE_CURRENT_SOURCE_DIR}/make_compiled_preamble.${SHELL_EXT}
      ${CMAKE_CURRENT_BINARY_DIR}/compiled_preamble_avx2.cpp ${COMPILER}
      ${PROJECT_SOURCE_DIR} ${CLANG} ${CMAKE_SYSTEM_PROCESSOR}
      "${PREAMBLE_HWY_AVX2_FLAGS}" "${MLX_HIGHWAY_INCLUDE_DIR}"
      get_prebuilt_preamble_avx2
    DEPENDS make_compiled_preamble.${SHELL_EXT} compiled_preamble.h
            ${COMPILE_DEPS})
endif()

if(PREAMBLE_HWY_AVX3_FLAGS)
  list(APPEND PREAMBLE_OUTPUTS
       ${CMAKE_CURRENT_BINARY_DIR}/compiled_preamble_avx3.cpp)
  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/compiled_preamble_avx3.cpp
    COMMAND
      ${SHELL_CMD}
      ${CMAKE_CURRENT_SOURCE_DIR}/make_compiled_preamble.${SHELL_EXT}
      ${CMAKE_CURRENT_BINARY_DIR}/compiled_preamble_avx3.cpp ${COMPILER}
      ${PROJECT_SOURCE_DIR} ${CLANG} ${CMAKE_SYSTEM_PROCESSOR}
      "${PREAMBLE_HWY_AVX3_FLAGS}" "${MLX_HIGHWAY_INCLUDE_DIR}"
      get_prebuilt_preamble_avx3
    DEPENDS make_compiled_preamble.${SHELL_EXT} compiled_preamble.h
            ${COMPILE_DEPS})
endif()

add_custom_target(cpu_compiled_preamble DEPENDS ${PREAMBLE_OUTPUTS})

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
          ${PREAMBLE_OUTPUTS})

if(MLX_ENABLE_X86_SIMD)
  if(MSVC AND NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(MLX_HIGHWAY_MSVC_DISPATCH_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/norms_highway_dispatch.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/quantized_highway_dispatch.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/rope_highway_dispatch.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/sdpa_highway_dispatch.cpp)
    set(MLX_HIGHWAY_MSVC_TARGET_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/norms_highway.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/quantized_highway.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/rope_highway.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/sdpa_highway.cpp)

    # Native MSVC needs per-target object builds because /arch:AVX2 is TU-wide.
    function(mlx_add_highway_msvc_target target_name suffix)
      add_library(${target_name} OBJECT ${MLX_HIGHWAY_MSVC_TARGET_SOURCES})
      target_link_libraries(${target_name} PRIVATE hwy::hwy)
      target_include_directories(
        ${target_name} PRIVATE ${PROJECT_SOURCE_DIR} ${MLX_HIGHWAY_INCLUDE_DIR})
      target_compile_definitions(
        ${target_name}
        PRIVATE MLX_USE_HIGHWAY_KERNELS
                ${MLX_HIGHWAY_TARGET_DEFINITIONS}
                MLX_HIGHWAY_MANUAL_TARGET
                MLX_HIGHWAY_TARGET_SUFFIX=${suffix}
                HWY_COMPILE_ONLY_STATIC
                NOMINMAX
                WIN32_LEAN_AND_MEAN
                FMT_UNICODE=0)
      if(BUILD_SHARED_LIBS)
        target_compile_definitions(${target_name} PRIVATE MLX_EXPORT)
      else()
        target_compile_definitions(${target_name} PRIVATE MLX_STATIC)
      endif()
      target_compile_options(${target_name} PRIVATE /bigobj /Zc:preprocessor)
    endfunction()

    mlx_add_highway_msvc_target(mlx_highway_msvc_avx2 _avx2)
    # Highway's AVX2 target assumes AVX2/BMI2/FMA/F16C. Limit /arch:AVX2 to the
    # AVX2 target bodies; public dispatch wrappers stay baseline-safe and only
    # call them after runtime CPU detection.
    target_compile_options(mlx_highway_msvc_avx2 PRIVATE /arch:AVX2)
    mlx_add_highway_msvc_target(mlx_highway_msvc_sse4 _sse4)
    target_compile_definitions(mlx_highway_msvc_sse4 PRIVATE HWY_WANT_SSSE3=1
                                                             HWY_WANT_SSE4=1)
    mlx_add_highway_msvc_target(mlx_highway_msvc_ssse3 _ssse3)
    target_compile_definitions(mlx_highway_msvc_ssse3 PRIVATE HWY_WANT_SSSE3=1)
    mlx_add_highway_msvc_target(mlx_highway_msvc_sse2 _sse2)

    target_sources(
      mlx
      PRIVATE ${MLX_HIGHWAY_MSVC_DISPATCH_SOURCES}
              $<TARGET_OBJECTS:mlx_highway_msvc_avx2>
              $<TARGET_OBJECTS:mlx_highway_msvc_sse4>
              $<TARGET_OBJECTS:mlx_highway_msvc_ssse3>
              $<TARGET_OBJECTS:mlx_highway_msvc_sse2>)
  else()
    target_sources(mlx PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/norms_highway.cpp)
    target_sources(mlx
                   PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/quantized_highway.cpp)
    target_sources(mlx PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/rope_highway.cpp)
    target_sources(mlx PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/sdpa_highway.cpp)
  endif()
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
