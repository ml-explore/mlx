
if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  set(COMPILER ${CMAKE_C_COMPILER})
  set(CLANG TRUE)
else()
  set(COMPILER ${CMAKE_CXX_COMPILER})
endif()

add_custom_command(
    OUTPUT  compiled_preamble.cpp
    COMMAND /bin/bash
              ${CMAKE_CURRENT_SOURCE_DIR}/make_compiled_preamble.sh
              ${CMAKE_CURRENT_BINARY_DIR}/compiled_preamble.cpp
              ${COMPILER}
              ${PROJECT_SOURCE_DIR}
              ${CLANG}

    DEPENDS make_compiled_preamble.sh
            compiled_preamble.h
            ${PROJECT_SOURCE_DIR}/mlx/types/half_types.h
            ${PROJECT_SOURCE_DIR}/mlx/types/fp16.h
            ${PROJECT_SOURCE_DIR}/mlx/types/bf16.h
            ${PROJECT_SOURCE_DIR}/mlx/types/complex.h
            ops.h
)

add_custom_target(
  cpu_compiled_preamble
  DEPENDS compiled_preamble.cpp
)

add_dependencies(mlx cpu_compiled_preamble)

target_sources(
  mlx
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/arg_reduce.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/binary.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/compiled.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/conv.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/copy.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/erf.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/fft.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/primitives.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/quantized.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/reduce.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/scan.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/select.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/softmax.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/sort.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/threefry.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/indexing.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/load.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/qrf.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/svd.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/inverse.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/compiled_preamble.cpp
)

if (IOS)
  target_sources(
    mlx
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/compiled_nocpu.cpp
  )
else()
  target_sources(
    mlx
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/compiled_cpu.cpp
  )
endif()
