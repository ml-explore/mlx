set(
  HEADERS
  ${CMAKE_CURRENT_SOURCE_DIR}/bf16.h
  ${CMAKE_CURRENT_SOURCE_DIR}/bf16_math.h
  ${CMAKE_CURRENT_SOURCE_DIR}/complex.h
  ${CMAKE_CURRENT_SOURCE_DIR}/defines.h
  ${CMAKE_CURRENT_SOURCE_DIR}/erf.h
  ${CMAKE_CURRENT_SOURCE_DIR}/reduce.h
  ${CMAKE_CURRENT_SOURCE_DIR}/utils.h
)

set(
  KERNELS
  "arange"
  "arg_reduce"
  "binary"
  "binary_two"
  "conv"
  "copy"
  "gemm"
  "gemv"
  "quantized"
  "random"
  "reduce"
  "scan"
  "softmax"
  "sort"
  "unary"
  "indexing"
)

function(build_kernel KERNEL)
  set(SRCFILE ${CMAKE_CURRENT_SOURCE_DIR}/${KERNEL}.metal)
  set(HEADERS_PADDED ${HEADERS})
  if(${KERNEL} STREQUAL "gemm")
    set(HEADERS_PADDED ${HEADERS_PADDED} ${CMAKE_CURRENT_SOURCE_DIR}/gemm/gemm.h)
  endif()
  if(${KERNEL} STREQUAL "conv")
    set(HEADERS_PADDED ${HEADERS_PADDED} ${CMAKE_CURRENT_SOURCE_DIR}/gemm/conv.h)
  endif()
  add_custom_command(
    COMMAND xcrun -sdk macosx metal -Wall -Wextra
                  -fno-fast-math
                  -c ${SRCFILE} 
                  -I${PROJECT_SOURCE_DIR} 
                  -o ${KERNEL}.air
    DEPENDS ${SRCFILE} ${HEADERS_PADDED}
    OUTPUT ${KERNEL}.air
    COMMENT "Building ${KERNEL}.air"
    VERBATIM
  )
endfunction(build_kernel)

foreach(KERNEL ${KERNELS})
  build_kernel(${KERNEL})
  set(KERNEL_AIR ${KERNEL}.air ${KERNEL_AIR})
endforeach()

add_custom_command(
  OUTPUT ${MLX_METAL_PATH}/mlx.metallib
  COMMAND xcrun -sdk macosx metallib ${KERNEL_AIR} -o ${MLX_METAL_PATH}/mlx.metallib
  DEPENDS ${KERNEL_AIR}
  COMMENT "Building mlx.metallib"
  VERBATIM
)

add_custom_target(
  mlx-metallib
  DEPENDS
  ${MLX_METAL_PATH}/mlx.metallib
)

add_dependencies(
  mlx
  mlx-metallib
)

# Install metallib
include(GNUInstallDirs)

install(
  FILES ${MLX_METAL_PATH}/mlx.metallib
  DESTINATION ${CMAKE_INSTALL_LIBDIR}
  COMPONENT metallib
)
