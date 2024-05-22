set(
  HEADERS
  ${CMAKE_CURRENT_SOURCE_DIR}/atomic.h
  ${CMAKE_CURRENT_SOURCE_DIR}/bf16.h
  ${CMAKE_CURRENT_SOURCE_DIR}/bf16_math.h
  ${CMAKE_CURRENT_SOURCE_DIR}/complex.h
  ${CMAKE_CURRENT_SOURCE_DIR}/defines.h
  ${CMAKE_CURRENT_SOURCE_DIR}/utils.h
)


set(
  KERNELS
  "arg_reduce"
  "conv"
  "fft"
  "gemv"
  "quantized"
  "random"
  "rms_norm"
  "layer_norm"
  "rope"
  "scaled_dot_product_attention"
)

if (NOT MLX_METAL_JIT)
set(
  KERNELS
  ${KERNELS}
  "arange"
  "binary"
  "binary_two"
  "unary"
  "ternary"
  "copy"
  "softmax"
  "sort"
  "scan"
  "reduce"
)
set(
  HEADERS 
  ${HEADERS}
  arange.h
  unary_ops.h
  unary.h
  binary_ops.h
  binary.h
  ternary.h
  copy.h
  softmax.h
  sort.h
  scan.h
  reduction/ops.h
  reduction/reduce_init.h
  reduction/reduce_all.h
  reduction/reduce_col.h
  reduction/reduce_row.h
)
endif()

function(build_kernel_base TARGET SRCFILE DEPS)
  set(METAL_FLAGS -Wall -Wextra -fno-fast-math -D${MLX_METAL_VERSION})
  if(MLX_METAL_DEBUG)
    set(METAL_FLAGS ${METAL_FLAGS}
        -gline-tables-only
        -frecord-sources)
  endif()
  add_custom_command(
    COMMAND xcrun -sdk macosx metal
                  ${METAL_FLAGS}
                  -c ${SRCFILE}
                  -I${PROJECT_SOURCE_DIR}
                  -o ${TARGET}.air
    DEPENDS ${SRCFILE} ${DEPS}
    OUTPUT ${TARGET}.air
    COMMENT "Building ${TARGET}.air"
    VERBATIM
  )
endfunction(build_kernel_base)

function(build_kernel KERNEL)
  set(SRCFILE ${CMAKE_CURRENT_SOURCE_DIR}/${KERNEL}.metal)
  build_kernel_base(${KERNEL} ${SRCFILE} "${HEADERS}")
endfunction(build_kernel)

foreach(KERNEL ${KERNELS})
  build_kernel(${KERNEL})
  set(KERNEL_AIR ${KERNEL}.air ${KERNEL_AIR})
endforeach()

file(GLOB_RECURSE STEEL_KERNELS ${CMAKE_CURRENT_SOURCE_DIR}/steel/*.metal)
file(GLOB_RECURSE STEEL_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/steel/*.h)

foreach(KERNEL ${STEEL_KERNELS})
  cmake_path(GET KERNEL STEM TARGET)
  build_kernel_base(${TARGET} ${KERNEL} "${STEEL_HEADERS}")
  set(KERNEL_AIR ${TARGET}.air ${KERNEL_AIR})
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
