set(
  HEADERS
  ${CMAKE_CURRENT_SOURCE_DIR}/atomic.h
  ${CMAKE_CURRENT_SOURCE_DIR}/bf16.h
  ${CMAKE_CURRENT_SOURCE_DIR}/bf16_math.h
  ${CMAKE_CURRENT_SOURCE_DIR}/binary.h
  ${CMAKE_CURRENT_SOURCE_DIR}/complex.h
  ${CMAKE_CURRENT_SOURCE_DIR}/defines.h
  ${CMAKE_CURRENT_SOURCE_DIR}/erf.h
  ${CMAKE_CURRENT_SOURCE_DIR}/indexing.h
  ${CMAKE_CURRENT_SOURCE_DIR}/reduce.h
  ${CMAKE_CURRENT_SOURCE_DIR}/unary.h
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
  "gemv"
  "quantized"
  "random"
  "reduce"
  "rope"
  "scan"
  "softmax"
  "sort"
  "ternary"
  "unary"
  "gather"
  "scatter"
)

function(build_kernel_base TARGET SRCFILE DEPS)
  add_custom_command(
    COMMAND xcrun -sdk macosx metal -Wall -Wextra
                  -fno-fast-math
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
