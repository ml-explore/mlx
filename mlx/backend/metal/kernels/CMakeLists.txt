set(BASE_HEADERS
    metal_3_1/bf16.h
    metal_3_0/bf16.h
    bf16_math.h
    complex.h
    defines.h
    erf.h
    expm1f.h
    utils.h)

function(build_kernel_base TARGET SRCFILE DEPS)
  set(METAL_FLAGS -Wall -Wextra -fno-fast-math -Wno-c++17-extensions)
  if(MLX_METAL_DEBUG)
    set(METAL_FLAGS ${METAL_FLAGS} -gline-tables-only -frecord-sources)
  endif()
  if(NOT CMAKE_OSX_DEPLOYMENT_TARGET STREQUAL "")
    set(METAL_FLAGS ${METAL_FLAGS}
                    "-mmacosx-version-min=${CMAKE_OSX_DEPLOYMENT_TARGET}")
  endif()
  if(MLX_METAL_VERSION GREATER_EQUAL 310)
    set(VERSION_INCLUDES
        ${PROJECT_SOURCE_DIR}/mlx/backend/metal/kernels/metal_3_1)
  else()
    set(VERSION_INCLUDES
        ${PROJECT_SOURCE_DIR}/mlx/backend/metal/kernels/metal_3_0)
  endif()
  add_custom_command(
    COMMAND xcrun -sdk macosx metal ${METAL_FLAGS} -c ${SRCFILE}
            -I${PROJECT_SOURCE_DIR} -I${VERSION_INCLUDES} -o ${TARGET}.air
    DEPENDS ${SRCFILE} ${DEPS} ${BASE_HEADERS}
    OUTPUT ${TARGET}.air
    COMMENT "Building ${TARGET}.air"
    VERBATIM)
endfunction(build_kernel_base)

function(build_kernel KERNEL)
  set(SRCFILE ${CMAKE_CURRENT_SOURCE_DIR}/${KERNEL}.metal)
  cmake_path(GET KERNEL STEM TARGET)
  build_kernel_base(${TARGET} ${SRCFILE} "${ARGN}")
  set(KERNEL_AIR
      ${TARGET}.air ${KERNEL_AIR}
      PARENT_SCOPE)
endfunction(build_kernel)

build_kernel(arg_reduce)
build_kernel(conv steel/conv/params.h)
build_kernel(gemv steel/utils.h)
build_kernel(layer_norm)
build_kernel(random)
build_kernel(rms_norm)
build_kernel(rope)
build_kernel(scaled_dot_product_attention sdpa_vector.h)
if(MLX_METAL_VERSION GREATER_EQUAL 320)
  build_kernel(fence)
endif()

set(STEEL_HEADERS
    steel/defines.h
    steel/utils.h
    steel/conv/conv.h
    steel/conv/loader.h
    steel/conv/loaders/loader_channel_l.h
    steel/conv/loaders/loader_channel_n.h
    steel/conv/loaders/loader_general.h
    steel/conv/kernels/steel_conv.h
    steel/conv/kernels/steel_conv_general.h
    steel/gemm/gemm.h
    steel/gemm/mma.h
    steel/gemm/loader.h
    steel/gemm/transforms.h
    steel/gemm/kernels/steel_gemm_fused.h
    steel/gemm/kernels/steel_gemm_gather.h
    steel/gemm/kernels/steel_gemm_masked.h
    steel/gemm/kernels/steel_gemm_segmented.h
    steel/gemm/kernels/steel_gemm_splitk.h
    steel/utils/type_traits.h
    steel/utils/integral_constant.h)

set(STEEL_ATTN_HEADERS
    steel/defines.h
    steel/utils.h
    steel/gemm/gemm.h
    steel/gemm/mma.h
    steel/gemm/loader.h
    steel/gemm/transforms.h
    steel/utils/type_traits.h
    steel/utils/integral_constant.h
    steel/attn/attn.h
    steel/attn/loader.h
    steel/attn/mma.h
    steel/attn/params.h
    steel/attn/transforms.h
    steel/attn/kernels/steel_attention.h)

build_kernel(steel/attn/kernels/steel_attention ${STEEL_ATTN_HEADERS})

if(NOT MLX_METAL_JIT)
  build_kernel(arange arange.h)
  build_kernel(binary binary.h binary_ops.h)
  build_kernel(binary_two binary_two.h)
  build_kernel(copy copy.h)
  build_kernel(fft fft.h fft/radix.h fft/readwrite.h)
  build_kernel(
    reduce
    atomic.h
    reduction/ops.h
    reduction/reduce_init.h
    reduction/reduce_all.h
    reduction/reduce_col.h
    reduction/reduce_row.h)
  build_kernel(quantized quantized.h quantized_utils.h ${STEEL_HEADERS})
  build_kernel(fp4_quantized fp4_quantized.h quantized_utils.h ${STEEL_HEADERS})
  build_kernel(scan scan.h)
  build_kernel(softmax softmax.h)
  build_kernel(logsumexp logsumexp.h)
  build_kernel(sort sort.h)
  build_kernel(ternary ternary.h ternary_ops.h)
  build_kernel(unary unary.h unary_ops.h)
  build_kernel(steel/conv/kernels/steel_conv ${STEEL_HEADERS})
  build_kernel(steel/conv/kernels/steel_conv_general ${STEEL_HEADERS})
  build_kernel(steel/gemm/kernels/steel_gemm_fused ${STEEL_HEADERS})
  build_kernel(steel/gemm/kernels/steel_gemm_gather ${STEEL_HEADERS})
  build_kernel(steel/gemm/kernels/steel_gemm_masked ${STEEL_HEADERS})
  build_kernel(steel/gemm/kernels/steel_gemm_splitk ${STEEL_HEADERS})
  build_kernel(steel/gemm/kernels/steel_gemm_segmented ${STEEL_HEADERS})
  build_kernel(gemv_masked steel/utils.h)
endif()

add_custom_command(
  OUTPUT ${MLX_METAL_PATH}/mlx.metallib
  COMMAND xcrun -sdk macosx metallib ${KERNEL_AIR} -o
          ${MLX_METAL_PATH}/mlx.metallib
  DEPENDS ${KERNEL_AIR}
  COMMENT "Building mlx.metallib"
  VERBATIM)

add_custom_target(mlx-metallib DEPENDS ${MLX_METAL_PATH}/mlx.metallib)

add_dependencies(mlx mlx-metallib)

# Install metallib
include(GNUInstallDirs)

install(
  FILES ${MLX_METAL_PATH}/mlx.metallib
  DESTINATION ${CMAKE_INSTALL_LIBDIR}
  COMPONENT metallib)
