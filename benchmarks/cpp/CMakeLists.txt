function(build_benchmark SRCFILE)
  get_filename_component(src_name ${SRCFILE} NAME_WE)
  set(target "${src_name}")
  add_executable(${target} ${SRCFILE})
  target_link_libraries(${target} PRIVATE mlx)
endfunction(build_benchmark)

build_benchmark(single_ops.cpp)
build_benchmark(irregular_strides.cpp)
build_benchmark(compare_devices.cpp)
build_benchmark(autograd.cpp)

# This is intentionally opt-in: it is an ARM I8MM evidence harness for mlx#4197,
# not a production MLX CPU backend or a portability requirement.
option(MLX_BUILD_I8MM_EVIDENCE
       "Build the Apple ARM I8MM affine-QMM evidence prototype" OFF)
if(MLX_BUILD_I8MM_EVIDENCE)
  if(NOT APPLE OR NOT CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "arm64")
    message(FATAL_ERROR "MLX_BUILD_I8MM_EVIDENCE requires an Apple arm64 host")
  endif()
  set_property(
    DIRECTORY
    APPEND
    PROPERTY CMAKE_CONFIGURE_DEPENDS
             "${CMAKE_CURRENT_SOURCE_DIR}/quantized_i8mm_evidence.cpp")
  file(SHA256 "${CMAKE_CURRENT_SOURCE_DIR}/quantized_i8mm_evidence.cpp"
       QUANTIZED_I8MM_EVIDENCE_SOURCE_SHA256)
  add_executable(quantized_i8mm_evidence quantized_i8mm_evidence.cpp)
  target_compile_features(quantized_i8mm_evidence PRIVATE cxx_std_17)
  target_compile_definitions(
    quantized_i8mm_evidence
    PRIVATE
      MLX_I8MM_EVIDENCE_SOURCE_SHA256="${QUANTIZED_I8MM_EVIDENCE_SOURCE_SHA256}"
  )
  target_compile_options(quantized_i8mm_evidence
                         PRIVATE "-march=armv8.6-a+i8mm")
endif()
