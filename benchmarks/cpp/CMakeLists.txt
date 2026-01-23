function(build_benchmark SRCFILE)
  get_filename_component(src_name ${SRCFILE} NAME_WE)
  set(target "${src_name}")
  add_executable(${target} ${SRCFILE})
  target_link_libraries(${target} PRIVATE mlx)
  # On Windows, copy the mlx DLL to the executable directory for runtime loading
  if(WIN32 AND BUILD_SHARED_LIBS)
    add_custom_command(
      TARGET ${target}
      POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:mlx>
              $<TARGET_FILE_DIR:${target}>)
  endif()
endfunction(build_benchmark)

build_benchmark(single_ops.cpp)
build_benchmark(irregular_strides.cpp)
build_benchmark(compare_devices.cpp)
build_benchmark(autograd.cpp)
