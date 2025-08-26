if(MLX_BUILD_CUDA)
  target_sources(mlx PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/nccl.cpp)
  find_package(NCCL)
  if(NCCL_FOUND)
    target_link_libraries(mlx PRIVATE ${NCCL_LIBRARIES})
    target_include_directories(mlx PRIVATE ${NCCL_INCLUDE_DIRS})
  else()
    message(
      STATUS
        "NCCL not found, using stubs. To run distributed with NCCL backend, install NCCL."
    )

    include(ExternalProject)
    ExternalProject_Add(
      nccl_stub
      SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/nccl_stub"
      BUILD_COMMAND ${CMAKE_COMMAND} --build .
      INSTALL_COMMAND "")
    set(NCCL_PATH
        "${CMAKE_CURRENT_BINARY_DIR}/nccl_stub-prefix/src/nccl_stub-build/")
    target_link_libraries(mlx PRIVATE ${NCCL_PATH}/libnccl.so)
    target_include_directories(mlx PRIVATE ${NCCL_PATH})
  endif()
else()
  target_sources(mlx PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/no_nccl.cpp)
endif()
