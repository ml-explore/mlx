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
      BUILD_COMMAND ${CMAKE_COMMAND} --build . --config Release
      INSTALL_COMMAND "")
    # Ensure nccl_stub is built before mlx compiles nccl.cpp (needs nccl.h)
    add_dependencies(mlx nccl_stub)
    set(NCCL_PATH
        "${CMAKE_CURRENT_BINARY_DIR}/nccl_stub-prefix/src/nccl_stub-build/")
    if(WIN32)
      # On Windows, shared libraries produce .dll and .lib (import library)
      target_link_libraries(mlx PRIVATE ${NCCL_PATH}/Release/nccl.lib)
    else()
      target_link_libraries(mlx PRIVATE ${NCCL_PATH}/libnccl.so)
    endif()
    target_include_directories(mlx PRIVATE ${NCCL_PATH})
  endif()
else()
  target_sources(mlx PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/no_nccl.cpp)
endif()
