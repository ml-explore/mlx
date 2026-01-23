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
    file(
      DOWNLOAD
      "https://raw.githubusercontent.com/NVIDIA/nccl/refs/tags/v2.27.5-1/src/nccl.h.in"
      "${CMAKE_CURRENT_BINARY_DIR}/nccl.h")
    add_library(nccl_stub OBJECT
                ${CMAKE_CURRENT_SOURCE_DIR}/nccl_stub/nccl_stubs.cpp)
    target_include_directories(
      nccl_stub
      PRIVATE ${CUDAToolkit_INCLUDE_DIRS}
      PUBLIC ${CMAKE_CURRENT_BINARY_DIR})
    target_link_libraries(mlx PRIVATE $<BUILD_INTERFACE:nccl_stub>)
  endif()
else()
  target_sources(mlx PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/no_nccl.cpp)
endif()
