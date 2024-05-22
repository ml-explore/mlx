
target_sources(
  mlx
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/load.cpp
)

if (MLX_BUILD_SAFETENSORS)
  MESSAGE(STATUS "Downloading json")
  FetchContent_Declare(json URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz)
  FetchContent_MakeAvailable(json)
  target_include_directories(
    mlx PRIVATE
      $<BUILD_INTERFACE:${json_SOURCE_DIR}/single_include/nlohmann>
  )
  target_sources(
    mlx
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/safetensors.cpp
  )
else()
  target_sources(
    mlx
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/no_safetensors.cpp
  )
endif()

if (MLX_BUILD_GGUF)
  MESSAGE(STATUS "Downloading gguflib")
  FetchContent_Declare(gguflib
      GIT_REPOSITORY     https://github.com/antirez/gguf-tools/
      GIT_TAG            af7d88d808a7608a33723fba067036202910acb3 
  )
  FetchContent_MakeAvailable(gguflib)
  target_include_directories(
    mlx PRIVATE
      $<BUILD_INTERFACE:${gguflib_SOURCE_DIR}>
  )
  add_library(
    gguflib STATIC
    ${gguflib_SOURCE_DIR}/fp16.c
    ${gguflib_SOURCE_DIR}/gguflib.c)
  target_link_libraries(mlx PRIVATE $<BUILD_INTERFACE:gguflib>)
  target_sources(
    mlx
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/gguf.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/gguf_quants.cpp
  )
else()
  target_sources(
    mlx
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/no_gguf.cpp
  )
endif()

