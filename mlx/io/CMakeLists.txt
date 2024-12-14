target_sources(mlx PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/load.cpp)

if(MLX_BUILD_SAFETENSORS)
  message(STATUS "Downloading json")
  FetchContent_Declare(
    json
    URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz)
  FetchContent_MakeAvailable(json)
  target_include_directories(
    mlx PRIVATE $<BUILD_INTERFACE:${json_SOURCE_DIR}/single_include/nlohmann>)
  target_sources(mlx PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/safetensors.cpp)
else()
  target_sources(mlx PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/no_safetensors.cpp)
endif()

if(MLX_BUILD_GGUF)
  message(STATUS "Downloading gguflib")
  FetchContent_Declare(
    gguflib
    GIT_REPOSITORY https://github.com/frost-beta/gguf-tools.git
    GIT_TAG a8001a0a6603a4f6a833678e5854f0e18332518d)
  FetchContent_MakeAvailable(gguflib)
  target_link_libraries(mlx PRIVATE gguflib)
  target_sources(mlx PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/gguf.cpp
                             ${CMAKE_CURRENT_SOURCE_DIR}/gguf_quants.cpp)
else()
  target_sources(mlx PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/no_gguf.cpp)
endif()
