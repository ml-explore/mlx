target_sources(
  mlx
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/load.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/safetensor.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/gguf.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/gguf_quants.cpp
)

MESSAGE(STATUS "Downloading json")
FetchContent_Declare(json URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz)
FetchContent_MakeAvailable(json)
target_include_directories(
  mlx PRIVATE
    $<BUILD_INTERFACE:${json_SOURCE_DIR}/single_include/nlohmann>
)

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
target_link_libraries(mlx $<BUILD_INTERFACE:gguflib>)
