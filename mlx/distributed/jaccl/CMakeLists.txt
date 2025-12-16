if(MLX_BUILD_CPU
   AND ${CMAKE_SYSTEM_NAME} MATCHES "Darwin"
   AND MACOS_SDK_VERSION GREATER_EQUAL 26.2)
  target_sources(mlx PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/jaccl.cpp)
else()
  target_sources(mlx PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/no_jaccl.cpp)
endif()
