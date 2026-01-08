if(MLX_BUILD_CPU AND ${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  # Check if infiniband header actually exists in the SDK being used
  include(CheckIncludeFileCXX)
  check_include_file_cxx("infiniband/verbs.h" HAVE_INFINIBAND_VERBS_H)

  if(HAVE_INFINIBAND_VERBS_H AND MACOS_SDK_VERSION GREATER_EQUAL 26.2)
    target_sources(mlx PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/jaccl.cpp)
  else()
    target_sources(mlx PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/no_jaccl.cpp)
  endif()
else()
  target_sources(mlx PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/no_jaccl.cpp)
endif()
