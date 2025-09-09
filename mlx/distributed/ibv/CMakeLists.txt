target_sources(mlx PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/ibv.cpp)
target_link_libraries(mlx PRIVATE rdma)
