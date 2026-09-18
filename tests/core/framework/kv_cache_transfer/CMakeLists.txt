include(cc_test)

cc_test(
  NAME
    host_kv_transfer_test
  SRCS
    host_transfer/transfer_test.cpp
  DEPS
    :kv_cache_transfer
    GTest::gtest_main
)

cc_test(
  NAME
    kv_redundancy_test
  SRCS
    kv_redundancy_test.cpp
  DEPS
    :kv_redundancy
    GTest::gtest_main
)

cc_test(
  NAME
    pd_route_test
  SRCS
    pd_route_test.cpp
  DEPS
    :kv_redundancy
    :pd_route_table
    :route_binder
    GTest::gtest_main
)

cc_test(
  NAME
    cache_directory_test
  SRCS
    cache_directory_test.cpp
  DEPS
    :cache_directory
    :pd_route_table
    :route_binder
    GTest::gtest_main
)

cc_test(
  NAME
    pd_route_transfer_test
  SRCS
    pd_route_transfer_test.cpp
  DEPS
    :pd_route_transfer
    GTest::gtest_main
)

# Checks the routing coordinates against the runtime shard class itself
# (KVShardLayout lives in :kv_cache), so it links :kv_cache like
# cache_layout_builder_test does.
cc_test(
  NAME
    kv_shard_contract_test
  SRCS
    kv_shard_contract_test.cpp
  DEPS
    :cache_directory
    :kv_cache
    GTest::gtest_main
)
target_link_libraries(kv_shard_contract_test
                      PUBLIC
                      Python::Python
                      $<$<BOOL:${USE_NPU}>:ascendcl>
                      $<$<BOOL:${USE_NPU}>:hccl>
                      $<$<BOOL:${USE_NPU}>:c_sec>
                      $<$<BOOL:${USE_NPU}>:nnopbase>)

# Builds real cache tensors through the layout builder, so it links :kv_cache
# like cache_layout_builder_test does.
cc_test(
  NAME
    pd_route_integration_test
  SRCS
    pd_route_integration_test.cpp
  DEPS
    :cache_directory
    :kv_cache
    GTest::gtest_main
)
target_link_libraries(pd_route_integration_test
                      PUBLIC
                      Python::Python
                      $<$<BOOL:${USE_NPU}>:ascendcl>
                      $<$<BOOL:${USE_NPU}>:hccl>
                      $<$<BOOL:${USE_NPU}>:c_sec>
                      $<$<BOOL:${USE_NPU}>:nnopbase>)

cc_test(
  NAME
    pd_topology_guard_test
  SRCS
    pd_topology_guard_test.cpp
  DEPS
    :pd_topology_guard
    GTest::gtest_main
)

cc_test(
  NAME
    kv_transfer_completion_test
  SRCS
    kv_transfer_completion_test.cpp
  DEPS
    :kv_transfer_completion
    GTest::gtest_main
)

cc_test(
  NAME
    reshard_planner_test
  SRCS
    reshard_planner_test.cpp
  DEPS
    :reshard_planner
    GTest::gtest_main
)

if(USE_NPU OR USE_MLU)
cc_test(
  NAME
    mooncake_transfer_engine_test
  SRCS
    mooncake_transfer_engine_test.cpp
  DEPS
    :kv_cache_transfer
    :xllm_server
    GTest::gtest_main
)

# Resolve static link order between xtensor and xllm_server for this test target.
target_link_libraries(mooncake_transfer_engine_test PRIVATE
  "$<LINK_GROUP:RESCAN,xtensor,xllm_server>")
endif()

if(USE_MLU)
cc_test(
  NAME
    compact_load_test
  SRCS
    host_transfer/compact_load_test.cpp
  DEPS
    :kv_cache_transfer
    GTest::gtest_main
)

cc_test(
  NAME
    compact_offload_test
  SRCS
    host_transfer/compact_offload_test.cpp
  DEPS
    :kv_cache_transfer
    GTest::gtest_main
)

endif()

if(USE_NPU OR USE_MLU)
cc_test(
  NAME
    kv_cache_store_test
  SRCS
    kv_cache_store_test.cpp
  DEPS
    :kv_cache_transfer
    GTest::gtest_main
)

cc_test(
  NAME
    basic_host_kv_transfer_test
  SRCS
    host_transfer/basic_transfer_test.cpp
  DEPS
    :kv_cache_transfer
    GTest::gtest_main
)

cc_test(
  NAME
    hierarchy_kv_cache_transfer_test
  SRCS
    hierarchy_kv_cache_transfer_test.cpp
  DEPS
    :kv_cache_transfer
    GTest::gtest_main
)
endif()
