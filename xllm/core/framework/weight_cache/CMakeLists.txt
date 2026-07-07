include(cc_library)

cc_library(
  NAME
    weight_mmap_cache
  HDRS
    weight_mmap_cache.h
  SRCS
    weight_mmap_cache.cpp
  DEPS
    :common
    :model
    glog::glog
    torch
)
