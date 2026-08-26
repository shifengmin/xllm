// hook_aclcreate.cpp - hook aclCreateTensor 记录 nullptr 返回
#define _GNU_SOURCE
#include <dlfcn.h>
#include <execinfo.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

typedef void* (*aclCreateTensor_t)(int64_t* dims,
                                   size_t dimNum,
                                   int dtype,
                                   int64_t* strides,
                                   size_t offset,
                                   int format,
                                   int64_t* storageDims,
                                   size_t storageDimNum,
                                   void* deviceData);

static aclCreateTensor_t real_fn = nullptr;
static int call_count = 0;

__attribute__((constructor)) static void hook_init() {
  FILE* f = fopen("/tmp/aclCreateTensor_loaded.log", "w");
  if (f) {
    fprintf(f, "hook_aclcreate.so loaded pid=%d\n", getpid());
    fclose(f);
  }
}

extern "C" void* aclCreateTensor(int64_t* dims,
                                 size_t dimNum,
                                 int dtype,
                                 int64_t* strides,
                                 size_t offset,
                                 int format,
                                 int64_t* storageDims,
                                 size_t storageDimNum,
                                 void* deviceData) {
  if (!real_fn) {
    real_fn = (aclCreateTensor_t)dlsym(RTLD_NEXT, "aclCreateTensor");
  }
  void* ret = real_fn(dims,
                      dimNum,
                      dtype,
                      strides,
                      offset,
                      format,
                      storageDims,
                      storageDimNum,
                      deviceData);
  call_count++;
  if (ret == nullptr) {
    FILE* f = fopen("/tmp/aclCreateTensor_fail.log", "a");
    if (f) {
      fprintf(
          f, "=== aclCreateTensor returned NULL! call#%d ===\n", call_count);
      fprintf(f,
              "dimNum=%zu dtype=%d format=%d offset=%zu deviceData=%p\n",
              dimNum,
              dtype,
              format,
              offset,
              deviceData);
      fprintf(f, "dims: ");
      for (size_t i = 0; i < dimNum && i < 8; i++)
        fprintf(f, "%ld ", (long)dims[i]);
      fprintf(f, "\nstrides: ");
      for (size_t i = 0; i < dimNum && i < 8; i++)
        fprintf(f, "%ld ", (long)strides[i]);
      fprintf(f, "\nstorageDimNum=%zu storageDims: ", storageDimNum);
      for (size_t i = 0; i < storageDimNum && i < 8; i++)
        fprintf(f, "%ld ", (long)storageDims[i]);
      fprintf(f, "\nbacktrace:\n");
      void* bt[25];
      int n = backtrace(bt, 25);
      backtrace_symbols_fd(bt, n, fileno(f));
      fprintf(f, "---\n");
      fclose(f);
    }
  }
  return ret;
}
