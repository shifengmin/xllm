#!/bin/bash
# GDB attach 到 rank0 主进程，崩溃后打印 SparseFlashAttention 的 tensor 详情
# 用法: bash gdb_detail.sh <main_pid>
PID=$1
XLLM=/usr/local/python3.11.15/lib/python3.11/site-packages/xllm/xllm
TS=$(date +%H%M%S)
OUT=/export/home/shifengmin.3/workspace/glm5_2_pd/logs/gdb_detail_${TS}.log

echo "[INFO] attaching gdb to pid $PID, output to $OUT"
gdb -batch \
  -ex "set pagination off" \
  -ex "set print thread-events off" \
  -ex "handle SIGSEGV stop print nopass" \
  -ex "attach $PID" \
  -ex "continue" \
  -ex "bt 30" \
  -ex "frame 5" \
  -ex "info args" \
  -ex "info locals" \
  -ex "print this" \
  -ex "print this->aclnnOpCache_->aclnnVariantPack.aclOutTensors.size()" \
  -ex "print this->aclnnOpCache_->aclnnVariantPack.aclInTensors.size()" \
  -ex "print this->aclnnOpCache_->aclnnVariantPack.aclOutTensors._M_impl._M_start[0]->tensor" \
  -ex "print this->aclnnOpCache_->aclnnVariantPack.aclOutTensors._M_impl._M_start[0]->atbTensor.desc.dtype" \
  -ex "print this->aclnnOpCache_->aclnnVariantPack.aclOutTensors._M_impl._M_start[0]->atbTensor.desc.format" \
  -ex "print this->aclnnOpCache_->aclnnVariantPack.aclOutTensors._M_impl._M_start[0]->atbTensor.desc.shape.dimNum" \
  -ex "print this->aclnnOpCache_->aclnnVariantPack.aclOutTensors._M_impl._M_start[0]->atbTensor.desc.shape.dims[0]" \
  -ex "print this->aclnnOpCache_->aclnnVariantPack.aclOutTensors._M_impl._M_start[0]->atbTensor.desc.shape.dims[1]" \
  -ex "print this->aclnnOpCache_->aclnnVariantPack.aclOutTensors._M_impl._M_start[0]->atbTensor.desc.shape.dims[2]" \
  -ex "print this->aclnnOpCache_->aclnnVariantPack.aclOutTensors._M_impl._M_start[0]->atbTensor.deviceData" \
  -ex "print this->aclnnOpCache_->aclnnVariantPack.aclInTensors._M_impl._M_start[0]->tensor" \
  -ex "print this->aclnnOpCache_->aclnnVariantPack.aclInTensors._M_impl._M_start[0]->atbTensor.desc.dtype" \
  -ex "print this->aclnnOpCache_->aclnnVariantPack.aclInTensors._M_impl._M_start[0]->atbTensor.desc.shape.dimNum" \
  -ex "print this->aclnnOpCache_->aclnnVariantPack.aclInTensors._M_impl._M_start[1]->tensor" \
  -ex "print this->aclnnOpCache_->aclnnVariantPack.aclInTensors._M_impl._M_start[2]->tensor" \
  -ex "print this->aclnnOpCache_->aclnnVariantPack.aclInTensors._M_impl._M_start[3]->tensor" \
  -ex "print this->aclnnOpCache_->aclnnVariantPack.aclInTensors._M_impl._M_start[4]->tensor" \
  -ex "print this->aclnnOpCache_->aclnnVariantPack.aclInTensors._M_impl._M_start[5]->tensor" \
  -ex "print this->aclnnOpCache_->aclnnVariantPack.aclInTensors._M_impl._M_start[6]->tensor" \
  -ex "print this->aclnnOpCache_->aclnnVariantPack.aclInTensors._M_impl._M_start[7]->tensor" \
  -ex "print this->aclnnOpCache_->aclnnVariantPack.aclInTensors._M_impl._M_start[8]->tensor" \
  -ex "print param_" \
  $XLLM > $OUT 2>&1

echo "[INFO] gdb finished, output in $OUT"
