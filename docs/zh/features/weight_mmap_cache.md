# 权重预处理缓存（Weight Mmap Cache）

## 概述

xLLM 支持将预处理后的模型权重（经过 TP 切分、量化融合等变换）持久化到文件。当服务异常退出后重启时，可直接从缓存文件加载已预处理好的权重，跳过耗时的预处理流水线，实现快速拉起。

## 功能介绍

模型权重加载时会执行多项预处理操作（Tensor Parallel 切分、FP8 requantize、Gate/Up 权重融合等），在大模型场景下这些操作耗时可达数十秒。Weight Mmap Cache 功能通过以下机制解决重启耗时问题：

- **首次加载**：正常执行预处理流水线，完成后将所有处理好的权重 tensor 序列化写入缓存文件
- **后续加载**：检测到有效缓存文件后，直接通过 mmap 读取已处理好的权重，仅执行 Host→Device 拷贝

### 设计特点

- **Per-Rank 独立缓存**：每个 TP rank 生成独立的缓存文件，天然适配分布式场景
- **自动失效重建**：基于模型文件 mtime、TP 配置、量化参数计算指纹 hash，配置变更时自动重建
- **故障安全**：使用原子 rename 保证写入完整性；任何缓存异常自动回退到正常加载路径

## 使用方式

最小配置只需开启 `enable_weight_mmap_cache`：

```shell
--enable_weight_mmap_cache=true
```

可选配置项：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enable_weight_mmap_cache` | bool | false | 开启权重缓存功能 |
| `weight_mmap_cache_dir` | string | "" | 缓存目录。为空时默认使用 `{model_path}/.xllm_cache/` |

### 启动示例

```shell
# 基础用法：缓存文件存放在模型目录下
./xllm --model_path=/data/models/Qwen2-72B \
       --enable_weight_mmap_cache=true

# 指定独立的高速存储目录
./xllm --model_path=/data/models/Qwen2-72B \
       --enable_weight_mmap_cache=true \
       --weight_mmap_cache_dir=/nvme_fast/xllm_cache
```

### JSON 配置文件用法

也可通过 `--config_json_file` 启用：

```json
{
  "enable_weight_mmap_cache": true,
  "weight_mmap_cache_dir": "/nvme_fast/xllm_cache"
}
```

## 缓存文件说明

缓存文件按如下路径组织：

```
{cache_dir}/{model_hash_prefix}/rank{rank}_tp{world_size}.cache
```

- `model_hash_prefix`：由模型路径的 XXH3 hash 生成，避免路径过长
- 每个 rank 拥有独立的缓存文件

### 缓存失效条件

以下任一条件变化时，缓存自动失效并重建：

- 模型权重文件变更（基于 mtime 检测）
- TP 并行度（world_size）变更
- 量化方法 / 量化位数 / group_size 变更
- 模型类型（model_type）或推理精度（dtype）变更

## 注意事项

1. **磁盘空间**：缓存文件大小约等于预处理后权重的总 CPU 内存占用（每个 rank 的切分后大小）
2. **首次启动**：第一次启动会在正常加载后额外执行一次 save 操作，耗时略有增加
3. **Rolling Load 互斥**：当 `enable_rolling_load=true` 时，weight cache 不生效（Rolling Load 有独立的加载路径）
4. **缓存目录权限**：确保 xLLM 进程对缓存目录有读写权限
5. **手动清理**：如需强制重建缓存，删除缓存目录下对应文件即可
