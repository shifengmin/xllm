# GLM-5.2 PD 分离（A3 双机）操作手册

Prefill 在 `11.87.191.98`，Decode 在 `11.87.191.83`。管控面（etcd + xllm-service）跑在 83 的 `fengmin-cann9-0801` 容器内，`--network=host`。

本文记录已验证的环境、脚本、踩坑和 Mooncake 去重修复，供复测复用。

## 1. 拓扑

```
curl /v1/chat/completions
        |
        v
11.87.191.83:48888  HTTP  (xllm-service，有 P+D 实例后才 listen)
11.87.191.83:48889  RPC
11.87.191.83:4389   etcd
        |
        +-- PREFILL  11.87.191.98  容器 fengmin-cann9-20260801  16x Ascend910
        +-- DECODE   11.87.191.83  容器 fengmin-cann9-0801      16x Ascend910
```

并行配置（脚本写死）：

| 角色 | 并行 | 其它 |
|---|---|---|
| Prefill | CP=2, TP=8, KV split=2 | prefix cache + chunked prefill |
| Decode | DP=2, TP=8, layerwise split=4 | schedule overlap |

单机 16 卡 PD **不要** 传 83+82 / 83+99 那种双机 ranktable。

HTTP `48888` 必须等 P、D 都注册后才 listen。RPC `48889` 可先探活。

## 2. 登录与容器

本机测试进程通常已经在 Prefill 容器内。83 必须用 `shifengmin.3`（在 docker 组），**不要用 root**：

```bash
SSH_KEY=/export/home/shifengmin.3/.ssh/id_rsa
ssh -i "$SSH_KEY" -o IdentitiesOnly=yes shifengmin.3@11.87.191.83
docker exec -it fengmin-cann9-0801 bash
```

98 宿主机 docker 需从 83 跳转（98 对本容器内 root 的 key 免密失败）：

```bash
ssh -i "$SSH_KEY" -o IdentitiesOnly=yes shifengmin.3@11.87.191.83 \
  ssh -o BatchMode=yes shifengmin.3@11.87.191.98 'docker ps --format {{.Names}}'
```

| 机器 | 测试容器 | 参考容器（有 hccn.conf） |
|---|---|---|
| 98 | `fengmin-cann9-20260801` | `fengmin-cann9` |
| 83 | `fengmin-cann9-0801` | `fengmin-cann9` |

83 上 `glm5_2_pd` 目录属 `zhouyu.474`，往里面写文件要用 `docker exec`（root）。

## 3. 权重与二进制

两边共用（或各自可见）的权重：

- `/export/home/models/GLM-5.2-W8A8-EcoTech`
- `/export/home/models/GLM-5.2-W8A8-EcoTech-MTP`

xllm ELF（**不要**用 `which xllm` 那个 Python wrapper）：

```text
/usr/local/python3.11.15/lib/python3.11/site-packages/xllm/xllm
```

本机增量编译产物：

```text
build/lib.linux-aarch64-cpython-311/xllm/xllm
```

CMake 目录：`build/cmake.linux-aarch64-cpython-311`。

脚本默认 `XLLM_BIN=$(repo)/xllm/xllm` **不存在**，启动前必须显式 export。

wheel 里的 ELF 带了本机 build 的 RPATH。83 上没有那份目录，会报 `libasio.so: cannot open shared object file`。把 98 的

`build/cmake.linux-aarch64-cpython-311/mooncake-common/libasio.so`

拷到 83 的 `glm5_2_pd/lib/`，并：

```bash
export LD_LIBRARY_PATH=/export/home/shifengmin.3/workspace/glm5_2_pd/lib:$LD_LIBRARY_PATH
```

## 4. 管控面（已在 83 容器内）

```text
./etcd --listen-peer-urls http://0.0.0.0:4390 \
       --listen-client-urls http://0.0.0.0:4389 \
       --advertise-client-urls http://11.87.191.83:4389

./xllm-service/build/xllm_service/xllm_master_serving \
  --etcd_addr=11.87.191.83:4389 \
  --http_server_port 48888 \
  --rpc_server_port 48889 \
  --tokenizer_path=/export/home/models/GLM-5.2-W8A8-EcoTech
```

日志：`/export/home/shifengmin.3/workspace/xservice.log`

**HTTP 48888 在没有可用 P+D 实例时不会 listen**，这是 xllm-service 的 readiness 逻辑，不是启动失败。

Prefill / Decode 的 `ETCD_ADDR` 必须是 `11.87.191.83:4389`，不要用 `127.0.0.1`。

**不要**杀 83 上的 etcd / `xllm_master_serving`。

## 5. `/etc/hccn.conf`

PD KV 传输要读该文件。测试容器默认没挂，从同机 `fengmin-cann9` 拷入（**按机器分别拷，IP 不同**）：

```bash
# 83
docker cp fengmin-cann9:/etc/hccn.conf /tmp/hccn.conf.83
docker cp /tmp/hccn.conf.83 fengmin-cann9-0801:/etc/hccn.conf

# 98（从 83 跳转）
ssh shifengmin.3@11.87.191.98 '
  docker cp fengmin-cann9:/etc/hccn.conf /tmp/hccn.conf.98
  docker cp /tmp/hccn.conf.98 fengmin-cann9-20260801:/etc/hccn.conf
'
```

校验：98 的 `address_0=11.98.191.11`，83 的 `address_0=11.83.191.11`，各 16 张卡。

`docker restart` 后 overlay 上的拷贝会丢，需再拷一次，或改为启动容器时 `-v /etc/hccn.conf:/etc/hccn.conf`。

## 6. 启动脚本

路径：

- Prefill 仓库：`scripts/glm5_2_pd/`（本机）
- Decode 拷贝：83 上 `/export/home/shifengmin.3/workspace/glm5_2_pd/`（83 家目录与 98 **不是**同一份盘）

环境变量：

| 变量 | 含义 | 本实验取值 |
|---|---|---|
| `XLLM_BIN` | C++ 可执行文件 | 见第 3 节 |
| `MODEL_PATH` / `DRAFT_MODEL_PATH` | 主模型 / MTP | 见第 3 节 |
| `HOST` | 本机容器 IP | 98 或 83 |
| `ETCD_ADDR` | etcd | `11.87.191.83:4389` |
| `NNODES` | 进程数 | 16 |
| `LOG_DIR` | 每 rank 一份日志 | 自定义绝对路径 |
| `EXTRA_XLLM_ARGS` | 追加 CLI | Decode **必须** `--max_memory_utilization=0.88 --rank_tablefile=/tmp/hccl_16p.json`（AIV + layerwise=2） |
| `NUM_SPECULATIVE_TOKENS` | MTP 长度 | 默认 3（已验证正确中文） |
| `KV_CACHE_TRANSFER_TYPE` | KV 传输后端 | layerwise 用 `LlmDataDist`（Mooncake 不兼容） |
| `LAYERWISE_SPLIT_SIZE` | Decode 层切分 | 已验证 `2`。`1` = 不切分。必须 >= 1 |

默认端口（P/D 错开，勿改到互相冲突）：

| | Prefill | Decode |
|---|---|---|
| brpc | 18994+ | 19994+ |
| master | 18888 | 19888 |
| transfer | 36100+ | 37100+ |
| disagg_pd | 8877 | 8878 |

建议环境变量：

```bash
export LD_PRELOAD=/usr/lib64/libtcmalloc.so.4:$LD_PRELOAD
export HCCL_EXEC_TIMEOUT=300
export HCCL_CONNECT_TIMEOUT=300
# Decode start_decode.sh 默认 HCCL_OP_EXPANSION_MODE=AIV（需 --rank_tablefile）。
# Prefill 仍 unset。Decode 上要关掉 AIV：
#   export HCCL_OP_EXPANSION_MODE=
```

机器重启后先做 NPU 初始化，否则 xllm 可能起不来：

```bash
python3 -c "import torch_npu
for i in range(16):
    torch_npu.npu.set_device(i)"
```

先起 Decode，后起 Prefill。就绪：两边 16 个 rank 日志均出现 `Brpc Server started`，随后 `11.87.191.83:48888` 可连。

### Decode（83 容器内）

```bash
cd /export/home/shifengmin.3/workspace/glm5_2_pd
export XLLM_BIN=/usr/local/python3.11.15/lib/python3.11/site-packages/xllm/xllm
export MODEL_PATH=/export/home/models/GLM-5.2-W8A8-EcoTech
export DRAFT_MODEL_PATH=/export/home/models/GLM-5.2-W8A8-EcoTech-MTP
export HOST=11.87.191.83
export ETCD_ADDR=11.87.191.83:4389
export LOG_DIR=/export/home/shifengmin.3/workspace/glm5_2_pd/logs/decode
export EXTRA_XLLM_ARGS='--max_memory_utilization=0.88 --rank_tablefile=/tmp/hccl_16p.json'
export NUM_SPECULATIVE_TOKENS=3
export KV_CACHE_TRANSFER_TYPE=LlmDataDist
export LAYERWISE_SPLIT_SIZE=2
export LD_LIBRARY_PATH=/export/home/shifengmin.3/workspace/glm5_2_pd/lib:${LD_LIBRARY_PATH:-}
./start_decode.sh
# 脚本默认 HCCL_OP_EXPANSION_MODE=AIV。关掉：export HCCL_OP_EXPANSION_MODE=
```

### Prefill（98 容器内）

```bash
cd /export/home/shifengmin.3/workspace/xllm_coding/xllm/scripts/glm5_2_pd
export XLLM_BIN=/usr/local/python3.11.15/lib/python3.11/site-packages/xllm/xllm
export MODEL_PATH=/export/home/models/GLM-5.2-W8A8-EcoTech
export DRAFT_MODEL_PATH=/export/home/models/GLM-5.2-W8A8-EcoTech-MTP
export HOST=11.87.191.98
export ETCD_ADDR=11.87.191.83:4389
export LOG_DIR=/export/home/shifengmin.3/workspace/xllm_coding/logs/glm52_pd_smoke/prefill
export EXTRA_XLLM_ARGS='--max_memory_utilization=0.88'
export NUM_SPECULATIVE_TOKENS=3
export KV_CACHE_TRANSFER_TYPE=LlmDataDist   # 必须与 Decode 一致
./start_prefill.sh
```

## 7. Smoke

```bash
curl -sS http://11.87.191.83:48888/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "GLM-5.2-W8A8-EcoTech",
    "max_tokens": 16,
    "temperature": 0,
    "messages": [{"role": "user", "content": "你好，请用一句话介绍你自己。"}]
  }'
```

再测 `stream: true`。更长 prompt 用来确认 KV transfer，不算最小 smoke。

## 8. 停测（只杀 worker）

不要用 `pkill -f instance_role=DECODE`：会误杀命令行里带这段字符串的 bash。按 ELF 路径扫：

```bash
# 98 Prefill
ps -eo pid,ppid,cmd | awk '/site-packages\/xllm\/xllm/ && /instance_role=PREFILL/ {print}'
kill $(pgrep -f '/site-packages/xllm/xllm .*instance_role=PREFILL' || true)

# 83 Decode（容器内）
kill $(pgrep -f '/site-packages/xllm/xllm .*instance_role=DECODE' || true)
```

每个 rank 常有 1 个父进程 + 1 个编译子进程，16 rank 看起来像 32 条，属正常。

## 9. 增量编译与部署（Mooncake 去重）

本机容器：

```bash
ninja -C /export/home/shifengmin.3/workspace/xllm_coding/xllm/build/cmake.linux-aarch64-cpython-311 xllm
cp /export/home/shifengmin.3/workspace/xllm_coding/xllm/build/lib.linux-aarch64-cpython-311/xllm/xllm \
   /usr/local/python3.11.15/lib/python3.11/site-packages/xllm/xllm
```

拷到 83（宿主机 `/tmp` 再 `docker cp` 进容器，因 `glm5_2_pd` 属 `zhouyu.474`）：

```bash
scp -i /export/home/shifengmin.3/.ssh/id_rsa -o IdentitiesOnly=yes \
  /export/home/shifengmin.3/workspace/xllm_coding/xllm/build/lib.linux-aarch64-cpython-311/xllm/xllm \
  shifengmin.3@11.87.191.83:/tmp/xllm.glm52
ssh -i /export/home/shifengmin.3/.ssh/id_rsa -o IdentitiesOnly=yes shifengmin.3@11.87.191.83 \
  'docker cp /tmp/xllm.glm52 fengmin-cann9-0801:/usr/local/python3.11.15/lib/python3.11/site-packages/xllm/xllm'
```

### 把本机编出的 wheel 装到 83

wheel 在 98 容器：

```text
/export/home/shifengmin.3/workspace/xllm_coding/xllm/dist/xllm_npu_torch2_9_0-0.11.0-cp311-cp311-linux_aarch64.whl
```

大文件用 `ssh cat` 传（`scp` 中断会截断，md5 对不上 pip 会报 `Wheel is invalid`）：

```bash
WHL=/export/home/shifengmin.3/workspace/xllm_coding/xllm/dist/xllm_npu_torch2_9_0-0.11.0-cp311-cp311-linux_aarch64.whl
ssh -i /export/home/shifengmin.3/.ssh/id_rsa -o IdentitiesOnly=yes shifengmin.3@11.87.191.83 \
  'cat > /tmp/xllm_npu_torch2_9_0-0.11.0-cp311-cp311-linux_aarch64.whl' < "$WHL"
# 两边 md5 一致后再 docker cp + pip
ssh -i /export/home/shifengmin.3/.ssh/id_rsa -o IdentitiesOnly=yes shifengmin.3@11.87.191.83 '
  docker cp /tmp/xllm_npu_torch2_9_0-0.11.0-cp311-cp311-linux_aarch64.whl \
    fengmin-cann9-0801:/tmp/xllm_npu_torch2_9_0-0.11.0-cp311-cp311-linux_aarch64.whl
  docker exec fengmin-cann9-0801 python3 -m pip install --force-reinstall --no-deps \
    /tmp/xllm_npu_torch2_9_0-0.11.0-cp311-cp311-linux_aarch64.whl
'
```

当前这份 wheel 是 11:15 的包，**不含**随后合入的 Mooncake 去重。装完必须再覆盖一次新 ELF（上面的 `xllm.glm52`），并保留 `LD_LIBRARY_PATH=.../glm5_2_pd/lib`（`libasio.so`）。

## 10. 排障

| 现象 | 处理 |
|---|---|
| 48888 连不上 | 先看 P/D 是否都 `Brpc Server started` 并注册 etcd |
| HCCL / transfer 失败 | 检查容器 `/etc/hccn.conf` 是否为本机 IP |
| `XLLM_BIN` 立刻退出 | 是否误用 Python wrapper；看 rank_0.log |
| `Available kv cache size must be greater than 0` | `--max_memory_utilization=0.91`（或更高） |
| `registerLocalMemory failed` / overlapped memory | Decode `layerwise_split_size=4` 下未持有层共用 scratch KV，同一 `data_ptr` 被登记多次。需带地址去重的 ELF（见第 11 节） |
| `libasio.so: cannot open shared object file` | 83 上设 `LD_LIBRARY_PATH=.../glm5_2_pd/lib` |
| etcd 连不上 | advertise 必须是 `11.87.191.83:4389` |
| 83 SSH Permission denied | 用户必须是 `shifengmin.3` + `id_rsa` |
| docker restart 后 PD 失败 | 重新拷 `hccn.conf` |
| 停测误伤管控面 | 只杀 xllm worker，不要杀 etcd / `xllm_master_serving` |
| `NPU out of memory` 但本进程只占了几 GB | 卡被**宿主机上其它任务**占用。容器内 `npu-smi info` 看到的 PID 是宿主机 PID。**不要杀别人的任务** |
| LlmDataDist 首包 Decode 进程消失 | `layerwise=4` 时 rank 3/7/11/15 报 `SVector at index out of range`；`layerwise=1` 无该异常但 Decode 仍会在首 token 后退出。Prefill 侧能跑完 1 个 token。见第 14 节 |

Prefill 无 layerwise split 时，78 层 × 3 个 tensor = 234 块独立 buffer，旧 ELF 也能注册成功。Decode 在 `buf_id=6`（第 3 层的 K，与 scratch 层地址相同）失败。

## 11. Mooncake 重复注册（根因与修复）

Decode 开了 `layerwise_split_size=4`。`allocate_kv_caches()` 里未持有的层全部 `create_shared_view()` 指向**同一块** scratch KV。`get_cache_tensors()` 仍按 78 层展开，Mooncake `registerLocalMemory` 不允许重叠区间。

修复：`MooncakeKVCacheTransferDefault::add_buf` 按 `data_ptr` 去重，复用已有 `buf_id`，只把新 region 交给 `register_memory`。日志类似：

```text
register_kv_cache_impl success, registered_layers=78, new_buffers=63, total_unique_buffers=63
```

（约 1/4 层真实 KV + 1 份 scratch；每层 K/V/indexer 共 3 块。）

MTP=3 时 draft KV 也会再走一遍注册，去重表跨 main/spec 共用。MTP 是否完全打通仍需实跑确认。

去重之后 **登记能成功，但 PD 推 KV 仍会失败**：Mooncake `move_memory_groups` 要求两边 `buffers.size()` 相同，并用同一 `buf_id` 索引远端。Prefill 无 layerwise 是 234 块，Decode `layerwise_split_size=4` 去重后是 63 块，日志为 `buffer count mismatch, local=234, remote=63`。端到端 smoke 可先 `LAYERWISE_SPLIT_SIZE=1`（两边都是 234）。layerwise=4 需要对齐逻辑 buf_id 与物理 MR 的别名，尚未做。

代码：`xllm/core/framework/kv_cache_transfer/mooncake_kv_cache_transfer.{h,cpp}`。

## 12. 脚本端口与角色摘要

`start_prefill.sh`：`--instance_role=PREFILL`，`--cp_size=2 --kv_split_size=2`。

`start_decode.sh`：`--instance_role=DECODE`，`--dp_size=2 --layerwise_split_size=${LAYERWISE_SPLIT_SIZE:-4}`。

## 13. 可复用经验清单

1. SSH 83 只用 `shifengmin.3`；动 98 宿主机 docker 从 83 再跳。
2. 永远跑 C++ ELF，不跑 Python wrapper；83 记得 `libasio.so` + `LD_LIBRARY_PATH`。
3. `hccn.conf` 按机器从 `fengmin-cann9` 拷，restart 后会丢。
4. GLM-5.2 W8A8 + MTP 在 16 卡上必须把 `max_memory_utilization` 提到约 0.91。
5. xllm-service 的 HTTP 口是 readiness 口，P+D 齐了才 listen。
6. 停测按 ELF 路径杀 worker，别 `pkill` 角色字符串，别动 etcd。
7. Decode layerwise split 会让未持有层共享 scratch，Mooncake 必须按地址去重。
8. 首次 PD smoke 先 `NUM_SPECULATIVE_TOKENS=0`，过了再开 MTP=3。
9. 83 与 98 家目录不是同一块盘；83 上写 `glm5_2_pd` 用 `docker exec` / `docker cp`。
10. 起服务前先 `npu-smi info`。容器与宿主机共享 NPU；HBM 已被其它 PID 占满时不要启动，更不要杀别人的进程。

## 14. 当前测试状态（2026-08-25）

已完成：

- Mooncake 按 `data_ptr` 去重：Decode `layerwise_split_size=4` 登记成功，`new_buffers=63`。
- 无 MTP 首次 chat（Mooncake + layerwise=4）：Prefill 推 KV 失败，`buffer count mismatch, local=234, remote=63`，随后 Prefill FATAL，xllm-service 摘掉实例。
- Decode `LAYERWISE_SPLIT_SIZE=1` + **Mooncake** 端到端（2026-08-25 12:14）：两边登记都是 `78/234/234`，建链成功（`prefill_kv_split_size: 2`）。Prefill 第 1 个 token 出来（TTFT ~6.6s）。Decode 仍在首包后整组退出，无 `buffer count mismatch`、无 `SVector`。说明 **layerwise=1 对齐 buffer 之后，Decode 照样死，且与传输后端无关**（Mooncake 与 LlmDataDist 现象相同）。
- **LlmDataDist 后端**（`KV_CACHE_TRANSFER_TYPE=LlmDataDist`，无 MTP，`max_memory_utilization=0.91`）：
  - Initialize / LinkLlmClusters 成功；P+D 注册后 HTTP `48888` listen。
  - Prefill 首 token 能出来（layerwise=4 时 TTFT ~7.3s，layerwise=1 时 ~0.58s）。
  - Decode 在收 KV / 首步 decode 时整组退出，xllm-service 摘实例，curl 返回 `Instance is failed and deleted`。
  - `LAYERWISE_SPLIT_SIZE=4`：rank 3/7/11/15 明确 `terminate ... std::out_of_range` / `SVector at index out of range`（ATB `SVector`）。
  - `LAYERWISE_SPLIT_SIZE=1`：无 `SVector` 异常，进程同样消失（TBE `main process disappeared`），日志停在 `LinkLlmClusters success`。
- **LlmDataDist，Prefill 关 CP/kvsplit**（`CP_SIZE=1 DP_SIZE=2 KV_SPLIT_SIZE=1`，Decode 仍 DP=2 TP=8 layerwise=1）：`prefill_kv_split_size: 1` 建链成功。Prefill 仍能出第 1 个 token（TTFT ~3.5s），Decode 同样在首包后整组退出。关 CP/kvsplit **不能**绕过 Decode 崩溃。
- **Mooncake + 两边 full TP=16**（`DP=1 CP=1 kv_split=1 layerwise=1`，无 MTP）：登记仍是 `78/234/234`，建链成功。Prefill 第 1 个 token 出来（TTFT ~3.0s），Decode 仍在首包后整组退出。full TP **不能**绕过 Decode 崩溃。

未完成：

- layerwise=1 时 Decode 首包静默退出（当前主线，见第 15 节）。
- 端到端 curl `/v1/chat/completions` 尚未成功。MTP=3 未测。

## 15. 待办：layerwise=4 构图失败（先记下，稍后再看）

Decode `layerwise_split_size=4` 时，**skip_topk × layerwise owner** 的 ATB `DecoderLayer()` 建不出图。

证据（`logs/decode/rank_3.log`，rank 3/7/11/15）：

- 加载权重阶段大量 `npu_deepseek_v32_decoder_layer_impl.cpp:992] node.operation is null`
- 成对出现（每层 `prefill_node_` + `decode_node_`），19 对 = 19 个 owned 层
- owner 公式是 `layer_id % 4`，这些 rank 拥有的正好是 layer 3/7/11…（GLM-5.2 的 `shared` indexer 层）
- 首包 `build_node_variant_pack` 往空 `variantPack` 写 `WEIGHT_COUNT_PER_LAYER + N`，`SVector at index out of range`

原因要点：

- ATB `runIndexer = !skipTopk && (!layerwise || owner)`，skip_topk owner 仍走 SparseFlashAttention，Q 走 `layerwise_split_q_*_owner`，但 indexer 链被跳过，图创不出来。
- `merge_loaded_weights()` 里 `init_layer()` 的返回值被丢掉，进程带着空 `operation` 起来。
- Mooncake `234 vs 63` 是独立问题（逻辑 `buf_id` 与物理 MR 未对齐），构图不过则即使对齐也跑不了 skip_topk decode。

稍后修：ATB skipTopk+layerwise owner 构图；`init_layer()` 失败必须 FATAL。

## 16. layerwise=1 Decode 挂掉（当前主线）

NPU plog（Decode rank0 `plog-247440`，full TP=16 / Mooncake / layerwise=1）：

| 时间 | 事件 |
|---|---|
| 12:24:15.51 | Mooncake HIXL `HcclCommPrepare` 成功（Prefill 正在 PUSH KV） |
| 12:24:16.63 | Prefill 打出第 1 个 token |
| 12:24:16.71 | Decode 首次 `CreateCCLbuffer` **1074790400 B（约 1.00 GB）** |
| 12:24:16.91–17.10 | 16 个 rank **同时**在建 `AllGather_..._sub_1` 的 P2P transport |
| 之后 | 进程消失，无 AIC/SVector；xllm 日志停在 `link_cluster` |

KV 占用：权重后剩余约 13.7 GB，`max_memory_utilization=0.91` 分给 KV **8.15 GB / 622 block**，理论剩余约 **5.5 GB**。1 GB CCL buffer 本身分配成功，死在随后的 AllGather `_sub_1` 建链。

含义：layerwise=1 不是卡在 skip_topk 的 SFA 内核（那是 layer 3），而是 **第一次执行 `decode_node_` 时懒创建 HCCL AllGather 子通信域**，整组在建链过程中退出。Prefill 空闲时从不跑 decode 图，所以只有首包才触发。Mooncake / LlmDataDist 都会走到这条 decode AllGather，所以两边现象一样。
