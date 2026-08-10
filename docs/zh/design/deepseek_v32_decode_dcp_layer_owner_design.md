# DeepSeek V3.2 Decode Layerwise Owner DCP 设计文档

## 文档状态

- 状态：设计已确认，第一阶段 eager ATB 源码实现完成，待 Ascend NPU 编译和端到端验证
- 目标平台：Ascend NPU / ATB
- 目标模型：DeepSeek V3.2 DSA
- 当前阶段：仅在 PD 混布实例验证 decode 组图，prefill 保持原路径，KV Cache 仍由外部框架在每个 rank 全量分配
- 待验证项：见“实现验证项”章节；不存在未决设计项

## 1. 背景

DeepSeek V3.2 的 decode DSA 需要先通过 LightningIndexer 从长上下文中选择固定数量的 cache，再执行 SparseFlashAttention。目标是在 Tensor Parallel（TP）基础上增加 Decode Context Parallel（DCP），降低未来每个 rank 的 KV Cache 存储量，并确保以下语义与无 DCP 基线一致：

- Indexer 选出的全局逻辑位置不重复、不遗漏；
- 当前 token 只出现一次并满足 causal；
- MTP 中每个 query 只能看到自己的 causal 前缀；
- top-k sharing 跨层、跨 MTP iteration 保持正确；
- 每个 TP rank 仍然使用自己的 attention Q heads 和投影权重完成本地 attention 后处理。

本文选择 layerwise owner 方案：DCP group 中每一层只有一个 owner rank 持有并访问该层 cache。owner 完成 Indexer 和 selected cache 打包，再把 selected cache 广播给组内其他 TP rank。所有 rank 使用自己的 TP Q head shard 执行 SparseFlashAttention。

## 2. 设计目标

### 2.1 最终目标

- 每个 DCP group 内，每层只有 owner 分配完整容量并读写该层的语义 cache；non-owner 仅为该层保留一个永不读取的 null block，以维持统一 `KVCache` tensor 接口并兼容融合 cache-write 算子。
- owner 使用完整的本层逻辑 cache 执行原始 PA_BSND LightningIndexer。
- owner 将选中的 cache 打包为 TND 布局并通过 DCP Broadcast 分发。
- 所有 rank 使用本地 TP Q heads 执行 TND SparseFlashAttention。
- owner 按层轮转，避免所有层的 cache 和 Indexer 都集中到同一设备。

### 2.2 快速验证目标

快速验证阶段不修改 KVCache 分配、block manager、scheduler、slot mapping、prefill 和 cache transfer：

- 外部框架仍在每个 rank 上分配并填充全量 cache；
- PD 混布实例中的 prefill 完整执行原有图，在所有 rank 写入全量 cache；
- decode ATB 图严格按照最终 sharding 数据依赖构造；
- 非 owner 在 decode 中不读、不写本层 cache；
- 非 owner 只通过 DCP Broadcast 获得 selected cache；
- 验证通过后，真实 sharding 只改变 cache 生命周期，不再改变 decode ATB 图。

## 3. 非目标

- 本阶段不减少实际 KV Cache 显存占用。
- 本阶段不修改 prefill 的 cache 路由。
- 本阶段不修改 PD、prefix cache 或 cache transfer 的 layer ownership。
- 本阶段不验证 PD 拆分部署和跨实例 cache transfer，只验证 PD 混布实例内由原有 prefill 生成的 cache 上的 decode。
- 本阶段不对单层 MTP draft model 做 DCP 或 cache sharding；draft model 继续使用原图和全量 cache。
- 本阶段不以性能收益作为通过标准；首先验证正确性和数据依赖。
- 本阶段不引入“两级 Indexer”路径。

## 4. 并行拓扑

### 4.1 DCP group

DCP group 从 attention TP group 中连续切分。`dcp_size` 表示每个 DCP group 的 rank 数，并要求：

```text
tp_size % dcp_size == 0
```

例如 TP group 为 `[0..15]`、`dcp_size=4` 时，DCP groups 为：

```text
{0, 1, 2, 3}
{4, 5, 6, 7}
{8, 9, 10, 11}
{12, 13, 14, 15}
```

组内 rank 持有不同的 TP attention head shard，因此 owner 不能只使用自己的 Q heads 代替其他 rank 完成 attention。

每个 DCP group 独立选举 layer owner：

```text
owner_local_rank(layer_id) = layer_id % dcp_size
```

该规则使用物理 decoder `layer_id`，不因 dense/MoE 层类型、Indexer top-k sharing pattern 或 MTP 层而改变。`owner_local_rank` 是 DCP communicator 内的 local rank，不是 global rank。每个 DCP group 都有自己的 owner，因此 cache 在不同 DCP group 之间仍会存在副本。

当前 `ATTN_CP` 采用的 strided 分组语义不能直接表示上述连续 DCP 子组。decode layerwise owner DCP 需要独立 mapping 和 communicator；不得通过修改 `ATTN_CP` 的既有语义影响 prefill CP。

第一版新增独立的实验配置 `decode_dcp_size`，不得复用 `cp_size` 或 `ATTN_CP`。初始化时从 attention TP communicator 取得其有序 rank 列表 `attn_tp_rank_ids`，按照该列表的相邻元素每 `decode_dcp_size` 个切分一个 DCP group。这里的“连续”指 attention TP communicator 中的 rank 顺序，不能假设 world rank 的数值连续。初始化必须校验：

```text
1 <= decode_dcp_size <= len(attn_tp_rank_ids)
len(attn_tp_rank_ids) % decode_dcp_size == 0
```

所有 rank 必须生成相同的分组和 communicator；Broadcast root 使用分组内 local rank。`decode_dcp_size=1` 关闭 owner-DCP 路径并保持原 decode 图，用于建立同配置基线。

### 4.2 Framework mapping 和 communicator

decode DCP 是框架的一等并行 mapping，新增 `ATTN_DECODE_DCP`。`MappingNPU` 在初始化时以 `ATTN_TP.rankIds` 的顺序生成其 `rankIds`、`groupId` 和 local `rank`；`ExternalCommManager` 基于该 mapping 注册并返回唯一的 `commDomain`/`hcclComm`。因此 ATB 参数中的 `dcpRank`、`dcpSize`、`dcpCommDomain` 和 `dcpOwnerRank` 都直接来自 `ATTN_DECODE_DCP`。

decoder layer wrapper 只能读取这个 mapping，不能自行按 global rank 创建 HCCL group 或拼接 commDomain。该边界保证 PD、多模型和未来真实 layer cache sharding 始终使用同一个拓扑事实来源。

## 5. Decode 数据流

### 5.1 所有 rank 的公共路径

所有 rank 执行：

1. Attention RMSNorm；
2. 保留现有融合 QKV down projection 和 Split；
3. 继续消费本地 latent Q，生成本地 `q_nope` 和 `q_rope`；
4. 等待 owner 广播 selected cache；
5. 使用本地 Q heads 执行 TND SparseFlashAttention；
6. 执行本地 V reprojection 和 O projection；
7. 进入原有 TP collective 和后续 decoder layer。

### 5.2 owner 路径

owner 额外执行：

1. 消费融合 QKV down projection 已产生的 MLA latent KV 和 RoPE K，并执行后续 KV norm/RoPE；
2. 写入本层 MLA cache 和 RoPE cache；
3. 在需要 Indexer 的层计算 Indexer Q、K 和 weight；
4. 写入当前 token 的 Indexer cache；
5. 使用原始 PA_BSND LightningIndexer 得到 raw 全局逻辑 top-k；
6. 仅在 `outputTopk` 层先 Broadcast top-k，并以 Broadcast 输出作为后续输入；
7. 将逻辑 top-k 位置映射到物理 cache slot；
8. Gather latent cache 和 RoPE cache；
9. 按 query 紧凑打包 selected cache；
10. Broadcast selected cache。

### 5.3 non-owner 路径

non-owner 不执行：

- 当前 token 的 MLA cache 写入；
- 当前 token 的 Indexer cache 写入；
- LightningIndexer；
- block table 寻址；
- paged cache Gather；
- 任何本层历史 cache 读取。

non-owner 仍执行现有融合 QKV down projection，因为当前 `in_q_proj_a_weight` 和量化路径一次产出 latent KV、RoPE K 与 latent Q；non-owner 仅继续消费 latent Q，KV/RoPE K 分支在 Split 后终止。该冗余计算不读写 cache，快速验证和未来真实 cache sharding 使用同一图。第一版不拆分融合权重或新增 Q-only projection；是否优化由后续 profiling 决定。

GLM-5 的 target model type 为 `glm_moe_dsa`，MTP model type 为 `glm_moe_dsa_mtp`。当前 NPU layer 对包含 `glm_moe_dsa` 的 model type 强制设置 `enableMlaPreprocess=false`，因此即使量化类型为 `w8a8_dynamic`，GLM-5 decode 也不会进入将 Q preprocess 与 cache write 融合的 `MlaPreprocessV2`，而是进入本文可按 owner/non-owner 拆分的普通 preprocess 图。第一版不需要为 GLM-5 新增 `MlaPreprocessV2` 特判或 Q-only 变体。

未来若扩展到实际启用 `MlaPreprocessV2` 的模型，non-owner 不必拆分该融合算子。真实 sharding 初始化时仍为每个 non-owned layer 创建完整类型的 `KVCache` tensor 对象，但其物理容量固定为一个 null block；owner layer 则按正常 block 容量分配。non-owner 将融合算子的 MLA/RoPE cache 输出写入自己本层的 null block，并丢弃 KV 分支，只继续消费本地 Q。所有 non-owner execution row 的 `new_cache_slot` 均设为 `0`，允许同一次算子执行中的多个 token 重复写同一 null slot；该 slot 的最终内容未定义且不得被任何后续节点读取。NPU wrapper 在 non-owner 路径中忽略 worker 传入的真实 `new_cache_slots`，改为绑定初始化时分配的持久 INT32 全零 buffer，并按当前 `execution_Tq` 取得 active view；owner 继续绑定真实 slots。zero-slot buffer 容量直接使用现有 `max_tokens_per_batch`，不新增 DCP 专用容量配置；MTP expanded verify 的 query token 同样计入该调度上限。worker、scheduler 和 block manager 均不感知这一替换。null block 和 zero-slot buffer 在 executor 生命周期内地址稳定，forward 不重新分配、不清零。该方案直接复用现有按层 `KVCache` 创建、索引和 ATB tensor 绑定接口，不需要独立 scratch allocator、跨 layer/stream 共享、cache-signature 管理或新 NPU 算子，也不属于第一版 GLM-5 的实现依赖。

non-owner 使用持久接收 buffer 作为 Broadcast 的输入/输出缓冲。该缓冲是最终方案的一部分，不是验证阶段为了绕过数据依赖而增加的临时路径。

ATB operation 可以保留 owner/non-owner 一致的外部 cache 输入签名。快速验证阶段 non-owner 绑定当前层全量 cache；真实 sharding 后直接绑定当前 non-owned layer 自己的一-block null `KVCache`。第一版 GLM-5 的 non-owner 图在构造期必须移除所有读取或写回这些 cache 输入的节点，不能依靠运行时条件跳过。未来 `MlaPreprocessV2` 兼容路径是唯一例外，但它只允许融合算子写本层 null block；Indexer、attention 和其他节点仍不得读取该 null block。tensor 是否已定义、tensor 容量和 shape 都不能用于判断 ownership，owner/non-owner 角色只能来自 `LayerCachePlacement`。

### 5.4 图结构

```text
                         +-------------------------------+
                         | owner-only                    |
hidden states ---------->| KV/Indexer projection        |
       |                 | ReshapeAndCache              |
       |                 | LightningIndexer             |
       |                 | logical pos -> physical slot |
       |                 | selected cache Gather/Pack   |
       |                 +---------------+---------------+
       |                                 |
       |                                 v
       |                    DCP Broadcast selected cache
       |                                 |
       v                                 v
local TP Q ----------------------> local TND SFA
                                          |
                                          v
                                local V reproj / O proj
                                          |
                                          v
                                  existing TP collective
```

owner 和 non-owner 可以构造不同的本地计算图，但同一 DCP group 内所有 rank 的 collective 数量、顺序、communicator 和 tensor shape 必须一致。

### 5.5 Top-k sharing 通信顺序

对每一层，DCP collective 顺序由该层静态的 top-k sharing 属性决定，不能由运行时 top-k 内容决定：

| 层类型 | owner 本地工作 | DCP collective 顺序 | 供后续层使用的 top-k |
|---|---|---|---|
| 不 sharing | 运行 Indexer、Gather selected cache | `Broadcast(topk_int32)`，随后 `Broadcast(selected_cache)` | 无；top-k Broadcast 只写本层内部 tensor |
| `outputTopk` | 运行 Indexer | `Broadcast(topk_int32)`，随后 `Broadcast(selected_cache)` | Broadcast 输出逻辑 shape `[Tq, index_topk]` |
| `skipTopk` | 使用上游 top-k Gather selected cache | `Broadcast(selected_cache)` | 复用上游 Broadcast 输出 |

所有 `skipTopk=false` 的层都必须先广播 owner 生成的 raw logical top-k，再执行 slot 映射和 cache Gather。`outputTopk=true` 时，Broadcast 以现有物理 ABI `[Tq, 1, index_topk]` 直接写 `out_topk_indices`，owner 和 non-owner 都将该 tensor 视为当前层对外输出；`outputTopk=false` 时，Broadcast 写只在本层使用的 `dcp_topk`，不改变 decoder layer 外部输出。owner 的 slot 映射和 cache Gather 必须消费 Broadcast 结果，而非绕过它直接消费 raw top-k。这样独立 Indexer 层和 top-k producer 层都能保证本层所有 rank 使用同一组 logical positions；producer 和 consumer 的 owner 不同时，`skipTopk` 也能从框架传递的同一 INT32 tensor 正确执行。单层 MTP draft model 沿用原图和全量 cache。

`index_n_heads` 只参与 Indexer score 计算，不是 top-k 输出维度。现有 ATB 接口若保留 key-head singleton，物理 tensor 可以是 `[Tq, 1, index_topk]`；该 `1` 不是 `index_n_heads`，组图中可以直接作为 `[Tq, index_topk]` 的兼容 view 使用。每个 query 因此仍只对应一组最多 `index_topk` 个 logical positions。

## 6. Cache 寻址和打包

LightningIndexer 输出序列内的全局逻辑位置。普通 Gather 访问展平后的 paged cache 时需要物理 slot：

```text
topk_physical_abi[Tq, 1, index_topk]
topk_flat_view[Tq * index_topk]
```

singleton 维只用于兼容现有 LightningIndexer、top-k sharing 和 SparseFlashAttention 接口。owner 在 slot mapping 前将其 view 为一维；短上下文再通过 host 生成的 `packed_gather_indices` 保留每个 query 的前 `K_i` 项。该 view 不产生数据拷贝，也不改变 top-k 顺序。

selected cache 的行顺序必须严格等于 LightningIndexer top-k 的原始 rank order。第一版禁止按 logical position 或 physical slot 排序，禁止去重，也禁止为了 Gather 连续性重排。logical-to-physical 映射只替换地址表示，不改变元素次序；因此 TND SFA 可以直接使用 identity sparse indices，在冻结相同 layer 输入/cache 的单层诊断中，selected cache 也能与 baseline 按行逐元素比较。若 profiling 证明随机 Gather 是瓶颈，后续优化必须同时引入显式重排和 sparse-index 反向映射，不能静默改变本契约。

```text
logical_block = logical_pos / block_size
block_offset = logical_pos % block_size
physical_block = block_table[batch_id, logical_block]
physical_slot = physical_block * block_size + block_offset
```

第一版优先使用现有算子和静态 LUT 组合：

```text
logical_block_lut[p] = p / block_size
block_offset_lut[p] = p % block_size
```

组图步骤：

1. Gather `logical_block_lut`；
2. Gather `block_offset_lut`；
3. 根据 `query_batch_ids` 选择 block table 行；
4. 使用 `Gather(axis=1, batchDims=1)` 得到 physical block；
5. 计算 physical slot；
6. 展平 paged cache；
7. 分别 Gather latent cache 和 RoPE cache；
8. Concat 为 `[sum(K_i), kv_lora_rank + qk_rope_head_dim]` 后只执行一次 BF16 Broadcast。

若组合图在 profiling 中开销明显，再融合为 `PagedTopKCacheGather`。融合算子不是功能验证的前置依赖。

### 6.1 Selected-cache 通信契约

owner 按 query 的原始顺序紧凑打包 selected cache。第 `i` 个 query 的连续区间为：

```text
begin_i = sum(K_0 ... K_(i - 1))
end_i = begin_i + K_i
```

单个 BF16 Broadcast 的 active tensor 为：

```text
selected_cache[T_active, cache_width]
T_active = sum(K_i)
cache_width = kv_lora_rank + qk_rope_head_dim
```

接收方在 Broadcast 后沿最后一维 split 为 latent cache 和 RoPE cache；latent cache 同时作为 SparseFlashAttention 的 K/V 输入。所有 rank 只使用同一个 active view，并从已有逐 query metadata 独立派生：

```text
actual_seq_lengths_query[i] = i + 1
actual_seq_lengths_key[i] = end_i
```

packed TND SFA 将每个 query row 视为一个独立、长度为 1 的 query sequence。即使多个 MTP query 属于同一个原始请求，也不能继续用原始 `q_len` 把它们合并成一个 TND sequence，因为每个 query 对应不同的 causal selected-cache segment。`actual_seq_lengths_query=[1,2,...,Tq]` 与 `actual_seq_lengths_key=cumsum(K_i)` 必须成对生成。

wrapper 为最大 `Tq_max * index_topk * cache_width` 预分配持久接收 buffer，但 Broadcast 和 SFA 仅绑定前 `T_active` 行；未使用尾部不得被 SFA、Gather 或任何 cache 读取节点消费。eager ATB 阶段允许 `T_active` 随 batch 变化；Graph Mode 仅改变 buffer/view 的 bucket 和 capture/update，不改变该紧凑布局或单次 Broadcast 契约。

### 6.2 第一版算子依赖

第一版不要求新增 CANN/ATB 自定义算子，使用现有能力完成：

| 功能 | 第一版实现 | 是否新增算子 |
|---|---|---|
| selected cache/top-k 分发 | `atb::infer::BroadcastParam`，每层静态设置 `rankRoot`，绑定 DCP `hcclComm/commDomain` | 否 |
| top-k 逻辑位置分块和块内偏移 | 静态 `logical_block_lut`、`block_offset_lut` 加 Gather | 否 |
| block table 到 physical block | `Gather(axis=1, batchDims=1)` | 否 |
| physical slot 计算 | 现有 Elewise Mul/Add | 否 |
| 过滤每个 query 的 padding top-k | host 生成 `packed_gather_indices`，图内 Gather | 否 |
| latent/RoPE cache 读取及打包 | Gather 加 Concat | 否 |
| 全局选择 | 现有 LightningIndexer | 否 |
| 本地 attention | 现有 TND SparseFlashAttention | 否 |

必须新增的是 DCP 连续子组 mapping/communicator、ATB tensor contract 和 owner/non-owner 组图逻辑，它们是框架与图构造能力，不是 NPU 算子。`TopkLogicalPositionToPhysicalSlot` 不作为第一版依赖；`PagedTopKCacheGather` 仅是 profiling 后可选的融合优化。

## 7. 多 Query Decode 和 Causal

本设计不感知 MTP，不增加 `is_mtp` 控制分支，也不为 speculative decode 构造另一套 owner、Indexer、Broadcast 或 SFA 图。DCP 只用于 target model 的 decode 和 speculative verify；本来只有一层的 MTP draft model 保持全量 cache 和现有执行图，不进入 DCP。第一版仍要求端到端支持 MTP，其中 target verify 以 `Tq > batch_size` 的通用多 query decode 形态进入 DCP 图。

图只依赖：

- query token 总数 `Tq`；
- query/key cumulative sequence lengths；
- block table 和当前 token slots；
- LightningIndexer 为每个 query 产生的 top-k。

假设一次请求的当前 MTP query 长度为 `q_len`，`final_kv_len` 已包含本轮全部当前 token，则第 `i` 个 query 的可见长度为：

```text
visible_len(q_i) = final_kv_len - q_len + i + 1
K_i = min(index_topk, visible_len(q_i))
```

约束：

- owner 可以先把本轮全部当前 token 写入 cache；
- LightningIndexer 必须在选择阶段按 query 应用 causal，而不是 top-k 后过滤；
- 每个 query 只保留前 `K_i` 个有效 Indexer 输出；
- selected cache 按 query 紧凑排列为 `[sum(K_i), kv_lora_rank + qk_rope_head_dim]`；
- packed TND SFA 将每个 query row 作为长度为 1 的独立 sequence，`actual_seq_lengths_query=[1,2,...,Tq]`，不得沿用原请求的多 token `q_len` 分组；
- TND SFA 的 `actual_seq_lengths_key` 为 `cumsum(K_i)`；
- TND SFA 使用每个 query 局部的 identity sparse indices；固定 `index_topk` 维度中 `[0, K_i)` 为 identity 有效前缀，`[K_i, index_topk)` 使用与 baseline 兼容的无效尾部编码。该编码可在端到端失败后用算子级测试确定；在短上下文覆盖宣告通过前必须完成验证；
- padding top-k 在 Gather 前必须被移除或替换为合法占位值，不能导致 LUT/cache 越界。

### 7.1 当前 token

当前实现已经按以下顺序处理当前 token：先执行 MLA `ReshapeAndCache`，再执行 Indexer K 的 `ReshapeAndCache`，最后让 LightningIndexer 读取更新后的 cache。DCP 不新增当前 token 的复制、拼接、强制选中或单独 Gather：

1. owner 保留原有 cache 写入顺序和 logical slot；
2. 当前 token 作为原始 LightningIndexer 的普通候选；
3. selected cache 完全由原始 top-k 决定，不在 top-k 后额外 prepend/append 当前 token；
4. target verify 一次写入多个 query token 时，`visible_len(q_i)` 继续由原有 causal metadata 限制每个 query 可见前缀。

因此当前 token 被选择时恰好出现一次，未被选择时不进入 selected cache；DCP 只限制 cache write/indexer/gather 的执行 rank，不改变该语义。

需要的 metadata：

```text
query_batch_ids[Tq]
valid_topk_count[Tq]
packed_gather_indices[sum(K_i)]
actual_seq_lengths_query[Tq]
actual_seq_lengths_key[Tq]
identity_topk[Tq, 1, index_topk]
```

### 7.2 `K_i` 的来源和责任边界

worker 不新增 `K_i` 输入，也不新增 DCP/MTP 专用 metadata。`K_i` 由 NPU decoder layer wrapper 在组图前从已有的 host sequence metadata 派生：

```text
visible_kv_seq_lens[Tq] = normalize(q_seq_lens, kv_seq_lens)
valid_topk_count[i] = K_i = min(index_topk, visible_kv_seq_lens[i])
```

归一化规则：

1. 若 `graph.expanded_kv_seq_lens_vec` 已存在且长度为 `Tq`，直接将其作为 `visible_kv_seq_lens`。当前 MTP/speculative verify 路径已经按 `kv_len - q_len + token_idx + 1` 生成该向量；
2. 否则由 `attention.host.q_seq_lens` 和 `attention.host.kv_seq_lens` 按同一公式展开；
3. 普通单 token decode 的 `q_len=1`，因此结果自然等于原有 `kv_seq_lens`。

wrapper 使用 `valid_topk_count` 在 host 侧生成 `packed_gather_indices`、`actual_seq_lengths_query`、`actual_seq_lengths_key` 和 collective 的有效元素数量，再将这些 tensor 作为 ATB 图输入。这样做有两个原因：

- `K_i` 是已有 sequence metadata 的纯派生值，不应让 worker 感知 DCP 或复制语义；
- 当前紧凑布局的 Broadcast 长度为 `sum(K_i)`，ATB 执行前必须确定 tensor shape 和 HCCL count，不能依赖图内数据相关 shape。

因此，“worker 提供逐 query 可见长度”是已有输入契约；“计算 `K_i` 和紧凑布局 metadata”属于 NPU layer wrapper 的职责。未来若 Graph Mode 使用固定 bucket/padded buffer，只改变通信 buffer 的容量，不改变 `K_i` 的来源和有效长度语义。

wrapper 必须同时从同一份 metadata 生成每个 query 对应的 block-table row，不能只计算 `K_i`：

```text
sequence-scoped input:
  for sequence s, token j in [0, q_len_s):
    visible_len[i] = kv_len_s - q_len_s + j + 1
    block_table_row[i] = s

expanded spec-verify input:
  visible_len[i] = graph.expanded_kv_seq_lens_vec[i]
  block_table_row[i] = i
```

expanded verify 的 block table 已由 MTP worker 为每个 query token 复制一行；普通 decode 和非 expanded 多 query 输入仍使用 sequence-scoped block table。`visible_len`、`block_table_row`、`packed_gather_indices` 和 `actual_seq_lengths_key` 必须由一次归一化过程共同生成，禁止分别按不同 batch 语义推导。否则 top-k 虽然正确，logical position 到 physical slot 的映射仍可能读取另一条 sequence 的 cache。

本设计不单独验证 CANN LightningIndexer 的多 query causal 契约。现有无 DCP 路径作为语义基线；DCP 路径只要求每个 query 的 top-k 与相同输入下的无 DCP baseline 完全一致。若 baseline 本身存在 causal 问题，应作为独立问题处理，不扩大本项目范围。

### 7.3 空 batch / 空 DP shard

当前框架不会以真正的零 token tensor 执行需要 collective 的空 shard：

- eager 中，worker 检测到 `num_sequences=0` 且 token 为空后，在并行 mapping 生效时补一个 fake token 和 position，但保留 `meta.num_sequences=0`；
- decoder layer 通过 `num_sequences==0` 识别 empty eager batch，并为缺失的 `q_cu_seq_lens`、block table、slot 等绑定现有 placeholder；
- Graph Mode 中保留 `actual_num_sequences=0`，但按 bucket 构造非零 padded token rows，并为 padding row 填充 `q_len=1`、`kv_len=1` 和默认 slot/block table。

DCP 必须复用这一协议，不在 wrapper 中跳过空 shard，也不发起 count 为 0 的 Broadcast。wrapper 同时维护：

```text
actual_query_count = 0
execution_Tq = hidden_states.size(0) >= 1
```

当 logical batch 为空时，DCP metadata 按现有 placeholder/padding 语义为每个 execution row 构造 dummy `visible_len=1`、`K_exec=1`、`query_batch_id=0` 和 identity sparse-index 有效前缀。owner 仍执行其静态 owner 图，所有 rank 以相同非零 shape 调用 top-k/selected-cache Broadcast 和本地 SFA；输出随后按现有 empty-shard 语义丢弃。不得让 owner 跳过 collective、non-owner 继续执行，也不得向 worker 增加 DCP 专用 fake input。

`K_exec=1` 不是空 rank 的性能优化；整个 step 仍受最慢 rank 限制。选择它是因为可直接复用短上下文的 compact-pack 路径，并避免为 `visible_len=1` 伪造 2048 个合法 logical positions、重复 cache rows 或越界 padding top-k。empty-shard 支持因此与短上下文尾部编码共用同一实现和验证结果。

Graph Mode 的具体 bucket buffer 在第二阶段实现，但其 empty-shard 语义必须与 eager 相同：logical count 为 0，execution count 非零。

## 8. Top-k Sharing

### 8.1 outputTopk 层

owner 将 LightningIndexer 结果写入内部 tensor，然后执行 INT32 DCP Broadcast。所有 rank 的 `out_topk_indices` 都绑定 Broadcast 输出，确保下一层或下一次 MTP iteration 拿到相同的全局逻辑位置。对于 `outputTopk=false` 的独立 Indexer 层，同一 Broadcast 写入本层内部 `dcp_topk`；它不向下一层传播，但仍是本层 slot 映射的唯一 top-k 输入。

`out_topk_indices` 的 shape 必须由模型和本次输入 metadata 直接构造：

```text
logical: [Tq, index_topk]
ATB-compatible physical view: [Tq, 1, index_topk]
```

所有 rank 都分配该 Broadcast 输出 tensor。shape 中不得使用 `index_n_heads`。也禁止通过 `kv_caches[layer_id].get_index_cache()` 或任意 cache tensor 的 shape 推导输出 shape；singleton 维由模型固定的 Indexer 输出 ABI 直接给出，否则 non-owner 在真实 sharding 下会形成隐藏 cache 依赖。该改动必须在快速验证阶段完成，即使此时 non-owner 仍有全量物理 cache 副本。

### 8.2 skipTopk 层

上游 producer 已将 top-k 广播到所有 rank。当前层 owner 直接读取 `in_shared_topk_indices`，从自己的本层 cache Gather selected cache。当前层不重复广播 top-k，只广播 selected cache。

producer 和 consumer 的 layer owner 可以不同，因为 producer top-k 已经复制到整个 DCP group。

## 9. ATB 图和 Wrapper 改造

### 9.1 参数

建议增加默认关闭的实验开关：

```text
XLLM_ENABLE_DECODE_DCP_LAYER_OWNER=0
```

该开关只改变 decode 语义的 SparseAttention。普通 prefill、chunked prefill 和 prefix cache 路径必须继续使用现有图，不允许因为开启实验开关而进入 owner cache 写入或 DCP Broadcast。

第一版运行时激活条件固定为：

```text
enable_decode_dcp =
    enableDecodeDcpLayerOwner &&
    (batch_forward_type.is_decode() ||
     graph.use_expanded_decode_for_spec_verify_attention)
```

第二项使用现有 attention execution mode 覆盖 MTP/spec verify，不检查 `is_mtp` 或另建 MTP 图。`PREFILL`、普通 `CHUNKED_PREFILL` 和 `MIXED` batch 全部保持原图；第一版不在同一个 `MIXED` batch 内只切分 decode sequence。

wrapper 的 node dispatch 也必须使用该条件，而不能只依赖 `batch_forward_type.is_decode()`：当 `use_expanded_decode_for_spec_verify_attention=true` 时，target verify 必须选择 DCP decode node，并以 decode 语义绑定 sequence metadata；普通 chunked prefill 仍选择 prefill node。该分流使用 `enableDecodeDcpLayerOwner && (is_decode() || use_expanded_decode_for_spec_verify_attention)`，不检查 MTP model 类型。

ATB 参数需要包含：

```text
enableDecodeDcpLayerOwner
dcpRank
dcpSize
dcpOwnerRank
dcpCommDomain
blockSize
```

`dcpRank`、`dcpOwnerRank`、`dcpSize`、`skipTopk` 和 `outputTopk` 都是 layer 初始化后不变的建图常量。每个 rank 对每个 target decoder layer 只构造其本地角色对应的 decode DCP 图：owner 图包含 cache write、Indexer、slot mapping 和 Gather；non-owner 图不包含这些节点，只包含同顺序 Broadcast 和本地 SFA。图缓存 key 必须纳入 DCP enable、local role、root、group size、top-k sharing 属性和 decode attention execution mode。不得将同一图通过运行时 `is_owner` 条件复用，也不得将 prefill 图或 MTP draft 图替换为 DCP 图。

### 9.2 持久 tensor

NPU layer wrapper 管理以下持久 tensor，并将其作为 ATB 输入：

```text
dcp_selected_cache_recv_buffer
dcp_topk_recv_buffer
logical_block_lut
block_offset_lut
identity_topk
query_batch_ids
valid_topk_count
packed_gather_indices
actual_seq_lengths_query
actual_seq_lengths_key
```

这些 tensor 的地址和 shape bucket 需要兼容 Graph Mode。快速验证可以先使用 eager ATB；Graph Mode 适配作为单独里程碑。

第一阶段的验收范围仅包含 eager ATB，但必须构造完整的 owner cache/indexer、selected cache pack、DCP Broadcast 和本地 TND SFA 数据流，不允许通过全量 cache 副本增加 fallback 或旁路。第二阶段复用同一组 ATB 节点和 tensor contract，仅补充 `Tq`、`sum(K_i)` 等动态维度的 shape bucket、持久 buffer 容量以及 ACL Graph capture/update 机制。

### 9.3 代码边界

计划修改：

- `xllm/core/layers/npu/npu_deepseek_v32_decoder_layer_impl.cpp`
  - 实验开关、decode DCP mapping、持久 buffer 和 metadata；
- `third_party/xllm_atb_layers/models/deepseekv2/layer/decoder_layer.cpp`
  - 将 DCP tensor 传入 SparseAttention；
- `third_party/xllm_atb_layers/models/deepseekv2/operation/sparse_latent_attention.cpp`
  - 拆分 common Q、owner cache/indexer、Broadcast 和本地 TND SFA。

本阶段不修改：

- KVCache/block manager；
- scheduler/worker 的 cache ownership；
- prefill cache 路由；
- PD/prefix/cache transfer。

### 9.4 真实 sharding 的 ownership 契约

后续真实 sharding 引入统一的 `LayerCachePlacement`，作为 layer ownership 的唯一来源。该对象由 attention TP/DCP mapping 和当前 rank 构造，至少提供：

```text
owner_local_rank(layer_id)
owns_layer(layer_id)
```

各模块的责任边界如下：

- Worker/KVCache allocator：`kv_caches` 仍保持 `n_layers` 个槽位，并对每层调用与模型 cache 类型匹配的创建流程。owned layer 分配正常 block 容量；non-owned layer 的 MLA、RoPE 和 Indexer cache tensor 只分配一个 null block，保持 `kv_caches[layer_id]`、dtype、layout 和 tensor role 接口不变；
- prefill/decode model layer：消费同一个 `LayerCachePlacement`，只允许 owner 写入本层语义 cache；未来融合 preprocess 的 non-owner null-block 写入不属于语义 cache，且不得被读取；decode 读路径继续使用本文定义的 owner-DCP 图；
- PD、prefix 和 hierarchy KV cache transfer：只注册、发送和接收本 rank owned layers，并允许 non-owned layer 的注册集合为空；
- block manager/scheduler：继续维护 sequence 级逻辑 block、block table 和 slot，不感知 layer owner，也不为不同 layer 分配不同逻辑 block id。

不得在 allocator、model 和 transfer 中分别重新实现 `layer_id % dcp_size`。初始化时应逐层校验每个 DCP group 恰好有一个 owner；transfer 还必须根据源端和目标端各自的 `LayerCachePlacement` 建立路由。由于 non-owned layer 的 tensor 仍然已定义，cache capacity 估算、Graph executor 地址收集、prefix cache 和 transfer 都禁止通过 `defined()`、`numel()`、shape 或 `kv_caches.front()` 推断 ownership 或有效 block 容量，必须显式使用 `LayerCachePlacement` 并选择 owned layer 的完整 cache。

## 10. 数据流等价性推导

### 10.1 证明范围和结论

本节证明的是 layerwise owner-DCP 与当前无 DCP decode 的算法语义等价。结论分为两层：

- 冻结某层输入、cache 和 metadata 时，owner 的 cache 状态、LightningIndexer logical top-k、logical-to-physical slot 映射以及 Gather 得到的 selected cache 必须与 baseline 精确一致；Broadcast 只复制这些值，不改变内容。
- baseline 使用 `PA_BSND` SFA，DCP 使用 packed `TND` SFA。两者在实数数学上计算相同 attention，但 BF16 kernel 的归约顺序可能不同，因此 attention、下游当前 token cache 和最终 logits 只保证数值容差内等价，不承诺 bitwise 一致。

当前仓库中尚不存在 `ATTN_DECODE_DCP`、`LayerCachePlacement` 或 owner-DCP ATB 节点实现，因此这里确认的是设计和待实现数据流的正确性，不能替代实现后的端到端验证。

### 10.2 baseline 单层 decode 数据流

对物理 decoder layer `l`、attention TP rank `r` 和 query row `i`，定义：

- `X_l`：进入本层的 hidden states；同一 attention TP group 内 token 顺序和输入相同；
- `Q_l^r`：rank `r` 的本地 TP Q-head shard；
- `C_l`：MLA latent cache，baseline 同时将它作为 SFA 的 K 和 V；
- `R_l`：RoPE K cache；
- `I_l`：Indexer K cache；
- `P_l[i,j]`：LightningIndexer 为 query `i` 返回的第 `j` 个 logical position。

当前 GLM-5 baseline 的顺序为：

```text
X_l
  -> input norm
  -> fused replicated Q-A/KV-A projection
       -> latent_q -> TP-sharded Q-B -> Q_l^r
       -> current latent KV ---------> ReshapeAndCache(C_l)
       -> current RoPE K ------------> ReshapeAndCache(R_l)
  -> replicated Indexer Q/K/weight projections
       -> ReshapeAndCache(I_l)
  -> LightningIndexer(I_l, block_table, q/kv lengths) -> P_l
  -> PA_BSND SparseFlashAttention(Q_l^r, C_l, R_l, P_l)
  -> local V reprojection
  -> TP row-parallel O projection
```

代码事实支持 cache 复制前提：

- `deepseek_v32_decoder_loader.cpp::merge_host_at_weights` 将完整的 `kv_a_proj_with_mqa` 和 `q_a_proj` 权重拼成融合 Q-A/KV-A 权重；
- `deepseek_decoder_loader_constants.h::WEIGHT_SHARD_W8A8` 只切分 Q-B、KV-B、O projection 等 TP-head 相关权重，Q-A、KV-A、KV norm 和 Indexer Q/K/weight 不在 shard 表中；
- `sparse_latent_attention.cpp::Preprocess` 明确保持 MLA/RoPE cache write、Indexer cache write、LightningIndexer、SFA 的先后顺序；
- GLM-5 当前设置 `enableQkvdownDp=false`，不存在把本设计中的 cache-producing A projection 分散到其他 DP rank 的路径。

因此在同一 attention TP group 内，`Q_l^r` 按 rank 分片，但写入 `C_l`、`R_l`、`I_l` 的候选值是复制且逐元素相同的。DCP owner 可以保存其中一份，不能只保存某个 TP Q-head shard。

### 10.3 owner-DCP 单层数据流

attention TP group 可以包含多个 DCP subgroup。每个 subgroup `G` 对每层各有一个 owner `o(l,G)`，但不同 subgroup 会各自执行同样的 owner 流程：

```text
all ranks in G:
  X_l -> replicated fused A projection -> local TP Q-B -> Q_l^r

owner only:
  write C_l/R_l/I_l
  -> original PA_BSND LightningIndexer -> P_l
  -> map P_l logical positions to physical slots
  -> Gather C_l/R_l in original top-k order
  -> compact pack selected_cache_l

collective:
  Broadcast(P_l) only when outputTopk
  Broadcast(selected_cache_l) on every DSA layer

all ranks in G:
  split selected_cache_l -> selected latent / selected RoPE
  -> packed TND SparseFlashAttention with local Q_l^r
  -> unchanged local V reprojection and TP row-parallel O projection
```

不同 DCP subgroup 不需要互相通信，因为整个 attention TP group 的 replicated cache、Indexer 输入和 logical top-k 相同。每个 subgroup 内广播同一 selected cache，最终覆盖 TP group 内所有本地 Q-head shards。

### 10.4 cache 状态归纳

对 layer `l` 的 owner cache 做 decode step 归纳。该归纳证明相同输入或实数数学语义下的结构等价；有限精度下前序 layer 的 SFA 误差传播按 10.9 节处理：

1. 基础状态：快速验证阶段 prefill 在所有 rank 写全量 cache，因此 owner 的 `C_l/R_l/I_l` 与 baseline 相同。未来真实 sharding 时，必须由 prefill ownership 路径保证 owner 获得同一完整初始 cache。
2. 归纳假设：step `t` 开始前，owner 的历史 cache 在所有可见 logical positions 上与 baseline 相同。
3. 当前写入：owner 使用与 baseline 相同的 `X_l`、replicated projection 权重、RoPE position 和 `new_cache_slots`，因此把相同的 MLA、RoPE 和 Indexer K 写入相同 physical slots。
4. 归纳结论：step `t` 写入后，owner cache 在 baseline 本轮可见的全部位置上仍相同；同一 layer 的 owner 不随 decode step 改变，因此下一步继续拥有完整历史。

MTP verify 一次写入多个 query token 时，owner 与 baseline 一样先写入本轮所有 token。cache 中存在后续 speculative token 不代表当前 query 可见；可见性仍由传给原始 LightningIndexer 的逐 query KV length 限制。

### 10.5 Indexer 和 logical-to-physical 映射等价

冻结本层输入后，owner LightningIndexer 的六个输入都与 baseline 相同：Indexer Q、Indexer K cache、Indexer weight、query lengths、key lengths 和 block table。因此：

```text
P_dcp[i, :] = P_baseline[i, :]
```

Indexer 多个 score head 只参与内部打分，最终每个 query 仍只有一组 logical top-k。DCP 不进行第二次选择、排序、去重或强制插入当前 token。

对 `P_l[i,j]=p`，DCP 使用与 paged cache 相同的寻址：

```text
logical_block = p / block_size
block_offset = p % block_size
physical_block = block_table[block_table_row[i], logical_block]
physical_slot = physical_block * block_size + block_offset
```

由于 baseline PA_BSND SFA 使用同一个 block table 解释同一个 logical position，DCP Gather 得到：

```text
selected_latent[i,j] = C_l[P_l[i,j]]
selected_rope[i,j] = R_l[P_l[i,j]]
```

这里右侧表示 logical cache 值，不表示直接以 logical position 索引物理 tensor。DCP pack 必须保持 query 顺序和 `j` 的原始 top-k rank 顺序。Concat、Broadcast 和 Split 只是表示变换，不改变元素值或顺序。

### 10.6 attention 数学等价

对 TP rank `r`、query `i`、本地 head `h` 和 selected position `j`，baseline 与 DCP 都计算：

```text
score[r,i,h,j] = scale * (
    dot(q_nope[r,i,h], selected_latent[i,j]) +
    dot(q_rope[r,i,h], selected_rope[i,j]))

output[r,i,h] = sum_j(
    softmax(score[r,i,h,:])[j] * selected_latent[i,j])
```

baseline 通过 `P_l + block_table` 在 PA cache 中间接取得 selected values；DCP 已把同样的 values 按 `j` 打包，并用 identity sparse indices `[0,K_i)` 访问。DCP 同时设置 `actual_seq_lengths_query=[1,2,...,Tq]` 和 `actual_seq_lengths_key=cumsum(K_i)`，使每个 query 只绑定自己的 selected-cache segment。两条路径的 Q、selected K/V、RoPE K、scale、causal 有效集合和排列顺序相同，因此实数数学结果相同。

DCP 不让 owner 独自执行 attention。每个 rank 仍使用自己的 `Q_l^r`、本地 V reprojection 和 O-projection shard，最后参加原 TP row-parallel collective，所以 TP 输出语义保持不变。

### 10.7 MTP causal、不重复和不遗漏

对 sequence `s` 的第 `j` 个 verify query：

```text
visible_len = kv_len_s - q_len_s + j + 1
K_i = min(index_topk, visible_len)
```

owner 调用原始 LightningIndexer 时使用该 query 的可见长度，因此 causal 过滤发生在 top-k 选择内部，而不是 DCP Gather 后。DCP 只保留该 query 的前 `K_i` 个有效结果，并按 query 分段 pack：

```text
segment_i = [prefix_sum(K)[i], prefix_sum(K)[i + 1])
```

由此得到：

- 当前 token 仅在原始 LightningIndexer 选中它时出现，DCP 不额外 append/prepend；
- 单个 query 内的 selected logical positions 与 baseline 完全相同，因此不新增重复，也不遗漏 baseline 候选；
- 不同 MTP query 的 causal 前缀天然重叠，同一历史 position 出现在不同 query segment 中是预期复制，不属于重复错误；
- sequence-scoped 和 expanded verify 必须使用 7.2 节定义的对应 block-table row，否则 causal 长度正确也可能 Gather 错 sequence。

### 10.8 top-k sharing 等价

`outputTopk` producer 将 owner 的 logical top-k 广播给 subgroup 内所有 rank，owner 自己也消费 Broadcast 输出。`skipTopk` consumer 直接使用该 tensor，在自己 layer 的完整 owner cache 上执行相同 logical-to-physical 映射。producer 和 consumer owner 可以不同，因为共享的是 sequence logical position，而不是 producer layer 的 physical slot。

### 10.9 数值等价边界

以下内容可以要求精确一致：

- 同一冻结输入下各 baseline TP rank 的 replicated MLA/RoPE/Indexer cache candidate；
- owner 与 baseline 的 logical top-k；
- logical-to-physical slot；
- owner Gather 后、Broadcast 前后的 selected cache bit pattern；
- subgroup 内所有 rank 的 Broadcast 输出。

以下内容只能要求数值容差：

- `PA_BSND` SFA 与 packed `TND` SFA 输出；
- DCP attention 误差传播后，下游 layer 为当前 token 新生成的 cache；
- 下游 Indexer score、attention 输出和最终 logits。

因此“selected cache 逐元素完全一致”只适用于冻结本层输入/cache 的单层对比。端到端连续执行时，前一层 SFA 的 BF16 差异会进入下一层 projection；即使算法相同，下游 selected cache value 也可能出现数值差异。logical top-k 通常仍应一致，但当第 `K`/`K+1` 个 score 接近 tie 时，小数值扰动可能改变等价候选的排名。遇到这种情况必须定位首个差异 layer，并结合 top-k 边界 score gap 判断，不能直接归因于 slot mapping 或 Broadcast 错误。

### 10.10 正确性不变量

1. 每个 layer、每个 DCP group 恰好有一个 owner。
2. 同一 DCP group 的所有 rank 具有相同 query row 顺序、sequence metadata 和 block-table 语义。
3. decode 中只有 owner 读写该层的语义 cache；未来融合 preprocess 允许 non-owner 重复写本层 null block，但其内容永不读取。
4. owner cache 在每个 decode step 开始前包含与 baseline 相同的全部历史 logical positions。
5. owner 使用与无 DCP 相同的输入和逻辑 cache 顺序执行原始 LightningIndexer。
6. top-k 在映射前保持 sequence logical position。
7. `visible_len` 和 `block_table_row` 必须由同一 query expansion 生成。
8. packed TND 必须将每个 query row 视为独立的长度 1 sequence，query/key cumulative lengths 分别为 `[1..Tq]` 和 `cumsum(K_i)`。
9. selected cache 的排列顺序与 owner top-k 顺序一致。
10. 每个 MTP query 的 selected cache 只来自其 causal 前缀；跨 query 的重叠位置允许重复打包。
11. non-owner 的 attention 输出只依赖本地 Q 和 Broadcast selected cache。
12. top-k sharing 的对外 tensor 在所有 DCP rank 上一致。
13. 同一 DCP group 内 collective 调用序列、shape、count、root 和 communicator 完全一致。
14. owner 轮转不改变算法语义；PA_BSND 到 TND 的 kernel 变化按数值容差验收。

## 11. 验证计划

### 11.1 第一阶段主验收：端到端

第一轮只在 PD 混布实例以 eager ATB 跑端到端对比：先完成相同 prompt 的原图 baseline，再开启 `decode_dcp_size=2` 跑 owner-DCP decode。首轮使用 `visible_len >= index_topk` 的长上下文普通 decode，避免短上下文尾部编码成为首个变量。

必需检查：

- prefill 保持原路径且 decode 无 collective hang；
- 最终 logits 在预设数值容差内；
- greedy token 序列完全一致；
- profiler 显示每层存在预期的 selected-cache Broadcast。

首轮不默认 dump 每层 tensor，也不要求先做单算子对比。端到端不一致、hang 或 profiler 发现依赖异常时，才进入以下诊断项。

### 11.2 可选诊断：算子级验证

- DCP Broadcast 支持按层变化的 root；
- INT32 top-k Broadcast；
- 逻辑位置到物理 slot 的组合图与 CPU 公式逐元素一致；
- paged Gather 与 CPU reference 一致；
- 同一 Q/top-k/cache 下，PA_BSND SFA 与 packed TND SFA 在允许误差内一致。
- 对每个 `K=1..index_topk`，验证 `[0,K)` identity sparse-index 前缀、`cumsum(K_i)` 和候选无效尾部编码下，packed TND SFA 与 PA_BSND SFA 一致；该项用于定位短上下文问题，不默认阻塞首轮 E2E。

### 11.3 可选诊断：单层和 tensor dump

- 构造不同 root 的相同 SparseAttention 输入；
- 冻结相同 layer input、cache 和 metadata，比较 baseline 和 DCP owner top-k，要求逐元素相等；
- 在同一冻结输入下比较 physical slot 和 selected cache，要求逐元素相等；
- 比较每个 TP rank 的 attention 输出；
- 检查 non-owner 图中不存在 cache read/write 和 LightningIndexer。

仅在端到端失败或需要定位时，开启逐层 dump，比较首个发生差异 layer 的 top-k、selected cache 和 attention 输出。若该 layer 的输入尚未发生差异，则 top-k、slot 和 selected cache 必须严格相等；若输入已受前序 TND SFA 数值差异影响，则 cache value 和 Indexer score 使用数值容差，并额外记录第 `K`/`K+1` 个 score gap。attention 输出使用与端到端相同的数值容差。

### 11.4 扩展端到端覆盖

覆盖：

- DCP size 2 和 4；
- 单请求和多请求 batch；
- context 小于、等于和大于 `index_topk`；
- block 边界前后；
- 普通 decode；
- MTP 多 query；
- top-k sharing producer/consumer owner 不同；
- 连续多步 decode。
- DP/EP 空 shard 的 fake-input decode，确认所有 DCP collective 使用非零一致 shape 且无 hang。

对比标准：

- 冻结单层输入/cache 的诊断中，owner top-k、physical slot 和 selected cache 与 baseline 逐元素完全一致；
- 连续端到端中，owner logical top-k 预期一致；若首次 mismatch 发生在输入已产生 BF16 差异且 top-k 边界接近 tie 的 layer，按 score gap 和最终输出容差分析，不将其直接判定为 DCP 寻址错误；
- 连续端到端的 selected cache value 允许继承前序 layer 的数值误差，但 pack 的 logical position、query segment 和元素顺序必须正确；
- 所有 rank Broadcast 输出完全一致；
- attention/final logits 不要求 bitwise 一致，但必须满足测试前固定的绝对误差和相对误差阈值；
- greedy token 序列完全一致；
- 无 collective hang。

该标准将离散语义和浮点实现差异分开：在相同输入下，top-k、cache 寻址、pack 顺序及 Broadcast 内容属于 DCP 正确性，必须严格一致；PA_BSND 到 packed TND SFA 的实现变化可能改变 BF16 浮点归约顺序并传播到后续 layer，因此连续端到端的 cache value、attention 和 logits 使用数值容差，但不允许改变 greedy token 结果。

### 11.5 非 owner 隔离验证

prefill 完成后，在测试模式下将每个 rank 的非 owned layer cache 填入 NaN 或固定毒值，再执行 decode。结果必须与未污染时一致。该测试用于证明 non-owner decode 图没有通过全量物理副本形成隐藏依赖。

Profiler 还应证明：

- 每层只有 owner 执行 LightningIndexer；
- 每层只有 owner 执行 decode ReshapeAndCache；
- 每层只有 owner 执行 paged cache Gather；
- 每个 DSA layer 只有一次 BF16 selected-cache Broadcast；
- `outputTopk` layer 额外一次 INT32 Broadcast。

## 12. 实施顺序

1. 锁定 DCP rank topology 和 owner 公式；
2. 重构 SparseAttention，拆出 common/owner/non-owner 路径；
3. 实现 top-k 逻辑位置到 physical slot 的组合图；
4. 实现 selected cache Pack/Broadcast/TND SFA；
5. 接入 top-k sharing；
6. 先进行长上下文普通 decode 的 eager 端到端对比；
7. 遇到问题时按需进行单算子对比或逐层 tensor dump；
8. 完成短序列、通用多 query compact pack 和 non-owner cache poisoning 验证；
9. 单独设计 Graph Mode bucket 和持久 buffer；
10. 正确性稳定后再修改真实 cache allocation、prefill 和 transfer。

## 13. 风险

- 当前 `ATTN_CP` mapping 可能与目标 DCP 子组拓扑不一致；
- LightningIndexer speculative causal 契约不明确；
- Broadcast 是否允许 owner/non-owner 使用不同 producer tensor，需要先做最小实验；
- packed TND SFA 与 PA_BSND SFA 可能存在非 bitwise 的数值差异；
- 若 packed TND 沿用原请求 `q_len` 而不是把每个 query row 展开为长度 1 sequence，会让多个 MTP query 错误共享 selected-cache segment；
- 短序列的 compact packing 会引入动态通信长度和 Graph Mode 分桶问题；
- selected cache Broadcast 通信量约为 `K * 576 * dtype_size`，可能成为性能瓶颈；
- owner 每层轮转可以平衡存储和总计算量，但不能隐藏单层 Broadcast 延迟；
- top-k tie-breaking 必须继承原始 LightningIndexer 结果，不能在 DCP 图内重新排序。

第一阶段不实现 `Q AllGather + owner SFA` 备选路径，也不设置运行时自动切换阈值。仅记录每层 selected-cache Broadcast 的 payload、耗时及其与计算的重叠情况。满 `index_topk=2048` 时，每个 query 的 BF16 payload 为 `2048 * 576 * 2 = 2.25 MiB`；是否需要备选路径必须在正确性验证和真实 cache sharding 后基于端到端 profiling 决定。

## 14. 决策记录

以下决策已按依赖顺序完成确认：

1. 已确认：DCP 是 TP group 内的连续子组。
2. 已确认：`dcp_size` 表示每个 DCP group 的 rank 数，并要求 `tp_size % dcp_size == 0`。
3. 已确认：owner 严格使用物理 decoder `layer_id % dcp_size`，不随层类型和 sharing pattern 改变。
4. 已确认：第一版端到端支持 MTP；DCP 只处理 target model 的通用多 query verify，单层 MTP draft model 保持原图和全量 cache。
5. 已确认：不增加 LightningIndexer 多 query causal 的独立 reference 测试，以现有无 DCP 路径作为语义基线。
6. 已确认：短序列必须进入 owner-DCP 路径，不允许回退到全量 cache baseline。
7. 已确认：worker 不新增 `K_i` 输入；wrapper 从已有逐 query 可见 KV 长度派生 `K_i=min(2048, visible_len)`。
8. 已确认并澄清：冻结相同 layer input/cache 时，top-k、physical slot 和 selected cache 逐元素完全一致；连续端到端中，PA_BSND/TND 的 BF16 差异可能传播到下游 cache 和近 tie 的 top-k，按首个差异 layer、score gap 和预设数值容差分析；greedy token 序列必须完全一致。
9. 已确认：第一阶段只验收 eager ATB；Graph Mode 是第二里程碑，但复用相同组图语义和 tensor contract。
10. 已确认：第一阶段不实现 Q AllGather 备选路径和自动切换阈值，只采集 Broadcast 性能数据，真实 sharding 后再决策。
11. 已确认：统一 `LayerCachePlacement` 是 ownership 唯一来源；allocator、prefill/decode 和 transfer 共同消费，block manager/scheduler 不感知 layer owner。
12. 已确认：第一版只在 PD 混布实例验证 decode；不做真实 cache sharding，不考虑 PD 拆分部署、跨 topology 路由和 cache transfer。
13. 已确认：owner-DCP 只在普通 `DECODE` 或 expanded decode/spec verify attention 激活；`PREFILL`、普通 `CHUNKED_PREFILL` 和 `MIXED` 保持原图。
14. 已确认并修正：Indexer 的多个 head 只参与 score，最终每个 query 只有一组 top-k；`outputTopk` 逻辑 shape 为 `[Tq, index_topk]`，当前 ATB ABI 可保留 `[Tq, 1, index_topk]` singleton view。所有 rank 分配输出，不使用 `index_n_heads`，也不从 index cache 推导 shape。
15. 已确认：owner/non-owner 保持统一 cache 输入接口；快速验证时 non-owner 绑定本层全量 cache，真实 sharding 后绑定本层一-block null `KVCache`。除未来融合 preprocess 可写 null slot 外，图中不得存在任何消费或写回该绑定的节点。
16. 已确认：当前 token 保持现有“cache write -> Indexer -> SFA”语义；DCP 不增加当前 token 的额外处理。
17. 已确认：target expanded verify 选择 DCP decode node；dispatch 使用 decode attention execution mode，不只检查 `is_decode()`。
18. 已确认：短上下文 TND SFA 的 identity sparse-index 尾部编码通过按需的 NPU 算子级测试确定；首轮先跑长上下文端到端，短上下文通过前必须完成该验证。
19. 已确认：`decode_dcp_size` 是独立实验配置；DCP group 按 attention TP communicator 的 rank 顺序连续切分，不复用 `cp_size` 或 `ATTN_CP`；值为 `1` 时关闭 DCP 并使用原 decode 图。
20. 已确认：selected cache 只执行一次 BF16 Broadcast；owner 按 query 顺序紧凑打包 active view `[sum(K_i), kv_lora_rank + qk_rope_head_dim]`，持久 buffer 仅提供容量，尾部不参与 SFA。
21. 已确认：`outputTopk` 层依次 Broadcast INT32 top-k 和 selected cache，top-k Broadcast 写对外 `out_topk_indices`；非 sharing 层采用相同顺序，但 top-k Broadcast 只写本层内部 `dcp_topk`；`skipTopk` 层复用上游已广播的 top-k，因此只 Broadcast selected cache。owner 的 slot mapping 始终使用对应的 Broadcast 输出。
22. 已确认：owner/non-owner 是 layer 初始化时的静态建图角色；图缓存不得通过运行时 owner 条件复用两种依赖图。
23. 已确认：decode DCP 新增框架一等 mapping `ATTN_DECODE_DCP`；由 `MappingNPU` 构造连续子组并由 `ExternalCommManager` 提供 communicator，layer 不自行创建 HCCL group。
24. 已确认：首轮验收先跑 eager 长上下文端到端；单算子对比和逐层 tensor dump 是失败后的可选诊断项，不作为首轮前置门槛。
25. 已确认：第一版能力必须支持 MTP，DCP 图不感知 MTP；target verify 按通用多-query decode 进入同一 owner-DCP 图，单层 `_mtp` draft model 保持原图和全量 cache。验收顺序上，首轮 E2E 先使用 PD 混布、eager、`decode_dcp_size=2`、batch=1、`visible_len >= index_topk` 的普通 target decode 建立基线，再在同一版本补充 MTP、短上下文、多 batch 和 `dcp_size=4` 覆盖。
26. 已确认：top-k 在 ATB 图和 Broadcast 中保持现有 `[Tq, 1, index_topk]` 物理 ABI；仅在 owner 的 slot mapping/Gather 前 view 为 `[Tq * index_topk]`，不引入 `index_n_heads`。
27. 已确认：DCP 兼容现有 empty-shard fake-input 协议；logical query count 可以为 0，但 ATB `execution_Tq` 至少为 1，所有 DCP rank 仍以一致的非零 tensor shape 执行 collective，不新增 worker 侧 DCP 特判。dummy row 使用 `K_exec=1`，目的是复用短上下文路径和简化合法寻址，不作为性能优化。
28. 已确认：selected cache 严格保持 LightningIndexer top-k 原始 rank order；第一版禁止排序、去重或 Gather 重排，不引入 sparse-index 反向映射。
29. 已确认：所有 rank 保留现有融合 QKV down projection 和 Split；non-owner 只消费 latent Q，KV/RoPE K 分支终止，不执行后续 cache/indexer 节点。第一版不拆融合权重或新增 Q-only projection。
30. 已确认：GLM-5 的 `glm_moe_dsa` 和 `glm_moe_dsa_mtp` 都命中现有 model-type 特判，`enableMlaPreprocess=false`；DCP 无需处理 `MlaPreprocessV2` 内部 cache write。
31. 已确认并修正：真实 sharding 初始化时仍为每个 non-owned layer 创建现有类型的 `KVCache` tensor，但仅分配一个 null block；owner layer 分配正常容量。未来 `MlaPreprocessV2` 直接写当前层 null cache，不再引入共享 dedicated scratch、独立 allocator 或 cache-signature 管理。
32. 已确认：所有 non-owner execution row 都使用 `new_cache_slot=0`，允许在一次 `MlaPreprocessV2` 中重复写同一 null slot；null slot 的最终内容未定义且永不读取，不为 execution rows 分配唯一 slot 或扩容 scratch。
33. 已确认并修正：每个 non-owned layer 的一-block null cache 随 layer cache 初始化并持久存在，executor 生命周期内地址稳定；每次 forward 不重新分配、不清零，以兼容后续 ATB Graph capture。
34. 已确认：non-owned layer 继续通过现有模型 `KVCache` factory 创建 MLA、RoPE 和 Indexer 三类 tensor，每类都分配一个 block；即使 Indexer null block 不参与计算，也不引入 optional tensor 分支或专用 null-cache 类型。
35. 已确认：worker 继续提供真实 `new_cache_slots` 且不感知 DCP ownership；未来 `MlaPreprocessV2` non-owner 路径由 NPU wrapper 将 slots 替换为持久 INT32 全零 buffer 的 `execution_Tq` active view，owner 仍使用真实 slots。
36. 已确认：zero-slot buffer 容量复用现有 `max_tokens_per_batch`，不新增 `decode_dcp_max_tq` 一类配置；MTP expanded verify 同样受现有 batch token 上限约束，Graph Mode 只选取对应 bucket 的 active view。
37. 等价性推导确认：packed TND SFA 必须把每个 query row 展开为独立的长度 1 sequence，使用 `actual_seq_lengths_query=[1..Tq]` 和 `actual_seq_lengths_key=cumsum(K_i)`；不得沿用原请求的 MTP `q_len` 分组。
38. 等价性推导确认：冻结同一 layer 输入时，Indexer、slot mapping、Gather 和 Broadcast 必须精确一致；PA_BSND/TND 的 BF16 kernel 差异及其下游传播按数值容差验收，不承诺整网 bitwise 一致。

## 15. 实现验证项

以下项目需要通过最小实验或端到端运行确认，但不改变已确认的组图语义：

1. 验证 ATB Broadcast 是否允许 root 和 non-root 绑定不同 producer tensor，并确认 owner/non-owner 的输入、输出 buffer 是否需要显式 Copy；无论具体 API 约束如何，Broadcast 输出仍是所有 rank 唯一的后续输入。
2. 在短上下文覆盖中确认 TND SparseFlashAttention 固定 `index_topk` 维度的无效 sparse-index 尾部编码；不允许因尾部编码问题回退到全量 cache 路径。
3. 未来扩展 `MlaPreprocessV2` 时，验证算子允许多个 execution row 重复写 `new_cache_slot=0`，且该重复写不影响 Q 输出；该项不是第一版 GLM-5 的前置条件。
4. 用冻结输入验证 packed TND SFA 在 `actual_seq_lengths_query=[1..Tq]`、`actual_seq_lengths_key=cumsum(K_i)` 时，每个 MTP query 只访问自己的 selected-cache segment，并确认 identity sparse indices 是 segment-local 编号。
5. 若端到端出现首层 top-k 差异，dump 同一 attention TP group 各 rank 的 fused A projection、Indexer Q/K/weight 和 baseline top-k，验证 replicated-cache 前提；正常首轮 E2E 不要求预先 dump。

## 16. 两种 DCP 方案对比

| 维度 | 两级 Indexer + sequence-sharded cache | Layerwise owner + layer-sharded cache |
|---|---|---|
| cache 分布 | 每层 cache 沿 context 分散到 DCP ranks | 每层完整 cache 只在一个 owner，层间轮转 owner |
| Indexer | 每个 rank 对本地 shard 取候选，AllGather 后再次全局选择 | 只有 owner 对完整本层 cache 执行一次原始 Indexer |
| 通信数据 | 每 rank 的候选 cache/index，需要汇总 `dcp_size * K` 候选 | owner 只 Broadcast 最终 `K` 个 selected cache；sharing producer 额外广播 top-k |
| 等价性条件 | 本地候选数至少为全局 `K`，还需严格处理 logical position、causal、当前 token 唯一归属和稳定 tie-breaking | Indexer 输入与无 DCP baseline 相同，直接继承原始 top-k 顺序和 causal 语义 |
| TP attention | 所有 rank 仍用本地 Q shard执行 SFA | 所有 rank 仍用本地 Q shard执行 SFA |
| MTP | target verify 需要逐 query 合并候选并再次选择 | target verify 作为通用多 query decode；单层 MTP draft model 不进入 DCP |
| 第一版算子依赖 | 现有 local LightningIndexer、AllGather、第二次 LightningIndexer；通常还需要候选重排/去重 metadata，后续可能融合 | 现有 LightningIndexer、Gather/Concat、Broadcast、TND SFA；无需新增算子 |
| 实现风险 | 高：sequence shard、当前 token 放置、两级 top-k 等价性和通信 shape 同时变化 | 中：主要风险是 owner 图分支、physical slot 映射和 Broadcast contract |
| 与快速验证的关系 | 需要先定义真实 sequence shard，无法在全量副本上忠实验证数据分布 | 即使外部仍有全量 cache，也能通过 owner-only 图和 poison 测试忠实验证最终 decode 数据依赖 |

因此第一版选择 layerwise owner。两级 Indexer 不作为 fallback 同时保留，避免出现两套难以对齐的 causal 和 top-k 语义。

## 17. 框架真实 sharding 实现进展（2026-08-10）

当前本地开发基线为 `codex/feat/deepseek-v32-decode-dcp`，基线提交
`6c530ec1`。框架真实 sharding 与 PD PUSH 适配正在该工作区以未提交改动
继续开发。

### 17.1 已实现

- 新增 `enable_decode_dcp_layerwise_kv_cache` 开关，默认关闭，并接入命令行、
  JSON 配置和 `ParallelArgs`。
- 新增统一的 `DecodeDcpLayerPlacement`，worker 分配与 Prefill PUSH 过滤共同
  使用 `owner = layer_id % decode_dcp_size`。
- target decode owner 层按公共逻辑 block 数分配完整 KV；non-owner 层仍创建
  相同 `KVCache` 接口，但 K/V、Indexer 和 scale tensor 的 block 维均缩为 1。
- 容量估算按每个 DCP rank 的 owner 层成本和所有 non-owner null block 成本
  计算，并取所有 DCP rank 可支持的最小公共 block 数，block manager 无需感知
  layer owner。
- target worker 的 block swap 跳过 non-owner 层。draft worker 不进入 layerwise
  ownership，继续保留单层全量 cache。
- beam search copy-on-write 的 Torch fallback 和 NPU fused BlockCopy 均读取
  `KVCache` 上持久化的 layer ownership；non-owner null cache 不执行 block
  copy，owner cache 保持原有复制语义。
- PD PUSH 从 decode response 获得 layerwise 开关和 `decode_dcp_size`，Prefill
  根据每个目标 decode TP rank 对应的 DCP local rank 逐层过滤；draft cache
  transfer 明确绕过过滤。
- LlmDataDist 使用逐层直接写已注册远端 cache 的协议，没有 receiver 侧层计数；
  因此 layer 过滤只改变 sender 的目标 key 集合，不引入额外完成消息或等待条件。

### 17.2 本轮发现并修复的原始问题

`SpeculativeEngine::calculate_kv_cache` 原先在 target/draft 共用设备时，仍按
target 的“所有层全量 KV”重新计算共同 block 数。该逻辑会覆盖 target
layerwise estimator 的结果，使 MTP 下显存收益丢失，也没有表达“target 按
owner/null sharding、draft 单层全量 cache”的真实分配。

当前已将 layerwise 容量公式抽成共享 helper。MTP 的公共 block 数现在对每个
DCP rank 使用以下成本计算并取最小值：

`non_owner_null_bytes + N * (owned_target_bytes + full_draft_bytes)`。

因此 DCP 仍不感知 MTP 的执行语义；唯一额外处理发生在显存容量估算，draft
cache 本身不做 layerwise sharding。

### 17.3 已补测试

- layer 数不能被 DCP size 整除时，以最大 owner footprint 决定公共 block 数；
- MTP full draft cache 参与 layerwise 公共容量计算；
- owner 层 tensor 使用完整 block 数，non-owner 层 tensor 只有 1 个 block；
- 配置开关的 JSON/命令行注册；
- PUSH 按目标 owner 过滤，配置关闭保持全量传输，draft cache 始终全量传输。

### 17.4 待验证

1. 已在远端 `/export/home/shifengmin.3/workspace/xllm-dcp-e2e` 完成
   `python setup.py build --device npu` 全量构建，主二进制、Python export 和
   `all_tests` 均通过。
2. 聚焦单元测试当前在通用 NPU test bootstrap 导入 `torch_npu` 前中止：测试
   二进制链接的 `/usr/local/libtorch_npu` 与 Python wheel 自带的 `torch_npu`
   动态库不匹配。该问题与本功能断言无关，且截至上游 main 的
   `736f998` 尚未统一两者的库来源；待使用匹配的 NPU runtime 环境后重跑。
3. 单机整机 8 卡/16 Die 运行 GLM-5 target decode，比较 DCP 关闭基线与
   layerwise 开启后的生成、block 数、每层实际 tensor shape 和 HBM 占用。
4. 开启 MTP 验证 target layerwise + draft 单层全量 cache。
5. PD 分离 PUSH 验证每个 decode rank 只接收 owner 层 target KV，draft KV
   全量传输，并确认无越界、等待或死锁。
