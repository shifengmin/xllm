# PD 路由验证方案：GLM 5.3 flash 场景（单测与 mock 设计）

## 溯源与状态

- 日期：2026-09-17
- 基线 commit：`200939593`
- 本文性质：**方案 review + 验证设计**（不含代码改动）
- 对象方案：`pd_transfer_redesign_proposal_20260917.md`
- 目标场景：prefill 8 卡 `TP8 + DCP4` → decode 8 卡 `DP4 + TP2 + DCP2`
- 前置事实：GLM 5.3 flash（`glm5_next` 家族）当前不支持 PD 分离，因此本方案**全部验证在 host 侧完成**，不依赖 NPU / RDMA / PD 框架

---

## 第一部分：方案 review

### 0. 结论

**方向正确，接口基本可用；有 5 处必须修（其中 2 处会在目标场景下直接失败），3 处建议改。**

| 类别 | 编号 | 问题 | 严重度 |
|---|---|---|---|
| 正确性 | **F1** | `S` 必须按 **(role, group)** 派生 `S_eff`；否则目标场景的 KDA 缓存会被 C2 误拒 | **阻塞** |
| 正确性 | **F2** | `G` 必须按 **(role, group)**；GLM5-next 的 MLA/indexer `G=1`，KDA `G=64` | **阻塞** |
| 正确性 | **F3** | 边表未定义 **DP 维度**；本场景 P(DP1)→D(DP4) 需要 1→4 展开 | 高 |
| 正确性 | **F4** | 缺"**本地物理行 ↔ 规范块**"映射；不补则 mock 期望与实现不符 | 高 |
| 正确性 | **F5** | `B_token` **不是新字段**，就是现有 `block_token_capacity`；S3 改动面被高估 | 中 |
| 简洁性 | **F6** | `CanonicalAddr` 可并入 `RouteBinder`，不必独立对外 | 低 |
| 效率 | **F7** | `RouteBinder::bind` 应按 `t` 预分桶，把 `O(#edges × #blocks)` 降到 `O(#blocks)` | 中 |
| 效率 | **F8** | 边表应按 `(本侧拓扑, 对侧拓扑)` 缓存，而非按 peer | 中 |
| 清晰性 | **F9** | 删除清单未说清 `repeat_count / local_stride / remote_stride` 应**保留** | 低 |

---

### F1（阻塞）：`S_eff` 必须按 role/group 派生

**问题**：模型的 C2 是 `S ≤ D`。但 `S`（`kv_split_size`）是**实例级**配置，而 `D = CP × D_tp` 依赖**每个缓存的全局 head 数 `G`**。GLM5-next 的 KDA 状态 `G = kda_num_heads = 64`，`TP=8` ⇒ `D_tp = max(8/64,1) = 1` ⇒ `D = 1`，而 `S = 4`。按原文 C2，这个**合法且在跑的配置**会被直接拒绝。

**证据 [读码]**：现有实现用一张硬编码白名单表达同一件事，注释说得很明确：

```cpp
// xllm/core/framework/block/block.h:75-88
// KV-split widens one source-side logical block across multiple destination
// blocks. This applies to ordinary KV and every grouped attention-cache pool,
// but not to the sequence-scoped embedding or recurrent-state slots.
inline constexpr bool is_kv_split_cache_block_type(BlockType type) {
  switch (type) {
    case BlockType::KV: case BlockType::SWA:
    case BlockType::C4: case BlockType::C128:   return true;
    case BlockType::EMBEDDING: case BlockType::LINEAR: return false;
  }
  return false;
}
```

**修法（已实现）**：把 `S` 的作用域限定到 group，派生

```
sequence_scoped            => S_eff = 1
full_sequence_replica      => S_eff = 1
D == 1                     => S_eff = 1
S <= D 且 D % S == 0       => S_eff = S
其余                        => 报错（不静默降级）
```

语义：**能被冗余度吸收的缓存才切分；不能被吸收的缓存整体复制到每个 rank**（`S_eff = 1` ⇒ `t ≡ 0`，每个 rank 持完整序列）。但"不能被吸收"分两类：**语义性**的（`sequence_scoped`，或由 `full_sequence_replica` 显式声明，如 indexer 池）退 1；**配置性**的（`S>1` 却除不尽 `D`）报错 —— 静默退 1 会让使用者误以为切分已经生效。

**S2 已修正原 F1 的两处结论**：

1. `S_eff` 的判定多出 `full_sequence_replica`，且它**不能**从 `G/TP/CP/S` 推导：indexer 的 `G=1, D=8` 按规则会得到 `S_eff=4`，而它必须整序列复制（kPool 的 top-k 读全序列）。因此它由 `GroupTopology` **声明**，理由写进注释。
2. "白名单由 `S_eff` 推导、改造后降级为 `EXPECT_EQ(is_kv_split_cache_block_type(t), S_eff(group) > 1)`" **作废**：MLA latent 与 indexer 同属 `BlockType::KV`、同 `group_id`，却一个 `S_eff=4`、一个 `S_eff=1`，该断言必然自相矛盾。两者的对应关系改由 `(namespace, role, group_id)` 粒度的 `GroupTopology` 表达：

| BlockType | GLM5-next `G` | `TP` | `D_tp` | `D` | `S=4` 时 `S_eff` | 现有白名单 | 一致 |
|---|---|---|---|---|---|---|---|
| KV（MLA latent） | 1 | 8 | 8 | 8 | **4** | true | ✅ |
| KV（indexer k） | 1 | 8 | 8 | 8 | **1**（`full_sequence_replica`） | true | ❌ 白名单无法表达 |
| LINEAR（conv / ssm） | 64 | 8 | 1 | 1 | **1** | false | ✅ |

---

### F2（阻塞）：`G` 按 role/group，不按实例

`describe_cache_tensor` 对不同 role 用不同的全局 head 数（`cache_layout_builder.cpp:379-420`）：

| role | 全局 head 数来源 | GLM5-next 取值 |
|---|---|---|
| KEY / VALUE（MLA） | `kv_head_count = n_kv_heads().value_or(n_heads())`，但 MLA 形状是 `[n_blocks, B, 1, kv_lora_rank]` ⇒ **G = 1** | **1** |
| INDEX / INDEX_SCALE | `describe_attention_heads(context, /*global=*/1)` | **1** |
| SSM | `linear_value_head_count` | **64** |
| CONV | `linear_key_head_count` / `linear_value_head_count` | **64** |

所以 `KvTopology` 必须拆成两部分：

```cpp
struct KvTopology {                 // 实例级
  int32_t dp_size, cp_size, tp_size;
  int32_t kv_split_size;            // S
  int32_t tokens_per_block;         // B_token
};
struct GroupTopology {              // 每个 (namespace, role, group_id)
  int32_t global_head_count;        // G
  int64_t head_bytes;               // 一个 head 的字节数
  CacheResourceScope scope;         // BLOCK / SEQUENCE
};
```

`KvRedundancy::derive` 的入参是 `(KvTopology, GroupTopology)`。

---

### F3（高）：DP 维度必须显式建模

目标场景 **P 有 DP1、D 有 DP4**。原文 `RouteEdge` 只有一个 `dp_rank` 字段，无法表达 1→4 展开，且若把 dp 塞进边表会让边数随 DP 增长。

**修法**：**边表只描述「拓扑局部」的 rank 配对，DP 偏移在绑定阶段加**：

```cpp
struct RouteEdge {                  // 全部索引都是 DP 组内局部量
  int32_t src_local_rank;           // = cp_P * TP_P + tp_P
  int32_t dst_local_rank;           // = cp_D * TP_D + tp_D
  int32_t head_begin, head_end;
  int32_t src_slice;                // t_P
  int32_t dst_slice;                // t_D
};
// 实际全局 rank：
//   src_global = src_dp * (CP_P*TP_P) + edge.src_local_rank
//   dst_global = dst_dp * (CP_D*TP_D) + edge.dst_local_rank
```

DP 配对 `(src_dp → dst_dp)` 由 `TransferKVInfo.dp_rank` 给出，属调度层，不进边表。这样一张边表可服务全部 DP 对，**边数不随 DP 增长**。

---

### F4（高）：必须显式定义「本地物理行 ↔ 规范块」

原文只说"规范块号由 token 区间导出"，但没说本地 `resource_id` 与它的关系。实际关系由 DCP 的分配方式决定：

**几何事实 [读码]**：

| 量 | 取值 | 证据 |
|---|---|---|
| 物理行（= 规范块）的 token 数 | `options_.block_size()` | `worker_impl.cpp:640` 把 `options_.block_size()` 作为 `block_token_capacity`；`init_key_cache_shape` 用 `kv_cache_cap.block_size()`； estimator 里 MLA 的 `kv_slot_size` **不乘** `kv_split`（`kv_cache_estimation.cpp:78-83`），`.block_size(options.block_size)`（`:695`） |
| BlockManager 的 block | `options_.block_size() * S` | `llm_engine.cpp:653` |
| KV 张量行数 | `kv_cache_cap.n_blocks()` | 由显存估算，与 `S` 无关 |

⇒ **BlockManager 的一个逻辑 block id `b` 覆盖 `S` 个规范块 `[b·S, (b+1)·S)`，由 DCP 组的 `S` 个 rank 各持一个。** 验算：总规范块数 = `n_blocks × S`，rank `j` 需要 `n_blocks × S / S = n_blocks` 行 ✓ 与张量行数精确吻合。

⇒ 本地物理行的映射为（**按 role 不同**，见 F4′）：

```
KV 类（行数 = n_blocks）      : canonical_block = local_row * S_eff + slice_rank
                              : local_row       = canonical_block / S_eff
INDEX（行数 = n_blocks × S）  : canonical_block = local_row          （复制，全序列）
SEQUENCE 类                   : 无 canonical block 概念，用 slot id
```

**这是 S3 必须新增的一步**（`CanonicalBlock::to_local / to_canonical` 需要按 role 选择映射），也是 mock 期望值的推导依据。

### F4′（新增，待实测）：index cache 是「切分」还是「复制」

`init_index_cache_shape`（`kv_cache_shape.cpp:388-403`）在 `supports_dsa_indexer_cache_sharding() && S > 1` 时把 **index 张量的行数 × S**；同一个条件下 `kv_cache_estimation.cpp:96-99` 又把 **index 的每 token 字节数 × S**（`supports_dsa_indexer_cache_sharding() = is_mlu() || is_npu()`，`platform.h:66-68`，目标平台为真）。

两处同时生效 ⇒ index 张量行数 = `n_blocks × S` = **全部规范块数**，配合估算侧的 S 倍预算，指向「**每个 rank 保留全部规范块的 index**」，即 **index cache 在 DCP 下是复制而非切分**（kPool 的 top-k 需要全局序列的 index）。

若成立，则 **indexer 的 `S_eff` 应为 1**（每 rank 持完整序列），而**不是**与 MLA latent 相同的 4。这会改变 T3 的 indexer 用例与边表。**必须用一次探针实测确认**（见第六部分）。

**S2 的处理**：建模上已按"复制"落地 —— indexer 组由 `GroupTopology::full_sequence_replica` 声明，`S_eff=1`、`N_rep=D`，行映射仍是统一公式 `row = canonical / S_eff`（`S_eff=1` 退化为 `row = canonical`，正好对应 index 张量行数 `= n_blocks × S`）。探针仍待做，但它只影响"这组是否真的该声明"，不再阻塞 L1/L2/L3 的形状。

---

### F5（中）：`B_token` 是已有字段

`CacheTensorManifest::block_token_capacity`（`cache_layout.h:64`）就是 `options_.block_size()`，**已经是 peer 无关的**。规范块不是新概念，而是**把已有字段提升为坐标系基准**。

⇒ 方案 §3.2 的 `CanonicalBlock::tokens_per_block` 直接绑定该字段；§5.3 新增的 `tokens_per_block` 校验改为校验 `block_token_capacity_P == block_token_capacity_D`（该字段已在 manifest 中并在 `reshard_planner.cpp:244-245` 被比对过）。**S3 的改动面因此比原方案小。**

---

### F6–F9（建议）

- **F6**：`CanonicalAddr` 标为 `RouteBinder` 的实现细节，不对外暴露（单测可直接测 `RouteBinder`）。
- **F7**：`bind` 前把请求的规范块列表按 `canonical % S_eff` **分桶**，每条边只遍历自己的桶 ⇒ 复杂度从 `O(#edges × #blocks)` 降到 `O(#blocks)`。目标场景下 `#blocks = N/128`，`#edges = 16`（MLA），收益线性。
- **F8**：边表按 `(本侧 KvTopology, 对侧 KvTopology)` 缓存。一个 P 实例若同时服务多个不同拓扑的 D 实例（异构舰队），可复用而不必每个 peer 重算。
- **F9**：`StridedRegionTemplate` 里的 `repeat_count / local_stride / remote_stride` 表达的是**块内 token 维**的压缩，**必须保留**（一个规范块内 `B_token` 个 token 的 head 区间仍需展开）。被删除的是**跨资源的区间求交与折叠公式**。

---

## 第二部分：GLM 5.3 flash 的 cache 特征 **[读码]**

模型：`model_type = glm5_next`（`xllm/python/models/glm5_next.py:404-455`），混合架构，45 层。

| 参数 | 取值 | 含义 |
|---|---|---|
| `n_layers` | 45 | |
| `kv_lora_rank` | 512 | MLA latent 维 |
| `qk_rope_head_dim` | 0（checkpoint 通常覆盖为 64） | MLA rope 维 |
| `qk_nope_head_dim` / `v_head_dim` | 256 / 256 | |
| `kda_num_heads` | 64 | KDA（线性注意力）head 数 → `linear_num_key_heads = linear_num_value_heads = 64` |
| `kda_head_dim` | 128 | |
| `short_conv_kernel_size` | 4 | conv state 长度 |
| `index_n_heads` / `index_head_dim` | 32 / 128 | DSA indexer |
| `layer_types` | `linear_attention` / `deepseek_sparse_attention` | 逐层异构 |
| `indexer_types` | `full` / `shared` | 逐层异构，`shared` 层复用前一层 indexer |

**逐层异构**：`layer_types` 决定该层是 KDA（linear）还是 DSA（full attention + indexer）。因此**同一个 rank 上不同层的 cache 角色集合不同**，路由必须按层索引（`RouteEdge` 已有 `layer` 维度，通过 `bind(layer)` 传入）。

### 每个 cache 组的路由参数

| group / role | `BlockType` | `G` | scope | P(TP8,S=4) `D` / `S_eff` / `N_rep` | D(TP2,S=2) `D` / `S_eff` / `N_rep` | DCP |
|---|---|---|---|---|---|---|
| MLA latent KEY | `KV` | **1** | BLOCK | 8 / **4** / 2 | 2 / **2** / 1 | ✅ |
| MLA latent VALUE | `KV` | **1** | BLOCK | 8 / **4** / 2 | 2 / **2** / 1 | ✅ |
| indexer k（`kPool`） | `KV` | **1** | BLOCK | 8 / **4** / 2 | 2 / **2** / 1 | ✅ |
| KDA conv state | `LINEAR` | **64** | SEQUENCE | 1 / **1** / 1 | 1 / **1** / 1 | ✗ |
| KDA recurrent state | `LINEAR` | **64** | SEQUENCE | 1 / **1** / 1 | 1 / **1** / 1 | ✗ |

> **场景可解的必要条件**：MLA/indexer 要支持 `S=4` 与 `S=2`，需要 `G=1`。
> 由 C1/C2：P 侧需 `max(8/G,1) ≥ 4` 且被 4 整除 ⇒ `G ∈ {1,2}`；D 侧需 `max(2/G,1) ≥ 2` ⇒ `G = 1`。**交集 = `G = 1`**。
> GLM5-next 的 MLA latent 与 indexer 都恰好是 `G=1` ✅；若换成 GQA（`G ≥ 4`），此场景**在模型上就不成立**，应在建链前被拒。

---

## 第三部分：mock 与单测设计

### 3.0 分层与被测对象

| 单测 | 被测 | 是否依赖 NPU/RDMA | 目标 |
|---|---|---|---|
| `kv_redundancy_test` | `KvRedundancy` / `KvLayoutIndex` | 否 | 穷举正确性 |
| `kv_split_scope_test` | `S_eff` 派生 + 与 `is_kv_split_cache_block_type` 的一致性 | 否 | F1 回归锚点 |
| `pd_route_glm53flash_test` | `PdRouteTable::build` | 否 | **目标场景边表 golden** |
| `route_binder_test` | `RouteBinder::bind` | 否 | 字节区间 golden |
| `mock_pd_transfer_test` | 边表 + binder + 本地 memcpy 模拟搬运 | 否 | **路由 + 传输端到端字节正确** |

### 3.1 公共 fixture

```cpp
// 目标场景拓扑
struct Scenario {
  // prefill: dp1 tp8 cp1 kv_split4
  KvTopology p{/*dp*/1, /*cp*/1, /*tp*/8, /*S*/4, /*B_token*/128};
  // decode : dp4 tp2 cp1 kv_split2
  KvTopology d{/*dp*/4, /*cp*/1, /*tp*/2, /*S*/2, /*B_token*/128};
};

// GLM5-next 的 cache 组
constexpr GroupTopology kMla   { /*G*/1,  /*head_bytes*/1024, BLOCK   };  // kv_lora_rank 512 × bf16
constexpr GroupTopology kIndex { /*G*/1,  /*head_bytes*/256,  BLOCK   };  // index_head_dim 128 × bf16
constexpr GroupTopology kConv  { /*G*/64, /*head_bytes*/(4*128*2), SEQUENCE };
constexpr GroupTopology kSsm   { /*G*/64, /*head_bytes*/(128*2),  SEQUENCE };
```

### 3.2 T1 `kv_redundancy_test`：穷举

```
参数化：G ∈ {1,2,4,8,16,32,64} × TP ∈ {1,2,4,8} × CP ∈ {1,2} × S ∈ [1, CP*max(TP/G,1)+1]
```

断言：

1. `derive` 成功 ⟺ `(G%TP==0 || TP%G==0) && 1≤S≤D && D%S==0`；
2. `H_c * H_l == G`（head 类无损铺满）；
3. 对每个 `(h, t)`：`writer_of` 恰好返回 1 个 rank，且其 `c == 0`；
4. 对每个 `(h, t)`：`copies_of` 返回恰好 `N_rep` 个 rank，`c` 互不相同且 ∈`[0,N_rep)`；
5. 对每个 `(h)`：`{ slice(r) : r ∈ G_cptp(dp,h) }` 恰好等于 `[0,S)`（不重不漏）；
6. 每个 `t` 恰好被 `N_rep` 个 rank 持有；
7. `rank(dp,cp,tp)` 与 `(h,t,c)` 互为双射。

**判别性负例**：`TP=8, G=2, S=3`（`D_tp=4`，`3 ∤ 4`）必须在 `derive` 失败，而非等到 `"no source writer"`。

### 3.3 T2 `kv_split_scope_test`：`S_eff` 与硬编码白名单的一致性

对 GLM5-next 的 5 个 cache 组，断言 `S_eff` 与 `is_kv_split_cache_block_type` 的映射一致（表见 F1）。这是**新旧语义的锚点**：改造后白名单应改为 `EXPECT_EQ(is_kv_split_cache_block_type(t), S_eff(g) > 1)`。

再补一组判别性反例：构造一个 `G = TP`（`D_tp = 1`）的 KV 组，断言 `S_eff = 1` 且**它不应被当作 `KV` 走 DCP 路径**——这正是当前白名单无法表达的情形（当前按 BlockType 判断，不按实际冗余度）。

### 3.4 T3 `pd_route_glm53flash_test`：目标场景边表 golden

**MLA latent / indexer（`G=1`, `S_eff_P=4`, `S_eff_D=2`）**

`head_pairs = {(hP=0, hD=0, [0,1))}`（两侧 `H_c = 1`）。`S_D | S_P` ⇒ 汇聚，`block_map(t_P) = {t_P % 2}`。

P 侧 rank 0..7 的 `(h, t, c)`：

| P rank | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| `t` | 0 | 1 | 2 | 3 | 0 | 1 | 2 | 3 |
| `c` | 0 | 0 | 0 | 0 | 1 | 1 | 1 | 1 |

⇒ **写者只有 rank 0..3**（`c=0`），rank 4..7 是冗余副本、不发任何边。

期望边表。按 F3，边表**不含 DP 维**，因此表本体只有 **4 条**；下表是"每个 `dst_dp ∈ [0,4)` 各一份"的展开视图（共 **4 × 4 = 16 条有效边**）：

| # | src_local | dst_local | head range | `t_P` | `t_D` |
|---|---|---|---|---|---|
| 1 | 0 | 0 | [0,1) | 0 | 0 |
| 2 | 1 | 1 | [0,1) | 1 | 1 |
| 3 | 2 | 0 | [0,1) | 2 | 0 |
| 4 | 3 | 1 | [0,1) | 3 | 1 |

断言：
- 表本体边数 == 4；按 `dst_dp` 展开后 == 16（4 × 4）；
- `{edge.src_local_rank} == {0,1,2,3}`，**不含 4..7**；
- 对每个 `(dst_dp, canonical block b ∈ [0,64))`：恰好 1 条边的 `t_P == b % 4` 指向 `dst_local == b % 2`；
- 对每个 `(dst_dp, dst_local)`：源集合 == `{t_P : t_P % 2 == dst_local}`。

**KDA conv / ssm（`G=64`, `S_eff=1`）**

`H_l^P = 8, H_c^P = 8`；`H_l^D = 32, H_c^D = 2`。head 类区间：P 的 `h` 覆盖 `[8h, 8h+8)`，D 的 `h` 覆盖 `[32h, 32h+32)`。

⇒ `head_pairs = {(0,0,[0,8)), (1,0,[8,16)), (2,0,[16,24)), (3,0,[24,32)), (4,1,[32,40)), (5,1,[40,48)), (6,1,[48,56)), (7,1,[56,64))}`

`S_eff = 1` 两侧 ⇒ `t ≡ 0`，`block_map` 只有 1 个目的。⇒ 每个 P rank 恰好 1 个目的 rank：

| src_local (P h) | dst_local (D h) | head range |
|---|---|---|
| 0 | 0 | [0,8) |
| 1 | 0 | [8,16) |
| 2 | 0 | [16,24) |
| 3 | 0 | [24,32) |
| 4 | 1 | [32,40) |
| 5 | 1 | [40,48) |
| 6 | 1 | [48,56) |
| 7 | 1 | [56,64) |

断言：表本体边数 == 8；按 `dst_dp` 展开后 == 8 × 4 = 32；**P 侧 8 个 rank 全部参与**（与 MLA 只有 4 个形成对照，是 `S_eff` 生效的判别性证据）；每条边的 head range 与上表一致。

> 实现时两组数都已固定：`pd_route_test.cpp` 同时断言表本体（4 / 8）与展开后的有效边数（16 / 32），后者用 `全局 rank = dst_dp * (CP*TP) + local_rank` 计算。

### 3.5 T4 `route_binder_test`：字节区间 golden

用 `BufferDirectory`（`resource_count = 8`, `resource_stride_bytes = 131072`, `buffer_bytes = 1048576`, `explicit_offsets = false`）构造两侧目录，对 `canonical block b ∈ [0,8)`、`layer = 0`、`dst_dp = 0` 断言：

| b | local row | local offset | remote row | remote offset | length |
|---|---|---|---|---|---|
| 0 | 0 | 0 | 0 | 0 | 131072 |
| 1 | 0 | 0 | 0 | 0 | 131072 |
| 2 | 0 | 0 | 1 | 131072 | 131072 |
| 3 | 0 | 0 | 1 | 131072 | 131072 |
| 4 | 1 | 131072 | 2 | 262144 | 131072 |
| 5 | 1 | 131072 | 2 | 262144 | 131072 |
| 6 | 1 | 131072 | 3 | 393216 | 131072 |
| 7 | 1 | 131072 | 3 | 393216 | 131072 |

（`local_row = b / 4`，`remote_row = b / 2` —— 这就是 F4 的映射。）

再补一组：
- **发散方向**：把两侧 `S` 对调（`S_P=2, S_D=4`），断言 `local_row = b/2`、`remote_row = b/4`，且每条边扇出 2 个目的；
- **`explicit_offsets = true`**（XTensor 形态）：断言偏移取自 `local_offsets[] / remote_offsets[]` 而非 `row × stride`；
- **`repeat_count / stride`**：断言一个规范块内 `B_token` 个 token 被压成 1 条模板（F9），并断言 `bind` 展开后的区间总数 == `B_token`。

### 3.6 T5 `mock_pd_transfer_test`：路由 + 传输的端到端字节验证

**这是满足"GLM 5.3 flash 尚未支持 PD 分离"约束的核心用例**：用真实内存 + 本地 memcpy 替代 RDMA，验证"边表 + 绑定 + 搬运"三段合起来正确。

```
1) 分配 P 侧 4 个 writer rank 的 host buffer（仅 writer，因为 c≠0 不参与）
     每个 rank: [resource_count=8][131072] 字节
2) 分配 D 侧 4(dst_dp) × 2(tp) = 8 个 rank 的 host buffer，清零
3) P 侧填充 pattern：
     pattern(rank, canonical_block, byte_index) =
         hash(rank, canonical_block, byte_index)   // 可复现的伪随机
     注意：canonical_block = local_row * 4 + slice(rank)
4) 对每个 dst_dp、每个 canonical block b ∈ [0, 64)：
     regions = RouteBinder::bind(edges, {b}, layer=0, dir_P, dir_D)
     对每条 region：memcpy(D[dst].buf + remote_offset,
                           P[src].buf  + local_offset, region.length)
5) 校验：对每个 dst_local、每个 b，把 D 的字节与 pattern(源 rank, b, ·) 比对
     期望值必须由**独立于实现的**参考函数计算（手工推演的 (src, local_row, remote_row) 表）
6) 反例：故意打乱一条边的 `t_P` → 断言校验失败（判别性，防止测试自身写反）
```

覆盖矩阵：

| 用例 | 目的 |
|---|---|
| `MlaLatent_S4toS2` | 主场景，汇聚方向 |
| `MlaLatent_S2toS4` | 发散方向 |
| `Indexer_S4toS2` | 不同 `head_bytes`（256 vs 1024） |
| `Kda_Ssm_NoSplit` | `S_eff=1`，每 P rank → 1 D rank，head 区间切分 |
| `Kda_Conv_NoSplit` | 同上，`head_bytes` 不同 |
| `MixedLayers` | 同一 rank 上 KDA 层与 DSA 层的角色集合不同（模拟 `layer_types`） |
| `DpExpansion_1to4` | P DP1 → D DP4 的 1→4 展开（F3） |
| `X tensor_ExplicitOffsets` | `explicit_offsets = true` 形态 |

### 3.7 T6 真实 PD 验收（后续，非本次交付）

GLM 5.3 flash 支持 PD 分离后：
1. P 侧导出每个 `(canonical block, group)` 的字节校验和；
2. D 侧导出收到的校验和；
3. 逐 `(block, group)` 比对；
4. 断言 `S_eff` 与配置一致（`kSplitScopeDump`）。

---

## 第四部分：把本次 review 落到方案里

需要回写 `pd_transfer_redesign_proposal_20260917.md` 的条目：

| 编号 | 回写位置 |
|---|---|
| F1 | §3.1 `KvRedundancy::derive` 增加 `S_eff`；§6 边界改为"由 `S_eff` 推导，不再按 BlockType 白名单" |
| F2 | §3.1 `KvTopology` 拆出 `GroupTopology`；§2 的 L0 层说明改为 `(实例级拓扑, 每组拓扑)` |
| F3 | §3.3 `RouteEdge` 改为拓扑局部 + 增加 DP 偏移公式；§5.1 说明 DP 配对属调度层 |
| F4 | §3.2 增加 `to_local / to_canonical`；§4 数据面补映射步骤 |
| F5 | §3.2 `tokens_per_block` 绑定 `CacheTensorManifest::block_token_capacity`；§5.3 校验改名 |
| F6/F7/F8 | §3.4 `RouteBinder` 增加分桶；§5.1 增加边表缓存键；`CanonicalAddr` 标 internal |
| F9 | §7.1 删除清单明确 `repeat_count / *_stride` **保留** |

---

## 第五部分：两个待确认问题的解释（已部分查实）

### 5.1 问题 3：indexer 的 `group_id`

**问题含义**：路由的键是 `group_id`（`KVTransferMapping.group_id` ← `cache_group_id(BlockType)`）。若 index cache 与 MLA latent 同组，则它们共用一条路由；若不同组，则各自独立。同时 `GroupTopology` 里的 `G`（从而 `D_tp`、`S_eff`）是按组的，若同组内不同 role 的 `G` 不同，路由就会冲突。

**已查实 [读码]**（`kv_cache/kv_cache_impl.cpp:141-155`）：

```cpp
add_tensor(KVCacheTensorRole::KEY,         get_k_cache(),              BlockType::KV);
add_tensor(KVCacheTensorRole::VALUE,       get_v_cache(),              BlockType::KV);
add_tensor(KVCacheTensorRole::INDEX,       get_index_cache(),          BlockType::KV);   // ← 同组
add_tensor(KVCacheTensorRole::INDEX_SCALE, index_scale.value(),        BlockType::KV);
add_tensor(KVCacheTensorRole::CONV,        get_conv_cache(),           BlockType::LINEAR);
add_tensor(KVCacheTensorRole::SSM,         get_ssm_cache(),            BlockType::LINEAR);
```

`add_tensor` 内 `cache_group_id(block_type)`（`:139`）。⇒ **`KEY / VALUE / INDEX / INDEX_SCALE / *_SCALE` 全部同属 `group_id = cache_group_id(BlockType::KV) = 0`**；`CONV / SSM` 同属 `group_id = 5`。

**结论与影响**：

1. GLM5-next 的 MLA latent（`G=1`）与 indexer（`G=1`）**在同一个组里且 `G` 相同** ⇒ 可共用一条路由，无冲突。
2. 但这是**巧合而非保证**。对「GQA 主 KV（`G=n_kv_heads`）+ 共享 head indexer（`G=1`）」的模型，同组内 `G` 不同 ⇒ `D_tp` 不同 ⇒ `S_eff` 可能不同 ⇒ **一条组级路由无法同时满足两者**。
3. 因此模型层需要：**路由按 `(namespace, role, group_id)` 计算**，并把「同组内所有 role 的 `(G, S_eff)` 必须一致」作为**显式断言**（GLM5-next 通过；其他模型若失败则报错而非静默）。
4. 对 mock 的影响：`GroupTopology` 的键从 `group_id` 细化为 `(group_id, role)`；T3 的 indexer 用例需按 §5.2 的结论修正。

### 5.2 问题 4：kPool 的 packed 宽度与 `S` 放大

**问题含义有两层**：

**(a) packed 宽度**：`init_index_cache_shape`（`kv_cache_shape.cpp:399-402`）

```cpp
const int64_t head_dim = model_args.index_head_dim();            // 128
const int64_t cache_head_dim =
    model_args.index_kpool_compress() ? head_dim * 2 + 1 : head_dim;
index_cache_shape_ = {index_block_count, kv_cache_cap.block_size(), 1, cache_head_dim};
```

开启时把 `[k(128), gate(128), valid(1)]` 打包进一行 ⇒ `head_bytes = 257 × dtype_size`；关闭时 `head_bytes = 128 × dtype_size`。这直接决定 mock 的 `resource_stride_bytes`。

**glm5_next 的取值存在歧义**：配置默认 `index_kpool_compress = false`（`glm5_next.py:446`、`model_args.h:189`），但紧邻的注释明确说 *"GLM-next kPool packs [k, gate, valid] ... into the index cache (its Python select_topk reads historical gate/valid from the cache)"*，且 `deepseek_v2_attention.cpp:56` 用 `use_kpool_indexer_ = has_indexer_ && args.index_kpool_compress()`。⇒ **代码默认关、真实 checkpoint 预期开**。mock 应对两种取值各出一个用例。

**(b) 是否随 `S` 放大**：**有两处独立的 ×S，且在目标平台上同时生效**（`supports_dsa_indexer_cache_sharding() = is_mlu() || is_npu()`，`platform.h:66-68`）：

| 位置 | 放大量 |
|---|---|
| `kv_cache_shape.cpp:390-392` | index 张量的**行数** × S |
| `kv_cache_estimation.cpp:96-99` | index 的**每 token 字节数** × S（进入 `standard_full_cache_block_size_in_bytes`，从而影响 `n_blocks`） |

⇒ index 张量行数 = `n_blocks × S` = **全部规范块数**，且估算侧按 S 倍为 index 计预算。**两者合起来指向「每个 rank 保留全部规范块的 index」**，即 **index cache 在 DCP 下是复制而非切分**（kPool 的 top-k 需要在全局序列上选）。

**若该推断成立，必须修正前面的表**：

| group | 原表 `S_eff` | 修正后 |
|---|---|---|
| MLA latent KEY / VALUE | 4 | **4**（不变，行数 = `n_blocks`） |
| indexer k | 4 | **1**（复制，行数 = `n_blocks × S`） |
| KDA conv / ssm | 1 | 1 |

同时**行 ↔ 规范块映射按 role 不同**（见 F4′）：KV 用 `row = canonical / S`，INDEX 用 `row = canonical`。

**定音探针（必须在写 mock 前跑）**：

1. 在目标配置（TP8 + DCP4）下单次启动，打印 `index_cache_shape()` 与 `k_cache_shape()`，确认 index 行数 == KV 行数 × S；
2. 打印 `kv_cache_cap.n_blocks()` 与 `num_indexer_layers()`；
3. 在 `filter_kv_split_infos` 入口打点，确认 `group_id == 0` 的 mapping 是否真的进入 remap —— 注意 `has_rank_preserving_kv_groups`（`disagg_pd_scheduler.cpp:162-173`：`block_type == KV || !is_kv_split_cache_block_type(...)`）在 GLM5-next 的分组集合（KV + LINEAR）上**恒为 true**，从而 `rank_local_mapping = true` 使 remap **被整体跳过**。若确实被跳过，则说明 **kv_split 的划分是在 D 侧分配 block id 时就完成的**（rank-preserving 契约），而不是在传输层做的 —— 这对「S3 应该在哪一层引入 canonical block」有直接影响。

---

## 第五部分补：旧计划器为什么不能作为 S2 的比对基准 **[读码]**

S2 原定的验收是"新边表与旧 `select_sources` 的 ACTIVE 集合逐边一致"。实施时发现**目标配置下旧路径无法表达**：

1. MLA 缓存在 `describe_cache_tensor` 里走 `describe_replicated_tensor`（`cache_layout_builder.cpp` 的 `enable_mla` 分支）：**整行**一个 span、`bytes_per_region = resource_stride_bytes`、`owner_tp_rank = 0`、`kind = REPLICATED`。也就是说描述符里**没有块维身份**。
2. `select_sources`（同构分支）要求源的 `cp_rank`、`kv_split_size`、`kv_split_rank` 与目的完全相同，再用 `only_static_owner`（此处 `kv_split_spans_cp_and_tp` 为假 ⇒ 为真）按 `tp_rank == span.owner_tp_rank` 去重。于是 `TP8 + DCP4` 下有：
   - `kv_split_rank = dcp_group_->rank() = rank % S`（运行时真实 DCP 分组，与模型 §5 的 `t` 一致）⇒ 目的 rank 1/2/3 的候选源只有 `{1,5}`/`{2,6}`/`{3,7}`，而它们的 `owner_tp_rank` 都是 0 ⇒ **0 个写者**（`"no source writer"`）；
   - fallback `kv_split_rank = rank / (world/kv) = rank/2` ⇒ 每个目的 rank 有 **2 个写者**（`"multiple writers"`）。

两条路都过不了覆盖率校验。**结论**：旧路径支持的形状只有 `G ≥ TP 且 S = 1`、`G < TP 且 S = 1`、sequence-scoped 三类；目标场景（MLA `G=1` 配 `DCP4/2`）恰恰是它不支持的。因此 S2 的验收改为 **golden + mock 逐字节**（T3/T4/T5），旧路径比对仅作可选补充。

顺便回答 §6 的问题 5（kv_split 的划分发生在哪一层）：MLA 的描述符里既然没有块身份，`kv_split` 的"哪些规范块归谁"就**不可能**由 manifest 表达，只能来自调度侧分配的 block id（即 rank-preserving 契约）。这与"canonical block 应引入在**分配层/契约层**"的判断一致，S3 应据此评估。

---

## 第六部分：待确认（更新后）

| # | 项 | 状态 |
|---|---|---|
| 1 | GLM 5.3 flash == `glm5_next` | ✅ **已确认**（用户） |
| 2 | `qk_rope_head_dim` | ✅ **已确认 = 0** ⇒ MLA V 张量维数为 0，实际只有 K（latent）承载 KV；`head_bytes(MLA) = kv_lora_rank × dtype_size = 512 × 2 = 1024` |
| 3 | indexer 的 `group_id` | ✅ **已查实**：与 MLA latent 同属 `group_id = 0`（`kv_cache_impl.cpp:147`）⇒ 需新增「同组内 role 的 `(G, S_eff)` 一致」断言 |
| 4a | kPool packed 宽度 | ⚠️ **代码默认 `index_kpool_compress = false`（宽度 128），但注释与 `use_kpool_indexer_` 表明真实 checkpoint 预期为 true（宽度 257）** ⇒ mock 两种各出一例 |
| 4b | index cache 是否随 `S` 放大 | ⚠️ **两处 ×S 均已定位且目标平台生效**；推断为「复制而非切分」⇒ indexer 的 `S_eff` 为 1。**S2 已按此建模**（`full_sequence_replica` 显式声明，见 F4′）；探针只需确认"该不该声明"，不再阻塞实现 |
| 5 | kv_split 划分发生在哪一层 | ⚠️ `has_rank_preserving_kv_groups` 在 GLM5-next 的分组集合上恒为 true ⇒ `filter_kv_split_infos` 被跳过 ⇒ 划分可能已在 D 侧分配 block id 时完成。**这决定 S3 应在传输层还是分配层引入 canonical block**，需探针确认 |
