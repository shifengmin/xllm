# xLLM PD 传输路由现状盘点与按冗余模型的简化方案

## 溯源与状态

- 日期：2026-09-17
- 基线 commit：`200939593`
- 本文性质：**现状盘点 + 简化方案**（不含代码改动）
- 目标模型：`kv_redundancy_model_and_pd_routing_20260917.md`
- 背景：本文自洽，不依赖任何既有设计文档的结论；所有论断以本树 `200939593` 的代码为准
- 证据口径：**[读码]** 静态阅读（本文全部结论未在 Ascend 上验证）

---

## 0. 结论摘要

| # | 结论 |
|---|---|
| **C1** | 当前存在**三套 rank 配对实现**（其中**两套是死代码**），以及**两套生产路径**（PUSH 走 manifest 计划、PULL 走 modulo）。 |
| **C2** | 复杂度的根因不是"kv_split 本身复杂"，而是**缺少 `D`（冗余度）与 `N_rep`（残余冗余度）这两个派生量**，导致只能靠 `cp_size` / `kv_split_size` 的大小关系去间接猜冗余，于是每个组合都要一条特例。 |
| **C3** | 引入 `(h, t, c)` 三索引 + `D` / `N_rep` 后，**至少 8 类特例分支可被删除或降级为推论**，两套死代码可直接删除。 |
| **C4** | 最后一层复杂度是**逻辑地址 peer 相关**：`logical_offset` 里没有规范块身份，而 `BlockLogicalBytes` 又随本侧 `kv_split_size` 变化（`llm_engine.cpp:653`），于是 `S_P ≠ S_D` 时两侧落在不同坐标系。解法不是"再补一条折叠公式"，而是引入**规范逻辑地址层**（模型文档 §7）：规范块尺寸与 `S` 无关、两侧恒等，物理资源几何只在 `bind` 阶段出现。 |
| **C5** | 落地上 **S0（删死代码）零风险可立即执行**；S1/S2 是纯逻辑层重构，可用现有 46 个 planner 单测 + 新增 `(h,t,c)` 判别性测试守住；**S3（规范地址层）是本方案的组成部分，不是外部依赖**。 |

---

## 1. 现状盘点

### 1.1 两条生产传输路径

| 路径 | 触发点 | 发起方 | 配对决策者 | 搬运原语 |
|---|---|---|---|---|
| **PUSH** | `llm_worker_impl.cpp:279` `push_kv_blocks_async` | P | `MooncakeKVCacheTransferBase::merge_kv_blocks`（`mooncake_kv_cache_transfer.cpp:635`）+ `ReshardPlanner` plan | `bind_outgoing_regions` + `move_memory_regions(WRITE)` |
| **PULL** | `disagg_pd_scheduler.cpp:1210` / `pd_ooc_scheduler.cpp:1818` `pull_kv_blocks` | D | `LLMEngine::pull_kv_blocks`（`llm_engine.cpp:814`） | `append_buffer_mappings` + `move_memory_groups(READ)` |

两条路径**互不复用配对逻辑**，且对同一件事（谁给谁）给出两套算法。

### 1.2 三套 rank 配对实现（两套死代码）

| # | 位置 | 机制 | 状态 |
|---|---|---|---|
| **A** | `KVCacheTransfer::merge_kv_blocks`（`kv_cache_transfer.cpp:266-373`） | 整除步进的 modulo 路由：`for (i = src_dp_local_tp_rank % dst_tp_size + dst_tp_size*dst_dp_rank; i < ...; i += src_tp_size)`；并先算 `linked_dp_ranks` 做过滤 | **不可达** |
| **B** | `push_route.cpp`（`get_dst_ranks` / `get_src_tp_ranks` / `use_push_owner`） | A 的抽取版（参数化为 `src_tp_size/dst_tp_size`，增加 `src_tp > dst_tp` 的 owner 分支） | **仅单测调用** |
| **C** | `MooncakeKVCacheTransferBase::merge_kv_blocks`（`mooncake_kv_cache_transfer.cpp:635-659`） | 按 `(dp_rank, 目的 tp_size)` **全扇出**到目的 DP 组内所有 tp rank，再由 `select_sources` 的 ACTIVE 集合筛选 | **生产路径** |
| **D** | `LLMEngine::pull_kv_blocks`（`llm_engine.cpp:814-843`） | `src_dp_worker_rank = dst_worker_rank % src_tp_size` | **生产路径（PULL）** |

**A 不可达的证据** **[读码]**：`KVCacheTransferFactory::create`（`kv_cache_transfer.cpp:375-409`）只构造 `MooncakeKVCacheTransferDefault` / `MooncakeKVCacheTransferXTensor`，二者共同基类 `MooncakeKVCacheTransferBase` 在 `mooncake_kv_cache_transfer.h:54` 覆写了 `merge_kv_blocks`。全树无其它 `KVCacheTransfer` 子类（`grep "public KVCacheTransfer"` 仅命中 `MooncakeKVCacheTransferBase`）。`push_kv_blocks_async`（`kv_cache_transfer.cpp:213`）内的 `merge_kv_blocks(...)` 是虚调用，必然派发到 C。

**B 不可达的证据** **[读码]**：`use_push_owner` / `get_dst_ranks` / `get_src_tp_ranks` 的生产调用点为 0；`mooncake_kv_cache_transfer.cpp:30` 仍 `#include "push_route.h"` 但未使用其中任何符号。

> A 与 B 是历史遗留的 same-modulo 逻辑的两次拷贝（一份内联、一份抽取），抽取后原实现未删。A 中还有一处语义混淆：`src_tp_size = src_world_size / src_dp_size / src_kv_split_size`（`kv_cache_transfer.cpp:282`）—— 把 `kv_split_size` 直接当作 TP 宽度的除数，即把 kv_split 当成"又一个 TP 维"。

### 1.3 kv_split 介入路由的**全部**位置

| 位置 | 做了什么 | 在模型中的本义 |
|---|---|---|
| `parallel_args.h:173-186` `kv_split_size_effective()` / `kv_split_rank()` | `kv_split=0 ⇒ 回退 cp_size`；rank 取自 `dcp_group_->rank()` 或 `rank/(world/kv)` | `S` 与 `t` |
| `llm_engine.cpp:646-653` | `block_size(kv_split_eff > 1 ? block_size * kv_split_eff : block_size)` —— **块大小按 kv_split 放大** | 逻辑块 vs 物理资源的分离 |
| `collective_communicator.cpp:582-589` | `dcp_size = (cp_size == 1) ? kv_split_eff : 1`；DCP 组 = **连续** rank 块，`dcp_rank = rank % dcp_size` | **完备组的运行时实现** |
| `filter_kv_split_infos`（`kv_cache_transfer.cpp:145-196`） | 重排 `remote_ids`：`local_ids[k] ↔ remote_ids[kv_split_rank + k*kv_split_size]`，并在越界时截断 `local_ids` | `S_P = S_D` 时 `t` 规则的特化 |
| `rotate_dst_rank`（`kv_cache_transfer.cpp:199-210`，调用点 `mooncake_kv_cache_transfer.cpp:677,971`） | `kv_split_size > 1` 时按 `kv_split_rank` 轮转目的顺序 | 与模型无关的**负载均衡**手段；模型下路由被块号唯一确定，不需要它 |
| `validate_transfer_mappings`（`kv_cache_transfer.cpp:33-93`） | `kv_split_size > 1` 时校验 `remote_count ∈ [local*S - S + 1, local*S]` | 对 `t` 分片的覆盖度检查 |
| `has_rank_preserving_kv_groups` + `rank_local_mapping`（`disagg_pd_scheduler.cpp:162-173, 668`） | 标记"目的端 id 已按 rank 保存"，从而**跳过** filter | 对 `t` 规则是否已由调度侧完成的旁路开关 |
| `merge_kv_blocks`(A) `src_tp_size` 除以 kv_split | 把 kv_split 折进 TP 宽度 | A 独有，死代码 |
| `RemoteWorker`/`PullKVBlocks` 路径 | 无 kv_split 概念（`kv_cache_transfer.cpp:118-133` 以 `kv_split_size=1` 校验） | PULL 不感知 kv_split |

### 1.4 head 维（TP / kv_head）介入路由的位置

| 位置 | 内容 |
|---|---|
| `describe_attention_heads`（`cache_layout_builder.cpp:118-200`） | `replica_count` / `first_global_head` / `owner_tp_rank = replicated ? global_head*replica_count : tp_rank` |
| `only_static_owner` 过滤（`reshard_planner.cpp:155`） | `manifest.coordinates.tp_rank != span.owner_tp_rank ⇒ skip` |
| `head_pairs` 隐式求交（`reshard_planner.cpp` `visit_overlaps` + `validate_tensor_pair`） | 用 `logical_offset` 区间表达 head 归属 |
| `owner_tp_rank` 的赋值（`cache_layout_builder.cpp:97,180,247,297`） | 描述"该 head 的静态持有者"（副本组最低 rank），本身是**数据事实**，却被 planner 当**写者策略**开关使用 |

**核心观察**：**head 类 `h` 在整条链路中没有名字。**它只以两种形态出现：构建期的 `first_global_head`，和 planner 期的 `owner_tp_rank`。前者是数据，后者被迫兼职"写者策略"——**同一个概念被两处以不同方式表达**，这就是 R2 的成因。

### 1.5 拓扑校验能拿到什么 **[读码]**

```cpp
// pd_topology_guard.h:24-27
struct PdTopo { int32_t dp_size = 0; int32_t tp_size = 0; };
// pd_topology_guard.cpp:57
topo->tp_size = static_cast<int32_t>(cluster_num / dp_size);  // 注释: "PD routing only needs the aggregate worker width per DP partition"
```

`PdTopo` **不知道 `kv_head_num`、不知道 `kv_split_size`、不知道 `cp_size`**。因此 `check_pd_topo` 只能在"dp/tp 是否相同"这个最粗粒度上返回 `ALLOW_HOMO` / `ALLOW_HETERO`（异体还要求 `kv_mode == "PUSH"`）。

这意味着**构建期无法提前拒绝 C1/C2 违规的配置**，只能等 planner 在覆盖率校验处报 `"no source writer"` / `"multiple writers"` —— 报错点远离根因。

### 1.6 路由相关的条件分支清单

| # | 位置 | 条件 | 存在原因 |
|---|---|---|---|
| 1 | `reshard_planner.cpp:99-102` | `cp % S == 0 \|\| S == cp*tp` | 用大小关系间接表达"KV 归属是否静态" |
| 2 | `reshard_planner.cpp:94-97` | `S == cp*tp` | 同上，单独抽成一个判据被复用 |
| 3 | `reshard_planner.cpp:104-113` | `same_partition_sizes \|\| (dst.cp==1 && dst.cp_rank==0 && …)` | 允许 D 端 CP 塌缩 |
| 4 | `reshard_planner.cpp:115-124` | `same_partition_sizes ? same_partition : …` | 同构 vs 塌缩分叉 |
| 5 | `reshard_planner.cpp:835` | `collapse_partitions = !same_partition_sizes(...)` | 选源两条路径的分叉点 |
| 6 | `reshard_planner.cpp:461-465` | `spans_cp_and_tp ? 1 : cp/S`（`expected_cp_count`） | 折叠时 CP 副本数的特判 |
| 7 | `reshard_planner.cpp:466-470` | `dst.S>1 && group.second != dst.kv_split_rank ⇒ skip` | 目的 KV-split 过滤 |
| 8 | `reshard_planner.cpp:442-456` | `(dp,cp)` 上 `kv_split_rank` 必须一致 | 一致性检查 |
| 9 | `reshard_planner.cpp:854` | `only_static_owner = true`（硬编码） | 同构分支的去重 |
| 10 | `reshard_planner.cpp:479`、`869` | `only_static_owner = !kv_split_spans_cp_and_tp(...)` | 动态归属时关闭去重 |
| 11 | `reshard_planner.cpp:66` | `CoverageKey = (dp_rank, kv_split_rank)` | 覆盖率按 DP × KV-split 分组 |
| 12 | `kv_cache_transfer.cpp:172-185` | `remote_idx = kv_split_rank + k*kv_split_size` | block 维交织 |
| 13 | `kv_cache_transfer.cpp:239-245` | `kv_split_size > 1` 才过滤 | 退化即拷贝 |
| 14 | `kv_cache_transfer.cpp:676,971` | `kv_split_size > 1` 才 rotate | 负载均衡 |
| 15 | `cache_layout_builder.cpp:379-420` | `enable_mla` 分支优先于 CONV/SSM | 潜在遮蔽风险：MLA 模型上后两者不可达 |
| 16 | `mooncake_kv_cache_transfer.cpp:287-400` | main / spec-draft 布局两分支 + `offset` 接续 | draft 缓存追加注册 |
| 17 | `mooncake_kv_cache_transfer.cpp:617-621`、`667` | `is_spec_draft` 选 layout | 同上 |
| 18 | `pd_topology_guard.cpp:65-80` | `same_dp && same_tp ? HOMO : (PUSH ? HETERO : DENY)` | 信息不足下的粗判 |

---

## 2. 复杂度根源（三条）

**R1 — 缺少 `D` / `N_rep`，冗余度不可见。**
代码里没有任何地方计算"本实例有几种冗余副本、切分后还剩几种"。于是分支 #1/#2/#3/#4 只能用 `cp_size`、`kv_split_size`、`tp_size` 的**大小关系**去间接猜"KV 归属是静态还是动态"。
在模型里这就是一句话：`D = CP × D_tp`，`S | D`，`N_rep = D/S`。

**R2 — head 类 `h` 没有名字，`owner_tp_rank` 被迫兼职。**
`owner_tp_rank` 本应只回答"该 head 的静态持有者是谁"，却被 planner 当作"谁应该写"的策略开关（`only_static_owner` + 分支 #9/#10）。它只在所有权**静态**时成立，因此当 `S == CP × TP`（KV-split 横跨 CP 与 TP，即 `G = 1` 且切分取满）时只能被整体关闭（`reshard_planner.cpp:479,869`）——同一个语义在代码里有两种互相打架的表达，去重责任散落到别处。

**R3 — 逻辑地址空间是"资源内、peer 相关"的。**
`expand_manifest`（`reshard_planner.cpp:150,166`）给 `logical_offset` 只加 `span.logical_offset_bytes + repeat × span.logical_stride_bytes`，**没有规范块身份项**；资源 id 只进物理寻址（`bind_regions`：`id * resource_stride`）。同时 `llm_engine.cpp:653` 把 `block_size` 乘上 `kv_split_size_eff`，使 `BlockLogicalBytes = B·S·G·bytes_per_head` **随本侧 `S` 变化**。

两者叠加的后果：`S_P ≠ S_D` 时，"块号 × 块逻辑尺寸"落在**两个不同的坐标系**里。此时任何"折叠公式"都只能从 `resource_stride_bytes` 的比值去反推 `fold`——这是**症状级补丁**：真正缺的是规范块这一层（模型文档 §7）。

**R4（工程）— 同一件事三处实现 + 两处死代码。**
修 A 不影响 C/D，反之亦然；PUSH 与 PULL 对同一问题给出两套算法，且没有任何一处保证二者一致。

---

## 3. 简化映射（逐条对应）

| 现有机制 | 在新模型中的位置 | 处置 |
|---|---|---|
| 分支 #1 `supports_kv_split_topology` | `S \| D`（C1） | **删除**，换成一条整除断言 |
| 分支 #2 `kv_split_spans_cp_and_tp` | 无概念 | **删除**。`D_tp > 1` 只是 `D` 的一个因子 |
| 分支 #3/#4 `supports_partition_{layout,pair}` | `S_P \| S_D` 或反之（C1） | **降级**为一条跨实例整除检查；CP 塌缩 = `CP_D = 1` 的普通取值 |
| 分支 #5 `collapse_partitions` | 不存在分叉：`t` 规则统一 | **删除** |
| 分支 #6 `expected_cp_count` | `N_rep = D/S` | **删除**，改为派生量 |
| 分支 #7/#8 目的 kv_split 过滤与一致性 | `t_D = j % S_D` | **删除**（块号自带） |
| 分支 #9/#10 `only_static_owner` | 源侧取 `c == 0` | **替换**为显式的"取 `c=0` 那个 rank" |
| 目的侧 `expand_manifest(..., false)` | 遍历 `c ∈ [0, N_rep)` | **替换**为显式循环 |
| 分支 #11 `CoverageKey=(dp, kv_split_rank)` | 覆盖率按 `(dp, h, t)` 校验 | **替换**（见 §4） |
| 分支 #12 `remote_ids[kv_split_rank + k*S]` | `t_D = t_P % S_D`（`S_D \| S_P` 时） | **收敛**为 block 路由的一个特化 |
| 分支 #13 过滤退化 | 无需过滤：`S=1 ⇒ t≡0` | **删除** |
| 分支 #14 `rotate_dst_rank` | 路由由块号唯一确定 | **删除**（负载均衡改由"片→rank"划分承担） |
| 分支 #15 `enable_mla` 遮蔽 | 与路由无关 | 独立处理 |
| 分支 #16/#17 spec-draft 布局 | 与路由正交 | 保留 |
| 分支 #18 `PdTopo` 粗判 | 用 `(G, TP, CP, S)` 精确判定 | **升级**：把 `G` 与 `S` 纳入 `InstanceInfo`，让 C1/C2 在**建链前**拒绝 |
| `logical_offset` 无规范块身份（R3） | **规范层**：`(b, g, tau)` 线性化地址（模型文档 §7） | **新增**：`logical_offset` 改为规范块坐标，物理几何下沉到 `bind` |
| `block_size × kv_split_size`（`llm_engine.cpp:653`） | **物理层**：`resource_stride_bytes` 留在 manifest | **解绑**：规范块尺寸与 `S` 无关，`S` 只决定"哪些块归谁" |
| 实现 A（base `merge_kv_blocks`） | — | **删除** |
| 实现 B（`push_route.cpp` + 其单测 + 无用 include） | — | **删除** |
| 实现 C（Mooncake `merge_kv_blocks`） | head 路由的"全候选"枚举 | **保留但简化**：只枚举 `head_pairs` 命中的 rank |
| 实现 D（PULL modulo） | block 路由 + head 路由 | **收敛**：PULL 复用同一边集合，只把原语从 WRITE 换成 READ |

**净效果**：18 类分支中 **10 类删除、5 类替换为派生量计算、3 类保留**；另外新增一条**规范逻辑地址层**取代 R3 的全部周边补丁；两套死代码（含约 200 行逻辑与 1 个测试文件）删除。

---

## 4. 目标形态（代码骨架）

```cpp
// ---- 纯算术层：单实例派生量（无 IO、无状态）----
struct KvRedundancy {
  int32_t G, TP, CP, S;
  int32_t Hl;      // 每 rank 本地 head 数
  int32_t D_tp;    // TP 冗余度
  int32_t Hc;      // head 类数
  int32_t D;       // 复合冗余度 = CP * D_tp
  int32_t N_rep;   // 残余冗余度 = D / S

  static Status derive(int G, int TP, int CP, int S, KvRedundancy* out);
  // C1: (G%TP==0 || TP%G==0) && S>=1 && S<=D && D%S==0
  // C2: S <= D
};

// ---- 索引层：rank <-> (h, t, c) ----
struct KvIndex {
  static int32_t head_class(const KvRedundancy&, int32_t tp);
  static int32_t slice(const KvRedundancy&, int32_t cp, int32_t tp);   // t
  static int32_t replica(const KvRedundancy&, int32_t cp, int32_t tp); // c
  static bool    find_writer(const KvRedundancy&, int32_t h, int32_t t,
                             int32_t dp, int32_t* rank);   // c == 0
  static void    find_copies(const KvRedundancy&, int32_t h, int32_t t,
                             int32_t dp, std::vector<int32_t>* ranks);
};

// ---- 规范地址层：peer 无关的块身份（模型文档 §7）----
struct CanonicalBlock {
  static constexpr int64_t kTokensPerBlock = /* B_token, 全集群常量 */;

  // token 起点 -> 规范块号；与本地 block id 无关
  static int64_t of_token(int64_t token_start);
  // 本侧本地 block id <-> 规范块号（本侧映射表）
  static int64_t to_local(int64_t canonical_block);
  static int64_t to_canonical(int64_t local_block);

  // 规范逻辑地址线性化：规范块尺寸与 S / TP / 物理几何无关
  static uint64_t linear(int64_t b, int32_t g, int32_t tau, int32_t G,
                         uint64_t bytes_per_head);
};

// ---- 路由层：head_pairs ⊗ block_map ----
struct PdRoute {
  // 建链期：一次性枚举，结果与请求无关
  static Status build(const KvRedundancy& src, const KvRedundancy& dst,
                      std::vector<RouteEdge>* edges);

  // 请求期：规范块号代入
  static void bind(const RouteEdge&, Span<const int64_t> canonical_blocks,
                   int64_t layer, std::vector<ByteRegion>* regions);
};

struct RouteEdge {
  int32_t src_rank, dst_rank, dp_rank;
  int32_t head_lo, head_hi;   // 全局 head 半开区间
  int32_t src_slice;          // t_P
  int32_t dst_slice;          // t_D
};
```

配套改动：

1. `InstanceInfo` 增加 `kv_head_num`、`kv_split_size`、`cp_size`、`tokens_per_block`，使 `check_pd_topo` 能直接跑 `KvRedundancy::derive` 并在建链前报出 C1/C2 违规（含 `B_token_P == B_token_D`，模型文档 §7.6）。
2. `only_static_owner` 退出 planner：`owner_tp_rank` 回归"段身份"，写者去重由 `KvIndex::find_writer` 承担。
3. `logical_offset` 改为**规范块坐标**（模型文档 §7.2）：`b × BlockLogicalBytes + tau × G × bytes_per_head + g × bytes_per_head`，其中 `BlockLogicalBytes` 与 `S` 无关、两侧恒等；`resource_stride_bytes` 等物理几何只在 `bind_regions` 使用。block 路由的输入相应地从本地 block id 改为 `CanonicalBlock::to_canonical(local_block)`。

---

## 5. 分阶段落地

| 阶段 | 内容 | 风险 | 验证 |
|---|---|---|---|
| **S0** | 删除实现 A（base `merge_kv_blocks`）、实现 B（`push_route.*` + `push_route_test.cpp` + 无用 include） | **零**（不可达代码） | 全树编译 + 现有单测全绿 |
| **S1** | 新增 `KvRedundancy` / `KvIndex` 纯算术层；不改调用方；为 `(h,t,c)` 写判别性单测 | 低（纯新增） | 新增单测：`G∈{1,2,4,8,TP/2,2TP}` × `TP∈{1,2,4,8}` × `S∈[1,D]` 全枚举，断言"每字节恰好一个写者"、`N_rep == D/S` |
| **S2** | 用 `KvIndex` 替换分支 #1–#14；`merge_kv_blocks`(C) 与 `pull_kv_blocks`(D) 共用 `PdRoute::build` | 中 | `reshard_planner_test` 46 项 + 新增 `(h,t,c)` 用例 |
| **S3** | 引入**规范块层**：`CanonicalBlock` + `logical_offset` 改为规范块坐标；新增 `B_token_P == B_token_D` 校验。**此后 `S_P ≠ S_D` 才被允许** | 中高（触及 `expand_manifest` 与覆盖率校验的坐标系） | 先跑 `S_P = S_D` 回归（规范坐标下应等价于现状）；再上 `S_P ≠ S_D` 的折叠用例 |
| **S4** | 解除 `block_size × kv_split_size` 绑定（`llm_engine.cpp:653`），让规范块与物理资源彻底解耦 | 中（触及 BlockManager / prefix cache 哈希） | prefix cache 命中率与 `n_blocks` 语义回归 |
| **S5** | 把 `PdTopo` 升级为携带 `G/S/B_token` 的精确门禁；删掉 planner 侧的补丁式校验 | 低 | 配置矩阵负例测试 |

**S0 与 S1 相互独立，可并行**；S2 依赖 S1；S3 依赖 S1（不依赖 S2，但建议后置）；S4 依赖 S3；S5 依赖 S1。

**S3 的正确性论证（为什么这次不会重蹈上次的失败）**：上次的做法是**沿用既有的 `logical_offset` 语义（`span.logical_offset_bytes + repeat × span.logical_stride_bytes`）**，而那个 stride 由 `resource_stride_bytes` 派生、随本侧 `S` 变化。于是两个"坐标系"必然分叉，任何折叠公式都只能去**拟合**这个分叉。本次改为：**先定义 peer 无关的规范块尺寸 `BlockLogicalBytes`，再让双方描述符都声明在这一个坐标系里**；物理几何（含 `resource_stride_bytes`）下沉为 `bind` 阶段的实现细节。坐标系只有一个，折叠就退化为纯区间求交 —— 不需要拟合，也就不存在"公式写反"的空间。

---

## 6. 验证清单

- [ ] **[读码→实测]** 确认实现 A 不可达：在 base `merge_kv_blocks` 加 `LOG(FATAL)`，跑 PD 端到端，若从不触发则删除
- [ ] **[读码→实测]** 确认 `push_route.*` 仅单测调用：删除后全树编译通过
- [ ] `reshard_planner_test` 46/46 保持全绿
- [ ] 新增 `KvRedundancy` 单测覆盖 `G < TP`（`D_tp > 1`）与 `G >= TP`（`D_tp = 1`）两侧，以及 `S \| D` 的正反例
- [ ] 新增判别性负例：`TP=8, G=2, S=3`（`S ∤ D_tp`）必须在 `derive` 阶段被拒，而非等到 `"no source writer"`
- [ ] **S3 等价性回归**：`S_P = S_D` 时，规范块坐标下的 writer 选择与交织顺序必须与改造前**逐块一致**（用现有 46 个测试 + 端到端 block id 序列比对）
- [ ] **S3 折叠用例**：`S_P ≠ S_D`（`S_D \| S_P` 与 `S_P \| S_D` 两个方向）下"每逻辑字节恰好一个写者"成立，且 `bind` 出的物理区间逐字节等价于手工推演
- [ ] 删除 `rotate_dst_rank` 后，PD 吞吐不退化（负载均衡由划分承担）
- [ ] Ascend 端到端：MLA latent / indexer / SSM 三类缓存逐字节
- [ ] clang-format 20.1.6

---

## 7. 风险与未决

1. **S3 的改动面**：`logical_offset` 语义变更会同时触及 `expand_manifest`、`validate_coverage_for_sources`、`select_collapsed_writers`、`bind_regions` 四处。缓解办法是**先做等价性回归**（`S_P = S_D` 下必须与现状逐块一致），确认坐标系切换本身无副作用，再放开 `S_P ≠ S_D`。不要把两件事合在一次改动里。
2. **S4 与 prefix cache**：解除 `block_size × kv_split_size` 绑定会改变 BlockManager 的块尺寸与 `n_blocks` 语义，进而影响 block 哈希与 prefix 命中。S4 应与 S3 分开评估，且规范层可以先只作用于传输侧描述符（S3 不强制要求 S4）。
3. **`rotate_dst_rank` 的删除需实测**：它的注释是"Rotate the dst-worker traversal order in push_kv_blocks per …"（`disagg_pd_config.cpp:50`），可能承载过负载均衡的实测收益；删除前应确认收益是否已由 `select_sources` 的 ACTIVE 筛选覆盖。
4. **`rank_local_mapping` 的语义**：它是"目的 id 已按 rank 保存"的旁路开关。在 `(h,t,c)` 模型下应由调度侧直接产出符合 `t` 规则的 id 序列，该开关可随之消失 —— 但它跨了 scheduler / transfer 两层，改动面需独立评估。
5. **死代码删除的审计成本**：A/B 虽不可达，但可能被下游分支（未在本树启用的构建配置）引用。需在 `USE_NPU` / `USE_MLU` / `USE_DCU` / 无加速器四种配置下各编译一次。
6. **未纳入本方案的已知问题**（与路由正交，另行处理）：
   - `enable_mla` 分支优先于 CONV / SSM（`cache_layout_builder.cpp:393`），MLA 模型上后两者不可达；
   - `manifest` 无版本协商，对端版本不符时只有硬失败路径；
   - `push_kv_blocks` 中 plan 为空但 mapping 非空时静默 `continue`（`mooncake_kv_cache_transfer.cpp:705-707`），丢数据无告警；
   - 描述符路径（`bind_outgoing_regions`）与 strided 路径（`move_memory_groups`）并存，由 S3 统一到同一坐标系后自然收敛。
