# PD 传输重构方案（基于 KV 冗余模型）

## 溯源与状态

- 日期：2026-09-17
- 基线 commit：`200939593`
- 本文性质：**重构方案**（设计 + 迁移计划，不含代码改动）
- 依据：`kv_redundancy_model_and_pd_routing_20260917.md`（模型）、`pd_routing_simplification_20260917.md`（现状盘点）
- 验收场景：`pd_route_verification_plan_glm53flash_20260917.md`（GLM 5.3 flash，P `TP8+DCP4` → D `DP4+TP2+DCP2`）
- 证据口径：**[读码]** 静态阅读

> **本文已按 review 修订**（详见验证文档第一部分）：`S_eff` 按组派生（F1）、`G` 按组（F2）、边表改为拓扑局部 + DP 偏移（F3）、补本地行↔规范块映射（F4）、`B_token` 绑定已有字段（F5）、`bind` 分桶（F7）、边表缓存键（F8）、`repeat/stride` 保留（F9）。

---

## 0. 目标与非目标

**目标**：让 PD 传输的"谁给谁搬哪一段"变成**由拓扑元组唯一决定的纯算术结果**，从而删除现有的特例分支、死代码与两套并行实现。

**非目标**：
- 不改 `MooncakeTransferEngine` 之下的 RDMA 会话/注册/搬运原语；
- 不改 XTensor 的页映射机制（它只是物理层的一种实现）；
- 不改 prefix cache 的 `remote_shared_num` 语义；
- 不动 sequence-scoped 缓存（SSM / CONV / LINEAR / EMBEDDING）的 slot 语义（见 §6）。

---

## 1. 核心简化：一句话 + 三个删除

> **把 manifest 从「逻辑归属的描述」降级为「物理缓冲的目录」；逻辑归属完全由拓扑元组 `(DP, CP, TP, G, S, B_token)` 决定，两侧独立推出同一张边表。**

由此得到三个删除：

| # | 删除 | 理由 |
|---|---|---|
| **D1** | **逻辑归属的全部表达**：`LogicalShardDescriptor` / `LogicalSpan.logical_*` / `owner_tp_rank` / `LogicalShardKind` | 归属 = `(h, t, c)` 三个纯算术索引，不需要描述符承载 |
| **D2** | **区间求交式规划器**：`ReshardPlanner` 的 `select_sources` / `select_collapsed_writers` / `validate_writer_coverage` / `has_logical_overlap` / `expand_manifest` / 全部 `supports_*` `same_partition*` `kv_split_spans_cp_and_tp` | 覆盖不变量在模型下按构造成立，退化为一条 O(1) 算术断言 |
| **D3** | **modulo 时代的两套配对实现**：base `KVCacheTransfer::merge_kv_blocks`、`push_route.{h,cpp}`、`filter_kv_split_infos`、`rotate_dst_rank`、`LLMEngine::pull_kv_blocks` 的取模路由 | 全部由同一张边表替代 |

**净效果**：路由逻辑从「三套实现 + 18 类特例分支 + 两处死代码」收敛为「一个纯算术模块 + 一张边表 + 一个绑定函数」。

---

## 2. 新分层

```
┌─ L0 拓扑层 KvTopology ─────────────────────────────────────────┐
│  纯数据：DP/CP/TP/G/S/B_token。由启动配置与 InstanceInfo 提供。 │
└────────────────────────┬───────────────────────────────────────┘
                         │ derive()  ── C1 整除 / C2 上限校验（fail-fast）
┌────────────────────────▼───────────────────────────────────────┐
│ L1 冗余层 KvRedundancy + KvLayoutIndex                         │
│  D_tp / H_l / H_c / D / N_rep；rank <-> (h, t, c)              │
│  纯算术，无状态，无 IO。可被单测穷举覆盖。                     │
└────────────────────────┬───────────────────────────────────────┘
                         │ build()
┌────────────────────────▼───────────────────────────────────────┐
│ L2 边表层 PdRouteTable                                         │
│  RouteEdge{src, dst, dp, head[lo,hi), t_P, t_D}                │
│  建链期算一次，与请求无关。                                    │
└────────────────────────┬───────────────────────────────────────┘
                         │ bind()
┌────────────────────────▼───────────────────────────────────────┐
│ L3 绑定层 RouteBinder                                          │
│  规范块列表 + 本侧/对侧物理目录 → ByteRegion 列表              │
│  规范坐标（peer 无关）↔ 物理坐标（peer 相关）的唯一换算点      │
└────────────────────────┬───────────────────────────────────────┘
                         │ move_memory_regions(opcode)
┌────────────────────────▼───────────────────────────────────────┐
│ L4 搬运层（现有，不动）MooncakeTransferEngine                   │
└────────────────────────────────────────────────────────────────┘
```

**关键分界线在 L2/L3 之间**：L2 以上只认规范块号与 head 类（peer 无关）；L3 以下只认 buffer id 与字节偏移（peer 相关）。**这条线一旦划清，"折叠公式写反"这一类问题在结构上不可能发生** —— 因为两侧共用同一个坐标系。

---

## 3. 关键数据结构

### 3.1 L0/L1：拓扑与冗余

**`KvTopology` 分两层**：实例级（并行与切分）+ 每组级（head 几何）。原因见 §6 与验证文档 F2：同一个 rank 上不同 cache 组的 `G` 不同（GLM5-next 的 MLA/indexer `G=1`，KDA `G=64`），而 `TP/CP/S` 是实例共享的。

```cpp
struct KvTopology {                      // 实例级
  int32_t dp_size = 0, cp_size = 0, tp_size = 0;
  int32_t kv_split_size = 0;             // S
  int32_t tokens_per_block = 0;          // B_token（= CacheTensorManifest::block_token_capacity）
};

struct GroupTopology {                   // 每个 (namespace, role, group_id)
  int32_t global_head_count = 0;         // G
  uint64_t head_bytes = 0;               // 一个 head 的字节数
  bool sequence_scoped = false;          // SSM / CONV / LINEAR / EMBEDDING：无块维
  bool full_sequence_replica = false;    // indexer kPool 等"有块维但必须全序列"的组
};

struct KvRedundancy {
  int32_t Hl = 0;      // 每 rank 本地 head 数   = max(G / TP, 1)
  int32_t D_tp = 0;    // TP 冗余度              = max(TP / G, 1)
  int32_t Hc = 0;      // head 类数              = TP / D_tp
  int32_t D = 0;       // 复合冗余度             = CP * D_tp
  int32_t S_eff = 0;   // 本组实际切分宽度
  int32_t N_rep = 0;   // 残余冗余度             = D / S_eff

  // C1: (G % TP == 0 || TP % G == 0)
  //     sequence_scoped / full_sequence_replica / D == 1  => S_eff = 1
  //     S <= D 且 D % S == 0                              => S_eff = S
  //     其余                                              => 报错（见 §6）
  static Status derive(const KvTopology&, const GroupTopology&, KvRedundancy*);
};

class KvLayoutIndex {
 public:
  KvLayoutIndex(const KvTopology&, const GroupTopology&, const KvRedundancy&);

  int32_t rank(int32_t dp, int32_t cp, int32_t tp) const;
  int32_t head_begin(int32_t h) const;      // h * Hl
  int32_t head_end(int32_t h) const;        // (h+1) * Hl

  int32_t head_class(int32_t tp) const;     // tp / D_tp
  // t 必须等于运行时的 DCP rank（见 §3.1 下方说明），只有两种形状：
  //   (a) S <= CP 且 CP % S == 0:  t = cp / (CP/S)
  //   (b) S == CP * TP:            t = cp * TP + tp
  int32_t slice(int32_t cp, int32_t tp) const;
  int32_t replica(int32_t cp, int32_t tp) const;

  // 写者去重：c == 0 的唯一 rank
  bool writer_of(int32_t dp, int32_t h, int32_t t, int32_t* rank) const;
  // 副本全填：c ∈ [0, N_rep)
  void copies_of(int32_t dp, int32_t h, int32_t t,
                 std::vector<int32_t>* ranks) const;
};
```

> **修正（2026-09-18，工作日志第 5 轮）**：`slice` 原来写作 `(cp*D_tp + tp % D_tp) % S_eff`，**与运行时契约不符**。
> 运行时的规范地址由 `KVShardLayout::globalize` 定义：`canonical = row * S + dcp_rank`
> （`KVShardLayout::owner_of(b) = b % S`），而 `dcp_rank` 由 `ContextParallelTopology` 给出：
> (a) `S <= cp_size && cp_size % S == 0` ⇒ DCP 划分 PCP，`dcp_rank = cp_rank / (cp_size/S)`；
> (b) `S == cp_size*tp_size` ⇒ DCP 覆盖整个 DP-local 域，`dcp_rank = cp_rank*tp_size + tp_rank`；
> 其它形状会让 `ContextParallelTopology` 的 `CHECK` 直接失败。
> NPU 解码路径就是这么用的（`qwen_dcp_attention.cpp`：`KVShardLayout(block_size, dcp_group.world_size(), dcp_group.rank())`）。
> 受影响的是"哪个 rank 属于哪个切片/副本"（pilot：`slice = cp_rank`，写者 `(cp=s, tp=0)`），
> 不变量与规模（`N_rep = D/S`、`Hc*Hl = G`、边表条数）不变。

### 3.2 L1：规范地址

```cpp
struct CanonicalBlock {
  int32_t tokens_per_block = 0;             // = manifest.block_token_capacity
  int32_t split = 1;                        // 本侧 S_eff

  // 规范块号由 token 区间导出，与 S / TP / 物理几何无关
  int64_t block_of_token(int64_t token_index) const;
  int64_t token_begin(int64_t block) const;   // block * tokens_per_block
  int64_t token_end(int64_t block) const;

  // 本地物理行 <-> 规范块（DCP 分配：一个逻辑 block 的 split 个规范块
  // 由本组 split 个 rank 各持一个）
  int64_t local_row(int64_t canonical_block) const {   // canonical / split
    return canonical_block / split;
  }
  int64_t canonical_of_row(int64_t row, int32_t slice_rank) const {
    return row * split + slice_rank;
  }
};

// (block, head, tau) -> 规范线性偏移；BlockLogicalBytes 与 S / 物理几何无关。
// 内部类型：只被 RouteBinder 使用，不对外暴露。
struct CanonicalAddr {
  int32_t kv_head_num = 0;
  uint64_t head_bytes = 0;                     // 一个 head 的字节数
  int64_t tokens_per_block = 0;

  uint64_t block_bytes() const {
    return static_cast<uint64_t>(tokens_per_block) * kv_head_num * head_bytes;
  }
  uint64_t linear(int64_t block, int32_t head, int32_t tau) const {
    return static_cast<uint64_t>(block) * block_bytes() +
           static_cast<uint64_t>(tau) * kv_head_num * head_bytes +
           static_cast<uint64_t>(head) * head_bytes;
  }
};
```

### 3.3 L2：边表

```cpp
struct RouteEdge {                    // 全部索引都是 DP 组内局部量
  int32_t src_local_rank = 0;         // = cp_P * TP_P + tp_P
  int32_t dst_local_rank = 0;         // = cp_D * TP_D + tp_D
  int32_t head_begin = 0, head_end = 0;   // 全局 head 半开区间
  int32_t src_slice = 0;                  // t_P
  int32_t dst_slice = 0;                  // t_D
};
// 全局 rank = dp * (CP * TP) + local_rank；DP 配对由调度层给出
// （TransferKVInfo.dp_rank），不进边表 —— 因此一张边表服务全部 DP 对，
// 边数不随 DP 增长，且 P(DP1)→D(DPn) 的展开天然可表达。

class PdRouteTable {
 public:
  // 只需两侧拓扑；两侧各算一次，结果逐边相同（可互为断言）。
  // 缓存键 = (本侧 KvTopology, 对侧 KvTopology)：同一拓扑的多个 peer 复用。
  static Status build(const KvTopology& src, const GroupTopology& src_group,
                      const KvTopology& dst, const GroupTopology& dst_group,
                      std::vector<RouteEdge>* edges);

  // 覆盖不变量：模型下按构造成立，此处只做 O(#edges) 断言
  static Status validate(const std::vector<RouteEdge>& edges,
                         const KvTopology& src, const GroupTopology& src_group,
                         const KvTopology& dst, const GroupTopology& dst_group);
};
```

`build` 的语义（即模型文档 §4.3）：

```
for (hP,hD,lo,hi) in head_pairs(src, dst):
  for tP in [0, S_P):
    src = src_idx.writer_of(dp, hP, tP)
    for tD in block_map(tP):                 // (tP + k*S_P) % S_D, k ∈ [0, m)
      for cD in [0, N_rep_D):
        dst = 唯一 rank，h(tp)=hD ∧ t(cp,tp)=tD ∧ c(cp,tp)=cD
        emit {dp, src, dst, lo, hi, tP, tD}
```

### 3.4 L3：绑定

```cpp
// 物理目录：manifest 精简后的形态，只描述"本侧哪些 buffer、多大、怎么寻址"
struct BufferDirectoryEntry {
  int64_t  buf_id = 0;
  int32_t  group_id = 0;
  int32_t  role = 0;                 // KVCacheTensorRole
  uint64_t resource_count = 0;       // 本侧物理行数
  uint64_t resource_stride_bytes = 0;// 一行（= 一个规范块）的字节数
  uint64_t buffer_bytes = 0;
  bool     explicit_offsets = false; // XTensor 页映射
};

class RouteBinder {
 public:
  // canonical_blocks: 本请求涉及的规范块号（升序）
  // local/remote    : 两侧物理目录
  // local_offsets/remote_offsets: explicit_offsets 时的页内字节基点
  // local_split/remote_split    : 两侧的 S_eff（用于 row <-> canonical 映射）
  static Status bind(const std::vector<RouteEdge>& edges,
                     Span<const int64_t> canonical_blocks,
                     int64_t layer,
                     const BufferDirectory& local,
                     const BufferDirectory& remote,
                     Span<const uint64_t> local_offsets,
                     Span<const uint64_t> remote_offsets,
                     int32_t local_split, int32_t remote_split,
                     std::vector<ByteRegion>* regions);
};
```

`bind` 的实现要点：

1. **按 `canonical % S_eff` 预分桶**请求的规范块列表（F7），使每条边只遍历属于自己的桶 ⇒ 复杂度 `O(#blocks)` 而非 `O(#edges × #blocks)`；
2. 对每条边、每个属于它的规范块：
   - `local_row = canonical / local_split`（若 `local_split == 1` 则 `local_row = canonical`）
   - `remote_row = canonical / remote_split`
   - `local_off = local.explicit_offsets ? local_offsets[local_row] : local_row * local.resource_stride_bytes`（remote 同理）
   - head 区间与块内 token 区间映射到资源内偏移（`CanonicalAddr`）
3. **块内 token 维仍用 `repeat_count / local_stride / remote_stride` 压缩**（F9）——被删除的只是跨资源的区间求交与折叠公式。

**没有任何区间求交、没有 sweepline、没有折叠公式。**

---

## 4. 数据面收敛：PUSH / PULL 合一

现状是两条各写一遍的路：

| | 现状 | 新 |
|---|---|---|
| PUSH | `merge_kv_blocks`(C) → `bind_outgoing_regions` → `move_memory_regions(WRITE)` | `PdRouteTable` → `RouteBinder::bind` → `move_memory_regions(WRITE)` |
| PULL | `LLMEngine::pull_kv_blocks` 取模 → `append_buffer_mappings` → `move_memory_groups(READ)` | `PdRouteTable` → `RouteBinder::bind` → `move_memory_regions(READ)` |

合并为一个入口：

```cpp
Status KVCacheTransfer::transfer(const std::vector<RouteEdge>& edges,
                                 Span<const int64_t> canonical_blocks,
                                 MoveOpcode opcode);   // WRITE=PUSH, READ=PULL
```

**方向只影响 opcode，不影响任何配对计算。** 这直接消除了"PUSH 与 PULL 对同一问题给两套答案"的现状。

---

## 5. 控制面收敛

### 5.1 建链只连有边的对端

现状：`LLMEngine::link_cluster` 让每个 D worker 与**全部** P worker 建链，再用 `CachePeerMode::ACTIVE / PLAN_ONLY` 区分。新方案：

```
边表算完后，只对 edges 非空的对端建链；无边的对端完全不建链。
```

⇒ **`CachePeerMode` 三态退化为"连 / 不连"两态**，`PLAN_ONLY` 及其分支、`cache_peer_links_[addr].mode` 的判断全部删除。

**边表缓存键 = `(本侧 KvTopology, 对侧 KvTopology)` + group**，不是 peer 地址。一个 P 实例若服务多个不同拓扑的 D 实例（异构舰队），同一拓扑的多个 peer 复用同一张表。

**DP 配对属调度层**：P(DP1)→D(DPn) 的 1→n 展开由 `TransferKVInfo.dp_rank` 决定，不进边表。绑定阶段把局部 rank 加上 DP 偏移：

```
src_global = src_dp * (CP_P * TP_P) + edge.src_local_rank
dst_global = dst_dp * (CP_D * TP_D) + edge.dst_local_rank
```

### 5.2 计划不再需要跨 RPC 传递

现状：D 侧算出 writer 集后通过 `SetCachePeer` 把 mode 告诉 P，P 侧再 `build_outgoing_plan` 生成并缓存 `ReshardPlanTemplate`。

新方案：`PdRouteTable::build` 是**两侧拓扑的纯函数**，两侧各自算一次即可得到逐边相同的边表，无需传递计划。控制面只剩：

| RPC | 内容 | 频率 |
|---|---|---|
| `GetCacheLayoutManifest` | **精简后的物理目录**（buffer 表 + 拓扑元组），仅用于寻址与校验 | 建链时一次 |
| `SetCachePeer` | **仅表示"我要跟你建链"**（不再携带 mode 与 plan） | 建链时一次 |

`ReshardPlanTemplate` / `StridedRegionTemplate` / `RequestRegionBinder` 全部删除；`MooncakeTransferEngineCore` 的 `cache_peer_links_` 退化为一个 `std::unordered_set<std::string>`（已连接的 addr）。

### 5.3 门禁提前到建链前

`PdTopo {dp_size, tp_size}` 扩展为完整 `KvTopology` + 每组 `GroupTopology`（含 `G`、`S`、`cp_size`、`tokens_per_block`），由 `KvRedundancy::derive` 在**建链前**拒绝 C1/C2 违规、`tokens_per_block_P ≠ tokens_per_block_D`、以及 `S_eff` 两侧不可配对（`S_eff_P ∤ S_eff_D` 且反之不成立）的组合。

⇒ 违规配置不再等到覆盖率校验才以 `"no source writer"` 的形式暴露。

---

## 6. 适用范围边界

### 6.1 `S_eff` 按组派生：显式声明优先，其余不匹配直接报错

C2（`S ≤ D`）必须作用在**每个 cache 组**上，因为 `D = CP × D_tp` 依赖该组的全局 head 数 `G`，而 `S` 是实例级的。GLM5-next 就是反例：KDA 状态 `G = kda_num_heads = 64`，`TP = 8` ⇒ `D = 1`，而实例 `S = 4` —— 按实例级 C2 会被误拒。

已实现的派生规则（`kv_redundancy.cpp`）：

```
sequence_scoped            => S_eff = 1    无块维可切
full_sequence_replica      => S_eff = 1    语义性全序列复制（见下）
D == 1                     => S_eff = 1    无冗余可消
S <= D 且 D % S == 0       => S_eff = S
其余                        => 报错          不做静默降级
```

语义：**能被冗余度吸收的缓存才切分；不能被吸收的缓存整体复制到每个 rank**（`S_eff = 1 ⇒ t ≡ 0`，每 rank 持完整序列）。但"能不能被吸收"只解释前三种情形；**配置与冗余度不匹配（`S>1` 又除不尽 `D`）是配置错误，应当显式报错**——静默退 1 会让使用者以为更宽的切分已经生效。若某个组确实只能全序列保留，必须由 `full_sequence_replica` **声明**，把原因写进注释。

**`full_sequence_replica` 的由来（indexer kPool）**：`groups.kv` 里的 MLA latent 与 indexer 同属 `BlockType::KV`、同 `group_id`，但 indexer 的 top-k 需要读取**全序列**的 gate/valid，因此它是有块维、却必须整序列复制的组。这带来两个后果：

1. `S_eff` **不能**从 `G/TP/CP/S` 推导出来（该组 `G=1, D=8`，按规则会得到 `S_eff=4`，而实际必须是 1），所以它必须是 `GroupTopology` 上的显式声明，而不是白名单或推导；
2. 原先设想的 `EXPECT_EQ(is_kv_split_cache_block_type(t), S_eff(group) > 1)`（验证文档 F1）**不成立**：同一个 `BlockType::KV` 组内 MLA latent 的 `S_eff=4`、indexer 的 `S_eff=1`。该断言作废，`is_kv_split_cache_block_type` 与 `S_eff` 不是同一件事，改造后由 `(namespace, role, group_id)` 粒度的 `GroupTopology` 取代。

行 ↔ 规范块的映射仍然统一为 `row = canonical / S_eff`：`S_eff=1` 时退化为 `row = canonical`，这正是 indexer 张量行数 `= n_blocks × S`（全部规范块）的含义。

### 6.2 按 scope 分类

| scope | 组 | `t` 维度 | 说明 |
|---|---|---|---|
| `BLOCK` | KV / SWA / C4 / C128 / MLA latent / INDEX | 有 | 主路径，`S_eff` 由 §6.1 推导 |
| `BLOCK` | spec-draft（MTP）缓存 | 有 | 独立 namespace 的第二套 `(G, TP, S_eff)` |
| `SEQUENCE` | SSM / CONV / LINEAR / EMBEDDING | **`t ≡ 0`** | per-sequence slot，只有 `h` 与 `c` 两个维度；通常 `G ≥ TP` ⇒ `D_tp = 1` ⇒ `S_eff = 1` |

对 `SEQUENCE` 组，模型只做一件事：**把 `owner_tp_rank` 去重替换为 `c == 0` 判定**，不引入 `t`。若某天出现 `G < TP` 的 sequence-scoped 缓存，`S_eff` 规则同样适用（此时 `D > 1`、可切分）。

---

## 7. 与现有代码的对应关系

### 7.1 删除

| 目标 | 位置 |
|---|---|
| `push_route.{h,cpp}` + `push_route_test.cpp` + 无用 include | `kv_cache_transfer/push_route.*`、`tests/.../push_route_test.cpp`、`mooncake_kv_cache_transfer.cpp:30` |
| base `merge_kv_blocks`（modulo 路由，约 108 行） | `kv_cache_transfer.cpp:266-373` |
| `rotate_dst_rank` | `kv_cache_transfer.cpp:199-210` + `.h:73` |
| `filter_kv_split_infos` | `kv_cache_transfer.cpp:145-196` + `.h:56` |
| `LLMEngine::pull_kv_blocks` 的取模配对 | `llm_engine.cpp:814-843` |
| `select_sources` / `select_collapsed_writers` / `validate_writer_coverage` / `validate_coverage_for_sources` / `has_logical_overlap` / `expand_manifest` / `validate_source_instance` / `validate_tensor_pair` / `compact_planned_regions`，以及 `bind_regions` 里的**区间求交与折叠**逻辑 | `reshard_planner.cpp` 主体。注意：`bind_regions` 的 `repeat_count / local_stride / remote_stride`（**块内 token 维**压缩）**保留**，只是被 `RouteBinder` 重新实现 |
| `same_partition` / `same_partition_sizes` / `kv_split_spans_cp_and_tp` / `supports_kv_split_topology` / `supports_partition_layout` / `supports_partition_pair` / `validate_compatibility` | `reshard_planner.cpp:80-139` |
| `CoverageKey` / `RegionGroups` / `AtomicLogicalRegion` / `PlannedAtomicRegion` / `BoundRegion` | `reshard_planner.cpp:33-66` |
| `LogicalShardKind` / `LogicalSpan` / `LogicalShardDescriptor` / `owner_tp_rank` | `logical_cache_layout.h` |
| `CacheTensorManifest` 的 `shard` / `logical_*` 字段 | `cache_layout.h:43-68` |
| `only_static_owner` 及其 3 个调用点 | `reshard_planner.cpp:155,479,854,869` |
| `CachePeerMode::PLAN_ONLY` 及其分支 | `mooncake_transfer_engine.{h,cpp}` |
| `ReshardPlanTemplate` / `StridedRegionTemplate` / `ExplicitResourceMapping` 的 plan 依赖 | `reshard_planner.h:24-81` |
| `rank_local_mapping` / `has_rank_preserving_kv_groups` | `disagg_pd_scheduler.cpp:162-173,668` |
| `RemoteWorker::pull_kv_blocks` 路径的 kv_split 无关校验 | `kv_cache_transfer.cpp:118-133` |

### 7.2 修改

| 目标 | 位置 | 改法 |
|---|---|---|
| `MooncakeKVCacheTransferBase::merge_kv_blocks` | `mooncake_kv_cache_transfer.cpp:635-659` | 退化为按边表分组 |
| `bind_outgoing_regions[_explicit]` | `mooncake_transfer_engine.cpp:493-548` | 退化为查表 + 偏移计算 |
| `push_kv_blocks` / `pull_kv_blocks` | `mooncake_kv_cache_transfer.cpp:607-720, 854-878` | 各自调用统一入口 `transfer(opcode)` |
| `validate_transfer_mappings` | `kv_cache_transfer.cpp:33-93` | 删除 kv_split 覆盖区间校验，只留 group 唯一性 |
| `PdTopo` / `check_pd_topo` | `pd_topology_guard.{h,cpp}` | 扩为 `KvTopology`，调用 `KvRedundancy::derive` |
| `CacheRegistrationContext` / `publish_cache_layout` | `mooncake_kv_cache_transfer.cpp:154-221, 409-445` | 只发布物理目录 + 拓扑元组；删除 spec fingerprint 拼接与 `layout_generation` 的语义负担 |
| `InstanceInfo` | `common/types.h:225` | 增加 `kv_head_num` / `kv_split_size` / `cp_size` / `tokens_per_block` |
| `configure_cache_layout` | `mooncake_kv_cache_transfer.cpp:154-221` | 只算物理几何 |

### 7.3 保留不动

- `MooncakeTransferEngine` 的会话、注册、`move_memory_regions`；
- `GlobalXTensor` 与 `explicit_resource_offsets`（物理层的一种实现，由 `BufferDirectoryEntry::explicit_offsets` 承接）；
- spec-draft 双布局机制（作为第二个 namespace）；
- `remote_shared_num` 与 prefix cache 游标推进；
- `is_spec_draft` 在 `push_kv_blocks` 中的 layout 选择语义（改为选第二套 `KvTopology`）。

### 7.4 新增

| 新增 | 规模 |
|---|---|
| `KvTopology` / `KvRedundancy` / `KvLayoutIndex` | 约 200 行，纯算术 |
| `CanonicalBlock` / `CanonicalAddr` | 约 60 行 |
| `PdRouteTable`（build + validate） | 约 150 行 |
| `RouteBinder` | 约 120 行 |
| `BufferDirectory` | 约 80 行 |

---

## 8. 迁移阶段与验收

| 阶段 | 内容 | 风险 | 验收 |
|---|---|---|---|
| **S0** | 删除 §7.1 中不参与运行时的死代码（`push_route.*`、base `merge_kv_blocks`、无用 include） | 零 | 四种构建配置全树编译；现有单测全绿 |
| **S1** | 新增 L0/L1（`KvTopology` / `GroupTopology` / `KvRedundancy` / `KvLayoutIndex` / `CanonicalBlock`）与穷举单测，不改调用方 | 低 | `G∈{1,2,4,8,16,32,64}` × `TP∈{1,2,4,8}` × `CP∈{1,2}` × `S` 全枚举：(a) 每 `(h,t)` 恰一个 `c=0` 写者；(b) `N_rep == D/S_eff`；(c) `H_c·H_l == G`；(d) 负例 `TP=8,G=2,S=3` 被拒；(e) `S_eff` 与 `is_kv_split_cache_block_type` 在 GLM5-next 的 5 个组上一致（验证文档 T1/T2） |
| **S2** | 新增 L2/L3（`PdRouteTable` / `RouteBinder` / `BufferDirectory`） | 中 | **GLM 5.3 flash 场景边表 golden 通过（验证文档 T3）**；拓扑矩阵下 `validate` 全覆盖不变量成立；T4 字节 golden（`S_P≠S_D` 双向、`explicit_offsets`、checkpoint 子单元）逐字节正确 |
| **S2'** | host 侧 mock 端到端：真实内存 + 本地 memcpy 代替 RDMA | 低 | **验证文档 T5 全绿**（含判别性反例）—— 这是"GLM 5.3 flash 尚不支持 PD 分离"约束下的主要验收手段 |

> **S2 的原定验收"与旧 `select_sources` 逐边一致、`S_P=S_D` 时与 `bind_outgoing_regions` 逐字节一致"已放弃**，原因是旧路径在目标配置下根本无法表达：MLA 走 `describe_replicated_tensor`（整行一个 span、`owner_tp_rank=0`），而 `select_sources` 的 `same_partition` + `only_static_owner` 去重在 `TP8 + DCP4` 下要么选中 0 个写者（`kv_split_rank = rank % S`，即运行时 DCP 分组）要么选中 2 个写者（fallback `rank/(world/kv)`）。旧路径支持的形状（`G ≥ TP` 且 `S=1`、`G < TP` 且 `S=1`、sequence-scoped）仍可比对，属可选补充，不构成 S2 验收。
| **S3** | 数据面切到 `transfer(opcode)`：先切 PULL，再切 PUSH（PUSH 有 layer synchronizer，最后切） | 中 | PD 端到端字节正确；PULL/PUSH 结果一致 |
| **S4** | 删除 D1/D2/D3 全部旧路径与 `rank_local_mapping`；`SetCachePeer` 去掉 mode/plan | 中 | 全树编译 + 单测 + 端到端 |
| **S5** | 门禁提前（`PdTopo` → `KvTopology`）；建链只连有边的对端 | 低 | 配置矩阵负例；建链 RPC 数下降可观测 |

**关键顺序原则**：S2 必须"新旧并行 + 逐边/逐字节比对"通过后才进入 S3；S3 先切 PULL（无 layer synchronizer，失败面小）。**不要在新旧路径切换的同时改变量语义**（例如同时解除 `block_size × kv_split_size` 绑定）——后者单列为 S6，独立评估。

| 阶段 | 内容 | 风险 |
|---|---|---|
| **S6**（可选，独立） | 解除 `block_size × kv_split_size`（`llm_engine.cpp:653`），让规范块与物理资源彻底解耦；`B_token` 收敛为调度侧基础块大小 | 中（触及 BlockManager / prefix cache 哈希） |

---

## 8.1 实施进展

### S0 已完成（2026-09-17）

删除的不可达代码：

| 项 | 证据 |
|---|---|
| `xllm/core/framework/kv_cache_transfer/push_route.{h,cpp}` | 全树生产调用点为 0；`mooncake_kv_cache_transfer.cpp` 只 `#include` 不调用 |
| `tests/core/framework/kv_cache_transfer/push_route_test.cpp` + 两处 CMake 条目 | 随库一并删除 |
| `KVCacheTransfer::merge_kv_blocks` 的基类实现（`kv_cache_transfer.cpp:266-373`，108 行 modulo 路由） | 唯一子类 `MooncakeKVCacheTransferBase` 已 `override`；工厂只构造 Mooncake 实现，故基类实现不可达。现改为 **纯虚**，杜绝后续后端再继承这段死逻辑 |
| `kv_cache_transfer.h` 中 `merge_kv_blocks` 的声明 | `= 0`，并加注释说明为何必须由后端实现 |

验证：`kv_cache_transfer.h` 与 `mooncake_kv_cache_transfer.h` 的参数列表规范化后**逐字符相同**（`virtual`/`override`/`= 0` 之外无差异），因此 `MooncakeKVCacheTransferDefault` 仍是具体类。

### S1 已完成（2026-09-17）

新增 `xllm/core/framework/kv_cache_transfer/kv_redundancy.{h,cpp}`（L0/L1，约 220 行）：

- `KvTopology`（实例级：dp/cp/tp/kv_split/tokens_per_block）
- `GroupTopology`（每组：global_head_count / head_bytes / sequence_scoped）
- `KvRedundancy::derive` —— 校验 C1（`G % TP == 0 || TP % G == 0`）与 C2（`S_eff` 必须整除并 ≤ `D`），派生 `Hl / D_tp / Hc / D / S_eff / N_rep`，并断言 `Hc × Hl == G`
- `KvLayoutIndex` —— `rank ↔ (h, t, c)` 双向映射、`writer_of`（`c == 0` 唯一写者）、`replicas_of`（`N_rep` 个副本）
- `CanonicalBlock` —— 规范块 ↔ 本地物理行（`local_row = canonical / S_eff`）

**`S_eff` 派生规则**（本方案的核心修正，见 §6.1）：

```
sequence_scoped            => S_eff = 1     （无块维可切）
D == 1                     => S_eff = 1     （无冗余可消）
S <= D 且 D % S == 0       => S_eff = S
其余                        => 报错（不做静默降级）
```

**该层零非标准库依赖**，可被穷举单测完全覆盖。

### S2 已完成（2026-09-18）

新增 L2/L3（`xllm/core/framework/kv_cache_transfer/`）：

| 文件 | 内容 |
|---|---|
| `pd_route_table.{h,cpp}` | `RouteEdge`（DP 组内局部 rank）、`PdRouteTable::build`（`head_pairs ⊗ block_map`，源侧 `c==0` 去重、目的侧副本全遍历）、`PdRouteTable::validate`（逐 `(目的 rank, 源片, head)` 覆盖不变量 + 源侧唯一写者） |
| `route_binder.{h,cpp}` | `RouteRegion`、`BufferDirectoryEntry`、`BufferDirectory`、`PeerCacheView`、`RouteBinder::bind`（规范块 → 物理字节区间，按 `t` 预分桶，`units_per_resource` 承载块内 token / checkpoint 行压缩，`explicit_offsets` 支持 XTensor） |
| `kv_redundancy.{h,cpp}` | `GroupTopology` 增加 `full_sequence_replica`（§6.1） |

配套单测（host 侧，不依赖 NPU/RDMA）：

| 测试 | 结果 |
|---|---|
| `tests/core/framework/kv_cache_transfer/pd_route_test.cpp` | ✅ **12/12 PASSED**：T3 golden（MLA / KDA / indexer）、拓扑矩阵不变量、T4 字节 golden（汇聚、发散、`explicit_offsets`、checkpoint 子单元）、T5 mock 端到端（真实内存 + memcpy，含"远端行基点错位必须被校验抓住"的判别性反例） |
| `tests/core/framework/kv_cache_transfer/kv_redundancy_test.cpp` | ✅ **11/11 PASSED**（新增 full-sequence-replica 用例） |

边表规模口径：F3 去掉 DP 维后，目标场景的边表本体是 **MLA 4 条 / KDA 8 条**；把同一张表套到 4 个 `dst_dp` 上才是交接文档所说的 **16 / 32 条有效边**。两者都在测试中固定。

### S3-2 已完成（2026-09-18）：manifest → `PeerCacheView` 适配器

新增 `cache_directory.{h,cpp}`，即 wire 表示（manifest）与 L3 物理视图之间的那道缝：

| 类型 | 作用 |
|---|---|
| `CacheTensorDeclaration` | 模型侧声明：`(cache namespace, role, group id)` + 该族的 `KvTopology` + `GroupTopology`。manifest 只描述字节，表达不了 `G` / `head_bytes` / `sequence_scoped` / `full_sequence_replica` / `B_token`，这些必须声明后与描述符对账 |
| `CacheRowBases` | 页映射（XTensor）张量的按物理行基点；非页映射张量给了就报错 |
| `PeerDirectory::describe` | 解释 manifest 并对账，产出每张 cache tensor 的 `PeerCacheView` |

对账项（每项都有负例）：坐标与声明拓扑一致；`resource_scope` 与 `sequence_scoped` 一致；`units_per_resource`（BLOCK 取 `block_token_capacity`、SEQUENCE 取 `physical_rows_per_resource`）与声明一致；span 的 `bytes_per_region` 即 `head_bytes`；span 数等于派生的 `H_l`；span 的 global head 区间恰是一个 head class（`owner_tp_rank == class × D_tp`，且 MAIN 下必须就是该 rank 的 class）；`head_bytes` 声明值非 0 时必须相符；页映射与 `row_bases` 一一对应。

**两个实现层面的发现**：

1. **COMPOSITE（CONV）描述符无法用"每条边一个 head 区间"表达**：`describe_conv` 把 `conv_key_a` / `conv_key_b` / `conv_value` 打进同一行，各 component 的 head 空间彼此独立、物理偏移还带 component 偏移，而 `RouteBinder` 的寻址只认"一个 head 区间 × 单元步长"。适配器**显式拒绝**（不静默降级）。S3-4 需要二选一：COMPOSITE 组继续走旧 planner，或给 `RouteEdge` / `PeerCacheView` 增加 per-component 字节偏移。**（第 4 轮补充：该分支只在 `enable_mla == false` 的实例可达；MLA 实例里 CONV 走整资源路径，见下。）**
2. **整资源（whole-resource）描述符只在本地只有 1 个 head 时可路由**：`describe_replicated_tensor` 只给一个覆盖整行的 span、不带 head 轴，且 `logical_offset` 恒为 0，所以"我持有哪个 head"只能由 rank 推出（`head_class_of(tp_rank) × H_l`），准入条件是 `H_l == 1`。**（第 4 轮修正：第 3 轮曾写成"`G == 1`"，那样会让 MLA 实例下的 SSM/CONV 只有 rank 0 通过；详见第 4 轮进展。）**

顺带给 `PeerCacheView` 增加了 `local_rank`（`cp_rank * tp_size + tp_rank`，未知为 `-1`）：`bind` 据此跳过不属于本源 rank 的边，并拒绝与 `dst_local_rank` 不符的目的视图，避免把区间落到别的 rank 的 buffer。S2 既有单测不受影响（默认 `-1` 即不校验）。

验证：`tests/core/framework/kv_cache_transfer/cache_directory_test.cpp` **18/18 PASSED**，与 S2 的 11 + 12 在容器内一并复跑全绿（共 41 个用例）。其中 MLA 夹具（`6513` 行 × `128` token、`TP8`、`kv_split=4`、`G=1`）派生 `S_eff=4`、`replica=2`，与实测 `index 26052 = 6513 × 4` 的分配几何一致；indexer 夹具的 `head_bytes = 514 = 257 × 2` 与 §6.2 实测的打包宽度一致。

夹具是**照 `cache_layout_builder.cpp` 公式手写的约定夹具**（host 侧手编回路不链接 torch），builder 本身由既有 `tests/core/framework/kv_cache/cache_layout_builder_test.cpp` 覆盖。真实构建里可再用 `torch::zeros` + `describe_cache_tensor` 生成 manifest 喂给适配器，属后续增强。

**未决项**：manifest 里的 `coordinates.kv_split_rank`（运行时取 DCP 分组 rank）尚未与 `KvLayoutIndex::slice_of(cp_rank, tp_rank)` 对账。两者必须在"规范块 ↔ 请求 block id"换算落地前统一，否则无法判断请求里的 id 属于哪个 rank 的切片 —— 属 S3-4/S3-5。

### S3-3 已完成（2026-09-18）：端到端 host 集成测试

`tests/core/framework/kv_cache_transfer/pd_route_integration_test.cpp` 把整条链路跑通并**逐字节**校验：
`torch::zeros` 造真实张量 → 真实 `describe_cache_tensor` → manifest（字段赋值照抄 `register_kv_cache`）→
`PeerDirectory::describe` → `PdRouteTable::build`/`validate` → `RouteBinder::bind` → host `memcpy`。
S3-2 遗留的"用真实 `describe_cache_tensor` 做 host 单测"由此关闭（容器内手编也能链 torch：见工作日志 §1.8）。

期望值的设计是关键：它不是 bind 算出来的，而是"规范内容函数 + 目的侧自己的描述符"两条独立信息合成 ——
字节内容只依赖 `(group, 规范资源, head, 子单元, 字节偏移)`，物理位置来自该侧描述符的 span，
`规范资源 = row × split + slice` 来自模型。因此字节落到错的坐标必然不匹配，未被写到的字节留在 poison 上也不匹配。

| 场景 | 覆盖 |
|---|---|
| MLA kv4 → kv4 | 等价锚点 |
| MLA kv4 → kv2 | **目标形状**（`S_P ≠ S_D`） |
| MLA kv2 → kv4 | 发散方向（一个源切片扇出到两个目的切片） |
| 非 MLA cp4/tp8/kv4 → cp4/tp4/kv2 | head class 交集（`H_l` 1→2）+ indexer 全副本扇出 + sequence-scoped SSM |

MLA 场景每个覆盖 KEY / INDEX / SSM / CONV 四个 role。**此处修正了 S3-2 的一条结论**：整资源 span 的准入条件
是"本地只有 1 个 head"（`H_l == 1`）而不是"`G == 1`"，且 head 身份取自 rank —— MLA 实例下所有 role 都走
`describe_replicated_tensor`，SSM/CONV 因此都是整资源 span；旧规则会让它们只有 rank 0 通过、链路根本建不起来。
`H_l == 1` 同时保住了 CONV 的安全：单 head 时资源内部没有 head 顺序可言，打包的 component 作为整体搬运。
⇒ **COMPOSITE（CONV）只在非 MLA 实例出现**；MLA 实例里 CONV 走整资源路径（`H_l == 1` 正确，`H_l > 1` 被拒）。

仍未覆盖：`MixedLayers`（多层不同 role 集合）、`DpExpansion`（`RouteEdge` 不含 DP 维，S2 单测已固定）、
XTensor `explicit_offsets` 的端到端（适配器与 T4 golden 已覆盖）、真实运行时（GLM5.3flash 尚不支持 PD 分离）。
实跑结果：`kv_redundancy_test` 11/11、`pd_route_test` 12/12、`cache_directory_test` 19/19、
`pd_route_integration_test` 4/4。

### S3-5 的切片契约修正（2026-09-18，已落地）

第 5 轮把 S3-5 的阻塞项钉死了：**`slice` 必须等于运行时的 `ContextParallelTopology::dcp_rank`**，
S2 原先的 `slice_of` 公式是另一套分组。证据、正确的两分支公式、以及要改的测试清单见
`pd_routing_s3_worklog_20260918.md` 第 5 轮（§3.1 上方也有摘要）。

要点：

- `CanonicalBlock`（`canonical = row*S + slice`）与 `KVShardLayout::globalize` **完全一致**，这块不用改；
  `N_rep = D/S`、`Hc*Hl = G`、每 `(h,t)` 单写者、边表条数也都不变。
- 要改的是 `KvRedundancy::derive`（新增 C3：`(S ≤ cp_size && cp_size % S == 0) || (S == cp_size*tp_size)`）
  与 `KvLayoutIndex` 的 `slice_of`/`replica_of`/`writer_of`/`replicas_of`。
- pilot 的真实拓扑是 **`cp_size=4` + `tp_size=8`（world 32）+ `kv_split=4`**（`world/kv_split = 8 = tp_size`
  ⇒ DCP 组 = 固定 tp、变动 cp），因此 `slice = cp_rank`、写者 `(cp=s, tp=0)`，`S_eff=4` 来自 `S | cp_size`。
- S2 的 rank golden 需要重算；`cp=1` 配 `S=4` 之类形状在 C3 下非法，相关夹具要改成 `cp_size=4`。
- **顺序**：先做这一步（③ 规范逻辑地址层），再做 ② 数据面切换。否则会把错块搬到对端。

> **2026-09-18 第 6 轮：已落地并全绿。** `KvRedundancy::derive` 加了 C3（配置的 split 必须是 DCP 形状：
> `S | cp_size` 或 `S == cp_size*tp_size`），`KvLayoutIndex` 的 `slice_of`/`replica_of`/`writer_of`/`replicas_of`
> 按 (a)/(b) 两分支重写；pilot 的 prefill 现在是 `cp=4 + tp=8 + kv_split=4`（写者 `local_rank = 8*cp`，
> `slice == cp`），decode 是 `dp4/cp1/tp2/kv2`（`slice == tp`）。S2 的 golden 与全部夹具已按新契约重算，
> 容器内 47 个用例全绿。**仍未做**：把 `ContextParallelTopology` 本体链进单测做运行时 oracle（需给手编
> harness 加 glog），以及数据面的"请求 block id ↔ 规范块"换算（与 S3-4 一起）。

### S3-4 已落地（2026-09-18，第 7 轮）：统一入口 + 开关，生产调用点待接

`xllm/core/framework/kv_cache_transfer/pd_route_transfer.{h,cpp}`：

```cpp
enum class RouteOpcode { PULL, PUSH };          // PULL = READ 取向，PUSH = WRITE 取向
struct RoutePeer { addrs; views; };             // 对端实例：rank → 地址 + 该实例公布的视图
struct RouteLeg  { opcode; local_rank; peer_local_rank; peer_addr; regions; };
class  PdRouteCache { find_or_build(...); };    // 按两侧 (拓扑, 组) 缓存边表（F8）
class  PdRouteTransfer {
  static bool plan(cache, opcode, local_rank, canonical_blocks, local, peer, legs, error);
  static bool transfer(..., const MoveFn& move, ...);   // plan + apply，数据面唯一入口
};
```

设计要点：

1. **`RouteLeg` = 一条腿 = 一对 (writer rank, reader rank)**，`regions` 已按 opcode 取向排好，调用方直接交给
   `move_memory_regions(peer_addr, regions, READ|WRITE)`（`RouteRegion` 与 `ByteRegion` 字段完全一致）。
   **因此 `filter_kv_split_infos` / `rotate_dst_rank` 那套"S==TP、rank 1:1 对齐"的隐含前提在 canonical 路径上
   不再存在**：一条目的腿的写者来自边表（每个 `(head class, source slice)` 的唯一副本 0 rank），
   写者的一条腿的读者来自"目的副本全遍历"，两端切片宽度可以不同。
2. **PULL 与 PUSH 是同一对区间，只换两半的归属**（`RouteBinder` 恒 destination-last）。这样两个方向可以
   互相核对，而不是各自被相信 —— 单测即按此跑两遍再逐字节比对。
3. **腿的枚举由边表驱动**，不由两侧视图的笛卡尔积驱动（后者会包含 head class 不相交的假配对）。
4. **完整性检查**：读侧必须拿到自己切片的每个块；写侧的冗余副本 rank 必须一条腿都不产生。
5. `PdRouteCache` 在校验失败时**不**写入缓存，`find_or_build` 首次即跑 `PdRouteTable::validate`。

**顺带加固**：`RouteBinder::bind` 现在拒绝"不属于目的 rank 切片的规范块"。此前若调用方分组错误，块会被写进
错 rank 的缓冲（`remote_row = block / S_D` 默认块属于该 rank），属静默错字节。

**顺带对账**（关闭 §8.1 S3-2 的未决项）：`PeerDirectory::describe` 在 MAIN 且本组 `S_eff == 配置 split` 时，
要求 `KvLayoutIndex::slice_of(cp,tp)` 等于 manifest 公布的 `coordinates.kv_split_rank`。测试 fixture 里原先那条
`(cp*tp+tp) % S` 的占位公式正是 S2 那套错分组，已按运行时两分支公式改正 —— 这条对账在夹具上**立刻**抓出了
4 个用例的不一致，证明它不是空转。

**开关**：`--pd_route=legacy|canonical`（默认 `legacy`，已注册进 `DisaggPDConfig::option_category`），工厂里解析；
非法值 `LOG(FATAL)`，`canonical` 目前**显式拒绝**并说明缺的输入（见工作日志第 7 轮）：本侧声明
（role → 组几何的生产映射尚未存在）与对端视图 / `cp_size` / DP 局部 rank 换算。**不做静默回落**。

**验证**（容器内 host）：`kv_redundancy_test` 12/12、`pd_route_test` 12/12、`cache_directory_test` 19/19、
`pd_route_transfer_test` 12/12、`pd_route_integration_test` 4/4 = **59 用例全绿**；
生产 TU（`kv_cache_transfer.cpp` / `mooncake_kv_cache_transfer.cpp` / `disagg_pd_config.cpp`）真实 flags 编译 rc=0。

**集成测试已改走统一入口**：`pd_route_integration_test.cpp` 现在对每个源 rank 调一次
`PdRouteTransfer::transfer(PUSH, ...)`（真实张量 → 真实 `describe_cache_tensor` → manifest → 适配器 →
**统一入口** → memcpy 逐字节），四个角色的搬运量与改前**逐位相同**。顺带修了夹具缺陷：buffer id 原先
每 rank 从 0 编号，而传输层按 id 单独寻址缓冲（生产中由 Mooncake 全局唯一），已改成全局计数器。

**仍未做**：生产调用点接线（上面那两项输入）、`ContextParallelTopology` 本体 oracle、
`MixedLayers` / `DpExpansion` / XTensor `explicit_offsets` 端到端。**运行时**仍未验证（GLM5.3flash 不支持 PD 分离）。

### 在开发机上的构建与验证（jd-node-98，aarch64 + Ascend）

环境：`quay.io/jd_xllm/xllm-ai:xllm-dev-a3-arm-cann9-20260911`（cmake 3.27.9 / ninja 1.11.1 / gtest 1.14.0），
工作树 `~/workspace/xllm-pdroute`（`xllm-dcp-fp32` 的副本，原树未被改动）。

| 验证项 | 结果 |
|---|---|
| `kv_redundancy.cpp` 用**项目真实编译命令**（取自 `compile_commands.json`，含全部 CANN / torch / torch_npu include 与宏）编译 | ✅ 通过 |
| `kv_redundancy_test.cpp` 编译 + 链接 vcpkg `libgtest`/`libgtest_main` | ✅ 通过 |
| 运行 `kv_redundancy_test` | ✅ **10 tests from 3 test suites，全部 PASSED**（含 `G×TP×CP×S` 穷举不变量扫描、GLM 5.3 flash 场景表、`TP=8,G=4,S=4` 与 `TP=8,G=2,S=3` 判别性负例） |
| 两处 CMakeLists 语法（用真实 cmake 3.27.9 + 桩宏 `include` 整个文件） | ✅ `CMakeLists syntax OK`，且新条目 `cc_library(kv_redundancy)` / `cc_test(kv_redundancy_test)` 字段正确 |
| `kv_cache_transfer.cpp` / `mooncake_kv_cache_transfer.cpp` 编译 | ⚠️ **环境性失败**：`platform/stream.h:33` 的 `#include <torch_npu/torch_npu.h>` 在三个可用镜像中都无法满足（实际文件在 `torch_npu/include/torch_npu/csrc/libs/torch_npu.h`）。**对照实验**：用同一套 flags 编译 `git show HEAD:` 取出的**改动前**同名文件，失败信息**逐字相同**，证明与本变更无关 |

复现命令（在 jd-node-98 上）：

```bash
# 手动编译 + 链接 + 运行新层单测（绕开需要整树 vcpkg 重装的 reconfigure）
sudo docker run --rm --privileged \
  -v ~/workspace/xllm-pdroute:/export/home/shifengmin.3/workspace/xllm-dcp-fp32 \
  -v ~/pdroute_tools/pdroute_container_manual.py:/tmp/manual.py:ro \
  --entrypoint bash \
  quay.io/jd_xllm/xllm-ai:xllm-dev-a3-arm-cann9-20260911 \
  -c 'cd /export/home/shifengmin.3/workspace/xllm-dcp-fp32 && python3 /tmp/manual.py'
```

> **未覆盖的一环**：`ninja kv_redundancy_test` 这条 CMake 驱动的完整构建在本环境跑不通 —— 原 `build/` 目录是在 `xllm-dcp-fp32` 路径下配置的，换路径后 vcpkg 需要从零重装 242 个 port（实测启动后即放弃）。因此 CMake 侧只做到**语法 + 条目正确性**验证，链接顺序等仍待一次正常 CI 构建确认。

---

## 9. 风险与回退

| 风险 | 缓解 |
|---|---|
| 规范坐标与现状不等价，导致静默错字节 | S2 以 golden + mock 逐字节验证（旧路径无法表达目标配置，见 §8）；`S_eff` 侧仍用 `S_P = S_D` 的退化配置做等价锚点 |
| 两侧独立推导边表出现分歧（实现/版本不一致） | `PdRouteTable::validate` 作为两侧互相断言；拓扑元组进 `fingerprint`，不一致直接拒链 |
| XTensor 页映射与规范块语义冲突 | `explicit_offsets` 作为 `BufferDirectoryEntry` 的一个标志位，规范层不感知；S1/S2 单测覆盖 XTensor 形态 |
| sequence-scoped 缓存被误纳入 `t` 规则 | §6 明确边界；`t` 规则只对 `CacheResourceScope::BLOCK` 生效 |
| COMPOSITE（CONV）组不在规范路由的表达范围内（§8.1 S3-2/S3-3） | 只在非 MLA 实例可达（`enable_mla == false`）；适配器显式拒绝而非静默降级。**S3-4 的决定（第 7 轮）：COMPOSITE 组留在旧 planner**，不给 `RouteEdge` / `PeerCacheView` 增加 per-component 字节偏移 —— 代价是 canonical 路径只覆盖 MLA 实例与不含 CONV 的组，收益是不把"component 局部 head 空间"这一维度塞进边表（它会让 `head_begin` / `head_end` 变成二维语义） |
| MLA 实例下的 SSM/CONV 是整资源 span，只有 `H_l == 1` 时可路由 | 适配器按 `H_l == 1` 准入并从 rank 取 head 身份；`H_l > 1` 时报错要求 producer 改成每 head 一个 span（GLM5-next `TP8` + `linear_*_head_count = 8` 落在 `H_l == 1`） |
| 发布侧 `coordinates.kv_split_rank` 与派生切片 `slice_of` 可能不一致 | **已修复并加了运行时对账**（第 6 轮修 `slice_of` = `dcp_rank`；第 7 轮 `PeerDirectory::describe` 在 `S_eff == 配置 split` 时要求 `slice_of(cp,tp) == coordinates.kv_split_rank`）。夹具里原有的占位公式被这条对账立刻抓出 4 处不一致 |
| 建链收敛后，运行期新增对端无法建链 | 保留 on-demand 建链路径：`PdRouteTable` 可在运行期对新的拓扑元组补算边表 |

**回退点**：S0 / S1 完全独立可回退；S2 是纯新增并行路径；S3 起才切换行为，切换前保留旧路径的编译开关（`--pd_route=legacy|canonical`）以便灰度与快速回退。
