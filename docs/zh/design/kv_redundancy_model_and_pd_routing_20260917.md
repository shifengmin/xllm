# KV 冗余模型与 PD 传输路由算法

## 溯源与状态

- 日期：2026-09-17
- 基线 commit：`200939593`
- 本文性质：**概念模型与算法规范**（先定语义，再谈实现；不含代码改动）
- 本文目标：把 PD 传输路由从"条件特例集合"收敛为"三个正交索引的笛卡尔积"，并给出配套的**规范逻辑地址层**（§7），使得 CP / TP / `kv_split_size` / `kv_head_num` 的**任意组合**（含 `S_P ≠ S_D` 的折叠）都不需要特例分支
- 实现方案：见 `pd_routing_simplification_20260917.md`
- 证据口径：**[推导]** 模型推演；**[读码]** 静态阅读；**[实测]** 本机手编链路

---

## 0. 一句话

**KV 传输路由 = 三个正交索引 `(h, t, c)` 的笛卡尔积** —— `h`（head 类）走全局 head 区间求交，`t`（序列片）走块号取模，`c`（冗余副本号）在源侧去重、在目的侧复制。除此之外没有任何自由度。

---

## 1. 第一性前提

**`kv_split` 存在的唯一目的是消除 KV 冗余。**

由此得到两条方法论约束：

1. 任何不能被"消除冗余"解释的机制（额外的相位比较、拓扑特判、折叠公式）都应被怀疑是建模不完整的症状，而不是需求。
2. 冗余必须先被**定义**，才能知道能消多少。因此下文先定义冗余度 `D` 与冗余组 `G`，再定义路由。

---

## 2. KV 冗余的定义

### 2.1 符号

| 符号 | 含义 |
|---|---|
| `G` | 全局 KV head 数（`kv_head_num`） |
| `TP` / `CP` / `DP` | 本实例的 TP / CP / DP 宽度 |
| `S` | `kv_split_size`（本实例） |
| `D_tp` | TP 造成的 KV 冗余度 |
| `D` | 本实例的复合冗余度 |
| `H_l` | 每 rank 本地 head 数 |
| `H_c` | head 类数 |
| `N_rep` | `kv_split` 打开后的残余冗余度 |

### 2.2 TP 冗余

一个 rank 持有哪些 head，由 TP 与 `G` 的关系唯一决定：

```
H_l  = max(G / TP, 1)     每 rank 本地 head 数
D_tp = max(TP / G, 1)     TP 冗余度（几个相邻 rank 持相同 head）
H_c  = TP / D_tp          head 类数 == min(G, TP)
```

- `G >= TP`：head 被 TP 切满，`D_tp = 1`，每 rank 持有 `G/TP` 个**互不相同**的 head；
- `G < TP`：head 数少于 rank 数，`D_tp = TP/G`，每 rank 持 **1** 个 head，每个 head 被 `D_tp` 个 rank 重复持有。

**不变量（两侧都成立）**：`H_c × H_l == G`。即 head 类**无损铺满** `[0, G)`，任意一侧都不重不漏。

**G_tp(h)**：head 类 `h` 对应的 `D_tp` 个**相邻** tp rank，覆盖全局 head 区间 `[h·H_l, (h+1)·H_l)`。

> **[推导]** "相邻"这一性质使 head 类与 `tp_rank` 之间是整除块关系，而非任意集合。

### 2.3 CP 冗余

**前提（本文采用）**：CP 复制 KV —— 每个 CP rank 持有完整序列。

于是同一 head 类下，所有 CP rank 的内容相同：

```
D = CP × D_tp             复合冗余度
```

**G_cptp(dp, h)**：`{ rank(dp, cp, h·D_tp + r) : cp ∈ [0,CP), r ∈ [0,D_tp) }`，基数恰为 `D`。组内所有 rank 的 KV **逐字节相同**。

> 若 CP 改为切序列而非复制，则 `D = CP × D_tp` 仍是 `S` 的**容量上界**（全序列可切成 `CP·D_tp` 份各持一份），但真正的冗余度只有 `D_tp`。本文按"CP 复制"前提写作，两种语义的区别见 §7。

### 2.4 三个正交索引

对 rank `= dp·(CP·TP) + cp·TP + tp` 定义：

```
h = tp / D_tp               head 类        → 决定"哪些 head"
r = tp % D_tp               TP 组内偏移
s = cp·D_tp + r             G_cptp 内序号 ∈ [0, D)
t = s % S                   序列片索引     → 决定"哪些块"
c = s / S                   冗余副本号     ∈ [0, N_rep)
```

`h` 与 `t` 分别是 head 维与序列维的**完整身份**；`c` 是同一份数据的副本编号。一个 rank 持有的 KV = 「head 类 `h` 的全部 head」×「序列片 `t` 的全部块」。

### 2.5 kv 完备组

**kv 完备组** = `G_cptp` 内 `s` 连续的 `S` 个 rank，即 `c` 相同的那些。

性质（这是 `kv_split` 的语义定义）：

- **不重**：组内 `t` 互不相同 ⇒ 块集合两两不相交；
- **不漏**：组内 `t` 取遍 `[0, S)` ⇒ 并集为完整序列；
- **无冗余**：组内任意两个 rank 的 KV 无重叠；
- **完整身份**：组内所有 rank 属于同一 `h` ⇒ 覆盖该 head 类的全部 head。

`G_cptp` 因此被切成 `N_rep = D / S` 个互为冗余副本的完备组：

```
N_rep = D / S           残余冗余度（= 打开 kv_split 后的冗余倍数）
```

### 2.6 冗余阶梯

| `S` | 残余冗余 `N_rep` | 每 rank 持有 | 含义 |
|---|---|---|---|
| `1` | `D` | 完整序列 | 全冗余（未利用） |
| `S` | `D/S` | `1/S` 序列 | 消掉 `S` 倍冗余 |
| `D` | `1` | `1/D` 序列 | 冗余全部消除（极致省显存） |

---

## 3. 约束（唯二前提）

### C1 整除

```
G % TP == 0  或  TP % G == 0            head 可均匀切分或复制
S | D                                   完备组可整除分组
S_P | S_D  或  S_D | S_P                 跨实例序列片划分嵌套
```

`S | D` 的必要性 **[推导]**：`G_cptp` 是**连续** rank 块与"连续 `S` 个"切法的组合；若 `S ∤ D`，会剩下既不成完备组也不成副本的 rank，冗余度不再均匀，覆盖率恒等式被破坏。

### C2 复合数量上限

```
1 <= S <= D
```

`kv_split_size` 不能超过冗余度 —— 没有冗余可消时再切只会增加通信。

**满足 C1 + C2，CP / TP / `kv_split` / `kv_head` 的任意组合都可用同一条算法求解，不需要任何特例分支。**

---

## 4. 路由算法

### 4.1 核心结论：路由 = head 路由 ⊗ block 路由

块 `j`、head 类 `h` 的传输对由两个**互相独立**的映射决定：

| 维度 | 决定量 | 规则 | 依赖 |
|---|---|---|---|
| **head** | `h_P → h_D` | 全局 head 区间求交 | 仅 `G`, `TP_P`, `TP_D` |
| **block** | `t_P = j % S_P` → `t_D = j % S_D` | 纯块号 | 仅 `S_P`, `S_D` |

正交性的**前提**是：head 归属在整个 `G_cptp` 内是常数。该前提由 §2.3 的构造保证（`G_tp` 内相同；同一 `G_tp` 的不同 CP rank 也持相同 head）。

```
head_pairs = { (hP, hD, lo, hi) :
    [lo,hi) = [hP·H_l^P, (hP+1)·H_l^P) ∩ [hD·H_l^D, (hD+1)·H_l^D)  ≠ ∅ }

block_map(tP) = { (tP + k·S_P) % S_D : k ∈ [0, S_D / gcd(S_P, S_D)) }
```

- `S_D | S_P` ⇒ `|block_map| = 1`，每个源片只对一个目的片（**汇聚**）；
- `S_P | S_D` ⇒ `|block_map| = S_D/S_P`，每个源片扇出（**发散**）；
- 互不整除 ⇒ 连接数退化为 `lcm`，应被 C1 禁止。

### 4.2 两条非对称规则（都是推论，不是特例）

- **源侧去重**：同一 `(h, t)` 有 `N_rep` 个冗余副本，内容相同，只取一个确定性写者（取 `c = 0`）。
- **目的侧复制**：同一 `(h, t)` 的 `N_rep` 个副本位于不同 rank 的显存上，都要填，因此 1 个源扇出到全部副本。

> 现有实现中这对非对称需求分别表现为"按 `owner_tp_rank` 过滤源"与"目的侧不过滤"，二者是同一模型事实（`c = 0` 唯一写者 / `c` 全遍历）的两半。

### 4.3 建链期：一次性枚举有向边

```text
for dp in [0, DP):
  for (hP, hD, lo, hi) in head_pairs:              # head 维
    for tP in [0, S_P):                            # block 维：源片
      src = 唯一 rank 满足 h(tp)==hP ∧ t(cp,tp)==tP ∧ c(cp,tp)==0
      for tD in block_map(tP):                     # block 维：目的片
        for cD in [0, N_rep^D):                    # 冗余维：目的副本全填
          dst = 唯一 rank 满足 h(tp)==hD ∧ t(cp,tp)==tD ∧ c(cp,tp)==cD
          edge[src] += (dst, head_range=[lo,hi), tD)
```

边集合**与请求无关**，可长期缓存在 `cache_peer_links_` 一类结构中。

### 4.4 请求期：块号代入

```text
for each (src, dst, head_range, tD) in edge[src]:
  for layer in [0, L):
    for j in request.blocks:
      if j % S_P != t_of(src): continue
      local_id  = src.local_block_of(j)
      remote_id = dst.local_block_of(j)
      emit ByteRegion(layer, head_range, local_id, remote_id)
```

请求期只做**一次查表 + 一次模运算**，没有区间求交、没有二分、没有相位比较。

### 4.5 规模上界

| 量 | 公式 |
|---|---|
| head 对数 | `#head_pairs`，嵌套时 = `max(H_c^P, H_c^D) / min(H_c^P, H_c^D)` 倍 |
| block 对数 | `max(S_P, S_D)`（C1 保证） |
| **总边数** | **`#head_pairs × max(S_P, S_D) × N_rep^D`** |
| 单实例上界（`S_P = S_D = S`） | `≈ min(H_c^P, H_c^D) × D^D` |

**连接数只由冗余度 `D` 决定**，与 TP、head 数的具体切法无关。这就是"复合数量上限"的准确含义。

---

## 5. 退化验证

| 场景 | `D_tp` | `D` | `H_c` | `S` | 结果 |
|---|---|---|---|---|---|
| `G >= TP`, `S=1` | 1 | CP | TP | 1 | 全部 CP rank 持完整序列（`N_rep = CP`） |
| `G >= TP`, `S=CP` | 1 | CP | TP | CP | 每 CP rank 持 `1/CP` 序列，`N_rep = 1`（经典 CP 切序列） |
| `G < TP`, `S=1` | TP/G | CP·TP/G | G | 1 | 全冗余未利用 |
| `G < TP`, `S=D_tp` | TP/G | CP·TP/G | G | D_tp | 每 head 类内切出 `CP` 个完备组，`N_rep = CP` |
| `G < TP`, `S=D` | TP/G | CP·TP/G | G | D | `N_rep = 1`，每 rank 持 `1/(CP·D_tp)` 序列（极致省显存） |
| `S_P ≠ S_D` | — | — | — | — | `S_D|S_P` 汇聚 / `S_P|S_D` 发散，边数均为 `max·N_rep^D` |

### 与当前实现的对照 **[读码]**

`collective_communicator.cpp:582-589`：

```cpp
const int32_t dcp_size = normalized_cp_size == 1 ? kv_split_size_effective() : 1;
const int32_t dcp_group_index = global_rank / dcp_size;
const int32_t dcp_group_start = dcp_group_index * dcp_size;
```

即 **DCP 组 = 连续的 `dcp_size` 个 rank**，`dcp_rank = global_rank % dcp_size`，且 CP>1 时 `dcp_size = 1`（DCP 被禁用）。

代入模型：令 `S = dcp_size`，则 **DCP 组 == 本模型的 kv 完备组**（`c` 相同的连续 `S` 个 rank）。并且：

- CP>1 ⇒ `S=1` ⇒ 不切分，与模型一致；
- CP=1 且 `G < TP` ⇒ `D = D_tp`，`S | D_tp` 是 DCP 组能落在同一 head 类内的**充要条件** —— 这正是 C1 中 `S | D` **[推导]** 的实证；

> 反例（**[推导]**）：`TP=8, G=2, S=3` ⇒ `D_tp=4`，DCP 组 `{0,1,2}` 跨入 head 类 0 与 1 的边界，head 类 1 只剩 `{3}` 单 rank 覆盖完整序列 —— 覆盖恒等式立刻破裂。C1 提前拒绝这种配置。

---

## 6. 与既有概念的映射

| 本模型 | 现有实现中的对应物 | 关系 |
|---|---|---|
| `D_tp` | `describe_attention_heads` 的 `replica_count` | `replica_count == D_tp` |
| `H_l` | `local_head_count` | 相同 |
| head 类 `h` | 隐含在 `first_global_head = h·H_l` 中，未显式命名 | **需显式化** |
| `S` | `kv_split_size_effective()` | 相同 |
| 完备组 | 隐式等于运行时 DCP 组（连续 `S` 个 rank） | **需显式化** |
| `N_rep` | **无对应概念** | 复杂度根源之一 |
| `D` | **无对应概念** | 复杂度根源之一 |
| writer 去重 | `only_static_owner` + `owner_tp_rank` | 本模型中是 `c = 0` 的推论 |
| 目的副本全填 | 目的侧 `expand_manifest(..., false)` | 本模型中是"遍历 `c`"的推论 |
| head 路由 | `ReshardPlanner` 的区间求交 | 保留，但输入换成 head 类而非原始 span |
| block 路由 | `filter_kv_split_infos` 的 `remote_ids[kv_split_rank + k·kv_split_size]` | 是 `S_P = S_D` 时本模型的 `t` 规则的一个特化 |

---

## 7. 规范逻辑地址空间

### 7.1 为什么必须有这一层

两件事同时成立：

1. head 侧的重分片要求"**同一份 KV 数据在两端被识别为同一份数据**"；
2. `kv_split` 切序列使两端的**物理资源几何不同**（每 rank 覆盖多少 token、`resource_stride_bytes` 多少，都随本侧 `S` 变化）。

这两个需求必须在**两个不同的坐标系**里各自满足。混用同一个坐标系，就会得到一个"peer 相关"的逻辑地址 —— 即"我方逻辑偏移 `[0,3)` 在你方对应到哪里取决于你的资源粒度"。

### 7.2 定义

对任意一个 KV 字节定义**规范逻辑地址**：

```
canonical_addr = (b, g, tau)
  b   : 规范块号 —— 由 token 区间导出（第 b 个 B_token 的 token 区间），与 S / TP / CP 无关
  g   : 全局 KV head 序号 ∈ [0, G)
  tau : 块内 token 序号 ∈ [0, B_token)
```

线性化：

```
linear_offset(b,g,tau) = b * BlockLogicalBytes
                       + tau * (G * bytes_per_head)
                       + g * bytes_per_head
BlockLogicalBytes      = B_token * G * bytes_per_head
```

**`B_token` 与 `BlockLogicalBytes` 是与任何 rank 的 `S`、`TP`、物理资源几何都无关的常量。**

本地的 block id 与规范块号之间通过**本侧**的映射表互换（`canonical_block_of(local_block_id)`），因此两侧的本地 id 空间可以完全不同。

### 7.3 两层分离

| 层 | 内容 | 是否 peer 相关 |
|---|---|---|
| **规范层** | `(b, g, tau)` → 逻辑区间；用于身份识别、区间求交、覆盖率校验 | **否** |
| **物理层** | `(buffer_id, resource_id, resource_stride_bytes, 资源内偏移)` | **是**（每侧自己的显存布局） |

描述符 `LogicalSpan` **只声明规范层**的区间；`bind_regions` **只做物理层**的换算：

```
physical_offset = resource_id * resource_stride_bytes
                + span_offset_in_resource
                + repeat * token_stride
```

`resource_stride_bytes` 由本侧 manifest 提供 —— **两侧不同是允许且正常的**。

### 7.4 规范块与物理资源必须解耦

当前 `llm_engine.cpp:646-653` 把 `block_size` 乘上 `kv_split_size_eff`，并令

```cpp
.block_size(kv_split_size_eff > 1 ? block_size * kv_split_size_eff : block_size)
```

于是 `block_token_capacity = block_size * S`，**逻辑块与物理资源被绑成同一个尺寸**：

```
BlockLogicalBytes = B * S * G * bytes_per_head        <-- 依赖本侧 S
```

`S_P ≠ S_D` 时两侧 `BlockLogicalBytes` 不同，`b * BlockLogicalBytes` 落到**不同坐标系** —— 这就是"拼装空间不一致"的准确成因。

模型要求把它们拆开：

| | 定义 | 谁决定 |
|---|---|---|
| **规范块** | `B_token` 固定为 token 语义上的一块，**与 `S` 无关** | 全集群常量（两侧 `block_size` 相同即可） |
| **物理资源** | 每 rank 覆盖多少 token、`resource_stride_bytes` 多大 | **本侧** `S` 与显存布局，写在 manifest 里 |

`S` 切分的是"**哪些规范块归哪个 rank**"，不改变块的尺寸。解耦后，原先需要逐条打补丁的三条约束自然成立：

| 原先的约束 | 在两层分离下 |
|---|---|
| receiver 的每资源逻辑宽度须 = `s × fold` | receiver 声明的就是它需要的规范块集合，宽度天然等于规范块宽度 |
| 逻辑空间须 resource-major | 规范层的 repeat stride 是 `BlockLogicalBytes`（两侧恒等）；物理步长只出现在物理层 |
| `bind` 的目的基点须按目的段 stride | 目的基点用的是**目的侧自己的** `resource_id × resource_stride`，与源侧无关 |

### 7.5 与 §4 路由算法的衔接

引入规范层后，block 路由写成规范块号的形式：

```
t_P(b) = b % S_P        t_D(b) = b % S_D
```

§4.4 中的 `j % S_P` 应理解为 `canonical_block_of(j) % S_P`。head 路由（§4.1）不变。**规范块号是唯一同时被两侧理解的块标识，因此它是路由的输入，而不是本地 block id。**

### 7.6 新增的跨实例前置条件

```
B_token_P == B_token_D
```

`B_token` 是规范块的定义域，两侧不一致则 `b` 无意义。该条件**取代**所有"按 size 关系猜折叠系数"的启发式 —— 后者是因为缺少规范层才不得不从 `resource_stride_bytes` 反推，属于症状级补丁。

---

## 8. 边界与未决

1. **CP 语义**：本文按"CP 复制 KV"写作。若 CP 切序列，`D` 与"冗余度"分离（见 §2.3 注），需在实现中显式区分 `capacity` 与 `redundancy` 两个量。
2. **`S ∤ D` 的非均匀分组**：本文直接禁止。若确需支持，需为"剩余 rank"单独定义语义（例如全部持完整副本），会重新引入特例。
3. **DP 维**：DP 之间是不同请求，不构成冗余，故所有索引都限定在一个 DP 组内。跨 DP 的 rank 配对属调度问题，不属路由问题。
4. **head 类非整除的跨实例组合**：若 `H_l^P ∤ H_l^D`，head 区间会出现部分重叠，此时 `head_pairs` 的自然语言是"区间交集"而非"整除块"。模型仍然正确（传输本来按 head 子区间表达），但连接数上界公式需按交集数重算。
5. **组级 `S_eff` 的语义性例外**：`S_eff` 通常由冗余度推导（见 §2.5 与方案 §6.1），但存在**有块维却必须保持全序列**的组（DSA indexer kPool 的 top-k 需要读全序列的 gate/valid）。这类组不能靠 `G/TP/CP/S` 推出 `S_eff=1`（其 `G=1, D=8` 按规则会得到 4），必须显式声明。实现上落在 `GroupTopology::full_sequence_replica`。
6. **`block_size` 放大的移除面**：解除 §7.4 的绑定会改变 BlockManager 的块尺寸与 `n_blocks` 的语义，触及调度与 prefix cache 的 block 哈希。该改动应在 S2 之后独立评估，但它不是"是否采用规范层"的前提 —— 规范层可以先只作用于传输侧描述符。
