# ship-indices 单路 PD KV 传输 · per-tensor 异构设计

> 基线：`200939593`（canonical 引入之前），worktree `xllm-ship-indices`，分支 `pd-ship-indices`。
> 目标：把 legacy transfer 演进成**一条** SGLang 式 ship-indices 路，取代 canonical + legacy，
> 支持 per-tensor 的 P/D 异构。**忽略 XTensor 专属路径**（只做 `MooncakeKVCacheTransferDefault`
> strided 路径）。

## 0. 一句话

Legacy 已经是"半 ship-indices"（D 发物理 `remote_ids`，P 按它定位）。把其中**唯一**写死
"两侧 kv-split 宽度相等（N==M）"的地方，换成**逐 tensor 各带自己的序列切分宽度 N**，配对公式
推广到任意 N↔M。indexer 的 replica 就是 N==1 的退化，不是分支。

## 1. 两条正交的切分轴（关键认知）

KV 传输的重排只有两条独立的轴，当前分属两处，成熟度不同：

| 轴 | 编码在哪 | per-tensor？ | 通用度 | 本次是否动 |
|---|---|---|---|---|
| **head / TP** | `LogicalShardDescriptor.kind`（`cache_layout_builder` 按 head 数 / tp 算 `SHARDED/REPLICATED`） | ✅ 已是，随 manifest 上 wire | ✅ `RequestRegionBinder` 已通用（TP 1↔4 非整除都行） | **不动** |
| **序列 / kv-split** | `build_step_transfer_info` 与 `filter_kv_split_infos` 的 `×N` stride | ❌ instance 级 `is_kv_split_cache_block_type` bool 门 | ❌ 只支持 N==M（实为"D 完整、P 挑"） | **推广** |

- head/TP 异构已经被第一条轴解决了——`RequestRegionBinder::bind`（reshard_planner.cpp:673）把
  `local_ids[i]↔remote_ids[i]` 位置配对后各自 × `resource_stride`，reshard 模板来自协商好的
  per-tensor manifest。**本设计完全不碰它。**
- 用户的 indexer 异构（P 切 / D 不切）落在**第二条轴**。这是本次全部工作量所在。

## 2. 序列维的现状：两处折叠，都写死 N==M

D 侧把**完整**的 canonical-ordered `remote_ids` 发给 P；P 分两步折成"本 step、本 rank 应写的子集"：

1. **主折叠 `BatchInputBuilder::build_step_transfer_info`**（batch_input_builder.cpp:289）
   每 step 跑；layerwise/composite 与 flat-KV 两个调用点（:1106 / :1160）都走它。
   - `remote_stride = uses_kv_split && !rank_local_mapping ? kv_split_size : 1`（:361）
   - `remote_idx = (local_idx - remote_origin) * remote_stride + offset`（:414）
   - `kv_split_size = util::kv_split_size_effective()`（:1110/:1164，本实例=P 的单一 split）
   - SWA sentinel `remote_ids[i]==uint64::max` 跳过（:420）
2. **次折叠 `filter_kv_split_infos`**（kv_cache_transfer.cpp:145，在 `push_kv_blocks_async` :239）
   - `remote_idx = kv_split_rank + k * kv_split_size`（:168），对已折叠的再挑一道。

两处都用同一个 **instance 级** `kv_split_size`，判据都是 per-group 的
`is_kv_split_cache_block_type(block_type)`（block.h:75）。**N==M 假设只活在这两处的 stride。**

## 3. 目标设计

> **两个宽度，来源不同**（这是全设计的枢纽，别混）：
> - **本侧 N**（P 自己的 split，用于 filter 的**选择**）：P 本地已知，**不需上 wire**。
>   per-tensor 化只需把判据从 per-group bool 换成"这个 tensor 本侧切几份"。
> - **对端 M**（D 的 split，用于 build_step 的**展开**）：折叠点跨层拿不到 → **走 per-group wire**（点 C）。

### 3.1 对端宽度 M：per-group wire（点 C 定论）

- **wire**：复刻 `remote_shared_num` 范式，在 `KVTransferGroup` proto（disagg_pd.proto:127-137）+
  `KVTransferMapping`（types.h:309）加 per-group `remote_kv_split`（= 该 tensor-group 在 **D 侧**的宽度 M）。
- **producer = D 侧** `disagg_pd_service_impl.cpp:235-289`：拼 group 时天然知道自己每组 split，填入。
  `group_id` 即 tensor-group 粒度 → **天然区分 indexer/MLA**（indexer 组 M 可为 1、MLA 组 M 可为 2）。
- 这就是"切分是 tensor 的属性"落到 wire 上的形式——每个 tensor-group 各带自己的 M。

### 3.2 本侧宽度 N：per-tensor 判据（替换 per-group bool）

- 现在：`uses_kv_split = is_kv_split_cache_block_type(block_type)`（per-group 静态 bool）→ 用 instance N。
- 改后：`N = 该 tensor-group 本侧切分宽度`；`uses_kv_split = (N > 1)`。本侧 N 来源：
  - MLA 主 KV / indexer（P 侧 cp+kv_split）：`N = kv_split_size_effective()`
  - indexer D 侧（kv_split=1）：`N = 1`
  - LINEAR / EMBEDDING（sequence-scoped）：`N = 1`
- rank 坐标统一取 `kv_split_rank()`（单轴，已确认；不 per-tensor 存 rank）。
- indexer 在某侧 `N=1` / `M=1` → 自动退化 1:1。**replica = 宽度==1，退化，不是新分支。**
- 【实现注：本侧 N 若能从 tensor 的注册几何直接判定（A 已证 indexer 行数=块×N），可不必额外上 wire；
  否则同样可从 per-group 结构带。落码时择一，不影响公式。】

### 3.3 配对公式推广到 N↔M（三点确认后的**最终形态**）

三个调查点回来后，公式落地形态被**架构约束钉死**（见 §7 已确认）。核心：执行模型是
"P 侧构建广播产物 → 每 worker 各自消费"，这把序列维映射**天然劈成两半**，恰好对上现有两处折叠：

- **展开（用对端宽度 M，rank 无关）→ 落在 `build_step_transfer_info`**：
  它的产物经共享内存**广播给所有 worker**，所以结构上**必须 rank 无关**。它能做的、且只能做的是
  "把 D 想要的 canonical 布局摊开"——即对每个逻辑块，按 **D 的宽度 M** 展开成 M 个 slice。
  当前 `remote_stride` 用的是本地 N（N==M 假设）；**改成用对端 M**（per-group，见 §3.5）。
- **选择（用本侧宽度 N + kv_split_rank，每 rank 不同）→ 落在 `filter_kv_split_infos`**：
  每个 worker 持自己的 `kv_split_rank`，从展开后的列表里挑 `c % N == kv_split_rank` 的那份。
  这一步**天然 rank 相关，无法上移到广播产物里**——所以 `filter` **不可退役**（点 B 定论）。

统一的 canonical 位置 c 表达（c = 序列里第几个逻辑块，A 已确认 indexer 行本就是 c 空间）：
- D 侧：canonical 块 c 住在 (行 `c / M`, slice `c % M`)。build_step 按 M 展开，产出 D-ordered 列表。
- P 侧：本 rank 持有 c ⟺ `c % N == kv_split_rank`。filter 按 N 选。
- 复合：本地第 k 个块 → `remote[k*M + (kv_split_rank 对应的 slice)]`。
  **N==M 时退化为今天的 `remote[kv_split_rank + k*N]`，逐字节相同**（等价锚点，§5）。
- 复用一个纯函数做除/模（不引 canonical 层，写独立小工具 + 单测钉死），**不手写两份**（漂移=静默错字节）。

> **关键修正（对比本文件早期版本）**：M 属于"展开"、N 属于"选择"，二者分居两处、都要改。
> 早期设想"把折叠合到一处"是**错的**——广播产物 rank 无关这一硬约束禁止把 N-选择上移。

### 3.4 head/TP 维：不动

`RequestRegionBinder` 已经对 head/TP 通用。序列维折叠只决定"哪些 (local_id, remote_id) 对进入
mapping"，head 维由 binder 在 mapping 之上处理。两维彻底解耦。

### 3.5 M 的传递路径：per-group wire（点 C 定论）

对端 per-tensor manifest 存在 engine 侧，`build_step_transfer_info` 跨层 + 多线程拿不到（透传指针
破坏分层）。最干净路径 = **复刻 `remote_shared_num` 的现成范式**：在 `KVTransferGroup` proto
（disagg_pd.proto:127-137）+ `KVTransferMapping`（types.h:309）加 per-group `remote_kv_split`（=M）。
- D 侧在 `disagg_pd_service_impl.cpp:235-289` 本就按 tensor-group 拼 group、知道自己每组 split；
  `group_id` 即 tensor-group 粒度 → **天然区分 indexer/MLA**。
- 折叠点已逐 group 迭代（batch_input_builder.cpp:303）→ wire 直达，零新依赖、零跨层指针。
- 本侧 N 仍来自本地 `kv_split`（filter 已有 `kv_split_rank`），无需上 wire。

## 4. indexer 异构走通（用户已确认序列维同构）

```
P(cp=2, kv_split=2):  MLA.N=2, indexer.N=2   → 都 c%2==kv_split_rank，各持一半
D(kv_split=1):        MLA.M=1, indexer.M=1    → 都整份（N→M=1 折叠对称）
```

- indexer 与 MLA 在**序列维用同一个 N↔M 公式**；差别只在 head 维（head 数不同），head 维由
  `RequestRegionBinder` 处理。→ **indexer 无需任何特判。**
- "index cache 行数 = 2× 逻辑块数"这一现象由 per-tensor manifest 的 `resource_count`/
  `resource_stride` 吸收（index 的行本就是 canonical 粒度），同一公式照跑。
  【开放假设 A：落到该代码时核对 indexer 分配的实际 row 语义，预期无分支。】

## 5. 正确性与验证（ship-indices 放弃了 canonical 的可验证 bijection，需主动补回）

- **等价性锚点**：N==M 场景，新公式必须逐字节复现今天 `remote_ids[kv_split_rank + k*N]` 的输出。
  两处折叠点各建一个 N==M 回归。
- **序列覆盖 CHECK**（新增，必须）：N↔M 折叠后，每个 (group, 目的行) 必须**恰好一个**源写者。
  缺写/重写要当场 CHECK 失败，不能静默。（canonical 靠 `validate` 的 bijection 免费拿到这个；
  ship-indices 没有，必须显式加。）
- **byte-oracle 单测**（决定性）：P-split ≠ D-split（如 2→1、2→4），两侧都开 prefix cache，
  断言每个目的字节 == 独立算出的源字节。含 indexer（N: P=2/D=1）与 MLA 混在同一请求。
- **degenerate**：indexer D 侧 N=1 直传、LINEAR/EMBEDDING N=1，都要在用例里覆盖。

## 6. 改动清单（落点，均在 `MooncakeKVCacheTransferDefault` strided 路径）

1. `cache_layout.h` / proto：per-group `remote_kv_split`（=M）加到 `KVTransferGroup`
   （disagg_pd.proto:127-137）+ `KVTransferMapping`（types.h:309），复刻 `remote_shared_num` 范式。
2. D 侧 `disagg_pd_service_impl.cpp:235-289`：拼 group 时按 tensor-group 填自己的 split 到 `remote_kv_split`。
3. `batch_input_builder.cpp:289 build_step_transfer_info`：`remote_stride` 从本地 N 改用 **per-group M**
   （`full_mapping.remote_kv_split`），判据从 `is_kv_split_cache_block_type` 换成 `M>1`。这是 **M-展开**。
4. `kv_cache_transfer.cpp:145 filter_kv_split_infos`：**保留**（点 B：不可退役）。它做 **N-选择**，
   stride 需与 build_step 的展开宽度 M 匹配、rank 用本侧 `kv_split_rank`；N≠M 时按 §3.3/§7 数值验证的
   `remote[k*M + rank对应slice]` 选。
5. 新增序列覆盖 CHECK（每 (group,目的行) 恰一源写者）；新增等价 + byte-oracle 单测。
6. **不碰**：`RequestRegionBinder`/reshard 内核、head/TP 维、XTensor 路径。

## 7. 三个确认点 —— 已由 subagent 查实并决策（2026-09-19）

三点均带 file:line 证据确认。三者**独立地合流到同一个自洽方案**，这是决策的最强信号。

**A — indexer 无需特判（已确认）。** indexer 物理行 = 逻辑块 × kv_split_size（`kv_cache_shape.cpp:389-392`；
2× 只是 N=2 特例，真实因子是 N），按 **canonical 块粒度**索引（`c = B*N + shard`，
`kv_shard_batch_metadata.cpp:129-142`），manifest 如实描述（`resource_count`=canonical 行数、
`physical_rows_per_resource=1`、BLOCK 作用域，`mooncake_kv_cache_transfer.cpp:326-372`）。
→ 按 c 的公式用 indexer 自己的 resource_count 直接适用。**唯一硬约束：公式必须在 canonical 块空间运算**
（正好与 B 的展开语义一致）。

**B — 两处折叠严格串联、各管一段、`filter` 不可退役（已确认，最关键）。**
build_step 产物经共享内存**广播给所有 worker → 必须 rank 无关 → 结构上只能做 M-展开、不能做 rank-选择**；
filter 每 worker 持自己 `kv_split_rank` → 才能做 N-选择。复合 `local k → remote[k*N + kv_split_rank]` 正确不抵消。
→ **M-展开落 build_step，N-选择落 filter，两处都要改，filter 保留。**（推翻了早期"合并到一处"的设想。）

**C — M 走 per-group wire（已确认）。** 对端 per-tensor manifest 在 engine 侧，折叠点跨层+多线程拿不到；
复刻 `remote_shared_num` 范式，在 `KVTransferGroup`/`KVTransferMapping` 加 per-group `remote_kv_split`（=M），
D 侧按 group 拼装时天然知道每组 split、`group_id` 即 tensor-group 粒度 → 天然区分 indexer/MLA、零跨层。

### 等价性数值验证（N==M=2，必须逐字节复现今天）

逻辑块 0,1,2；D 展开列表 `remote=[r00,r01, r10,r11, r20,r21]`（r_{c,slice}）。
- build_step（M=2 展开）：local k=0 → 取 `remote[0..1]`=r00,r01；k=1 → r10,r11；k=2 → r20,r21。
- filter（N=2，kv_split_rank=1 选）：`remote[1 + k*2]` → r01, r11, r21。
- 结果：本 rank 写 D 的每个块的 slice 1。**与今天 `remote[kv_split_rank + k*N]` 完全一致。** ✓

### 目标场景数值验证（indexer：P N=2 → D M=1）

D 不切（M=1），`remote=[r0, r1, r2]`（每块一行）。
- build_step（M=1 展开）：local k → 取 `remote[k]` 单个。得 r0,r1,r2。
- filter（N=2，kv_split_rank=t 选）：P 的 rank t 只持有 `c%2==t` 的逻辑块。
  → rank 0 写 r0,r2；rank 1 写 r1。两 rank 合起来恰好覆盖 D 的 r0,r1,r2，**无缺无重**。 ✓
- 对称性：indexer 和 MLA 在此场景走**同一** (M=1 展开, N=2 选择)，仅 head 维不同（binder 处理）。✓

## 8. 增量顺序（每步独立可测）

1. 加 `remote_kv_split` 字段（proto `KVTransferGroup` + `KVTransferMapping`）+ producer 填值
   + consumer 拷贝 + step-copy 透传（**行为保持**：折叠点仍用 instance `kv_split_size` 参数，
   字段无人消费，等价锚点必须全绿）。✅ 已完成 2026-09-19。
2. 配对公式换成读 per-tensor N（仍 N==M，字节不变）。
3. 推广 N↔M（真正的新逻辑）+ 序列覆盖 CHECK + byte-oracle。
4. indexer P切/D不切 端到端用例。
