# PD 路由重构 · S3 工作日志（自主轮次用）

> 本文是**跨轮次/跨会话的durable memory**。每轮结束后必须追加「进展」条目；不要依赖对话上下文。
> 计划与约束的权威版本仍是 `pd_routing_handover_20260917.md`（§7.4）与
> `pd_transfer_redesign_proposal_20260917.md`（§8/§9）。本文只记录**执行状态**。

## 0. 目标（S3）

把 S2 的三层（`PdRouteTable` / `RouteBinder` / `BufferDirectory`）**接入生产数据面**，并在接线前
解除阻塞项。按价值排序：

| # | 任务 | 验收 | 状态 |
|---|---|---|---|
| S3-0 | **解除生产 TU 的编译阻塞**：镜像里 `<torch_npu/torch_npu.h>` 路径不符（真身在 `torch_npu/include/torch_npu/csrc/libs/torch_npu.h`）。做一个 include shim 后，让 `kv_cache_transfer.cpp` / `mooncake_kv_cache_transfer.cpp` 能在容器内编过 | 两个 TU 用真实 flags 编译成功（或证明失败发生在无关的第三方头） | ✅ 第 1 轮 |
| S3-1 | **探针 §6.1 / §6.2 / §6.3**：index 行数比（切/复制）、kPool 打包宽度、`filter_kv_split_infos` 是否被跳过 | 有可复现的实测输出（日志或探针打印），结论写回 `pd_route_verification_plan` | ✅ 第 2 轮（靠既有实测存档） |
| S3-2 | **适配器**：`CacheTensorManifest`/`ParallelCoordinates` → `BufferDirectoryEntry` / `PeerCacheView`，字段映射见 handover §7.4 | host 单测：用**真实 `describe_cache_tensor`** 造 manifest，再与手算期望逐字段比对 | ✅ 第 3 轮（夹具手写）；真实 builder 输出由 S3-3 覆盖（第 4 轮） |
| S3-3 | **host 集成测试**：manifest → 适配器 → `PdRouteTable::build` → `RouteBinder::bind` → memcpy 端到端 | 目标场景逐字节正确（T5 的"真实 manifest"版本） | ✅ 第 4 轮（4 个场景，真实 `describe_cache_tensor`） |
| S3-4 | **数据面切换**：`transfer(edges, canonical_blocks, opcode)` 统一入口，先 PULL 后 PUSH；用 `--pd_route=legacy\|canonical` 开关，默认 legacy | 编译通过 + 单测；运行时行为未验证（见约束） | ☐ |
| S3-5 | **规范逻辑地址层**：`logical_offset` 改规范块坐标，`bind` 只做物理换算 | `S_P = S_D` 等价锚点 + `S_P ≠ S_D` 折叠用例 | ⚠️ 前置阻塞已定位（第 5 轮）：**物理切片 = `ContextParallelTopology::dcp_rank`**，S2 的 `slice_of` 必须改；修法已定案，代码待改 |

## 1. 硬约束（每轮先读，别再试错）

1. **cmake/ninja 重配置不可用**：`vcpkg install` 的 `detect_compiler` 找 `aarch64-linux-gnu-gcc`（镜像没有）；
   加 `-DVCPKG_MANIFEST_INSTALL=OFF` 后卡在容器外的 `FETCHCONTENT_SOURCE_DIR_LIBTORCH`。
   ⇒ 只能手编（`~/pdroute_tools/s2_host_test.py` 的模式）。
2. **21:58 预编译树不可靠**：`xllm/core/common/libcommon.a` 是坏档案（malformed archive），
   镜像里没有 `-lcust_opapi` ⇒ **依赖旧 planner 的测试链接不起来**。新层刻意只依赖 std + gtest。
3. **GLM5.3flash 目前不支持 PD 分离** ⇒ 真实 PD 端到端（T6）现在做不了。
4. **生产 TU 当前编不过**（约束 1 之外的独立问题：`<torch_npu/torch_npu.h>`），S3-0 就是为它准备的。
5. 远端开发机：`jd-node-98`（`ssh -F ~/work/.ssh-xllm-config`，跳板 `jd-jump`）。
   注意 shell 里 `ssh` 默认用 ControlMaster，写 `~/.ssh/cm-*` 会被沙箱拒绝；需要时加
   `-o ControlMaster=no -o ControlPath=none`。远端执行用 `rrun`（见全局规范）。
6. 构建沙箱 `~/workspace/xllm-pdroute`：base `72e0ea817`（相关目录与基线 `200939593` 无差异），
   S2 文件靠 scp 同步；**权威副本是 git 分支 `pd-routing-s0s1`（origin）**。
7. 容器镜像：`quay.io/jd_xllm/xllm-ai:xllm-dev-a3-arm-cann9-20260911`。
8. **torch 手编链接可用**（第 4 轮验证）：真实 `describe_cache_tensor` + `torch::zeros` 在容器内能编能链：
   `-L<site-packages>/torch/lib -Wl,-rpath-link,<site-packages>/torch.libs
   -Wl,-rpath,<torch/lib>:<torch.libs> -ltorch -ltorch_cpu -lc10`。
   `torch.libs` 是必须的（`libtorch_cpu.so` 依赖的 openblas 需要里面的 `libgfortran-e1b7dfc8.so.5.0.0`，
   少了会报未定义符号）。容器里 `import torch` 会失败（缺 NPU 运行时），所以路径从 compile flags 里的
   `/torch/include` 反推，不要 import。harness 已支持：`s2_host_test.py` 的 `"torch": True` 目标
   （`pd_route_integration_test`）。

## 2. 锁定决策（不要重开）

- `S_eff` 分母不整除 ⇒ **报错**；语义性全序列保留必须显式声明（`sequence_scoped` / `full_sequence_replica`）。
- `full_sequence_replica` 承载 indexer kPool（同组内 MLA latent 与 indexer 的 `S_eff` 不同）。
- **取消**与旧 `ReshardPlanner` 的逐边比对（旧路径无法表达 `TP8+DCP4` 的 MLA）；S2 验收 = golden + mock。
- F1 的 `EXPECT_EQ(is_kv_split_cache_block_type(t), S_eff > 1)` 断言作废。

## 3. 每轮收尾清单

- [ ] 更新本文「进展」；若有结论性发现，同步进 `pd_route_verification_plan` / `handover`。
- [ ] 代码改动跑 `s2_host_test.py`（或等价手编）并在容器内全绿；用 **clang-format 20.1.6**
      （`~/.cache/pre-commit/repoyk0pgd44/py_env-python3.14/bin/clang-format`）格式化。
- [ ] 提交到分支 `pd-routing-s0s1`；提交信息遵守 `scripts/lint/check-commit-msg.sh`
      （`type: 至少四个词并以句点结尾.`，不要写 scope，钩子未装但保持一致）。
- [ ] 推送：`GIT_SSH_COMMAND="ssh -o ControlMaster=no -o ControlPath=none" git push`。

## 4. 进展

### 2026-09-18（第 1 轮）
- 接手：S0/S1 `9605a7c6a` + S2 `abfcdc422` + 存档文档 `621a82b56` 已在 origin `pd-routing-s0s1`。
- 建立本文；目标见 §0，硬约束见 §1。
- **S3-0 ✅ 完成**：镜像里 `<torch_npu/torch_npu.h>` 的真实位置是
  `/usr/local/python3.11.15/lib/python3.11/site-packages/torch_npu/include/torch_npu/csrc/libs/torch_npu.h`。
  建 `/tmp/torch_npu_shim/torch_npu/torch_npu.h`（内容一行 `#pragma once` + 一行 `#include "<真实路径>"`），
  再用 `compile_commands.json` 的真实 flags 加 `-I/tmp/torch_npu_shim` 编译：
  `kv_cache_transfer.cpp` 与 `mooncake_kv_cache_transfer.cpp` **都 rc=0**（`-fsyntax-only`）。
  ⇒ 生产 TU 的编译阻塞解除，S3-4 可以被编译验证（仍不可运行时验证）。
- 工具：`~/pdroute_tools/probe_compile.py`（可传任意 TU 名，自动建 shim 并编译），
  运行方式 `rrun -F ~/work/.ssh-xllm-config jd-node-98 < run_probe_compile.sh`。
- 顺带更正 `pd_routing_handover` §5.8（该结论已过时）。
### 2026-09-18（第 2 轮）——S3-1 ✅ 三个探针全部落实（靠既有实测存档，未起新实例）

来源：用户上一轮会话的存档 `~/work/glm5-next-dcp-session/DCP4容量实测-中断存档.md`
（生产 DCP4 配置，`max_memory_utilization=0.62`；对应树 `xllm-glm53` @ `72e0ea817`）与
`_resume/DCP×PD兼容性与linear-cache静态审查-20260915.md`。

| 探针 | 结论 | 证据 |
|---|---|---|
| §6.1 index 切分还是复制 | **复制**（每 rank 保留全部规范块）⇒ `S_eff = 1` | 启动日志：`kv_cache_shape.cpp:195 Initializing indexer cache with shape: [26052 128 1 257]`，同一条日志 `blocks: 6513`、实例 `kv_split=4`；`26052 = 6513 × 4` = 全部规范块。代码侧 `index_block_count *= kv_split_size_effective()` |
| §6.2 kPool 打包宽度 | **257**（`index_kpool_compress = true`，即 `[k(128), gate(128), valid(1)]` 打包） | 同一形状最后一维 `257`（非 128）。⇒ mock/测试应以 257 为准，`head_bytes = 257 × dtype_size` |
| §6.3 kv_split 划分在哪一层 | **在 D 侧分配 block id 时完成**（rank-preserving 契约），传输层只做 1:1 块映射 | `rank_local_mapping = instance_info_.kv_split_size > 1 && has_rank_preserving_kv_groups(resp)`，对普通 KV 组恒真 ⇒ `filter_kv_split_infos` 的 remap 被整体跳过 |

**附带确认了 L3 的几何**（与 S2 实现逐位吻合）：
- KV：rank 行数 `n_blocks = 6513`，逻辑块 = `block_size(128) × kv_split(4)` = 512 token ⇒ `row = canonical / S_eff`（`S_eff=4`）；
- index：行数 `26052 = n_blocks × S` ⇒ `S_eff=1` 时 `row = canonical`。

**对 S3 范围的关键影响**（来自兼容性审查存档）：
1. 旧 PD 路径**只有**「两端 `kv_split` 完全相同」可用，且前置条件是
   `kv_split == cp_size × tp_size == world / dp`（因为等 size 时强制 `same_partition`，D 只把 rank 对齐的 P 设为 ACTIVE）。
2. 其余组合：P 分片/D 全量 ⇒ planner 放行但**块映射层 `CHECK` abort**；反向 ⇒ 建链期报错；两端不同 k ⇒ 建链期报错。
3. ⇒ 目标场景 `P: TP8 + kv_split4`（`S=4 ≠ cp*tp=8`）**落在旧路径支持范围之外**。这不是 bug，正是重构要打开的
   新形状；但也意味着 **S3-4 要替换掉 `filter_kv_split_infos` / `rotate_dst_rank` 那套 "S == TP 且 1:1 rank 对齐" 的隐含前提**，
   不能只做"换一条算路"。
4. `fingerprint`（`model_type:n_layers:kv_head_count:head_dim:index_head_count`）**不含 kv_split**，
   所以跨实例的 `kv_split`/`B_token` 不一致必须由新的 `KvTopology` 校验兜住（S5 的门禁提前）。

**本轮结论**：S3-1 关闭（无需起实例）；`pd_route_verification_plan` 的 §6.1/6.2/6.3 状态已更新为实测结论。

### 2026-09-18（暂停点，S3-2 进行中）

> ⚠️ 本节是**暂停时的设计稿**，其中的 API 草案（`CacheGroupGeometry` + 单个 `KvTopology` + 扁平 `page_bases`）
> 已在第 3 轮按下面的实现调整；结论与待办仍然有效。实现结果见「第 3 轮」。

**用户要求暂停以便压缩上下文**（目标已 pause，未推进到下一轮）。S3-2 的设计已经定下来，但**代码尚未落盘**，
下次从这里继续（重新 resume 目标即可）：

**要新增的文件**（放在 `xllm/core/framework/kv_cache_transfer/`）：

- `cache_directory.h`：
  - `struct CacheGroupGeometry { int32_t role; int32_t group_id; GroupTopology group; };`（模型侧声明，
    manifest 不含 `G` / `head_bytes` / 是否全序列复制 —— 这正是 §6.3 的结论：块身份不在 manifest 里）
  - `class PeerDirectory final`：
    `static bool describe(const WorkerCacheLayoutManifest& manifest, const KvTopology& topology,
    const std::vector<CacheGroupGeometry>& geometry, const std::vector<uint64_t>& page_bases,
    PeerDirectory* directory, std::string* error);`
    + `const PeerCacheView* find(CacheNamespace, int64_t layer_id, int32_t role, int32_t group_id) const;`
    + `views()/size()`。
- `cache_directory.cpp`：只做"解释 manifest + 与模型侧声明对账"，至少校验：
  1. `manifest.coordinates` 的 `cp/tp/kv_split` 与传入 `KvTopology` 一致；
  2. `group.sequence_scoped == (tensor.shard.resource_scope == SEQUENCE)`；
  3. `units_per_resource`：SEQUENCE 取 `physical_rows_per_resource`，BLOCK 取 `block_token_capacity`
     且**必须等于** `topology.tokens_per_block`（跨实例 `B_token` 前置条件）；
  4. `head_bytes` 自洽：整行 span（`describe_replicated_tensor`，REPLICATED、`bytes_per_region == resource_stride_bytes`）
     或每 head 一个 span（`bytes_per_region == head_bytes`）；
  5. `local_heads = resource_stride_bytes / (units × head_bytes)` 必须是整数，且
     **等于 `KvRedundancy::derive(topology, group).local_head_count()`**；
  6. 分片组：span 数 == `local_heads`，且各 span 的 `logical_offset_bytes / head_bytes`（= global head）
     恰好等于该 rank 的 head class 区间 `[head_begin(head_class_of(tp_rank)), +local_heads)`；
     并据此校验 `max(global_head)+1 == G`；
  7. `explicit_offsets` 时 `page_bases` 必须给出（长度 ≥ resource_count），否则必须为空。
- `cache_directory_test.cpp`：手工构造**遵循生产描述符规则**的 manifest（MLA replicated / indexer
  `full_sequence_replica` / 分片 KV / SSM checkpointed / explicit_offsets）+ 负例（head_bytes 不符、
  scope 不符、`units_per_resource` ≠ `tokens_per_block`、local head 数与声明不符）。

**本轮已读到的关键事实**（写适配器时直接用，不必再读）：

| role | 描述符形状（`cache_layout_builder.cpp`） |
|---|---|
| MLA（`enable_mla`） | `describe_replicated_tensor`：**一个 span**，`bytes_per_region = resource_stride_bytes`，`owner_tp_rank = 0`，`kind = REPLICATED` |
| INDEX / INDEX_SCALE | 同 MLA 分支 → 同样走 replicated（实际生产 `index_kpool_compress=true`，宽度 257） |
| 普通 KV（`describe_attention_heads`） | 每 local head 一个 span：`logical_offset = global_head × head_bytes`，`bytes_per_region = head_bytes`，`repeat_count = token_count`，`logical_stride = G × head_bytes`，`physical_stride = tensor_stride_bytes(token_axis)` |
| SSM（`describe_ssm`） | 每 local head 一个 span：`repeat_count = linear_ssm_checkpoint_stride`（本例 3），`logical_stride = linear_value_head_count × head_bytes`，`physical_stride = tensor_stride_bytes(tensor, 0)`，`resource_scope = SEQUENCE` |
| CONV（`append_conv_component`） | key/value 两个 component **追加到同一个 descriptor**：每 local head 一个 span，`bytes_per_head = linear_key_head_dim × element_size`，`repeat_count = tensor.size(1)`（state_count），`physical_offset = component_offset_bytes + local_head × bytes_per_head`，`logical_stride = global_head_count × bytes_per_head` |
| 其他（无 head 轴） | `describe_replicated_tensor` 的整资源复制 |

⚠️ CONV 的 descriptor 里 **key 与 value 的 span 混在同一个 descriptor**，且 `physical_offset_bytes` 带
component 偏移 ⇒ 适配器对分片组的"span 数 == local_heads"校验**不能直接套用到 CONV**，要么按
`logical_tensor` 分组后分别校验，要么对 CONV 放宽为"按 component 分组校验"。下次先确认
`describe_conv` 的整体拼装（本文件 `append_conv_component` 之后的部分尚未读完）。

**未验证事项**：S3-2 现在只是设计；`describe_cache_tensor` 的 **torch 链接测试**是否可行尚未验证
（若可行，比手工 fixture 更强；若不可行，按既有 `reshard_planner_test.cpp` 的 fixture 约定写，并在文档标注
"约定一致、非 builder 实测"）。

### 2026-09-18（第 3 轮）——S3-2 ✅ 适配器落地

**新增/改动文件**：

| 文件 | 内容 |
|---|---|
| `xllm/core/framework/kv_cache_transfer/cache_directory.h` | `CacheTensorDeclaration`（模型侧声明：`(namespace, role, group_id)` + `KvTopology` + `GroupTopology`）、`CacheRowBases`（页映射按行基点）、`class PeerDirectory final`（`describe` / `find` / `size` / `at`） |
| `xllm/core/framework/kv_cache_transfer/cache_directory.cpp` | 解释 manifest + 对账，产出 `PeerCacheView`（含 `local_rank`） |
| `xllm/core/framework/kv_cache_transfer/route_binder.{h,cpp}` | `PeerCacheView::local_rank`（默认 `-1` = 不校验）；`bind` 跳过异源 rank 的边、拒绝目的视图与 `dst_local_rank` 不符 |
| `tests/core/framework/kv_cache_transfer/cache_directory_test.cpp` | 18 个用例（8 正例 + 10 负例），含"S3-3 迷你版"：两个真实 manifest → 建表 → 绑定 → 逐字节覆盖断言 |
| 两处 `CMakeLists.txt` | `cc_library(cache_directory)` / `cc_test(cache_directory_test)` |

**与暂停稿设计的差异（都是被实现逼出来的）**：

1. `CacheGroupGeometry` → `CacheTensorDeclaration`：声明里带上该族的 `KvTopology`。理由：SPEC_DRAFT 的
   manifest 坐标描述的是 MAIN 的拓扑，draft body 有自己的 TP；同时 MAIN 的声明必须与 manifest 坐标一致（已校验）。
2. 扁平 `page_bases` → `CacheRowBases` 列表（按 `(namespace, layer, role, group_id)` 查）。理由：页基点是按张量
   给的，XTensor 下 `buffer_id = 0`、`buffer_bytes` 是整个全局区，一张表无法表达多张张量。
3. 检查项比设计多了三条：`layout` 必须是 binder 真正寻址的那种（每 head 一个 span、head 在资源内连续、
   `repeat_count == units`）、`owner_tp_rank == class × D_tp`、`local_rank` 与坐标一致。前两条把"约定"
   变成"可证明"，第三条顺带补上了 S2 的一个洞（详见下）。

**本轮确认的两个实现约束**（已写进 proposal §8.1 与 handover §7.4）：

1. **COMPOSITE（CONV）不在规范路由的表达范围内**。读完 `describe_conv` 后确认：三个 component 共用一个
   descriptor，`logical_offset` 是 **component 局部**的 head 索引，`physical_offset` 带 component 偏移，而
   `RouteBinder` 只会按 `(head - rank_head_begin) × head_bytes` + `unit × local_heads × head_bytes` 寻址。
   适配器**显式拒绝**（错误信息里点名 composite），S3-4 需要决定：COMPOSITE 组继续走旧 planner，还是给
   `RouteEdge` / `PeerCacheView` 增加 per-component 字节偏移。
2. **整资源（whole-resource）描述符只对 `G == 1` 可路由**。`describe_replicated_tensor` 只给一个覆盖整行的
   span、不带 head 轴；`G > 1` 时无法说明自己持有哪些 head。生产三个来源（MLA latent、INDEX/INDEX_SCALE、
   无 head 轴 role）都是 `G == 1`，所以不影响现状。唯一需要小心的歧义：`units == 1 && H_l == 1 && G > 1`
   （例如 `checkpoint_stride = 1` 的 SSM）时，单 span 既像整资源又像"一个 head 一行"，代码按**每 head**解释
   （`whole_resource` 分支要求 `G == 1`），并在该分支之外保留了这一退化路径。

**顺带补的 S2 洞**：`PeerCacheView` 之前不携带 rank，`bind` 无法判断传入的 `local` 是哪个源 rank —— 传入
整张边表时，属于别的源 rank 的边会用**本视图的 buffer** 配上**那条边的源 rank 几何**算出区间，静默错字节。
现在 `local_rank >= 0` 时跳过异源边（覆盖校验仍会抓住"没有写者"），`remote.local_rank` 则必须等于
`dst_local_rank`。默认 `-1` 保持 S2 既有单测不变。

**验证**（容器内，`~/pdroute_tools/s2_host_test.py`，已加 `cache_directory_test` 目标）：

```
kv_redundancy_test  11/11 PASSED
pd_route_test       12/12 PASSED
cache_directory_test 18/18 PASSED
```

夹具与实测的交叉验证：MLA 夹具（`6513` 行 × `128` token、`TP8`、`kv_split=4`、`G=1`）派生
`S_eff=4`、`replica=2`，与实测 `index 26052 = 6513 × 4`（§6.1）的分配几何一致；indexer 夹具
`head_bytes = 514 = 257 × 2` 与 §6.2 的打包宽度一致。

**仍未做的**：

- 夹具是照 builder 公式手写的，没有链 torch 跑真实 `describe_cache_tensor`。仓库里
  `tests/core/framework/kv_cache/cache_layout_builder_test.cpp` 已证明"torch 张量 + builder"在真实构建下可行
  （它链接 `:kv_cache` + ascendcl），所以真实构建的 `cache_directory_test` 可以同样用 `torch::zeros` 造 manifest；
  host 手编回路要链 torch 才行，本轮没做。
- `coordinates.kv_split_rank`（运行时 DCP rank）与 `KvLayoutIndex::slice_of` 尚未对账。**这是 S3-4/S3-5 的前置**：
  在"规范块 ↔ 请求 block id"换算落地前必须统一，否则无法判断请求里的 id 属于哪个 rank 的切片。
- `bind` 的 `local_rank` 校验只覆盖 MAIN 命名空间；SPEC_DRAFT 视图填 `-1`（其描述符自带 placement，而坐标是
  MAIN 的），调用方需自行过滤边。

### 2026-09-18（第 4 轮）——S3-3 ✅ 端到端 host 集成测试（真实 builder 输出）

**新增**：`tests/core/framework/kv_cache_transfer/pd_route_integration_test.cpp`（4 个场景，全部逐字节比对）、
`tests/.../CMakeLists.txt` 的 `pd_route_integration_test`（链接 `:kv_cache`，与 `cache_layout_builder_test` 同款），
以及 harness 的 torch 目标（配方见 §1.8）。**S3-2 遗留的"真实 `describe_cache_tensor`"要求由此关闭。**

链路：`torch::zeros` → 真实 `describe_cache_tensor` → manifest（字段赋值照抄 `register_kv_cache`）→
`PeerDirectory::describe` → `PdRouteTable::build`/`validate` → `RouteBinder::bind` → host memcpy。

**判别性设计**：期望值不由 bind 算出，而是"规范内容函数 + 目标侧自己的描述符"两条独立信息合成：

- `content_byte(group, 资源, head, 子单元, 偏移)` 只依赖逻辑身份；
- 每个字节的物理位置由**该侧描述符**的 span 推出（整资源 span 按其 head 轴展开）；
- `资源 = row * split + slice` 由模型（`KvLayoutIndex` / `CanonicalBlock`）推出。
⇒ 字节落到错的 (资源, head, 子单元, 偏移) 必然不匹配；没被写到的字节留在 poison 上也不匹配。

| 场景 | 覆盖 |
|---|---|
| `AnchoredEqualSplitReproducesTheSourceLayout`（MLA，kv4 → kv4） | 等价锚点（handover 要求每步保留） |
| `TargetSplitMismatchFoldsFourSlicesIntoTwo`（MLA，kv4 → kv2） | **目标形状** `S_P ≠ S_D`（旧路径不支持的形状） |
| `ReverseSplitMismatchExpandsTwoSlicesIntoFour`（MLA，kv2 → kv4） | 发散方向：一个源切片扇出到两个目的切片 |
| `ShardedHeadsReshardAcrossHeadClasses`（非 MLA，cp4/tp8/kv4 → cp4/tp4/kv2） | head class 交集（`H_l` 1→2）、indexer 全副本扇出、sequence-scoped SSM |

MLA 场景每个都覆盖 4 个 role：KEY（MLA latent）、INDEX（全序列副本）、SSM（sequence-scoped）、
CONV（**整资源打包行**）。非 MLA 场景不构造 CONV，因为那时 builder 会给出 COMPOSITE 描述符、规范路由按设计拒绝
（`RejectsCompositeCacheGroups` 已断言）。

**修正 S3-2 的一条结论（重要）**：整资源 span 的准入条件不是"`G == 1`"，而是"**本地只有 1 个 head**"，
且**该 head 的身份来自 rank**（`head_class_of(tp_rank) * H_l`），不来自 span。原因：`describe_replicated_tensor`
在 MLA 实例里会给**所有** role 整资源 span（`enable_mla` 分支最先命中），包括 SSM/CONV；这些 rank 物理上只持有
自己的本地 head，而描述符的 `logical_offset` 恒为 0、无法表达"我持有哪个 head"。按旧规则（要求 `G == 1`）时
MLA 实例下的 SSM/CONV 只有 rank 0 能通过，整条链路建不起来 —— 本轮第一次跑集成测试就撞上：
`role 4, group 1: the descriptor holds head class 0 but tp rank 1 owns class 1`。
现在 `H_l == 1` + rank 推出 `head_begin` ⇒ MLA 下的 SSM/CONV 可路由（两侧都必须 `H_l == 1`，否则报错要求
producer 改成每 head 一个 span）。`H_l == 1` 这个限制同时保住了 CONV 的安全：只有一个 head 时资源内部没有 head
顺序可言（打包的 key_a/key_b/value 作为一个整体搬运），不会猜错 component 布局。

⇒ **结论修正：COMPOSITE（CONV）只在非 MLA 实例出现**（`enable_mla == false` 时 `describe_conv` 才可达）；
MLA 实例里 CONV 走整资源路径，`H_l == 1` 时正确、`H_l > 1` 时被拒。

**仍未覆盖 / 未验证**（文档继续标注）：

- `MixedLayers`（同一 rank 上不同层的 role 集合不同）：夹具是单层；目录按 `(namespace, layer, role, group)`
  查找，多层只是多几份拷贝，未单独建例。
- `DpExpansion`（P DP1 → D DP4）：`RouteEdge` 不含 DP 维（F3 的设计），DP 展开在 S2 单测里已固定；本集成测试
  只覆盖 DP=1。
- `XTensor explicit_offsets` 的端到端：适配器字段由 `cache_directory_test`、字节 golden 由 `pd_route_test` 覆盖，
  集成测试用的是按 stride 寻址的缓冲。
- 真实运行时（RDMA / NPU / 真实调度 block id）仍未验证：GLM5.3flash 尚不支持 PD 分离（§1.3）。

**验证汇总**（容器内，`~/pdroute_tools/s2_host_test.py`）：`kv_redundancy_test` 11/11、`pd_route_test` 12/12、
`cache_directory_test` 19/19、`pd_route_integration_test` 4/4 **全绿**。
搬运量打印：MLA kv4→kv4 6.6 MB、kv4→kv2 8.9 MB、kv2→kv4 6.6 MB、非 MLA 12.6 MB（含 indexer 全副本扇出）。

### 2026-09-18（第 5 轮）——S3-5 前置：**物理切片 = DCP rank**，S2 的 `slice_of` 与之不一致

> 本轮只做定位与定案（代码未改，保持全绿）。**这是 S3-4 接线前必须修的正确性问题**：不修就会静默搬错块。

**问题**：`KvLayoutIndex::slice_of(cp, tp) = (cp*D_tp + tp % D_tp) % S_eff`（proposal §3.1 原文）。
它假设"一个 DCP 组 = 同一 head class 的 `D_tp` 个 TP 副本"，于是把切片分配给 `tp % S_eff`。

**事实（三处独立证据）**：

| 证据 | 内容 |
|---|---|
| `kv_shard_layout.cpp:63-70` | `globalize(local_slot) = (local_block_id * dcp_size + dcp_rank) * block_size + offset` ⇒ **规范块 = 行 × S + `dcp_rank`**（这正是 `CanonicalBlock` 的公式，`CanonicalBlock` 本身没错），且 `owner_of(b) = b % S` |
| `context_parallel_topology.cpp:43-73` | DCP 只有两种合法形状：(a) `S ≤ cp_size && cp_size % S == 0` ⇒ DCP 组是 PCP 组的划分，`dcp_rank = cp_rank / (cp_size/S)`；(b) `S == cp_size*tp_size` ⇒ DCP 覆盖整个 DP-local 域，`dcp_rank = cp_rank*tp_size + tp_rank`。**其它形状直接 `CHECK` 失败** |
| `layers/npu_torch/qwen_dcp_attention.cpp:121-125` | NPU 路径就是 `KVShardLayout(block_size, dcp_group.world_size(), dcp_group.rank())`，而 `parallel_args.kv_split_rank()` 返回 `dcp_group_->rank()` ⇒ **NPU 的物理切片就是 `ContextParallelTopology::dcp_rank`** |
| 归档（`glm5-next-dcp-session/_remote/glm5_3_flash_dcp_analysis.md:250`） | "DCP group 为 strided 分组（`global_rank % (world/kv_split)`）" ⇒ 组 = `{r : r % (world/kv) = g}` |

**对 pilot 的影响**：`world/kv_split = 8` 且 `tp_size = 8` ⇒ 组 = 固定 tp、变动 cp ⇒ **`cp_size = 4`（PCP=4），`dcp_rank = cp_rank`**。
所以 §6.5 里"P: TP8 + kv_split4"的写法应更正为 **`cp_size=4` + `tp_size=8`（world 32，`kv_split_size` 不设 ⇒ effective = cp_size = 4）**；
`S_eff = 4` 的来源是 `S | cp_size`（case a，`pcp_per_dcp = 1`），不是"TP 冗余里挤出来的"。
MLA 组（`G=1`）：`D_tp = 8`、`D = cp*D_tp = 32`、`N_rep = 8`。S2 模型给的 `slice = tp % 4`、写者 `tp ∈ {0,1,2,3}`（cp=0）**都是错的**；
正确是 `slice = cp_rank`、写者 `(cp=s, tp=0)` ⇒ 局部 rank `s*8`。

**不受影响的部分（好消息）**：`CanonicalBlock`（`canonical = row*S + slice`）与 `owner_of` 的**数值口径完全正确**；
`N_rep = D/S`、`Hc*Hl = G`、每 `(h,t)` 恰一个写者、边表规模（`Hc_pairs × S × period × N_rep`）**都不变**。
变的只是"哪个 rank 属于哪个切片/副本"，因此 S2 的**边表 golden 里的具体 rank 需要重算**，
而所有不变量型断言（`kv_redundancy_test` 的穷举矩阵）保持不变。

**修法（下一轮执行，已定案）**：

1. `KvRedundancy::derive` 增加 C3 校验：`(S ≤ cp_size && cp_size % S == 0) || (S == cp_size*tp_size)`，否则报错
   （报错信息要点名 DCP 拓扑，因为绕过它会在 `ContextParallelTopology` 里 `CHECK` 崩溃）。
2. `KvLayoutIndex` 的 `slice_of` / `replica_of` 改成按 (a)/(b) 两分支：
   - (a) `w = cp_size/S`：`slice = cp_rank / w`，序列副本组 `j = cp_rank % w`；
     `replicas_of(h,t) = {(cp = t*w + j, tp = h*D_tp + k) : j∈[0,w), k∈[0,D_tp)}`，`writer_of` 取序首；
   - (b) `slice = cp_rank*tp_size + tp_rank`，`replicas_of(h,t)` 只有 `dp_local == t` 那一个 rank（此时必有 `G=1`）。
3. 受影响的测试与夹具（同一批改完再提交）：
   - `pd_route_test`：T3 的两张 golden 表（MLA 4 条 / KDA 8 条）与 T4 字节 golden 的 rank 需按新规则重算；
   - `kv_redundancy_test`：穷举矩阵要按 C3 过滤（并把被拒形状写成负例）；
   - `cache_directory_test` 的 e2e 用例、`pd_route_integration_test` 的 MLA 场景（当前 `cp=1` 配 `S=4` 在 C3 下**非法**）
     要改成 `cp_size=4`（如 `cp4/tp2/kv4` 或 `cp4/tp8/kv4`）；顺带让集成测试变成"运行时忠实"的拓扑。
4. 文档：proposal §3.1 的 `slice/replica` 公式、handover §6.5 的目标场景描述、验证文档 F4 的行↔块映射说明。

**教训**：S2 的三层只验证了"自洽"（同一条公式在两侧一致就过），没有任何一条断言把公式钉在
`ContextParallelTopology` / `KVShardLayout` 这个运行时契约上。补 C3 与上述夹具时，应把"`slice` 必须等于
`ContextParallelTopology::dcp_rank`"写成一条**显式断言**（在集成测试里用 `ContextParallelTopology` 算出期望切片，
而不是用 `KvLayoutIndex` 反推），这样下次不一致会立刻暴露。

