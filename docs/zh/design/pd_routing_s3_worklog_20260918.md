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
| S3-4 | **数据面切换**：`transfer(edges, canonical_blocks, opcode)` 统一入口，先 PULL 后 PUSH；用 `--pd_route=legacy\|canonical` 开关，默认 legacy | 编译通过 + 单测；运行时行为未验证（见约束） | ⚠️ **第 7~10 轮**：入口 + F8 表缓存 + 开关 + 生产声明映射 + id↔规范块换算 + **对端实例装配**已落地，69 用例全绿；**只剩把三者接进 `push_kv_blocks_async`**（步骤见第 10 轮末尾，无未知项） |
| S3-5 | **规范逻辑地址层**：`logical_offset` 改规范块坐标，`bind` 只做物理换算 | `S_P = S_D` 等价锚点 + `S_P ≠ S_D` 折叠用例 | ⚠️ **换算规则已查清并落地**（第 9 轮：`canonical_blocks_of_request` + 运行时 oracle，67 用例全绿）；**尚未接进数据面** |

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
9. **运行时本体可以链进手编 harness（第 9 轮验证）**：`kv_shard_layout.cpp` 与
   `context_parallel_topology.cpp` 只依赖 glog，链接行加
   `libglog.a` + `libgflags.a`（都在 `vcpkg_installed/arm64-linux/lib/`）即可；
   harness 的 `TARGETS` 支持 `"glog": True` 与 `"defines": [...]`（`kv_shard_contract_test` 即用此）。
   注意真仓库构建里 `context_parallel_topology` 属于很重的 `:parallel_state`（且依赖坏档案 `:common`），
   所以该 oracle 目前只在手编 harness 里跑，用 `XLLM_HAVE_CONTEXT_PARALLEL_TOPOLOGY` 宏开关。
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

### 2026-09-18（第 6 轮）——S3-5 切片契约修正落地（C3 + `KvLayoutIndex`），全绿

**改了什么**（`kv_redundancy.{h,cpp}`）：

- `KvRedundancy::derive` 增加 **C3**：配置的 `kv_split` 必须是运行时可放置的 DCP 形状
  （`S | cp_size`，或 `S == cp_size * tp_size`），否则报错并点名 DCP。
  注意 C3 校验的是**实例配置值**（`kv_split_size_effective`）而不是每组的 `S_eff` —— DCP 组是按实例建的。
  副作用：C3 + C2 合起来让"配置 `S>1` 但本组 `D==1` 退化为 1"这条分支几乎不可达（`cp=1` 时配置 S 只能是 1 或 `tp`，
  而后者要求 `G==1`），文档里保留该分支但注明。
- `KvLayoutIndex` 的 `slice_of` / `replica_of` / `writer_of` / `replicas_of` 改成两分支：
  - (a) `S | cp_size`：`w = cp/S`，`slice = cp/w`，`replica = (cp%w)*D_tp + tp%D_tp`，
    副本集 = `{(cp = slice*w + j, tp = h*D_tp + k) : j∈[0,w), k∈[0,D_tp)}`；
  - (b) `S == cp*tp`：`slice = cp*tp_size + tp`，`replica = 0`，副本集只有 `dp_local == slice` 的那个 rank。
- `KvRedundancyTest.DerivesTheGlm53FlashPilotScenario` 现在显式钉住契约：prefill（cp4/tp8/kv4）
  `slice == cp`、`replica == tp`、写者 `local_rank == 8*cp`；decode（dp4/cp1/tp2/kv2）`slice == tp`、写者即本 rank。
  另新增 `RejectsSplitThatIsNotADcpShape`（并断言 `cp=4` 下同一 split 合法）。

**测试夹具同步改完**（都要按 DCP 合法形状重算）：

| 文件 | 改动 |
|---|---|
| `pd_route_test.cpp` | pilot prefill `cp1/tp8/kv4` → `cp4/tp8/kv4`；MLA golden 的 4 条边 → src `0/8/16/24`、dst `0/1/0/1`；KDA 与 indexer golden **不变**（边数 8/2、rank 与 head 区间一致）；fan-out 边数 `8 → 32`（目的侧 8 个副本）；mock 的写者 rank 改为 `(block%4)*8`、源缓冲 32 个 rank；拓扑矩阵扩到 `cp ∈ {1,2,4}` |
| `cache_directory_test.cpp` | MLA 夹具与该 mini e2e 用例改成 `cp4/tp8`（源 32 个 rank，写者 `(cp=slice, tp=class*D_tp)`）；`replica_count` 断言 2 → 8 |
| `pd_route_integration_test.cpp` | 三个 MLA 场景的 `cp_size` 1 → 4 |
| `kv_redundancy_test.cpp` | sequence-scoped / full-sequence-replica / no-redundancy 三个夹具改成 DCP 合法形状；C2 负例改用例 (b) 的形状（`cp1/tp8/S8`，`G=4/2`）；穷举矩阵扩到 `cp ∈ {1,2,4}` |

**验证**（容器内，`~/pdroute_tools/s2_host_test.py`）：`kv_redundancy_test` **12/12**、`pd_route_test` **12/12**、
`cache_directory_test` **19/19**、`pd_route_integration_test` **4/4** —— 47 个用例全绿。
搬运量：MLA kv4→kv4 6.6 MB、kv4→kv2 35 MB、kv2→kv4 26 MB、非 MLA 12.6 MB（目的侧副本变多）。

**未做 / 下一步**：

- **还没把运行时当真 oracle**：`context_parallel_topology.cpp` 需要 glog，手编 harness 要多带一个库，
  本轮用"显式断言 pilot 的 `slice == cp` / `== tp`"代替。下一轮建议给它加 glog 链接，
  在单测里直接用真实 `ContextParallelTopology` 反推期望切片（`cp ∈ {1,2,4}` × `S ∈ {1,2,4,8}` 全枚举比对）。
- **S3-5 的"规范逻辑地址层"其余部分未接线**：`logical_offset` 走规范块坐标、请求 block id ↔ 规范块的换算
  （`KVShardLayout::globalize/localize`）还没有进入数据面 —— 与 S3-4 一起做。
- S3-4 的开关（`--pd_route=legacy|canonical`，默认 legacy）与统一 `transfer(opcode)` 入口未开始。

### 2026-09-18（第 7 轮）——S3-4 的入口落地：`PdRouteTransfer` + `--pd_route`，59 用例全绿

**新增/改动文件**：

| 文件 | 内容 |
|---|---|
| `xllm/core/framework/kv_cache_transfer/pd_route_transfer.{h,cpp}` | **新增**：`PdRouteMode`（`legacy`/`canonical` + 解析）、`RouteOpcode`（PULL/PUSH）、`RoutePeer`（对端实例的 addrs + 视图）、`RouteLeg`（一条 (writer rank, reader rank) 腿，含传输就绪的 `RouteRegion`）、`PdRouteCache`（按两侧形状缓存边表，F8）、`PdRouteTransfer::plan/transfer/apply` |
| `route_binder.cpp` | **加固**：`bind` 现在校验"传入的规范块确实属于目的 rank 的切片"。此前若调用方分组错误，块会被静默写进错 rank 的缓冲（`remote_row = block / S_D` 假定块属于该 rank） |
| `cache_directory.cpp` | **对账**（关闭第 3 轮遗留项）：MAIN 且本组 `S_eff == 配置 split` 时，`KvLayoutIndex::slice_of(cp,tp)` 必须等于 manifest 公布的 `coordinates.kv_split_rank`（= 运行时 DCP rank）。`S_eff == 1` 的组没有切片可分，跳过 |
| `config/disagg_pd_config.{h,cpp}` | `--pd_route`（默认 `legacy`，注册进 `option_category`，flag/json/回写三处齐全） |
| `kv_cache_transfer.cpp` | 工厂里解析并校验 `--pd_route`；非法值 `LOG(FATAL)`（不猜、不回落）；`canonical` 目前显式拒绝并说明缺什么（见下） |
| `tests/.../pd_route_transfer_test.cpp` | **新增 12 用例**；`cache_directory_test.cpp` / `pd_route_integration_test.cpp` 的 fixture 改按真实 DCP 公式公布 `kv_split_rank` |

**两个方向同时验证（这是本轮最关键的判别性设计）**：每个场景都跑两遍 —— 从 writer 侧每个 rank 各 `transfer(PUSH)` 一次，从 reader 侧每个 rank 各 `transfer(PULL)` 一次 —— 然后

1. 两份目的侧内存都必须等于**独立期望值**；
2. 两份目的侧内存必须彼此逐字节相等。

期望值只由模型（`KvLayoutIndex` + `writer_of` + `CanonicalBlock`）与两侧声明的几何推出，**从不看边表或 region**，
所以"选错 rank / 错切片 / 错 head / 错子单元"都会被抓到。PUSH/PULL 相等这条则把 `RouteBinder` 的
"destination-last（`local` = writer）"取向与 `move_memory_regions` 的 READ/WRITE 取向钉在一起：
PULL 的 region 就是同一对区间把两半**对调**，不是重新推导。

场景（`cp4` 为 DCP 合法形状的前提）：等价锚点 `kv4→kv4`、收拢 `kv4→kv2`、发散 `kv2→kv4`、
非 MLA 头分片 `cp4/tp8/kv4 → cp4/tp4/kv2`；每个都带一个 sequence-scoped family（checkpoint 子单元）。
外加：单 writer 扇出到 8 个 tp 副本的腿数 golden、PULL 的多写者选择、副本 rank 不推（且不是错误）、
表缓存复用（同形状 1 张表、换形状 2 张表）、非递增/负块号报错、对端缺视图报错、传输失败即失败、
"另一目的切片的块"被 `bind` 拒绝、开关解析拒绝大小写/空串。

**验证**（容器内，`~/pdroute_tools/s2_host_test.py`）：`kv_redundancy_test` 12/12、`pd_route_test` 12/12、
`cache_directory_test` 19/19、`pd_route_transfer_test` 12/12、`pd_route_integration_test` 4/4 —— **59 用例全绿**。
生产 TU 编译（S3-0 的 shim + 真实 flags，`-fsyntax-only`）：`kv_cache_transfer.cpp`、`mooncake_kv_cache_transfer.cpp`、
`disagg_pd_config.cpp` **全部 rc=0**，即新增头文件与配置项在真实 include 环境下编得过。

**为什么生产调用点还没接（诚实记录）**：canonical 路线要的输入，数据面现在拿不到：

1. **本侧声明**（每个 `(namespace, role, group)` 的 `global_head_count` / `sequence_scoped` /
   `full_sequence_replica`）。`configure_cache_layout` 有 `ModelArgs` + `ParallelArgs`，
   `publish_cache_layout` 有全部张量的 role/group，但**role → 组几何**的映射目前只存在于
   `pd_route_integration_test.cpp` 的手写 fixture 里，生产侧没有这个函数。写错它 = 静默搬错块，
   而它无法在本轮验证（GLM5.3flash 不支持 PD 分离，见 §1.3）。
2. **对端视图**：对端 manifest 有（`MooncakeTransferEngine::cache_peers_` 的 `destination_manifest`），
   但 `TransferKVInfo.remote_instance_info` 只给 `dp_size` / `kv_split_size` / `addrs` / `cluster_ids`，
   不含 `cp_size`、不含各 rank 的视图；`addrs` 的下标语义是"对端实例内的**全局** rank"
   （`dp*(cp*tp) + local`），要靠对端 manifest 的 `coordinates` 才能换算成 DP 组内局部 rank。
3. `RouteRegion` → `ByteRegion` 是**逐字段拷贝**（两者字段完全一致），这层已经是 1:1 的；
   `RouteLeg` 已经按 `peer_addr` + opcode 组织好，接上就是 `move_memory_regions(addr, regions, opcode)`。

⇒ 因此本轮把开关接成"要么 legacy、要么**显式拒绝**"，而不是让它静默退化成 legacy（那种回落会让人以为
canonical 已经生效）。下一轮把这些输入补齐后再打开 `canonical`。

**本轮的第二个改动**：`pd_route_integration_test.cpp` 的数据面调用点从"直接调 `RouteBinder::bind`"改成
**走统一入口** `PdRouteTransfer::transfer(PUSH, rank, canonical_blocks, local, peer, move, ...)`：
每个源 rank 只调一次，由入口自己推导 `(writer rank, reader rank)` 腿、选出该腿的规范块、并把 region 按
opcode 取向排好。顺带修了夹具的一个真实缺陷：**buffer id 之前是每 rank 从 0 开始编号的**，而传输层是按
id 单独寻址缓冲（生产里 Mooncake 按注册顺序给全局唯一 id）；改成全局计数器后，四个角色的搬运量
（`9437184 / 16842752 / 8192 / 6144`）与重构前**逐位相同**，即入口复现了原先逐字节验证过的行为。

**其它可做的增强（未做）**：
- 把 `ContextParallelTopology` 本体链进单测当运行时 oracle（harness 需要加 glog 链接），
  现在用的是"C3 + 显式断言 `slice == cp` / `== tp`"。
- `bind` 的 `local_rank` 守卫仍只覆盖 MAIN（SPEC_DRAFT 视图 `local_rank == -1`）；
  `plan` 用"同族只有一个未命名视图才接受"来兜底，多于一个直接报错。


### 2026-09-18（第 8 轮）——S3-4 输入 (a)：**生产侧的 role → 组几何声明**，61 用例全绿

上一轮列出生产接线缺两项输入。本轮把 **(a) 本侧声明** 做完并验证；**(b) 对端视图** 的接法也已定案（见下）。

**新增**：`declare_cache_group(const CacheTensorLayoutContext&, KVCacheTensorRole, GroupTopology*, std::string*)`
（放在 `cache_directory.{h,cpp}`，**不依赖 torch**）。它是 `describe_cache_tensor` 的"声明半边"：
描述符说一个 rank 的字节怎么排，它说这个组暴露几个逻辑 head、有没有块维度、是否全序列留在每个 rank。
两者由同一个 role + 同一份 layout context 推出，所以布局改了必须同时改。

| role（按 `describe_cache_tensor` 的分支镜像） | `global_head_count` | `sequence_scoped` | `full_sequence_replica` |
|---|---|---|---|
| `is_kv_head_role`（KEY/VALUE/KEY_SCALE/VALUE_SCALE/CACHE_SCALE）且 `kv_head_count > 0` | `enable_mla ? 1 : kv_head_count` | false | false |
| INDEX / INDEX_SCALE | 1 | false | **true** |
| SSM（且 `linear_value_head_count > 0`） | `linear_value_head_count` | true | false |
| CONV（且两个 linear 计数都 > 0） | `linear_value_head_count` | true | false |
| 其它（WINDOW/SWA/KV_STATE/SCORE_STATE/COMPRESS_* 等无 head 轴者） | **报错拒绝** | — | — |

三条本轮查证的依据（都不是猜的）：

1. **MLA 下 KV 组的 `G` 必须是 1**：`enable_mla` 时 builder 走 `describe_replicated_tensor`（整资源、无 head 轴），
   适配器要求 `H_l == 1`；`G=1` 给出 `D_tp = tp`、`Hc = 1`，正是第 6 轮 `DerivesTheGlm53FlashPilotScenario`
   钉住的契约（`slice == cp`、写者 `local_rank = 8*cp`）。若错用 `G = kv_head_count`，`D_tp` 变 1、`Hc` 变 8，
   写者会变成 `(cp=t, tp=h)` —— 路由**完全不同**且不会报错，所以这一条必须由声明钉死。
2. **INDEX_SCALE 与 INDEX 同宽**：`KVCacheShape::init_index_cache_scale_shape()` 直接用
   `(*index_cache_shape_)[0]` 作行数（`kv_cache_shape.cpp:406-414`），而 `init_index_cache_shape()` 在
   `supports_dsa_indexer_cache_sharding() && kv_split_size_effective() > 1` 时把行数乘上 `kv_split`
   （同文件 387-404）。⇒ 实测的 `26052 = 6513 × 4` 表示 index 持有**全部规范块**（`S_eff = 1`）⇒
   `full_sequence_replica = true` 对 INDEX **和** INDEX_SCALE 都成立。
3. **SSM/CONV 的 `G` 在 MLA 与非 MLA 下相同**（`linear_value_head_count`）：builder 的 MLA 分支只改描述符
   *种类*，不改组的 head 语义；整资源形态下 `head_begin = tp_rank / tp_redundancy`，只有 `G = linear_value_head_count`
   才能让每个 rank 认领自己的那个 head（`G=1` 会让所有 rank 都宣称 head 0 ⇒ 静默错路由）。

**两条安全性说明**（写进函数注释）：

- **反向错误会响**：把实际被切分的 index 池声明成 `full_sequence_replica` 时，目的缓冲的行数不够，
  `bind` 会在"映射到物理行超出缓冲"处失败 —— 是响亮的失败，不是静默错字节。反过来（把全序列的声明成切分）
  没有任何检查能抓，所以默认取 `true` 是安全方向。若某平台的 `supports_dsa_indexer_cache_sharding()` 为假，
  canonical 会在此处**响亮失败**而不是搬错行；该平台需要另行声明，已记录为已知限制。
- **非 MLA 的 CONV** 是 COMPOSITE 描述符，canonical **按设计拒绝**（第 3/4 轮结论）。声明照给，
  好让那句拒绝只存在于适配器一处。

**顺带**：`is_kv_head_role` 从 `cache_layout_builder.cpp` 的匿名命名空间搬到 `kv_cache_tensor_role.h`
（`inline`），builder 与新声明共用一份，杜绝两处漂移。

**验证**（关键：让**适配器**当声明的 oracle，而不是手写期望）：

- `cache_directory_test` 新增 2 个用例（共 **21/21**）：11 组 `(role, enable_mla)` 的字段级期望 +
  拒绝路径（5 个无 head 轴的 role、`linear_value_head_count == 0` 的 SSM、`kv_head_count == 0` 的 KEY、空输出指针）。
- `pd_route_integration_test` **把声明换成生产函数** `declare_cache_group`（夹具只再提供拓扑与命名空间）：
  于是"真实张量 → 真实 `describe_cache_tensor` → `PeerDirectory::describe` 接受"这一步就成了对**生产声明映射**的
  校验。4 个场景（MLA 与非 MLA × 4 个 role）继续逐字节全绿，搬运量与第 7 轮逐位相同
  （`9437184 / 16842752 / 8192 / 6144`）。
- 容器内总计 **61 用例全绿**：`kv_redundancy_test` 12、`pd_route_test` 12、`cache_directory_test` 21、
  `pd_route_transfer_test` 12、`pd_route_integration_test` 4。
- 生产 TU 真实 flags 编译：`kv_cache_transfer.cpp`、`mooncake_kv_cache_transfer.cpp`、`disagg_pd_config.cpp`、
  **`cache_layout_builder.cpp`** 全部 rc=0。

**输入 (b) 对端视图——接法已定案（下一轮执行）**：

1. 注册期：`get_mooncake_tensors(cache)` 给出 `(role, group_id, sequence_scoped)`，`pending_registration_context_`
   给出 `CacheTensorLayoutContext` 与拓扑 ⇒ 用 `declare_cache_group` 造 `CacheTensorDeclaration` 列表，
   与本侧 manifest 一起交给 `PeerDirectory::describe` ⇒ **本侧视图**（顺便对账 manifest，第 3 轮的适配器就是干这个的）。
2. 传输期：对端 manifest 在 `MooncakeTransferEngine::cache_peers_[addr].destination_manifest`
   （由 `SetCachePeer` 写入）—— 需要一个 getter；对端**拓扑**从 manifest 的 `ParallelCoordinates` 取
   （`InstanceInfo` 只有 `dp_size`/`kv_split_size`，没有 `cp_size`）。
3. `RoutePeer.addrs` 的下标是**对端 DP 组内的局部 rank**，而 `InstanceInfo.addrs` 是按对端的**全局** rank
   （`dp*(cp*tp) + local`）排的 ⇒ 要按对端 `cp_size*tp_size` 做一次下标的换算，不能直接照搬。

**新发现的 S3-5 设计缺口（本轮定案，未写码）**：`plan()` 目前只接受**一份** `canonical_blocks`，但请求在
生产里是按 group 给的**本 rank 物理行号**（`KVTransferMapping.local_ids`），而同一个 group 内不同族的
`split` 不同（KEY `S_eff=4` vs INDEX `S_eff=1`），同一条规范块在不同族的行号不同。接法是：

- 调用方对请求里的每个族把行号换算成规范块（`CanonicalBlock(tokens_per_block, split_f).canonical_of_row(row, slice_f)`），
  再对同一 group 的各族取**并集**传给 `plan()`；`plan` 内部本来就按 `block % split == slice` 过滤，所以并集安全，
  且完备性检查仍逐族成立（INDEX 的并集最大，恰好覆盖 KEY 需要的全部块）。
- 校验点：`split`/`slice` 只能来自**声明**（不能被目的端反推），这正是第 3 轮"块身份不在 manifest 里"的直接后果。

**仍未做**：上面的 (b)+并集换算的代码、`ContextParallelTopology` 本体 oracle、`MixedLayers`/`DpExpansion`/
XTensor `explicit_offsets` 端到端。**运行时**仍未验证（GLM5.3flash 不支持 PD 分离，§1.3）。

#### 第 8 轮补充：S3-5 的"请求 id ↔ 规范块"换算**卡在一个未解的事实上**（必须先把这一条查清，再写码）

上面把并集接法写成了"已定案"，但随后追 id 的真实语义时发现**缺一个关键事实**，先把已经查实的部分与缺口列清楚，
避免下一轮照着一个可能错的假设写码。

**已查实的数字与代码路径**（GLM5-next pilot，`kv_split = 4`，`block_size = 128`）：

| 事实 | 出处 |
|---|---|
| 调度侧 block 大小 = `block_size × kv_split` = **512 token** | `distributed_runtime/llm_engine.cpp:653`（`options.block_size(kv_split_size_eff > 1 ? block_size * kv_split_size_eff : block_size)`），即 S6 要解的那个绑定 |
| 缓存 tensor 的**行** = `kv_cache_cap.block_size()` = **128 token** | `kv_cache_estimation.cpp:695`（`.block_size(options.block_size)`，用户配置值）+ `init_key_cache_shape` 用 `kv_cache_cap.block_size()` 作 dim 1 |
| KV 行数 = `n_blocks` = 6513；INDEX 行数 = `n_blocks × S` = 26052，两者每行都是 128 token | `init_key_cache_shape` / `init_index_cache_shape`（后者乘 `kv_split_size_effective()`，见上文依据 2）；实测日志 `[26052 128 1 257]` 与 `blocks: 6513` |
| 一个调度 block（512 token）在 4 个 DCP rank 上各驻留 1/4，即 rank 的 KV 行 `r` 持有规范块 `r*S + slice` | `kv_shard_layout.cpp` 的 `globalize`（第 5 轮已钉死） |
| 传输映射的 `local_ids` = `sequence->kv_state().blocks(type)` 的 **block id**，且**同一个 id 列表被套用到该 group 的每个 buffer**（KEY 与 INDEX 同为 `BlockType::KV` ⇒ 同一 group） | `batch_input_builder.cpp:334-340`、`mooncake_kv_cache_transfer.cpp::append_buffer_mappings`（`buffer_mapping.local_ids = mapping.local_ids`，按 `buffer.group_id` 取表） |

**缺口**：上面两行合起来是矛盾的 —— 一次 id 映射不可能同时满足

- KV：id `r` = 本 rank 的第 `r` 行 = 规范块 `r*S + slice`（128 token，**每个调度 block 只取 1/4**）；
- INDEX：id `r` = 第 `r` 行 = 规范块 `r`（INDEX 是全序列副本，26052 行覆盖**全部**规范块，且 DSA 的 top-k 需要整条序列的 gate/valid）。

要么 KV 行其实是 512 token（那 `n_blocks` 就该是 6513 个 512-token 行，与"INDEX 行数 = n_blocks × S"的 4 倍关系另有解释），
要么 INDEX 的行号不是 id 本身（例如需要 `id*S + slice` 的换算，而 `append_buffer_mappings` 里看不到这个换算），
要么 INDEX 在 MLA 实例里其实走的是**另一个 group**（`DeepSeekV4KVCacheImpl` 给 INDEX 的是 `compressed_block_type_`，与
`KVCacheImpl` 把它放进 `BlockType::KV` 不同 —— 而实测实例走的是后者）。

**结论 / 下一步**：

- 这一条**不是**可以靠"两侧一致就自洽"糊过去的：它决定 `local_ids` 是不是规范块、以及 INDEX 到底要不要按 `id*S+slice`
  展开。写错会静默搬错行，而 §1.3（GLM5.3flash 不支持 PD 分离）意味着没有实机可以兜底。
- 因此 **canonical 不能接线**，直到查清：建议下一轮按 `qwen_dcp_attention.cpp` / DSA indexer kernel 的
  index 写入寻址 + `kv_state().blocks()` 的 id 空间两条线取证（读码即可，不需要实例），并在文档里钉一条
  "id → 规范块"的显式断言（类似第 6 轮把 `slice` 钉在 `dcp_rank` 上）。
- 第 7~8 轮已落地的部分（入口、开关、声明、加固、对账）不受此影响：它们的输入都是**规范块**，与 id 语义无关。

### 2026-09-18（第 9 轮）——S3-5 的 id↔规范块语义**查清了**（有代码证据），并引入运行时 oracle，67 用例全绿

上一轮把"请求 id ↔ 规范块"标成阻塞项。本轮把它查到底，结论是**可以接线了**，而且顺手把两个老欠账用
**运行时本体**（而不是自洽公式）验掉。

#### 1. 决定性证据：indexer 的 block table 就是"逻辑块 × dcp_size + j"

`xllm/core/layers/common/kv_shard_batch_metadata.cpp:129-144`：

```cpp
torch::Tensor expand_kv_shard_indexer_block_table(const torch::Tensor& logical_block_table,
                                                  const KVShardLayout& layout) {
  torch::Tensor shard_offsets = torch::arange(layout.dcp_size(), ...);
  torch::Tensor expanded = logical_block_table.unsqueeze(-1) * layout.dcp_size() + shard_offsets;
  ...
  return expanded.flatten(1);
}
```

配合 `KVShardLayout`（`kv_shard_layout.cpp`，第 5 轮已读）：

- `logical_block_size() = physical_block_size × dcp_size` = `128 × S` = **512 token**，恰好等于
  `llm_engine.cpp:653` 把 BlockManager 的 `block_size` 放大成 `block_size × kv_split` 的结果；
- `globalize(local_slot) = (local_block_id × dcp_size + dcp_rank) × 128 + offset`
  ⇒ rank 的**物理行** `r` 持有规范块 `r×S + dcp_rank`；
- `localize/owns/owner_of` 是它的逆。

⇒ **两族的行空间关系彻底确定**：

| 族 | 行 = 什么 | 行数 | 与请求 id 的关系 |
|---|---|---|---|
| KV（MLA latent；`S_eff = S`） | 物理行 `r` = 逻辑块 `r` = 规范块 `r×S + slice` | `n_blocks` = 6513 | 请求 id **就是**行号（`block.id()` 来自同一 BlockManager） |
| INDEX / INDEX_SCALE（`full_sequence_replica`，`S_eff = 1`） | 行 = **规范块**（每 128 token 一个） | `n_blocks × S` = 26052 | 请求 id `b` 要展开成 `b×S + j, j∈[0,S)` |

⇒ **S3-5 的换算规则**：请求的 id 是**逻辑块**（KV 侧行号），规范块集合 = `{id×S + j : j ∈ [0,S)}`；
随后每个族用自己的 `split`/`slice` 从中筛选（KV 取 `canonical % S == slice` 的那一个，就是 `id` 本身；
INDEX 因为 `split = 1` 全取，即它的全部 26052 行）。序列型组（LINEAR/EMBEDDING）没有块维度，slot id 直接是规范单位。

**由此也确认了旧路径的一处隐患**（记录，不在本轮修）：`append_buffer_mappings` 把**同一份 `local_ids`**
套到该 group 的每个 buffer 上，而 KEY 与 INDEX 同属 `BlockType::KV`。对 INDEX 而言这些 id 少乘/漏展开
（应为 `id×S + j`），即旧路径在 DCP 分片下对 indexer 池的寻址是**可疑的**；这正是 canonical 路径要显式做对的地方。

#### 2. 新增 `canonical_blocks_of_request(...)`（`cache_directory.{h,cpp}`，纯 std）

签名：`(const std::vector<CacheGroupRequest>& groups, const std::vector<CacheTensorDeclaration>& local, std::vector<int64_t>* out, std::string* error)`。
行为即上面的规则：块维度组按 `topology.kv_split_size`（实例级 = DCP size）展开，序列型组原样；
结果排序去重；对"模型没声明的 group"与"同 group 内族之间 scope 不一致"报错。

#### 3. 运行时 oracle（本轮第二个交付）

手编 harness 加了 glog 链接（`libglog.a` + `libgflags.a`，见 §1.9），于是可以把**运行时本体**链进单测：
`kv_shard_layout.cpp` + `context_parallel_topology.cpp`（两者只依赖 glog）。

新测试 `tests/core/framework/kv_cache_transfer/kv_shard_contract_test.cpp`（**6 用例**）：

1. `CanonicalBlockInvertsTheRuntimeShardLayout`：对 `S ∈ {1,2,4,8}`、每个 dcp_rank、每个 local row 与
   offset ∈ {0,1,127}，断言 `CanonicalBlock::canonical_of_row/local_row/owns` 与
   `KVShardLayout::globalize/owner_of/localize` **逐点一致**，且别的 rank 的切片 `localize` 返回
   `kInvalidSlot`（`S=1` 时无"别的切片"，已按此收窄）。
2. `ExpandsRequestIdsTheWayTheIndexerBlockTableDoes`：用 §1 的展开式手算期望，并逐族核对筛选结果
   （KV 每切片恰好 1 个/逻辑块；INDEX 的 `local_row == canonical`）。
3. `KeepsSequenceScopedIdsWhole`、4. `RejectsAGroupTheModelDoesNotDeclare`、
   5. `RejectsAGroupWhoseFamiliesDisagreeOnScope`。
6. `SliceIsTheDcpRankTheRuntimeBuilds`（`-DXLLM_HAVE_CONTEXT_PARALLEL_TOPOLOGY`）：
   对 `cp ∈ {1,2,4}` × `tp ∈ {1,2,8}` × `S ∈ {1,2,4,8}` 的所有**运行时可接受形状**（C3 两分支）逐 rank 断言
   **`KvLayoutIndex::slice_of(pcp_rank, tp_rank) == ContextParallelTopology::dcp_rank()`**，
   并检查每个 slice 都有 rank 持有。**这就是第 5 轮"必须把 slice 钉在运行时上"的正解**，
   等于把当年那条错公式的回归钉死了。

**一处主动放弃的断言（记录理由）**：本想把"共享 slice 的 rank 集合 == `dcp_group_ranks()`"也钉上，
但运行时 DCP 组的**组成**随形状不同（`S | cp_size` 时是对 PCP 组的划分并共享 dcp_rank；
`S == cp_size×tp_size` 时覆盖整个 DP-local 域、每 rank 各自一个 dcp_rank），
读码无法在本轮把两种形状的统一判据钉死，试了两版都被 oracle 打回。
**该断言与路由契约无关**（路由只需要"每 rank 的 slice 身份" + 由冗余模型导出的副本枚举），
故按"不做无法证实的断言"原则删除，并在测试里写明原因。

#### 4. 验证

容器内 `kv_redundancy_test` 12、`pd_route_test` 12、`cache_directory_test` 21、**`kv_shard_contract_test` 6**、
`pd_route_transfer_test` 12、`pd_route_integration_test` 4 = **67 用例全绿**。
（生产 TU 编译验证在第 8 轮为 rc=0；本轮的改动只在 `cache_directory` 与测试，不影响那四个 TU。）

#### 5. 下一步（S3-4 生产接线，输入已齐）

1. 注册期：用 `declare_cache_group` 造声明 → 本侧 `PeerDirectory`（顺带对账 manifest）。
2. 传输期：对端 manifest getter（`MooncakeTransferEngine::cache_peers_`）；对端拓扑取 manifest 的
   `ParallelCoordinates`；`InstanceInfo.addrs` 的"实例全局 rank"→"DP 组内局部 rank"下标换算。
3. 请求侧：用 `canonical_blocks_of_request`（本轮交付）把 `mapping.local_ids` 换算成规范块，
   交给 `PdRouteTransfer::transfer(opcode)`（第 7 轮交付）。
4. 之后才允许把 `--pd_route=canonical` 的 `LOG(FATAL)` 换成真正的分支。

### 2026-09-18（第 10 轮）——S3-4 接线第 2 步：`build_route_peer`（对端实例装配），69 用例全绿

上一轮说"只剩生产调用点接线，三项输入已齐"。本轮把其中**最容易静默搬错字节的一环**做掉并验证：
调度器给的对端实例地址表是**按实例全局 rank**（`dp*(cp*tp) + local`）排的，而路由寻址的是**DP 组内局部 rank**。

**新增** `build_route_peer(instance_addrs, dp_rank, local_rank_count, manifests, declarations, row_bases, RoutePeer*, error)`
（`pd_route_transfer.{h,cpp}`：入口属于路由层，视图装配复用 `PeerDirectory`）：

1. `peer.addrs[local] = instance_addrs[dp_rank * local_rank_count + local]` —— 这一步错就是把字节搬到**另一个 DP 组**的
   worker 上，且所有后续检查都发现不了（地址本身有效），所以转换只允许存在这一份；
2. 对每个局部 rank 用 `PeerDirectory::describe(该 rank 的 manifest, 声明, 页基点, ...)` 造视图（== 第 3 轮适配器），
   再把各 rank 的视图拼成一列 —— 每个视图自带 `local_rank`，正是 `PdRouteTransfer::plan` 需要的形态；
3. 三重对账：地址表必须覆盖所请求的 DP 组；`manifests` 数必须 == `local_rank_count`；**每个视图自报的 rank 必须等于它被归到的那个 rank**
   （防调用方把 manifest 顺序传错）。

**验证**（host，`cache_directory_test` 新增 2 用例 → **23/23**）：

- `AssemblesThePeerInstanceOfOneDpGroup`：`cp1/tp2` 的实例、4 个地址（2 个 DP 组），断言 `dp_rank=1` 取到
  `worker-{0,1}.dp1`（**不是**局部 rank 0/1 的 `worker-*.dp0`），视图数 == 2、各自 `local_rank` 正确；
  再取 `dp_rank=0` 验证另一对。
- `RejectsAPeerInstanceThatDoesNotAddUp`：地址不足该 DP 组、manifest 顺序传错（视图自报 rank 不符）、某 rank 无 manifest，
  三条都报错且错误信息可辨。

**顺带**：写这个用例时先踩了一次自己的坑 —— 把 `&manifests.back()` 存进 `manifest_pointers` 的同时继续
`emplace_back` 到同一个 vector，**扩容后指针全部悬空**，适配器报出
`unsupported cache layout schema version 933675520`（读到了已释放内存）。已 `reserve` 并在测试里写明原因。
这也说明 `build_route_peer` 的 `const WorkerCacheLayoutManifest*` 入参对调用方有同样的要求，已写进函数注释。

**验证汇总**：容器内 `kv_redundancy_test` 12、`pd_route_test` 12、`cache_directory_test` **23**、
`kv_shard_contract_test` 6、`pd_route_transfer_test` 12、`pd_route_integration_test` 4 = **69 用例全绿**。
生产 TU 真实 flags 编译：`kv_cache_transfer.cpp`（已包含 `pd_route_transfer.h`，因此也验证了新头文件的依赖链
`cache_directory.h → cache_layout.h → pb`）、`mooncake_kv_cache_transfer.cpp`、`cache_layout_builder.cpp` 全部 rc=0。

**剩下的最后一步（下一轮，已无未知项）**：

1. `MooncakeTransferEngineCore::CachePeerLink` 里**保留对端 manifest**（现在只留旧 `ReshardPlanTemplate`），
   加 `peer_cache_layout(remote_addr)` 取值；`set_cache_peer()` 已经在收到 manifest 的那一刻，改动是纯增量的
   （**注意**：哪个方向发 manifest 由既有协商决定，本轮只读码确认，运行时行为未验证）。
2. `MooncakeKVCacheTransferBase` 在 `publish_cache_layout(tensor_manifests, context)` 时用
   `declare_cache_group` 造 `declarations_`（`context.tensor_layout` 给 head 计数，
   `context.coordinates` 给拓扑，`tensor.role/group_id` 给族身份；注意 `block_token_capacity` 在生产里传的是
   `options_.block_size()` = **128** = 规范块大小，与第 9 轮的模型一致），并顺手造出本侧 `local_directory_`。
3. `push_kv_blocks_async` 的 canonical 分支：用 `canonical_blocks_of_request`（第 9 轮）把 `mapping.local_ids`
   换算成规范块 → `build_route_peer`（本轮）装配对端 → `PdRouteTransfer::transfer(PUSH, ...)`（第 7 轮），
   `MoveFn` 里把 `RouteRegion` 逐字段拷成 `ByteRegion` 调 `move_memory_regions(..., WRITE)`。
   **还需一处设计**：`RouteRegion` 不含 layer，而 push 侧要按 layer 调 `synchronize_layer`；
   需要按 buffer 所属 layer（`BufLayout::layers[][]` 有 `buf_id`→layer 的映射）把腿的 region 分组后逐层发送。
4. 全部就绪后才把 `--pd_route=canonical` 的 `LOG(FATAL)` 换成真正分支。

**仍未验证（原因）**：真实运行时（RDMA/NPU/真实调度 block id/协商方向）——GLM5.3flash 尚不支持 PD 分离（§1.3）；
本轮新增的 `build_route_peer` 只有 host 单测，没有装配进任何生产调用点。
