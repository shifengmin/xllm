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
| S3-4 | **数据面切换**：`transfer(edges, canonical_blocks, opcode)` 统一入口，先 PULL 后 PUSH；用 `--pd_route=legacy\|canonical` 开关，默认 legacy | 编译通过 + 单测；运行时行为未验证（见约束） | ⚠️ **PUSH 已接线、待运行时验证**（第 12~13 轮；71 用例全绿 + 三个生产 TU rc=0）。**PULL 按用户要求停放**在本地分支 `pd-routing-pull-wip`（`8526462c5`，未推送）——实现已起草且编译通过，但未跑 host 全量、未验证。详见第 14 轮 |
| S3-5 | **规范逻辑地址层**：`logical_offset` 改规范块坐标，`bind` 只做物理换算 | `S_P = S_D` 等价锚点 + `S_P ≠ S_D` 折叠用例 | ✅ **已落地并接进数据面**（第 9 轮：`canonical_blocks_of_request` + 运行时 oracle 用 `KVShardLayout`/`ContextParallelTopology` 钉死；第 12 轮接进 PUSH） |

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
7b. **probe 必须"先解包再编译"（第 12 轮踩坑）**：`stage.sh` 只把 tarball scp 到 `/tmp/s2sync.tgz`，
   解包+拷进 worktree 是远端脚本的事；而 `probe_compile.py` 从 **worktree** 读源文件。
   若只 stage 不 unpack，"编译通过"其实编译的是上次留下的旧文件。
   现在 `run_probe3.sh` / `remote_stage_and_test.sh` 都先 `tar xzf` + `cp` 再编译；
   并且 **`config/disagg_pd_config.{h,cpp}`、`kv_cache_transfer.{h,cpp}` 必须进 staging**
   （第 7~11 轮漏了 config，导致 `kv_cache_transfer.cpp` 的 rc=0 不可信）。
   自检：`tar tzf /tmp/s2sync.tgz | grep -c config` 应为非 0。
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

### 2026-09-18（第 11 轮）——S3-4 接线第 3 步：对端 manifest 保留、本侧声明/视图、层化发送计划，71 用例全绿

第 10 轮列出 4 条待办，本轮做掉 1、2 和 3 里**唯一有逻辑、可单测**的那段（层化分组）。

**(1) 保留对端 manifest**（`mooncake_transfer_engine.{h,cpp}`）

旧路径拿到对端 manifest 后只留了由它构建的 `ReshardPlanTemplate`，manifest 本身丢弃。canonical 路线要从
**对端自己的布局**推导边表，所以给 `CachePeerLink` 增加 `std::optional<WorkerCacheLayoutManifest> manifest;`
（`set_cache_peer()` 里赋值，纯增量），并加取值口
`peer_cache_layout(remote_addr)`（core + engine 各一层转发）。

> ⚠️ 未验证：**哪个方向发 manifest** 由既有协商决定（`SetCachePeer` 由收到 manifest 的一方调用，
> 于是那一方的 `cache_peer_links_` 里才有对端布局）。本轮只读码确认这条链，运行时行为未验证（§1.3）。

**(2) 本侧声明 + 本侧视图**（`mooncake_kv_cache_transfer.{h,cpp}`）

`publish_cache_layout()` 末尾（MAIN 与 SPEC_DRAFT 都会走到，声明按累积后的 manifest 重建）：

- 对 `local_cache_layout_.tensors` 的每个族用 `declare_cache_group(context.tensor_layout, role, ...)` 造声明；
  拓扑取自 `registration_context.coordinates`（DRAFT 用自己的 placement，与既有注释一致），
  `tokens_per_block` 取自 `tensor_layout.block_token_capacity` —— 生产传的是 `options_.block_size()` = **128**
  = 规范块大小，与第 9 轮的模型一致；
- 用 `PeerDirectory::describe(local_cache_layout_, declarations_, row_bases_, ...)` 造本侧视图；
- **两条降级路径**（都是 `LOG(WARNING)` + `canonical_ready_ = false`，**不影响 legacy**）：
  ① 模型没钉住几何的族（WINDOW/SWA/KV_STATE 等，`declare_cache_group` 本来就会拒绝）——DSV4 实例就是这种，
  必须不能让 legacy 起不来；② 含 `explicit_resource_offsets` 的族（XTensor）——页基点还没接，
  显式标注而不是塞错误的基点。

**(3) 层化发送计划 `flatten_route_for_layers(...)`**（`pd_route_transfer.{h,cpp}`，纯函数）

`RouteRegion` 只有 buffer id，而 push 侧必须**按 layer 调 `synchronize_layer`**。`BufLayout::layers[][]` 有
`buf_id → layer` 的映射，所以：把腿的 region 按"**缓冲所属层**"折叠成
`RouteLayerBatch{layer_id, peer_addr, regions}`，按 **layer 升序**排序、同层内保持 plan 的腿序
（同一 (layer, peer) 的两条腿合并成一次传输调用）。region 的 buffer 查不到层 → **报错**而不是丢弃
（丢弃会在目的端留一个没有任何检查能发现的洞）。

**验证**：`pd_route_transfer_test` 新增 2 用例（12 → **14**）：

- `FlattensALegIntoAscendingLayerBatches`：2 层 × 2 peer × 4 个 region，断言输出为 4 个 batch、
  顺序是 (layer0,peer3)、(layer0,peer5)、(layer1,peer3)、(layer1,peer5)，且每个 region 恰好出现一次；
- `RejectsARegionWhoseBufferHasNoLayer`：查不到层的 buffer 报错且不产生 batch。

容器内总计 **71 用例全绿**（12+12+23+6+14+4）。生产 TU 真实 flags 编译：`mooncake_transfer_engine.cpp`、
`mooncake_kv_cache_transfer.cpp`、`kv_cache_transfer.cpp` 全部 rc=0。

**剩下的最后一步（第 12 轮，已无未知项）**：把三者串进 `push_kv_blocks_async` 的 canonical 分支 ——
`canonical_blocks_of_request`（第 9 轮）→ `build_route_peer`（第 10 轮，peer manifest 由本轮的
`peer_cache_layout` 取）→ `PdRouteTransfer::plan`（第 7 轮）→ `flatten_route_for_layers`（本轮）→
逐 batch `synchronize_layer` + `move_memory_regions(..., WRITE)`（`RouteRegion` 逐字段拷成 `ByteRegion`），
然后把 `--pd_route=canonical` 的 `LOG(FATAL)` 换成真正分支。
`canonical_ready_` 为假时必须显式拒绝（不能静默走 legacy）。

### 2026-09-18（第 12 轮）——S3-4 接线**完成**：canonical 走生产数据面，并修掉一个会让验证失真的 harness 缺陷

第 11 轮列的"最后一步"本轮做完：canonical 已经是一个**可选的生产数据面**（默认仍是 legacy）。

**(1) 开关变成真分支**（`kv_cache_transfer.{h,cpp}`）

- 工厂不再 `LOG(FATAL)`：解析 `--pd_route` 后 `transfer->set_canonical_route(route_mode == CANONICAL)`；
- `push_kv_blocks_async` 开头分流：canonical ⇒ 直接调新虚函数并返回，**完全跳过**
  `validate_transfer_mappings` / `filter_kv_split_infos` / `rotate_dst_rank` / `merge_kv_blocks`
  —— 那套"rank 1:1 对齐"的前提对 canonical 不成立；
- 新增虚函数 `push_kv_blocks_canonical(infos, parallel_args, layer_synchronizer, is_spec_draft)`：
  **基类默认实现报错拒绝**（"this backend has no canonical data plane"），
  `MooncakeKVCacheTransferDefault` 覆盖，`MooncakeKVCacheTransferXTensor` 不覆盖（XTensor 本来就是
  `canonical_ready_ = false`）。**没有任何静默回落**。

**(2) canonical push 实现**（`mooncake_kv_cache_transfer.cpp`）

链路：`canonical_blocks_of_request`（第 9 轮）→ `peer_cache_layout` 取对端每个局部 rank 的 manifest（第 11 轮）
→ `build_route_peer`（第 10 轮）→ `PdRouteTransfer::plan(PUSH)`（第 7 轮）→ `flatten_route_for_layers`（第 11 轮）
→ 逐 batch `synchronize_layer(layer)` 后 `move_memory_regions(peer_addr, regions, WRITE)`，
其中 `RouteRegion` 用文件内 `to_byte_region()` **逐字段**转换（字段若被改名会编译失败，不会静默错位）。

三处防御按"响亮失败"写：

1. `canonical_ready_ == false` ⇒ 直接报错并说明用 legacy；
2. 对端某个局部 rank 没有公布的布局 / 地址表不覆盖该 DP 组 / manifest 与模型不符 ⇒ 请求失败（不猜、不跳）；
3. **命名空间隔离**（本轮发现并修的真实缺口）：一次 push 只搬一个命名空间。draft body 的 tensor 注册在
   SPEC_DRAFT 下且与 MAIN 共用 group id，而 `plan` 会遍历全部本地族 —— 不隔离就会让 draft push 顺手把主缓存也搬一遍。
   现在按 `cache_namespace` 过滤 `declarations_` / `local_views` / `peer.views` 三处。
   （SPEC_DRAFT 视图 `local_rank == -1`，`plan` 仍按"同族只有一个未命名视图"兜底，多于一个直接报错。）

**(3) 修掉一个会让验证失真的 harness 缺陷（重要，影响此前若干轮的结论）**

`stage.sh` 只把 tarball **scp 到 `/tmp/s2sync.tgz`**，真正解包并拷进 worktree 的是另一个脚本；
而 `run_probe_compile.py` 是**从 worktree 读源文件**编译的。于是"先 stage 再跑 probe"实际编译的是**上一次解包留下的旧文件**。
本轮首次真正验证时立刻暴露：

- `mooncake_kv_cache_transfer.cpp` 报 `PdRouteTransfer has not been declared`（新代码根本没被编译）；
- 补上 include 后 `kv_cache_transfer.cpp` 报 `DisaggPDConfig has no member pd_route`
  —— 因为 **`config/` 目录从来没进过 staging**（`stage.sh` 里加 config 的补丁被 `if 'config' not in s` 误判跳过，
  文件里 `~/.ssh-xllm-config` 就含 "config" 子串）。

修法：`stage.sh` 增加 `lib/kv_cache_transfer.{h,cpp}` 与 `config/disagg_pd_config.{h,cpp}`；
`run_probe3.sh` 与 `remote_stage_and_test.sh` 都先**解包+拷贝**再编译；并用 `tar tzf | grep -c config` 自检。

**结论**：此前各轮"生产 TU 编译 rc=0"的结论里，`kv_cache_transfer.cpp` 那一条在
**第 7 轮（引入 `--pd_route` 读配置）之后到本轮之前是不可信的**（旧 config 头 + 旧源文件）。
`mooncake_kv_cache_transfer.cpp` / `cache_layout_builder.cpp` / host 单测不受影响（后者一直从 staged `lib/`、`kvcache/` 编译）。
**本轮已用修好的流程重新验证全部三个 TU：rc=0。**

**验证汇总**：容器内 **71 用例全绿**（12+12+23+6+14+4）；
生产 TU 编译（修好的流程）：`mooncake_transfer_engine.cpp`、`mooncake_kv_cache_transfer.cpp`、`kv_cache_transfer.cpp` **全部 rc=0**。

**仍未验证（必须标注）**：canonical 分支**没有运行时验证** ——
① GLM5.3flash 尚不支持 PD 分离（§1.3）；② 协商方向（谁给谁发 manifest）只读码确认；
③ `push_kv_blocks_canonical` 没有 host 单测（它直接依赖 Mooncake 引擎与 NPU layer synchronizer），
其可单测的部分（`canonical_blocks_of_request` / `build_route_peer` / `plan` / `flatten_route_for_layers`）已各自覆盖；
④ **canonical PULL 未接**：`pull_kv_blocks_async` 只拿到**单个源 rank 的地址**与 mappings，
   没有源实例的 `InstanceInfo`（addrs/dp_size），所以接 PULL 需要额外把源实例信息带到 pull 侧。
   这与 S3-4 原计划的"先 PULL 后 PUSH"顺序相反，原因是**数据可得性**而不是偏好，已在文档标注。

### 2026-09-18（第 13 轮）——查清 manifest 协商方向（此前标为"未验证"），并为 canonical PULL 备好两处数据

第 12 轮把 canonical PUSH 接进生产，并在末尾标了三条未验证项，其中"**谁给谁发 manifest**"本轮查清了，
顺带发现 canonical PULL **不需要改 RPC**，只差两处"取到却丢掉"的数据。

**(1) 协商方向（读码确认，不再是猜）**

- `LLMEngine::link_cluster(...)`（`distributed_runtime/llm_engine.cpp:936-991`）把**源实例的全部 rank**
  （`cluster_ids/addrs/ports`）交给**每一个** D worker ⇒ 每个 D worker 在 `link_clusters` 时就拿到
  **源实例的地址表，且下标就是源实例的全局 rank**（`cluster_ids[source_rank]` 顺序构造）。
- `MooncakeTransferEngine::link_sessions`（`mooncake_transfer_engine.cpp:759-841`）对**每个**源 rank
  `fetch_cache_layout(...)`（即 `GetCacheLayoutManifest` RPC）拿到对端 manifest，用它跑
  `planner.select_sources` 选出 ACTIVE，然后对每个 rank 调 `set_remote_peer(..., *local_manifest, mode)`
  —— **把本侧（D）的 manifest 通过 `SetCachePeer` 发给源（P）**。

⇒ 结论：

- **PUSH（P→D）的数据来源是成立的**：`SetCachePeer` 收端是 P，因此 P 的 `cache_peer_links_`
  才会持有 D 的 manifest ⇒ 第 12 轮的 `peer_cache_layout(D_addr)` 在 PUSH 侧能取到值。
  （第 11/12 轮标注的 ⚠️"协商方向未验证"由此**关闭**。）
- **PULL（D←P）的数据也已经在手上，只是被丢掉了**：D 在 `link_sessions` 里 fetch 了**全部**源 manifest，
  但只用于 `select_sources`，函数结束就释放；而 `link_clusters` 拿到的源地址表也没有留存。

**(2) 本轮把两处数据留下来**（纯增量，编译验证 rc=0）

- `MooncakeTransferEngineCore` 新增 `peer_layouts_`（addr → manifest）与
  `set_peer_cache_layout(addr, manifest)`；`peer_cache_layout(addr)` 改为读它。
  **两个方向都写这里**：入站 `SetCachePeer`（P 侧知道 D 的布局）与 `link_sessions` 的 fetch（D 侧知道 P 的布局）。
  `CachePeerLink::manifest`（第 11 轮加的第二个真相源）已删除，避免两处状态不一致；
  ABSENT 模式会同时擦除 `peer_layouts_`。
- `MooncakeKVCacheTransferBase` 新增 `linked_source_addrs_`，在 `link_clusters` 成功后保存源实例地址表
  （顺序 = 源实例全局 rank），正是 `build_route_peer` 需要的形态。

**(3) canonical PULL 还差什么（已无未知项）**

`pull_kv_blocks_async` 的调用链只把**单个源 rank 的 addr** 传到 transfer 层
（`worker_service.cpp:795` → `WorkerClient` → `WorkerImpl` → `KVCacheTransfer::pull_kv_blocks_async`），
缺的是 **`src_dp_rank`（DP 配对）**：它在 `LLMEngine::pull_kv_blocks(src_dp_size, src_dp_rank, src_cluster_ids, src_addrs, dst_dp_rank, mappings)`
里是参数，但转发到 worker 时只传了 `addrs[src_worker_rank]`。
其余输入都已具备：源实例地址表（本轮的 `linked_source_addrs_`）、源实例 dp_size/cp/tp（任一源 manifest 的
`ParallelCoordinates`）、对端视图（本轮的 `peer_layouts_`）。
⇒ 下一轮把 `src_dp_rank` 沿调用链带下来（或在 pull 请求里带上）即可接通 canonical PULL。

**验证**：容器内 **71 用例全绿**；三个生产 TU（`mooncake_transfer_engine.cpp`、`mooncake_kv_cache_transfer.cpp`、
`kv_cache_transfer.cpp`）用修好的"先解包再编译"流程全部 rc=0。

**仍未验证（原因）**：真实运行时（§1.3，GLM5.3flash 不支持 PD 分离）；
`src_dp_rank` 的取值语义（读码是 `LLMEngine::pull_kv_blocks` 的入参，来自 `disagg_pd_scheduler.cpp:1210` 的
`src_dp_rank`，其与 `dst_dp_rank` 的配对规则未在运行时验证）。

### 2026-09-18（第 14 轮）——按用户要求**暂停 PULL**；PUSH 状态盘点与"打通"的准确含义

用户指示：**PULL 先不做**，先确认 PUSH 是否已打通。本轮不加新功能，只做状态盘点 + 把已起草的 PULL 代码**停放**。

**(1) PULL 代码已起草但按指示停放**

第 13 轮判定"PULL 只差 `src_dp_rank`"；本轮进一步发现**连这个都不需要**：
`pull_kv_blocks_async` 拿到的那个 `src_addr` 本身就是"源 DP 组的一个 rank"，
而该 rank 的 manifest `coordinates` 里有 `dp_rank`/`cp_rank`/`tp_rank`/`cp_size`/`tp_size`
（第 13 轮已把所有源 manifest 留在 `peer_layouts_`）⇒ 可以直接把 `src_addr` 所属的整个源 DP 组
按 `local_rank = cp_rank*tp_size + tp_rank` 重新拼出来，不需要改 RPC/签名。

于是本轮写出了一版完整实现（`pull_kv_blocks_canonical`：按 `src_addr` 定位源 DP 组 →
`build_route_peer` → `canonical_blocks_of_request` → `PdRouteTransfer::transfer(PULL, ...)`
+ `move_memory_regions(READ)`；另把 `local_rank_`/`local_rank_count_` 在 `configure_cache_layout` 存下来、
把 `PdRouteCache` 提为成员供两个方向共用（F8））。

**按用户指示停放**：这些改动**不在** `pd-routing-s0s1` 上，而是单独放在**本地分支 `pd-routing-pull-wip`**
（commit `8526462c5`，**未推送**）。这样做的好处：① 评审分支 `pd-routing-s0s1` 保持在第 13 轮那个
"PUSH 已验证"的干净状态；② 起草的代码不丢，随时 `git cherry-pick 8526462c5` 或 `git diff a1c4186b6..pd-routing-pull-wip` 取回。
该分支只做过**编译验证**（三个生产 TU rc=0），**未跑 host 全量**、**未做运行时验证**。

**(2) "PUSH 已打通"的准确含义（不要过度解读）**

已经做到的：

| 层 | 状态 |
|---|---|
| 开关 | `--pd_route=legacy\|canonical`（默认 legacy）；工厂解析、非法值 `LOG(FATAL)`；**无静默回落** |
| 生产调用点 | `push_kv_blocks_async` 分流到 `push_kv_blocks_canonical`，**完全跳过** `filter_kv_split_infos`/`rotate_dst_rank`/`merge_kv_blocks` |
| 全链路 | 请求 id → 规范块（`canonical_blocks_of_request`）→ 对端每个局部 rank 的 manifest（`peer_cache_layout`）→ `build_route_peer` → `plan(PUSH)` → 按 layer 折叠（`flatten_route_for_layers`）→ 逐层 `synchronize_layer` + `move_memory_regions(WRITE)` |
| 数据可得性（读码） | ① P 侧有 D 的 manifest：`link_sessions` 会把本侧 manifest 通过 `SetCachePeer` 发给源；② D 的地址表来自 `TransferKVInfo.remote_instance_info.addrs`（下标 = 实例全局 rank）；③ 本侧声明/视图在 `publish_cache_layout` 时建好 |
| 编译 | 三个生产 TU（`mooncake_transfer_engine.cpp`/`mooncake_kv_cache_transfer.cpp`/`kv_cache_transfer.cpp`）真实 flags 编译 rc=0（用第 12 轮修好的"先解包再编译"流程） |
| host 单测 | **71 用例全绿**；其中 `pd_route_transfer_test`（14）覆盖 PUSH/PULL 双向逐字节、腿枚举、层化折叠；`cache_directory_test`（23）覆盖声明映射与对端装配；`kv_shard_contract_test`（6）用运行时本体钉 `slice == dcp_rank` 与 `CanonicalBlock ↔ KVShardLayout`；`pd_route_integration_test`（4）真实张量→统一入口→memcpy 逐字节 |

**没有做到的（关键，别当成"已验证可用"）**：

1. **运行时零验证**：GLM5.3flash 尚不支持 PD 分离（§1.3），没有实机可跑；RDMA/NPU/真实调度 block id 全部未跑过。
2. `push_kv_blocks_canonical` **本身没有单测**（它直接依赖 Mooncake 引擎与 NPU layer synchronizer）；
   它调用的每个纯函数都有单测，但"串起来是否对"只有编译验证。
3. 两处**读码得到的假设**未在运行时确认：
   ① `InstanceInfo.addrs` 的下标就是"实例全局 rank = `dp*(cp*tp) + local`"（依据 `merge_kv_info` 的用法）；
   ② 每个目的局部 rank 的 manifest 都能从 `peer_cache_layout` 取到（依据 `link_sessions` 的 `SetCachePeer` 方向）。
4. XTensor（`explicit_resource_offsets`）与角色几何未钉住的族（WINDOW/SWA/KV_STATE…，即 DSV4 实例）
   在 canonical 下**显式拒绝**（`canonical_ready_ = false` + 警告），只能走 legacy。
5. canonical 与非 canonical 的**等价性**只在 host 上验证过（`S_P = S_D` 锚点 + 折叠/发散场景），
   没有在真实 PD 上对照过字节。

**结论**：PUSH 方向**代码链路完整、可编译、可单测的部分全绿**，但**不能称为"已验证打通"** ——
准确说法是"**已接线、待运行时验证**"。要真正验收 PUSH，最小路径是：先把 canonical 在**单机 co-located
的 P/D（或 mock 引擎 + 真实 tensor）**下跑一次字节比对，再上真实 PD；S3-4 的"运行时可验证"门槛仍受 §1.3 限制。

**(3) 剩余清单（下一轮的入口）**

- 取回 `pd-routing-pull-wip`（`8526462c5`）→ 跑 host 全量 → 补 `src_addr` 定位的 host 单测 → 再决定是否并入主线；
- 若继续 PUSH 方向：想清楚"用 mock 引擎给 `push_kv_blocks_canonical` 做单测"是否可行
  （它是 `MooncakeKVCacheTransferDefault` 的成员，`MooncakeTransferEngine` 的方法是 virtual，理论上可派生 mock）；
- 未覆盖的 T5 项：`MixedLayers`、`DpExpansion`、XTensor `explicit_offsets` 端到端；
- `ContextParallelTopology` 的 DCP **组组成**判据（第 9 轮主动放弃）仍未钉死，与路由契约无关。

## 5. 剩余工作清单（第 15 轮盘点，按"是否依赖实机"分组）

用户要求先列清单再动工。下列每项都注明了**能否在本地/容器内闭合**。

### A. 把 PUSH 的可验证性推到顶（**不依赖实机**，优先级最高）

| # | 事项 | 为什么 | 闭合标准 |
|---|---|---|---|
| A2 | **`canonical` 下 PULL 静默走 legacy**（本轮新发现，见下） | 开关语义不自洽：`--pd_route=canonical` 只改了 PUSH，PULL 无分支 ⇒ 混合语义，且**静默** | canonical 下 PULL 要么显式拒绝（loud fail）要么接通；有单测钉住 |
| A1 | 用 **mock `MooncakeTransferEngine`** 给 `push_kv_blocks_canonical` 做单测 | 它是唯一"编译过但没测过"的成员；其调用的纯函数都测过，但"串起来是否对"没测过 | 真实 `describe_cache_tensor` 造 manifest + mock 记录 `(addr, regions, opcode)`，断言逐层 byte region 集合 == 纯函数期望 |
| A3 | 两处**读码假设**加运行时前哨 | ① `InstanceInfo.addrs` 下标 == 实例全局 rank；② 每个目的局部 rank 的 manifest 可 `peer_cache_layout` 取到。二者错了会**搬错字节**而不是报错 | 加 CHECK/计数+明确日志，使实机上第一时间炸掉；host 上用 mock 覆盖"缺失 manifest"分支 |
| A4 | canonical ↔ legacy **等价性**再做交叉对照 | 目前只在 host 上跑过锚点与折叠场景，没跟 legacy 的纯函数（`filter_kv_split_infos`/`rotate_dst_rank`）对过 | 多拓扑（`S_P=S_D` 锚点、DP 扩张、TP 扩张）下逐块坐标一致 |

**A2 的证据**（本轮核过）：`KVCacheTransfer::pull_kv_blocks_async`（`kv_cache_transfer.cpp:120-140`）
**没有** `canonical_route_` 分支，直接调 `pull_kv_blocks`；全文件 `canonical` 只出现在 `:228`（PUSH 分流）
与 `:320`（`set_canonical_route`）。⇒ 开着 canonical 的实例，PUSH 走新路径、PULL 走旧路径，
且没有任何日志提示这一点。

### B. PULL 方向（用户已停放，等指示）

- B1 取回 `pd-routing-pull-wip`（`8526462c5`）→ 跑 host 全量 → 补 `src_addr → 源 DP 组` 定位的单测 → 决定是否并入主线。
- B2 canonical 下 PULL 的语义与 A2 是同一处；做完 A2 再决定 B 的合并时机。

### C. T5 覆盖面（host 可闭合）

- C1 `MixedLayers`（同请求跨层组）；
- C2 `DpExpansion`（`S_P ≠ S_D` 的扩张方向）；
- C3 XTensor `explicit_offsets` 端到端 —— 当前**显式拒绝**（`canonical_ready_=false`）：要么支持，要么把"不支持"写进配置校验/文档，不能只留一条警告。

### D. 交付物与文档

- D1 **运维可见的适用范围**：哪些模型/拓扑/开关组合能开 canonical，不满足时在哪一行、以什么形式拒绝。
- D2 `handover` / `redesign_proposal` 的状态回写（现在文档还停在 S2 口径）。
- D3 写清"**验收 PUSH 需要什么**"：模型（需支持 PD 分离）、拓扑（prefill `cp4/tp8/kv4` × decode `dp4/cp1/tp2/kv2`）、字节比对方法。

### E. 运行时验证（**唯一真正的验收**，受 §1.3 阻塞）

- E1 最小路径：单机 co-located P/D，或**进程内两个引擎**（mock 引擎 + 真实 tensor）跑一次 canonical PUSH 字节比对 —— 不必等 PD-capable 模型。
- E2 真机 PD：需一个支持 PD 分离的模型 + 上述拓扑。
- E3 真机上与 legacy 做 A/B 字节对照。

### 建议次序

A2（便宜且修的是"静默走错路径"）→ A1 → A3 → E1（受资源决策约束）→ 其余。
**A 组不依赖实机，做完即到 PUSH 上的可验证上限；E 组才是"打通"二字的本体。**

### 2026-09-18（第 15 轮）——用户改向：**用 GLM-5.2（支持 PD 分离）+ 4 层减层模型做单机端到端验证**

**(1) 用户指令（原文要点）**

- 用 **GLM-5.2 python**（支持 PD 分离）端到端验证 **prefill/decode DCP 异构 + indexer full sequence** 的逻辑；
- GLM-5.2 单实例占满单机、PD 需两机，"当前没有资源" ⇒ **抽一个 4 层减层模型，单机即可测**（后续确认：直接用现成的 4 层模型）。
- GLM-5.2 **没有 linear 层**；linear（SSM/CONV 角色）的处理相对独立，等 glm-next 具备 PD 分离再说。

⇒ 这条指令把 §5 的 **E 组（运行时真机验收）** 从"受 §1.3 阻塞"变成当前主线；A1（mock 单测）暂停。

**(2) 本轮已完成的 A2（先落地，已提交 `d71abf6c0`）**

`--pd_route=canonical` 之前**静默**让 PULL 走 legacy（`kv_cache_transfer.cpp:120-140` 无 canonical 分支）。
现在加了 `pull_kv_blocks_canonical` 虚接口（基类默认 `LOG(ERROR)` + `false`）并在 `pull_kv_blocks_async` 分流；
三个生产 TU 编译 rc=0。**dispatch 本身仍无单测**（要链 folly/torch/threadpool，见第 15 轮 A1 的否决结论）。

**(3) A1 的"mock 引擎单测"路线**当轮否决**（证据）**

- `KVPushSynchronizerImpl` = `NPULayerSynchronizerImpl`：构造函数**无条件** `aclrtCreateEventWithFlag` +
  `aclrtGetCurrentContext` 并 `CHECK` 成功 ⇒ host 上任何派生 fake 构造即死；`synchronize_layer` 也**不是虚函数**。
- 引擎侧可注入（`unique_ptr<MooncakeTransferEngine>` + 虚 `move_memory_regions`），但 `peer_cache_layout` 非虚
  （可绕过：`set_peer_cache_layout` 是 public）+ 仍需链 `mooncake_transfer_engine.cpp`（mooncake/brpc/protobuf）。
- ⇒ 要"纯 mock"必须给 NPU 运行时类加抽象接口，代价与不可验证性都太高。**改为**：把 canonical push 的编排抽成纯函数
  （`plan_canonical_push`，落 `pd_route_transfer`，host 可测），glue 只剩 `synchronize_layer` + `move_memory_regions`。
  **该重构本轮未开工**（被用户改向打断）。

**(4) 环境侦察结论（本轮最有价值的产出，后续不用重查）**

| 事实 | 值 |
|---|---|
| 空闲机型 | **jd-node-83 全空（16/16 chip free，~3GB/64GB 占用）**；.82 10/16 free；.99 7/16；.100 8/16；**98 已满（每 chip 61/64GB）** |
| 卡 | 每机 8 NPU × 2 chip = 16 chip，64GB HBM/chip |
| 减层模型 | `GLM-5.2-W8A8-EcoTech-4layers`：`num_hidden_layers=4`、`indexer_types=['full','full','full','shared']`、`num_nextn_predict_layers=0`。98 的 `/export/home/models/` 有（**symlink** 到 `../GLM-5.2-W8A8-EcoTech/`），CFS `/mnt/cfs/9n-das-admin/llm_models/GLM-5.2-W8A8-EcoTech-4layers` 有**真文件**（5 个 shard ≈ 20GB） |
| CFS 权限漂移 | 83 上普通用户**读不了** CFS 该目录（`drwxr-x--- 1080`），容器内 root 可以；`/mnt/cfs` 本身在 83 可穿越 |
| `/export/home` | **各机本地 xfs**（83 是 `/dev/md0`）⇒ 83 上没有 GLM-5.2 权重，只有 `-MTP` |
| 现成 PD 配方（**已验证**，别人的） | `/export/home/shifengmin.3/workspace/glm5_2_pd/`：`README.md`（19KB 操作手册，root 600）、`start_prefill.sh`、`start_decode.sh`、`start_colocate_83.sh`、`test_smoke_83.sh`、`hccl_atb_env.sh`、`hook_aclcreate.so` |
| 配方拓扑 | Prefill `cp_size=2, kv_split_size=2, tp=8`；Decode `dp_size=2, tp=8, layerwise_split_size=4`（`kv_split` 默认 1）⇒ **S_P=2 vs S_D=1 已是 DCP 异构** |
| 管控面 | 83 容器 `fengmin-cann9-0801`：etcd `4389`、`xllm_master_serving` `48888/48889`（HTTP 口是 readiness，P+D 齐了才 listen）。**别杀** |
| KV 后端 | `LlmDataDist`（layerwise 用；Mooncake 不兼容）/ **Mooncake（非 layerwise）** |
| 配方已知坑（= S2/S3 要解决的） | ① Decode `layerwise>1` 时未持有层共享同一 scratch KV ⇒ Mooncake `registerLocalMemory` 重叠报错，需按 `data_ptr` 去重；② 去重后 P/D **buffer 数不匹配**（`local=234, remote=63`）⇒ `move_memory_groups` 要求两边 `buffers.size()` 相同，**逻辑 buf_id 与物理 MR 别名未对齐**；③ 结论：`layerwise=1` 时两边都是 234，legacy 路径能通 |
| 构建树 | 98 上多棵树有可用 `build.ninja`，`ninja` 会**重跑 cmake 并成功**（§1.1 的"重配置不可用"只对某些容器成立）。`xllm-pdroute` = **我的探针树**（已含我的文件、ELF Sep 16、`glm5_2.py` 在）；README 那套来自 `xllm_coding/xllm`（HEAD `b448eb5a`，**仅 root 可见**，与我的基线不同源）；ccache 目录是**空的** |
| 我的 delta 相对分支基线 `200939593` | **37 个文件，全部落在 kv_cache_transfer 层内**（+ `config/disagg_pd_config.*`、`kv_cache/cache_layout_builder.cpp`、`kv_cache/kv_cache_tensor_role.h`、测试、CMakeLists、设计文档）；**不含** `llm_engine.cpp` 等外部文件 ⇒ 移植成本低 |

**(5) 本轮发现的疑点（等用户裁决，影响后续路线）**

主线**本来就有** `kv_cache_transfer/push_route.{h,cpp}`（由 `feat: support heterogeneous qwen3.5 PD disaggregation. (#1995)`
引入），被我的第一个提交 `9605a7c6a`（"drop dead routing code"）**删除**，`push_route_test.cpp` 一并删。
而 README 那套 GLM-5.2 PD 依赖主线近期的 `align pd link_cluster with kv-split owners` / `gate pd link_cluster on
tp-invariant kv cache` 两个提交。⇒ **"验证 P/D DCP 异构 + indexer full sequence"到底验哪条实现**：
(a) 我这条 `--pd_route=canonical`；(b) 主线 `push_route` + link_cluster gating；(c) 只验这个**配置/语义**本身（不绑实现）。
**未定，下一步先问清再动手**（编译 + 部署成本高，走错方向代价大）。

### 2026-09-18（第 16 轮）——运行时验证的执行计划与当前停点（**压缩上下文后的续接点**）

**用户裁决（第 15 轮的三个问题）**

1. 端到端验证目标 = **(a) 我这条分支的 `--pd_route=canonical`**（不是主线 `push_route`，也不是"只验配置语义"）。
2. 通过判据 = **字节级比对 KV 传输结果**（不是"输出不乱码"）。
3. 用户补充指令：**编译成功后先停下**，等用户压缩上下文再继续。

**(1) 本轮已完成**

| 事项 | 状态 / 位置 |
|---|---|
| 83 侧容器准备 | 已创建容器 **`fengmin-pdroute-83`**（83 上，`--network=host --privileged`），挂载：`/export/home`、`/mnt/cfs/9n-das-admin/llm_models`、`/etc/hccn.conf`、Ascend driver/add-ons、`npu-smi`、`/var/log/npu`、`/runtime`、`/var/queue_schedule`。镜像 `quay.io/jd_xllm/xllm-ai:xllm-dev-a3-arm-cann9-20260801` |
| 容器内已核实 | `hccn address_0=11.83.191.11` ✅；CFS 4 层模型可读（5 shard）✅；`npu-smi info -l` 8 NPU/16 chip ✅；`torch 2.9.0 + torch_npu post2`，`device_count=16` ✅；etcd / `xllm_master_serving` 二进制在 `/export/home/shifengmin.3/workspace/{xllm-pack-probe-one-20260805/xllm_pack, xllm-service/build/xllm_service}/` ✅；**镜像里没有 xllm python 包**（必须装 wheel）|
| 模型完整性 | CFS `GLM-5.2-W8A8-EcoTech-4layers`：index 2472 个张量，引用 **5 个 shard 全在**，共 **20.48GB**；layers=[0,1,2,3]；非层键只有 `lm_head/embed_tokens/norm`；`indexer_types=['full','full','full','shared']`；**无 MTP**（`num_nextn_predict_layers=0`）⇒ 必须 `num_speculative_tokens=0` |
| 98 侧构建树 | `xllm-pdroute` 已通过 **git bundle**（本地 `72e0ea817..pd-routing-s0s1`，541K）切到我的分支 **`d68d7eda2`**，工作区干净；`push_route.*` 已不在磁盘上 |
| 构建流水线 | 容器 `fengmin-cann9-20260801`（98）内后台跑 `/tmp/build_and_wheel.sh` → 先 `ninja -C build/cmake.linux-aarch64-cpython-311 xllm`，成功后 `SKIP_TEST=1 python setup.py bdist_wheel --device npu`。日志 `/tmp/xllm_build2.log`，标记 `NINJA_EXIT=` / `WHEEL_EXIT=`；wheel 落 `xllm-pdroute/dist/*.whl` |

**(2) 关键接线事实（本轮查清，后续直接用）**

- **python 模型开关**：`--model_impl=python` + 环境变量 `XLLM_PYTHON_MODEL_PATH=<site-packages>`（即包含 `xllm` 包的那层目录；`launch_server.py:_ensure_python_model_path` 就是这么设的）。
  证据：`model_config.cpp:143 is_python_model_impl()`、`glm_5_3_flash.md:151 --model_impl=python`。
  ⇒ 直接调 ELF 时要显式给这两个。
- **`kv_cache_transfer_type` 在我这条线里不存在**（`grep` 全树无此符号）——那是 README 那棵 b448eb5a 树的开关。我的树里 `KVCacheTransferFactory::create` 在 NPU 分支**无条件**造 Mooncake transfer
  ⇒ 启动命令**不要**传 `--kv_cache_transfer_type`（传了会因未知 flag 出问题），canonical 只走 Mooncake。
- 管控面端口占用：83 上 **4389 / 48889 已被别人的 `fengmin-cann9-0801` 占用**（`--network=host`）⇒ 我用**独立端口**：etcd `5389`、master `58888/58889`；P brpc `28994+`、D brpc `29994+`；transfer `46100+/47100+`；disagg_pd `9877/9878`。

**(3) 计划拓扑（单机 83，16 chip 足够）**

| 角色 | ranks | 并行 | 结果 S（DCP 切片数） | 设备 |
|---|---|---|---|---|
| Prefill | 4 | `dp=1 cp=2` ⇒ tp=2，`--kv_split_size=2` | **S_P=2**（形状 (a)：`S \| cp`） | chip 0–3 |
| Decode | 2 | `dp=1 cp=1` ⇒ tp=2，`--kv_split_size=1` | **S_D=1**（形状 (a)） | chip 4–5 |

⇒ **S_P=2 vs S_D=1 = DCP 异构**（prefill 两切片 → decode 单份，属"折叠"方向）。后续可加做 D `kv_split=2`（`cp=2, tp=1`，S_D=2 == cp*tp）作为**同构对照**。

**(4) 下一步（压缩上下文后从这里继续）**

1. 等 `WHEEL_EXIT=0`；记录 wheel 路径与大小。
2. `docker cp` wheel 到 `fengmin-pdroute-83`，`pip install --force-reinstall --no-deps`；
   校验 `python3 -c "import xllm; print(xllm.__file__)"` 与 `site-packages/xllm/xllm`（ELF）存在。
3. 在容器内起 etcd(`5389`) + `xllm_master_serving`（`--etcd_addr=11.87.191.83:5389 --http_server_port 58888 --rpc_server_port 58889 --tokenizer_path=<4 层模型>`）。
4. 先起 Decode（2 ranks）再起 Prefill（4 ranks），都带：`--model_impl=python`、`XLLM_PYTHON_MODEL_PATH=<site-packages>`、
   `--enable_disagg_pd=true --instance_role={DECODE,PREFILL}`、`--pd_route=canonical`、`--num_speculative_tokens=0`、
   `--model=/mnt/cfs/9n-das-admin/llm_models/GLM-5.2-W8A8-EcoTech-4layers`、`--npu_kernel_backend=ATB`。
   就绪判据：各 rank 日志出现 `Brpc Server started`，`58888` 在 P+D 齐了之后 listen。
5. Smoke：`/v1/models` + 短 chat（4 层模型输出不会正常，只看**是否报错/乱码崩溃**与日志）。
6. **字节级验证**（用户要的判据，设计待定）：初步方案 = 在 canonical push 前后加**临时诊断**（env 门控），
   P 侧把每个 `RouteRegion` 的**源字节**按 `(role, canonical_block, slice, offset)` 落盘；
   D 侧在 KV 传输完成后把本地物理缓存按同一坐标落盘；两边文件拿到 Mac 上逐字节比对。
   注意：D 侧没有 canonical 传输动作（PUSH 方向由 P 发起），所以 D 侧的取样点要挂在传输完成之后
   （候选：`kv_transfer_completion.*`，或 decode worker 收到完成信号处）；PULL 已按 A2 显式拒绝（loud），
   若运行中出现 canonical PULL 会立刻在日志里暴露 —— 这本身也是一条要观察的信息。
7. 所有结论（含失败原因）回写本文件并 `git push origin pd-routing-s0s1`。

**(5) 待澄清/风险**

- 我的基线 200939593（+本分支）**是否支持 GLM-5.2 python + PD 端到端**尚未验证；72e0ea817 明显更旧（连 `kv_cache_transfer_type` 都没有）。
  若 EP/PD 在 python 模型路径上编译或运行失败，**下一步的兜底**是：把这 21 个文件的 delta 移到 83/98 上那棵已验证的 b448eb5a 树
  （`xllm_coding/xllm`，仅 root 可见）或 `xllm-coding/xllm`，用它们的 runtime + 我的 routing。
- 4 层模型 `layer3` 是 MoE（`n_routed_experts=256`，`first_k_dense_replace=3`）⇒ 小 tp（2）时要注意 `--ep_size` 是否需要显式给。
- 83 是共享机（别人有容器在跑，但卡是空的）：我只用 chip 0–5，绝不 kill 别人的进程。

### 2026-09-18（第 17 轮）——构建路径的坑与**暂停点（用户要求先压缩上下文）**

用户指令：**"启动标准构建前，先等我压缩上下文"**。本轮把构建折腾清楚后**主动停掉**在跑的构建，
状态全部落在下面（压缩后按 §(5) 直接续接）。

**(1) 上一轮的"编译成功"是假成功（重要，别再被骗）**

`ninja -C .../xllm-pdroute/build/cmake.linux-aarch64-cpython-311 xllm` 报 `NINJA_EXIT=0`，
但那份 build dir 是从 `xllm-dcp-fp32` **整份拷来的**，`CMakeCache.txt` 里
`CMAKE_HOME_DIRECTORY=/export/home/shifengmin.3/workspace/xllm-dcp-fp32`（`CMAKE_CACHEFILE_DIR` 同样），
ninja 文件里所有源码路径都写死指向 `xllm-dcp-fp32`。⇒ **那次编译编的是 dcp-fp32 那棵树**，
产出的 ELF（`xllm-dcp-fp32/build/lib.linux-aarch64-cpython-311/xllm/xllm`，14:03，559MB）**不含我的改动**。
（注：探针 `run_probe3.sh` 用的是 `-v $HOSTWORK:/export/home/.../xllm-dcp-fp32` 的**挂载覆盖**，
所以探针确实编到了我 staged 的文件 —— 探针结论仍然有效。）

**(2) 两棵树的 git 关系（踩过的坑）**

`xllm-pdroute` 与 `xllm-dcp-fp32` **共用同一份 git worktree 元数据**
（`.git` 指向 `xllm-coding/xllm/.git/worktrees/xllm-dcp-fp32`）。
所以在 `xllm-pdroute` 里做的 `git checkout -f pd-routing-s0s1` **也改了共享的 HEAD/index**，
导致 dcp-fp32 的文件仍停在 72e0ea817 而 HEAD 变成我的分支（172 个 dirty）。
**已修正**：在真实树 `xllm-dcp-fp32` 里也执行了 `git checkout -f pd-routing-s0s1` ⇒ 现在两棵树都在
我的分支 `d68d7eda2`、工作区干净、`push_route.*` 不在磁盘上。

**(3) 已查清的三处构建机关（后续直接用）**

| 机关 | 事实 | 正确做法 |
|---|---|---|
| `xllm_ops` 预编译 | `third_party/xllm_ops` 在这棵树里**不是真子模块**（只有一个内嵌目录，**没有 `build.sh`**）。CMakeLists.txt:81 在 `ENV{XLLM_OPS_GIT_HEAD_CACHED}` 与 `git -C third_party/xllm_ops rev-parse HEAD` 不等时会跑 `build.sh` → `error code 127` 直接失败 | 导出 **`XLLM_OPS_GIT_HEAD_CACHED=$(git -C third_party/xllm_ops rev-parse HEAD)`**（这里会解析成超项目的 HEAD，即 `d68d7eda2…`）→ 日志出现 `xllm_ops git HEAD unchanged; skipping precompile` ✅ 已验证有效 |
| `setup.py` 的子模块门禁 | `scripts/build_support/utils.py:504 _validate_submodules_or_exit`：`git submodule status --recursive` 每行首字符 `-`（未在 `.git/config` 注册）即 `exit(1)`。`git submodule init` **不解决**（这些目录没有自己的 `.git`） | 已在该函数开头插入 **`PDROUTE_BUILD_ONLY_BYPASS` 早返回**（**仅本地树、未提交、后续要还原**）。真正合入前必须还原 |
| torch/torch_npu 路径 | 只用 ninja 触发 reconfigure 时，`find_package(Torch)` 会算空（`-I/include`）且 `$ENV{PYTORCH_NPU_INSTALL_PATH}` 若为 `/usr/local/libtorch_npu` 在该容器**不存在** ⇒ 133 条 `fatal error: torch/torch.h: No such file or directory` | 正确环境：`PYTHON_EXECUTABLE=/usr/local/python3.11.15/bin/python3`、`PYTORCH_NPU_INSTALL_PATH=/usr/local/python3.11.15/lib/python3.11/site-packages/torch_npu`（`.../torch_npu/include/torch_npu/csrc/...` 与 `.../torch_npu/lib/libtorch_npu.so` 都在）。**用户指示改用标准构建 `python setup.py build --device npu`**（它会自己 setup 这些） |

`CMakeCache.txt` 已被污染 ⇒ 我把它改名成 `CMakeCache.txt.broken` 并删掉 `CMakeFiles/`，
下一次 configure 会从干净状态开始（对象文件保留，成功过的目标不会重编）。

**(4) 本轮结束时状态**

- 标准构建**已启动过又被按要求停掉**（停在 `dependencies.sh` 装 Mooncake 依赖阶段：
  `yalantinglibs`、Go 工具链、dnf 元数据；日志 `/tmp/std_build.log` 尾部是 dnf 下载）。
  进程树已终止（复查只剩我自己那次 `pgrep` 的 shell）。
- 83 容器 `fengmin-pdroute-83`：镜像 `xllm-dev-a3-arm-cann9-20260801`，`--network=host`，
  挂 `/export/home`、`/mnt/cfs/9n-das-admin/llm_models`、`/etc/hccn.conf`、Ascend driver/add-ons、`npu-smi`、
  `/var/log/npu`、`/runtime`、`/var/queue_schedule`。已核实：hccn `address_0=11.83.191.11`、CFS 4 层模型可读、
  16 chip 可见、torch/torch_npu 可导入。**还缺 `custom_xllm_math`（135MB 编译好的 xllm 自定义算子）** ——
  `cop` 脚本已写好（`copy_vendor.sh`：从 `fengmin-cann9-0801` `docker cp` 出来再 `docker cp` 进我的容器），**尚未执行**。
- 83 侧启动脚本：本地已写好 `pdroute83/env.sh`（模型/端口/topology/ATB+HCCL 环境）与
  `pdroute83/start_control.sh`（etcd 5389 + `xllm_master_serving` 58888/58889，**独立端口**，避开别人占用 4389/48888 的实例）；
  **还差 `pdroute83/start_workers.sh`（P/D 启动）与 `smoke.sh`**，写完 `scp` 到 83 的 `workspace/pdroute83/`。
- 83 上别人的 `etcd(4389)` 与 `xllm_master_serving(48888)` **在跑，别动**；83 卡是空的，我只用 chip 0–5。

**(5) 压缩上下文后的续接步骤（照做即可）**

1. **跑标准构建**（在 98 的容器里）：
   ```bash
   rrun -F ~/work/.ssh-xllm-config jd-node-98 < ~/work/std_build.sh     # 会 docker cp + docker exec -d
   # 日志：容器内 /tmp/std_build.log；标记 BUILD_EXIT= / WHEEL_EXIT=；wheel 落在
   #   /export/home/shifengmin.3/workspace/xllm-dcp-fp32/dist/*.whl
   ```
   （`std_build.sh` 里已含：清 cache、正确 torch/torch_npu 环境、`XLLM_OPS_GIT_HEAD_CACHED`、
   `SKIP_TEST=1 python setup.py build --device npu` 后接 `bdist_wheel`。）
2. 构建成功后：`cp wheel` → 83 → `docker cp` 进 `fengmin-pdroute-83` → `pip install --force-reinstall --no-deps`；
   并执行 `copy_vendor.sh` 补 `custom_xllm_math`。
3. 写 `start_workers.sh` + `smoke.sh`，`scp` 到 83，起 control(D) → P，做 smoke。
4. 字节级验证探针（设计见第 16 轮 §(4).6）。
5. 所有结论回写本文件并 push。

---

## 第 18 轮（2026-09-18 14:16–14:40）：标准构建打通；python+PD 的 DCP 契约彻底查清

用户裁定：验 **(a) 我这条 `--pd_route=canonical`**，判据 = **KV 传输结果的字节级比对**。
本轮把"标准构建"这条路彻底走通，并把 python 路径下 P/D 该怎么配查到了源码级。

### (1) 三个"门"逐个解决（都写清了真因，后续不用重查）

**(a) 依赖门**：`python setup.py build` 在 `dependencies.sh -y` 上 dnf 失败
（`Unable to find a match: boost1.78-devel msgpack-devel` / `epel-release`）。
**真因不是缺依赖**，而是 `utils.py:_get_required_dependency_files()` 检查的路径与安装路径**不一致**：
它查 `/usr/local/lib/cmake/yalantinglibs/config.cmake`，而实际装在 `/usr/local/yalantinglibs/`。
- 解法①：把 yalantinglibs 按标准 prefix 布局软链到 `/usr/local`
  （`/usr/local/lib/cmake/yalantinglibs` → `…/yalantinglibs/lib/cmake/yalantinglibs`，以及 7 个 include 项）。
- 解法②：装 **Go 1.25.9** 到 `/usr/local/go`。注意 `Mooncake/dependencies.sh` 的 `GOVER=1.25.9`，
  而 `_is_mooncake_go_ready()` 要求**精确版本相等**（复制别的容器的 1.25.10 不行）；
  `https://dl.google.com/go/go1.25.9.linux-arm64.tar.gz` 可达（55MB）。
- 验收方式：直接调 utils.py 自己的函数 ⇒ `missing dependencies: NONE`、`go ready: True`。
- 六个必查项（记下来）：yalantinglibs cmake config、zstd.h、libzstd.so、xxhash.h、libxxhash.so、msgpack.hpp。
  本容器 zstd/xxhash 在 `/usr/include` + `/usr/lib64`，msgpack.hpp 在 `/usr/local/include`。

**(b) xllm_ops 门**：`CMakeLists.txt:72` 的判据是
`NOT DEFINED XLLM_OPS_GIT_HEAD_CACHED OR NOT XLLM_OPS_GIT_HEAD STREQUAL XLLM_OPS_GIT_HEAD_CACHED`；
`XLLM_OPS_GIT_HEAD` 由 `:51` 的 `execute_process` 得到（= `git -C third_party/xllm_ops rev-parse HEAD`，
而该目录**不是真子模块**：真正的 `.git` 在 `third_party/xllm_ops/xllm_ops/.git`，内容指向不存在的
`../../.git/modules/third_party/xllm_ops` ⇒ git 向上冒泡返回**本仓 HEAD**）。
- cache 被删后只能靠 **env** 满足；但 `utils.py:665` 在 marker 不匹配时会 **主动 `os.environ.pop`** 该 env。
- 且 `third_party/xllm_ops/build.sh` **不存在** ⇒ 一旦触发预编译即 `error code: 127`（假"编译成功"的真凶）。
- **解法**：把 `$ASCEND_OPP_PATH/vendors/custom_xllm_math/.xllm_ops_git_head` 写成**当前 repo HEAD**
  （= 一次成功预编译后 CMake 自己会写的内容；`CMakeLists.txt:106`）。ops 早已编译安装（`custom_xllm_math` 135MB）。
- 注意：**marker 不是 CMake 读的**，读它的是 utils.py；CMake 只认 env/cache。两侧都要满足。

**(c) CPU 过订阅**：`nproc=12`，原 `MAX_JOBS=64` ⇒ 改 `MAX_JOBS=12`。

### (2) 设备映射（同机混布 P/D 的依据）

`distributed_runtime/master.cpp:449` + `platform/device_name_utils.cpp:40`：
`device_idx = node_rank % visible_device_count`，`visible_device_count = Platform::device_count()`
（受 `ASCEND_RT_VISIBLE_DEVICES` 影响）。
⇒ 同一容器里跑 P 和 D，**每个进程只暴露自己的那几张卡**即可：
P `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3`（nnodes=4，node_rank 0..3 → device 0..3）；
D `=4,5`（nnodes=2，node_rank 0..1 → device 0..1 → 物理 4,5）。互不冲撞。

### (3) 标志集（旧二进制 `--help` + 源码双重确认）

存在：`--pd_route`（默认 `legacy`，取值 `legacy`/`canonical`）、`--kv_cache_transfer_mode`（PUSH/PULL）、
`--communication_backend`、`--python_model_path`、`--enable_pd_ooc`、`--kv_push_dst_rotate`、
`--model_impl`、`--backend`、`--host/--port/--master_node_addr/--etcd_addr`、`--nnodes/--node_rank`、
`--cp_size/--dp_size/--kv_split_size`、`--npu_kernel_backend`（AUTO/ATB/TORCH）、
`--enable_disagg_pd`、`--instance_role`、`--disagg_pd_port`、`--transfer_listen_port`、
`--enable_prefix_cache`、`--enable_chunked_prefill`、`--enable_schedule_overlap`、
`--max_memory_utilization`、`--block_size`、`--num_speculative_tokens`、`--draft_model`、`--indexer_cache_dtype`。
**不存在**：`--layerwise_split_size`、`--kv_cache_transfer_type`、`--dispatch_policy`。
（第 15 轮记录里"`communication_backend` 不在生产代码里"是**错的**——`--help` 里有；当时 grep 没覆盖到定义点。）

### (4) python 路径的 DCP 契约（**本轮最重要的产出**，直接决定 P/D 配置）

- `master.cpp:395 resolve_npu_kernel_backend_for_options()`：`--model_impl=python` **强制**
  `npu_kernel_backend=TORCH`（并打日志）。⇒ 启动脚本直接传 `TORCH`（传 ATB 会被覆盖）。
- `xllm.cpp:740`：python 路径期望的 OPP vendor 顺序是
  `glm_next_transformer, custom_transformer, custom_xllm_math`；但 83 上与**已能跑通 GLM-5.2 的**
  `fengmin-cann9-0801` 一致，只有 `custom_transformer` + `custom_xllm_math`（缺 `glm_next_transformer`
  在该代码里是**安全 no-op**，因为只保留 `op_api/lib/libcust_opapi.so` 真实存在的 vendor）。
- **DCP 契约**（`framework/parallel_state/context_parallel_topology.h` 注释即规范）：
  `pcp_size = cp_size`，`dcp_size = kv_split_size_effective()`；`cp_rank` = PCP rank，`kv_split_rank` = DCP rank。
  合法形状：`dcp_size | pcp_size`（partitions_pcp），或 `dcp_size == pcp_size * tp_size`（= dp_stride）。
  `dcp_rank` = 物理 KV slice 的属主；`KVShardLayout.logical_block_size = physical_block_size * dcp_size`，
  rank `r` 拥有每个逻辑块内 `[r*B, (r+1)*B)` 这一段。
- **NPU 上不物化 dcp 通信组**：`collective_communicator.cpp:542` 的 dcp 组创建在
  `if constexpr (Platform::is_mlu())` 里；另一处 NPU 创建分支（:586）显式写了 `!is_python_model_impl(...)`。
  ⇒ python 侧 `distributed.dcp_group()` 恒为 `None`。
- 因此 `xllm/python/model_executor/executor.py:74` 的 `SfaDcpAttentionBackend`（条件：
  `cp_size == 1 && dcp_group is not None && dcp_group.size() > 1`）在 **NPU + python 下不可达**；
  实际总是 `NpuPagedAttentionBackend`。**这不影响 PD 路由要从 P 搬到 D 的物理 KV 布局**（那是另一条路）。
- **python 的 prefill KV 分片走另一条路**：`core/runtime/py_executor_impl.cpp:214` 要求
  `enable_mla && cp_size > 1 && kv_split_size > 1`（且 prefill/chunked_prefill）才构建
  `kv_shard_batch_metadata`，把 `new_cache_slots` 本地化到本 rank 的物理 slice。
  ⇒ **P 侧必须 `cp_size > 1` 且 `kv_split_size > 1`**，否则根本不发生 DCP 分片。
- `parallel_args.h:177 kv_split_rank()`：无 dcp_group 时退化为 `rank / (world_size / kv)`；
  在 `partitions_pcp` 情形与拓扑 `dcp_rank` **完全一致**（P：`rank/2` → 0,0,1,1）。
- D 侧（`kv_split=1`）不需要分片元数据；`worker_impl.cpp:1408` 只在 `dcp_group != nullptr` 时本地化槽位。

### (5) 本轮确定的 P/D 配置（S_P=2 vs S_D=1，真异构）

| 角色 | ranks | cp | dp | kv_split | tp | 可见卡 | S |
|---|---|---|---|---|---|---|---|
| PREFILL | 4 | 2 | 1 | 2 | 2 | 0,1,2,3 | **2** |
| DECODE | 2 | 1 | 1 | 1 | 2 | 4,5 | **1** |

拓扑推导（P）：`world=4, dp=1, pcp=2` ⇒ `tp = 4/(1*2) = 2`，`dp_stride = 2*2 = 4`；
`dcp=2 ≤ pcp=2` 且 `2 % 2 == 0` ⇒ `partitions_pcp`；`pcp_per_dcp=1` ⇒ `dcp_rank = pcp_rank`，
`kv_split_rank = rank/2`（0,0,1,1）✅ 两者一致。
映射：P rank0/1（pcp=0, tp=0/1）持 slice 0；rank2/3（pcp=1）持 slice 1；
D rank0/1（tp=0/1）各持**整块**。
⇒ canonical 路由必须把**同一逻辑块的两个 P slice 拼成 D 的整块**（不是恒等搬），
这正是要字节级验证的 reshard；`INDEX/INDEX_SCALE` 角色还带 `full_sequence_replica`（indexer full sequence）。

### (6) 83 侧脚手架已落盘（`/export/home/shifengmin.3/workspace/pdroute83/`）

`env.sh`、`start_workers.sh`（decode 先起、8s 后 prefill）、`stop_workers.sh`（按 ELF 路径匹配，避免自杀）、
`watch.sh`（扫 `Brpc Server started`）、`smoke.sh`（先 `/v1/models` 再 chat）、`npu_init.sh`、`preflight.sh`、`start_control.sh`。
端口 **5389 / 58888 / 58889**（避开别人在跑的 4389 / 48888 / 48889，**别杀**）。
P：brpc `28994+`、transfer `46100+`、disagg `9877`、master `18888`；
D：brpc `29994+`、transfer `47100+`、disagg `9878`、master `19888`。
`HCCL_IF_BASE_PORT` P=48439 / D=48539 分开；`unset HCCL_OP_EXPANSION_MODE`（不走 AIV ⇒ 不需要 ranktable）。

### (7) 83 预检结果

16/16 chip 空闲（~3GB/64GB）；hccn `address_0=11.83.191.11`；torch 2.9.0+cpu / torch_npu 2.9.0.post2，16 卡；
4 层模型 181 个 shard 齐全 + tokenizer；`custom_xllm_math` 已从 `fengmin-cann9-0801` 复制进我的容器
（两边 vendor 集合一致）；`etcd` / `xllm_master_serving` 二进制在位；**xllm 包尚未安装**（等 wheel）。

### (8) 构建状态

`SKIP_TEST=1 python setup.py build --device npu` → 越过两个门后在编译，
`[621/1396]`（12 核；ccache 目录空 ⇒ 全量编译，预计较久）。bdist_wheel 紧随其后。
产物：`build/lib.linux-aarch64-cpython-311/xllm/xllm` + `dist/*.whl`。

### (9) 仍需注意的树上残留

- `scripts/build_support/utils.py` 里的 `PDROUTE_BUILD_ONLY_BYPASS`（子模块门早返回）**仍未提交**，
  合入前要么删掉、要么按规范改写。
- 树上未跟踪残留：`third_party/dependencies.sh`、`xllm/core/framework/kv_cache_transfer/push_route.{h,cpp}`
  （CMake 不引用它们，**不参与编译**，属主线遗留副本）。

---

## 第 19 轮（2026-09-18 14:40–15:05）：链接缺陷（真 bug）修复；构建复跑中；GitHub 分叉的真相

### (1) **本轮的硬产出：发现并修复我分支里一个真实缺陷**（之前一直没暴露）

第 18 轮的全量编译跑到 `[1396/1396]` **全部编译通过**，但**链接失败**：

```
FAILED: build/lib.linux-aarch64-cpython-311/xllm/xllm
/usr/bin/ld: kv_cache_transfer.cpp: undefined reference to `xllm::pd_route_mode_name(xllm::PdRouteMode)'
mooncake_kv_cache_transfer.cpp: undefined reference to `xllm::declare_cache_group(...)'
mooncake_kv_cache_transfer.cpp: undefined reference to `xllm::PeerDirectory::describe(...)'
mooncake_kv_cache_transfer.cpp: undefined reference to `xllm::canonical_blocks_of_request(...)'
mooncake_kv_cache_transfer.cpp: undefined reference to `xllm::build_route_peer(...)'
mooncake_kv_cache_transfer.cpp: undefined reference to `xllm::PdRouteTransfer::plan(...)'
mooncake_kv_cache_transfer.cpp: undefined reference to `xllm::flatten_route_for_layers(...)'
collect2: error: ld returned 1 exit status
```

**根因**（不是"符号没写"，`pd_route_mode_name` 定义在 `pd_route_transfer.cpp:331`，7 个符号全都存在）：

`xllm/core/framework/kv_cache_transfer/CMakeLists.txt` 里
`pd_route_table` / `route_binder` / `cache_directory` / `pd_route_transfer` **四个 `cc_library` 都声明了**，
但**最后一个 `kv_cache_transfer` 目标的 DEPS 里一个都没引用**。
CMake 里没有依赖边的静态库**根本不会被 ninja 构建** ⇒
`pd_route_transfer.cpp` / `cache_directory.cpp` / `pd_route_table.cpp` / `route_binder.cpp`
**四个实现文件从未被编译**（`.o` 不存在，`find`/`ar t` 全查不到），
只有头文件参与编译，所以语法检查**全过** ⇒ 直到链接才炸。

**为什么以前没发现**：第 15/17 轮只做了"3 个生产 TU 编译 rc=0"，**从来没有链接过**。
这是"编译通过 ≠ 能跑"的教科书案例。

**修复**（一行，`CMakeLists.txt` 的第 147 行前后）：

```diff
   SRCS
     ...
     $<$<BOOL:${USE_NPU}>:mooncake_weight_transfer.cpp>
   DEPS
+    :pd_route_transfer
     :cache_layout
     :common
```

（`:pd_route_transfer` 自己 DEPS `:cache_directory` / `:kv_redundancy` / `:pd_route_table` / `:route_binder`，
`:cache_directory` 又 DEPS `:cache_layout` / `:route_binder` / `:pd_route_table` ⇒ 一条边足够，传递闭包覆盖全部 7 个符号。）
备份文件：`CMakeLists.txt.bak-r18`（同目录）。

### (2) **另一个坑：构建期间提交会让 bdist_wheel 挂掉**（自伤，已记牢）

第 18 轮第一次构建时，`setup.py build` 编译成功后我**提交了 round-18 工作日志**，
repo HEAD 从 `d68d7eda2` → `ac972b728`。随后脚本里的 `bdist_wheel` 再跑一遍
`utils.py:_ensure_xllm_ops_rebuild_state()`：

```
ℹ️ Installed xllm_ops marker does not match third_party/xllm_ops HEAD. A rebuild is required.
   installed git HEAD: d68d7eda22de090ef84e8e17eb406705772b8e22
   source git HEAD:    ac972b7284fa44c6b71b8d4fb3323a353de1820d
```

⇒ marker 与 HEAD 不匹配 ⇒ 重新触发预编译 ⇒ `third_party/xllm_ops/build.sh` 不存在 ⇒ cmake `error 127`。

**结论/规矩**：
- `xllm_ops` 门的输入是 **repo HEAD**（`git -C third_party/xllm_ops rev-parse HEAD` 会向上冒泡到本仓）。
  **构建期间不要 commit / checkout / rebase。**
- 如果非要动，动完**立刻**把 `$ASCEND_OPP_PATH/vendors/custom_xllm_math/.xllm_ops_git_head`
  写成**新的** HEAD，再进行下一次 `setup.py`（`utils.py` 也会读这个 marker 来决定是否 pop 环境变量）。
- 重跑只需要跑 `setup.py build` + `bdist_wheel`，**不要**像 `std_build.sh` 那样
  `rm -rf CMakeFiles`（那会把 1396 个 `.o` 全丢掉、退化成全量重编）。本轮用的是 `resume_build.sh`（不删任何东西）。

### (3) 当前构建状态（**续接点**）

- 脚本：`~/work/resume_build.sh`（本机）→ 容器内 `/tmp/resume_build_inner.sh`，日志 `/tmp/std_build_resume.log`。
- 它做的事：① 把 ops marker 对齐到当前 HEAD（`ac972b728`）② 导出第 18 轮那套环境（`PYTHON_EXECUTABLE`
  /`PYTORCH_NPU_INSTALL_PATH`/`MAX_JOBS=12`/`XLLM_OPS_GIT_HEAD_CACHED`）③ `setup.py build --device npu`
  ④ `setup.py bdist_wheel --device npu` ⑤ `ls -l dist/*.whl`。
- 15:00 时进度 **`[697/1380]`**，`undefined reference` 计数 **0**，无 `error:`（CMakeLists 变更导致 reconfigure，
  目标数从 1396 变 1380，许多 TU 被重编）。
- 等待器：本机后台 job 跑 `~/work/wait_build2.sh`（每 20s 查 `WHEEL_EXIT`，最多 100 分钟）。
- 产物预期：`$TREE/build/lib.linux-aarch64-cpython-311/xllm/xllm`（ELF，带 debug 约 **559MB**）+
  `$TREE/dist/xllm_npu_torch2_9_0-*.whl`。

### (4) GitHub 远端分叉的真相（**虚惊一场，不要 force push**）

`git fetch shifengmin` 在重试 **~39 次**后成功（GitHub SSH 大量 `message authentication code incorrect`，
小流量 `ls-remote` 有时能过、`fetch` 的 pack 传输几乎必挂；试过 `IPQoS=0`/换 MAC/关压缩，只有
`-o IPQoS=0 -o TCPKeepAlive=yes` 偶尔能过）。fetch 后看清：

| | commit | 内容 |
|---|---|---|
| GitHub `pd-routing-s0s1` tip | `8ac7b411b` | **只改工作日志**，追加第 16+17 轮（+129 行）|
| 它上面一个 | `e057637a8` | 同样只改工作日志（第 16 轮）|
| 我本地 tip | `ac972b728` | **只改工作日志**，追加第 18 轮（+126 行）|
| **共同基点** | `d68d7eda2` | 文件到第 1013 行完全相同（第 15 轮末尾）|

⇒ **两边都是在同一个基点上"往文件尾部追加"**，只是分别发生在**不同的 clone**（所以 `8ac7b411b`
在本树对象库里原本不存在）。**两边都是文档，没有任何内容冲突**，只是追加位置相同。

**处置（下次做）**：把本地那**唯一一个** commit rebase 到远端之上，冲突时取"远端文件 + 追加第 18/19 轮"，
结果 = 第 15→16→17→18→19 轮连续完整，且 push 是 **fast-forward，无需 force**：

```bash
cd $TREE
git stash push -- scripts/build_support/utils.py       # 保住 PDROUTE_BUILD_ONLY_BYPASS
git rebase --onto 8ac7b411b d68d7eda2 pd-routing-s0s1  # 只 replay ac972b728
#   冲突时：git checkout 8ac7b411b -- <worklog>，再把本轮文本追加进去，然后 git add + git rebase --continue
git stash pop
# 然后把 ops marker 重新对齐到新 HEAD（见 (2)），再 push
GIT_SSH_COMMAND="ssh -o ControlMaster=no -o ControlPath=none -o IPQoS=0 -o TCPKeepAlive=yes" \
  git push shifengmin pd-routing-s0s1
```

**注意**：rebase 会改 HEAD ⇒ **必须在构建结束之后再做**（见 (2)）。

### (5) 字节级验证的落地设计（代码已写好，等跑通就插桩）

> **⚠️ 第 20 轮已纠正本条**：下面的判据只对 **key/value** 成立；**index/indexer_scale 的行号是 canonical 块号**
> 且**每个 P rank 都复制一份**。以第 20 轮 (1) 的裁定表为准。

判据**不依赖我自己的路由代码**，而是用 **token 语义**：

> `DECODE.physical_block[r]  ==  PREFILL(dcp_rank = r % S).physical_block[r // S]`

推导：`KVShardLayout` 里 `localize_slots()` 返回的物理块号 = **逻辑块号**；
canonical block = `logical * S + slice`（`cache_directory.h` 注释：
"A block-scoped group's id names one *logical* block, which spans `kv_split_size` canonical blocks"）。
所以 P（S=2）rank 上物理块 `b` 装的是 canonical `2b+j`；D（S=1）物理块 `r` 装的是 canonical `r`。
并且 P 的 `tp_rank = rank % 2`、`dcp_rank = rank // 2` ⇒ D 的 rank `d` 对应
`prefill_rank = (r % 2) * 2 + (d % 2)`。

已写好的两个文件（在本机 `~/work/pdtrace/`，**未提交**）：

- `_pd_trace.py`：env `XLLM_PD_TRACE_DIR` 开关；每个 rank 每层每个 slot（`key/value/index/conv/ssm/indexer_scale`）
  逐**物理块**算 sha256 写 JSONL，行内含 role/rank/layer/slot/shape/dtype/block/nbytes/sha256；
  从 `/proc/self/cmdline` 解析 `instance_role/node_rank/kv_split_size` 做身份；
  `XLLM_PD_TRACE_BLOCKS`（默认 64）限制每张量哈希的块数，避免拖慢。
- `compare_kv.py`：按上式比对，输出 match rate、per-slot 统计、首批 mismatch 明细。
- `executor_hook_tail.py`：**追加到** `xllm/python/model_executor/executor.py` 末尾的临时代码，
  包一层 `ModelExecutor.execute`（前后各 dump 一次 + `bump()`）。取
  `P = prefill 调用后的 after dump`、`D = decode 调用前的 before dump`（两者数据都稳定）。
- 用完必须**删掉**这个 tail + `_pd_trace.py`（或明确标注为验证专用）。

### (6) 部署路线（98 → 83，已验证通路）

- **CFS 不共享**：`/mnt/cfs/9n-das-admin` 在 98 和 83 上都可写，但**互相看不到**对方写的文件
  （各机自己的挂载）⇒ 不能当中转盘。
- **98 → 83 直连 SSH 可用**：`ssh -i /export/home/shifengmin.3/.ssh/id_rsa -o IdentitiesOnly=yes
  shifengmin.3@11.87.191.83` 从 98 上直接通。
- 脚本已写好（本机 `~/work/`）：
  - `deploy_stage98.sh`：容器内 → 98 host `/tmp`（wheel + `libasio.so`）→ scp 到 83 `/tmp` → 两边 `md5sum` 对比。
  - `deploy_install83.sh`：83 host → `docker cp` 进 `fengmin-pdroute-83` → `pip install --force-reinstall --no-deps`
    → 校验 `$SP/xllm/xllm`、`$SP/xllm/python/models/glm5_2.py`、`$SP/xllm/libasio.so`，
    并确认新 ELF 的 `--help` 里有 **`--pd_route`**。
  - 参照别人的教训：大文件用 `ssh cat` 比 `scp` 稳（`scp` 中断会出"Wheel is invalid"）；装完若报
    `libasio.so: cannot open shared object file`，把 `libasio.so` 放到 `LD_LIBRARY_PATH` 里
    （`/export/home/shifengmin.3/workspace/pdroute83/lib/`）。
- **`libasio.so` 的位置**：`$TREE/build/cmake.linux-aarch64-cpython-311/mooncake-common/libasio.so`。

### (7) 83 侧脚手架（已上传到 `/export/home/shifengmin.3/workspace/pdroute83/`）

`env.sh`、`start_workers.sh`（decode 先起、8s 后 prefill；`--npu_kernel_backend=TORCH`；`--pd_route=$PD_ROUTE`）、
`stop_workers.sh`（按 ELF 路径匹配，避免自杀）、`watch.sh`（扫 `Brpc Server started` + etcd keys）、
`smoke.sh`（先 `/v1/models` 再 chat）、`npu_init.sh`、`preflight.sh`、`start_control.sh`、
`run_all.sh`（停残留 → 起管控面 → 起 P/D → 等 300s → 报告）、`check_env2.sh`、`check_env3.sh`。

要点：端口 **5389/58888/58889**（避开别人在跑的 4389/48888/48889，**别杀**）；
P：brpc `28994+`、transfer `46100+`、disagg `9877`、master `18888`、`HCCL_IF_BASE_PORT=48439`；
D：brpc `29994+`、transfer `47100+`、disagg `9878`、master `19888`、`HCCL_IF_BASE_PORT=48539`；
`unset HCCL_OP_EXPANSION_MODE`（不走 AIV ⇒ 不需要 ranktable）；容器内**没有 `ss`**（`start_control.sh` 已改成回退 netstat/跳过）。

### (8) 83 预检结论（已核实）

16/16 chip 空闲（~3GB/64GB）；`hccn address_0=11.83.191.11`；`torch 2.9.0+cpu` / `torch_npu 2.9.0.post2`，16 卡；
4 层模型 181 个 shard + tokenizer 齐；`custom_xllm_math`（135MB）已从 `fengmin-cann9-0801` 复制进我的容器，
两边 vendor 集合**完全一致**（只有 `custom_transformer` + `custom_xllm_math`；
python 路径期望的 `glm_next_transformer` 缺失是**安全 no-op**）；
`etcd` / `xllm_master_serving` 二进制在位；`/usr/lib64/libtcmalloc.so.4` 存在；**xllm 包尚未安装**。

### (9) 续接清单（按顺序）

1. 等 `wait_build2.sh` 报 `WHEEL_EXIT=0`；若链接仍报 undefined，先查是不是又没被链接（`ar t` 看 archive）。
2. `rrun ... < deploy_stage98.sh` → `rrun -F ~/work/.ssh-xllm-nocm jd-node-83 < deploy_install83.sh`
   （注意 `deploy_install83.sh` 是**在 83 的 host 上**跑，不是容器里）。
3. 83 容器内：`bash npu_init.sh 0,1,2,3,4,5`（重启后必须）→ `bash run_all.sh`。
4. 就绪后 `bash smoke.sh`（先 `/v1/models` 拿真实 model 名）。
5. 通了之后再插桩做字节比对：把 `_pd_trace.py` 装到 83 的 `$SP/xllm/python/`，把 `executor_hook_tail.py`
   追加到 `$SP/xllm/python/model_executor/executor.py`，`export XLLM_PD_TRACE_DIR=...`，重跑，
   `python3 compare_kv.py <dir>`。
6. 最后（**构建空闲时**）做 (4) 的 rebase + push，并把本机 `~/work/pdtrace/` 之外的工具一起归档说明。

### (10) 仍未清掉的树上残留（合入前必须处理）

- `scripts/build_support/utils.py` 里的 `PDROUTE_BUILD_ONLY_BYPASS`（子模块门早返回）**未提交**。
- `xllm/core/framework/kv_cache_transfer/CMakeLists.txt.bak-r18`（我本轮建的备份）。
- 未跟踪残留：`third_party/dependencies.sh`、`push_route.{h,cpp}`（CMake 不引用、不参与编译）。
- `~/work/pdtrace/` 三个文件是**验证专用**，不要直接合进主线（或明确标注）。

---

## 第 20 轮：cache 行索引语义的最终裁定 + 字节比对器重写并自测（2026-09-18 15:0x）

等构建期间（`[897/1380]`，undefined=0）做的**纯本机**工作，没有碰正在构建的树。

### (1) 纠正第 18 轮的判据：行索引语义**按 slot 分两类**，不能只写一条

第 18 轮记的 `DECODE.physical_block[r] == PREFILL(dcp=r%S).physical_block[r//S]` 只对 **key/value 成立**，
对 **index/indexer_scale 是错的**。最终裁定如下（两条独立证据互相印证）：

**证据 A：paged-slot 契约。** `build_kv_shard_batch_metadata()`（`kv_shard_batch_metadata.cpp:145`）用
`KVShardLayout(options_.block_size(), S, dcp_rank)`，而 `localize_kv_shard_slots` 的算法是

```
logical_offsets  = global_slot % (block_size * S)
owner_rank       = logical_offsets / block_size          ← 谁拥有这一片
local_offset     = logical_offsets % block_size
logical_block_id = global_slot / (block_size * S)
local_slot       = logical_block_id * block_size + local_offset
```

⇒ 分配器的"块"就是**逻辑块**（`block_size * S` 个 token），`slot_mapping` 的块号是逻辑块号，
某个 rank 的 cache **行号 = 逻辑块号**，行内只有它自己那 `block_size=128` 个 token。

**证据 B：shape + 我自己的 canonical 定义。**
`init_key_cache_shape`（MLA）= `[n_blocks, block_size, 1, kv_lora_rank]` ⇒ shape[0] = 逻辑块数；
`init_index_cache_shape` = `[n_blocks * S, block_size, 1, head_dim]`（`Platform::supports_dsa_indexer_cache_sharding()`
在 NPU 返回 true，`platform.h:66`）⇒ **index cache 的 shape[0] 是 canonical 块数**。
而 `cache_directory.cpp:573` 我自己写的注释是 "One logical block spans one canonical block per DCP rank"，
`canonical = id * S + offset`；python 侧 `expand_indexer_block_table()` 对**每个逻辑块展开出全部 S 列**
（`logical * S + slice`，slice 遍历 0..S-1），也就是**每个 P rank 的 indexer block table 都覆盖全部 S 片**。

**裁定表**：

| slot | cache 行号语义 | decode(行) | prefill 哪个 rank / 哪一行 |
|---|---|---|---|
| `key`, `value` | **逻辑块号**，行内是本 rank 的 128 token 切片 | `c` | rank `(c%S)*tp + tp(D)`，行 `c // S` |
| `index`, `indexer_scale` | **canonical 块号**，且**每个 P rank 都有一份（复制的全序列）** | `c` | 任意 rank，行 `c` |

这正是用户要验的"**prefill/decode DCP 异构 + indexer full sequence**"：MLA 的 KV 是真分片，
indexer cache 是 **S 倍分配但全序列复制**，所以任何 rank 都能不靠集合通信跑完整序列的 indexer。
验证时**必须额外断言** index 行在 4 个 P rank 上互相一致——这条现在写进断言里了。

### (2) 追踪器重写：按 metadata 选行，不再盲扫前 64 行

原版 `_pd_trace.py` 只 hash `tensor[:64]`。KV pool 有上千块，请求实际用的块几乎不可能落在 0..63，
**盲扫必然采不到数据**。新版改为从 batch metadata 取该 step 真正用到的块：

* `metadata.block_table` = 逻辑块号 → 取 `{b}`（给 key/value）
* 再取 `{b*S + s | s < S}`（`S = metadata.kv_split_size`，给 index）
* 两边 hash 的都是这两个索引空间的**并集**，比对脚本按 slot 语义各取所需
* 没有可用 metadata 时才回退到前 `XLLM_PD_TRACE_BLOCKS`（默认 512）行
* 记录里带上 `kv_split` / `is_prefill` / `call`，便于判断 chunked prefill 的第几 chunk

hook 点：`ModelExecutor.execute`。C++ 是 `py_executor_.attr("execute")(tokens, positions, metadata,
embedding, sync)`（`py_executor_impl.cpp:353`），**实例属性查找** ⇒ 在类上打补丁生效。
hook **自包含**（函数内 `import os`，`try/except ImportError` 退化为 no-op），因为 `executor.py` 里
**没有** `import os`，不能在文件头加 import 之外的东西。

### (3) 比对器重写：先"从字节反推映射"再对照理论，且**每个 D rank 都必须过**

两个关键设计：

1. **不预设 P↔D 的块号约定**。先建 `(layer, slot, sha256) -> [(rank, block)]` 反查表，再对每个 decode 行
   报出"预测的 (rank,row)"与"实际命中的 (rank,row)"。若不一致，把发现的映射按计数打印——失败自带诊断。
2. **判定取最差 rank，不是最好 rank**。第一版按"最佳 dump"打分，结果 decode rank1 的错片被 rank0 的
   100% 掩盖，**坏用例误判为 PASS**。改成：同一 (tag, call) 下**所有 rank 都通过**才算通过。

### (4) 自测（用合成 trace 验比对器，避免浪费一次真跑）

`~/work/pdtrace/make_synthetic.py` 造 S=2/tp=2/4 个 P rank/2 个 D rank 的 dump，三个用例：

| 用例 | 结果 |
|---|---|
| 正确（key/value 分片 + index 复制） | **PASS**，36/36 |
| 故意把 D rank1 的 key 块 2 写成 dcp=0 的片 | **FAIL**，定位到 rank1/key/mismatch=2，映射打印 `(1,1)->(0,0)` |
| 把 D rank0 的一个 key 行改成查不到的值（KV 没到） | **FAIL**，报 `absent=1` |

三种路径都对。注意合成里 `ambig` 非 0 是**正常的**：MLA 只有 1 个 kv head，两个 TP rank 的
latent 完全相同 ⇒ 同一 digest 有 2 个候选 (rank,row)，`(rank,row)` 命中即可，不算失败。

### (5) 把插桩接进 83 的脚手架（已上传，未执行）

新增/改动（本机 `~/work/pdroute83/`，已同步到 83 的 `/export/home/shifengmin.3/workspace/pdroute83/`）：

* `env.sh`：加 `PD_TRACE`（默认 0）/`PD_TRACE_DIR`/`XLLM_PD_TRACE_BLOCKS`，`PD_TRACE=1` 时导出
  `XLLM_PD_TRACE_DIR` 并建目录。**`/export/home` 是 bind-mount 进容器的**，所以 trace 目录 host 和
  容器都能看见，不需要 docker cp。
* `trace_install.sh`：把 `_pd_trace.py` 装进 `$SP/xllm/python/`，把 hook 追加到
  `$SP/xllm/python/model_executor/executor.py`（**先备份 `executor.py.pre_pdtrace`**，marker 幂等），
  装完用 `ast.parse` 校验两个文件都能解析。
* `trace_remove.sh`：从备份还原 + 删 `_pd_trace.py` + 断言 marker 清干净。
* `run_trace.sh`：清 trace 目录 → 装 hook → `PD_TRACE=1 run_all.sh` → `smoke.sh` → 列出行数。
* `mk_stage83.sh`（本机）：把 `pdroute83/*.sh` + 两个 trace payload 用 **base64 内联**生成
  `stage83.sh`，一次 `rrun` 写进 83。**这样上传不会留下 `._*` 的 AppleDouble 垃圾**
  （之前 10 个脚本每个都配了一个 `._xxx.sh`）。

payload 已在容器里用 3.11 校验：两个文件 `ast.parse` 都 OK。

### (6) 构建与下一步

构建仍是 `[897/1380]`、undefined=0、无 error，等在跑的 `wait_build2.sh`。

构建成功后的顺序（**中途绝不动 HEAD**）：

1. `rrun jd-node-98 < deploy_stage98.sh` → `rrun jd-node-83 < deploy_install83.sh`
2. 容器内 `bash npu_init.sh 0,1,2,3,4,5` → `bash run_all.sh`（**先不开 trace**，先确认能起来）
3. `bash smoke.sh` 通了之后 → `bash run_trace.sh` → 把 trace 目录取回来 → `compare_kv.py --list` 再正式比对
4. 比对通过后：`trace_remove.sh` 还原，清 `PDROUTE_BUILD_ONLY_BYPASS` 等残留，再对齐 98 的树到 GitHub 版本。

---

## 第 21 轮：给比对器补"作用域"，消掉一类误报（2026-09-18 15:2x）

第 20 轮的比对器有一个**会误报 FAIL** 的漏洞：它把"decode 侧某行的字节在 prefill 里查不到"一律记成
`absent` 并判失败。但 decode 的 `block_table` 里可能有**padding lane**（ACL graph 补位、分配器复用、
或某 lane 指向根本没被 prefill 处理的块），这种"查不到"**不是传输 bug**。

### 修法：把"批次声明了哪些逻辑块"也打进 trace，用它判作用域

`_pd_trace.py` 每次 dump 多写一条 `slot="__meta__"` 的记录，带上该 step 的**逻辑块列表**
（`block_table` ∪ 由 `slot_mapping // (block_size*S)` 反推的块，两者取并集做冗余）。
比对器据此分两类计数：

* `absent_in`（**在作用域内**却查不到字节）→ 真丢数据 → **判 FAIL**
* `abs_oos`（**不在作用域内**）→ padding/复用 lane → **只报告，不判失败**

作用域判据：canonical `c` 对应 prefill 逻辑块 `c // S`，只要 `c // S` 在**任一** P source 的
`__meta__.blocks` 里就算在作用域内；同时要求 `c` 落在 D 自己的 canonical 集合里。
两类 slot 用的是同一个条件（因为都回落到 `c // S`）。

### 自测扩到 4 个用例（`make_synthetic.py` 现在一次生成 4 个目录）

| 用例 | 构造 | 期望 | 实测 |
|---|---|---|---|
| `ok` | 正确的分片 + 复制 | PASS | PASS（36/36） |
| `padded` | 在 `ok` 基础上让 decode 多声明一个 P 从没碰过的块 `99` | **仍 PASS** | **PASS**，`abs_oos=2` |
| `bad` | D rank1 的 key 块 2 写成 dcp=0 的片 | FAIL | FAIL，`mismatch=2`，映射打印 `(1,1)->(0,0)` |
| `absent` | 在 `ok` 上把 D rank0 一个**真在作用域内**的 key 行改成查不到 | FAIL | FAIL，`absent_in=1` |

`padded` 是关键回归：改动前它会误判 FAIL。现在 4/4 都符合预期。

### 其它

* 两个 payload 已重新上传到 83，容器内 Python 3.11 `ast.parse` 通过
  （`_pd_trace.py` 8694B md5前缀 `c90ed6cc73f4`，`executor_hook_tail.py` 1578B `f4a8d3d1c58d`）。
* 构建 `[988/1380]`、undefined=0、无 error。

---

## 第 22 轮：跑之前的三个致命细节（prompt 长度 / 路由是否真被采用 / 日志签名）（2026-09-18 15:2x）

构建还在跑（实测 **25.75 边/分钟**，剩 253 边，ETA ~10min）。本轮全是"跑之前必须想清楚"的事。

### (1) 致命细节一：短 prompt 让整个验证**等于没做**

原计划用 `smoke.sh` 的 `你好，请用一句话介绍你自己。`（~15 token）。但一个物理块 = 128 token、
一个**逻辑块 = 128 * S = 256 token**，所以 15 token 的请求**只落在一个 canonical 块上**，
而且只落在 `dcp_rank = 0` 那**一个**分片上 —— 整个"异构 S_P=2 vs S_D=1 重分片"的验证就退化成
"一个 slot 的一行字节相等"，`c // S` 这个映射根本没被走到。

所以新增 `smoke_long.sh`：把一句话重复到 **~1200+ token**（≈5 个逻辑块），这样

* canonical 块覆盖 `0..9`，**两个分片都被走到**（`c%2 = 0` 和 `1`）；
* 逻辑块跨多个，`c // S` 的跨块映射（row 0,1,2,…）被真正检验；
* chunked prefill 若把 prefill 切成多 chunk，比对器取**每个 rank 的最高 call**（= 最终状态），
  正好覆盖全部分片。

并且 `run_trace.sh` **只发这一个请求**：trace 记的是每个 rank **最后一次** dump 的 block table，
发第二个请求会把比对需要的 block table 覆盖掉。

### (2) 致命细节二：`canonical` 可能**静默退化**成 legacy，实例照样能起来

读了 `mooncake_kv_cache_transfer.cpp` 的注册路径，发现 canonical 的可用性是**注册期**决定的
（`canonical_ready_`），而且失败只打 **`LOG(WARNING)`**，不 fatal：

| 位置 | 触发条件 | 日志 |
|---|---|---|
| `:474` | 任何 tensor 带 `explicit_resource_offsets` | `The canonical route does not build GlobalXTensor page bases yet; use pd_route=legacy.` |
| `:498` | `declare_cache_group` 拒绝某个 role | `The canonical route cannot serve this instance: ...` |
| `:515` | `PeerDirectory::describe` 看不懂自己的布局 | `The canonical route cannot interpret this rank's own published layout: ...` |

退化之后实例**照常启动、照常接请求**，直到 push 时才报错：

* `:805` `LOG(ERROR) pd_route=canonical cannot serve this instance: this rank's published cache layout was not declared or could not be interpreted.` → push 返回 false ⇒ **KV 一个字节都没传**。

派发点在 `kv_cache_transfer.cpp:233`：`if (canonical_route_) { push_kv_blocks_canonical(...); }`，
`canonical_route_` 只反映**命令行**（`:325` `set_canonical_route(route_mode == CANONICAL)`），
和 `canonical_ready_`（布局）是**两件事** —— 所以"flag 写了 canonical"绝不等于"走的是 canonical"。

**因此新增 `diagnose.sh`**：按"日志出现顺序"逐条 grep 上面所有签名 + `Create Mooncake KVCacheTransfer, pd_route=...`
（`:303`，确认 flag 解析成什么）+ push 期错误 + `Brpc Server started` + FATAL/Check failed/Traceback，
最后给一句"结论提示"。这样跑完第一件事就是看它，而不是瞎猜。

### (3) 致命细节三：`plane=canonical ... success=` 是 VLOG(1)，默认看不见

canonical push 的成功日志是 `VLOG(1) << "[Mooncake][PDTransfer] direction=push, plane=canonical, requests=..., success=..."`
（legacy 那条是 `direction=push, destinations=...`）。默认 VLOG 关着，所以"没看到 canonical push 日志"
**不能**推出"没走 canonical"。`env.sh` 加了 `PD_VLOG`（默认 0，设了才导出 `GLOG_v`），
需要时用它把这条证据打开；不需要时靠字节比对反证（字节对上 ⇒ 路由确实跑了）。

### (4) 其它确认

* 构建产物路径确认：wheel 在 `$TREE/dist/*.whl`（`bdist_wheel` 才创建，现在还没有）；
  ELF 由 `setup.py build` 最后产出；`libasio.so` 已在
  `$TREE/build/cmake.linux-aarch64-cpython-311/mooncake-common/libasio.so`（1.8MB）。
* 内层脚本尾部顺序确认：`BUILD_EXIT=...` → `setup.py bdist_wheel` → `WHEEL_EXIT=...` → `ls dist/*.whl`。
* 注意 `XLLM_OPS_GIT_HEAD_CACHED` 是在**构建启动时**用 `git -C third_party/xllm_ops rev-parse HEAD`
  钉住的（该目录不是真子模块 ⇒ 冒泡成 xllm 的 HEAD）。它在 cmake configure 时与当前 HEAD 比对，
  所以**只要 HEAD 在 configure 之前不动**就不会触发 ops 预编译 —— 结论不变：构建期间别动 HEAD。
* `rrun ... docker exec -i C bash -s < npu_init.sh 0,1,2,3,4,5` 这种写法**传不了参数**
  （stdin 已经是脚本本身），用脚本里的默认 `DEVICES=0,1,2,3,4,5`。

---

## 第 23 轮：**链接成功**，并在二进制里证实路由代码真的进了（2026-09-18 15:25）

### (1) 结果

* `BUILD_EXIT=0 2026-09-18T15:25:18+08:00` —— **第 18 轮那个链接缺陷确实是唯一的拦路虎**：
  给 `kv_cache_transfer` 目标补上 `:pd_route_transfer` 之后，7 个 `undefined reference` 全部消失，
  一次通过。之前 `[0/2] → [1380]` 的增量段实测 **25.75 边/分钟**（240s 走 66 边）。
* 产物 ELF：`$TREE/build/lib.linux-aarch64-cpython-311/xllm/xllm`，**563,018,848 字节**（15:23:29）。
* `bdist_wheel` 接着在跑（正在编 TileLang kernels），wheel 落在 `$TREE/dist/*.whl`。

### (2) 在**二进制里**证实路由代码被链接进去了（不只是"编译过了"）

这是比"退出码 0"强得多的证据：以前 4 个路由 `.cpp` **根本没被编译**，所以下面这些字符串和符号
**不可能**出现在产物里。现在全都在：

日志字符串（`strings -a`，命中数）：

| 字符串 | 命中 |
|---|---|
| `pd_route=canonical cannot serve this instance` | 1 |
| `The canonical route cannot serve this instance` | 1 |
| `The canonical route failed to write to` | 1 |
| `direction=push, plane=canonical` | 1 |
| `Create Mooncake KVCacheTransfer, pd_route=` | 1 |
| `The canonical route cannot interpret this rank` | 1 |
| `canonical route does not build GlobalXTensor` | 1 |

符号（`nm -C`，命中数）：

| 符号 | 命中 |
|---|---|
| `canonical_blocks_of_request` | 1 |
| `build_route_peer` | 1 |
| `flatten_route_for_layers` | 8 |
| `pd_route_mode_name` | 1 |
| `PdRouteTransfer::plan` | 1 |
| `declare_cache_group` | 1 |
| `PeerDirectory::describe` | 1 |

`flatten_route_for_layers` 有 8 处是因为它被内联进了各个调用点（canonical/主 cache/spec draft 等）。
验收脚本：本机 `~/work/verify_binary.sh`（rrun 到 98 上跑）。

### (3) 顺带确认的部署前提（在 98 host 上实测）

`/export/home/shifengmin.3/.ssh/id_rsa` 在位（2643B）；`md5sum/scp/ssh/base64` 全有；
`/tmp` 可写；**98 → 83 直连可达**（`REACHED A03-R40-I191-83-4100038.JD.LOCAL`）。
另外给 `env.sh` 补了 `LD_LIBRARY_PATH=$LIBDIR`（wheel 不一定带 `libasio.so`，
部署时是单独 stage 到 `pdroute83/lib/` 的，之前 `start_workers.sh` 没把它加进搜索路径）。

### (4) 下一步

等 `WHEEL_EXIT=0` → `deploy_stage98.sh` → `deploy_install83.sh` →（容器内）`npu_init.sh`
→ `run_all.sh` → `smoke.sh` → 通了再 `run_trace.sh` + `smoke_long.sh` → `compare_kv.py`。
构建已完成，所以**现在可以安全地**用 `~/work/align_remote_tree.sh`（不带参数先只报告）对齐 98 的树。

---

## 第 24 轮：wheel 落地与两个"看起来像网络问题、其实是别的"的坑（2026-09-18 15:36）

### (1) wheel 成功

`dist/xllm_npu_torch2_9_0-0.11.0-cp311-cp311-linux_aarch64.whl`，**576,356,500 字节**，
md5 `ec794733df99a2e01dc1cd59521143a9`。传到 83 后**自己校验**：

* 83 上 md5 一致；
* `zipfile.ZipFile(...).testzip()` 返回 `None`（无坏条目）；
* 257 个条目，`xllm/xllm` 在、`xllm/python/model_executor/executor.py` 在（trace 的 hook 目标）、
  并且 wheel **自带一份 `libasio.so`**。

`libasio.so` md5 `d694863db9b16f3a0c7273b521e525cf`（1,832,160 字节）。

### (2) 坑一：长传输会把脚本"截断"，但 rrun 仍然报 exit 0

第一次跑 `deploy_stage98.sh`：输出在 **wheel 的 `md5 OK` 之后就直接结束**，
`libasio.so` 根本没传（83 上 `ls: cannot access '/tmp/libasio.so'`），
**但 rrun 报的是 `exit code: 0`**。

结论：**绝不相信长远程命令自己的退出码**。脚本没有 bug，是 576MB 传输把会话耗到了尽头，
后面的语句没执行。已改成：

* `deploy_stage98.sh` 接受 `wheel|libasio|all`，**实践中一次只传一个文件**；
* 新增 `send_libasio.sh`（单文件 + 3 次重试 + md5）；
* 新增 `verify_stage83.sh`（在 83 上独立核对两个文件的 md5 + wheel zip 完整性）。
* `libasio.so` 已用单文件方式补传并 VERIFIED。

### (3) 坑二：**改 wheel 文件名会让 pip 直接拒绝**（之前的归因是错的）

`pip install /tmp/xllm_pd.whl` 报：

```
ERROR: Invalid wheel filename (wrong number of parts): 'xllm_pd'
```

PEP 427 要求文件名是 `{name}-{version}-{python}-{abi}-{platform}.whl`，把 wheel 改名成
`xllm_pd.whl` 就**破坏了这个结构**，pip 连解包都不会尝试。

**这推翻了工作日志早先的一条归因**：以前记的是"scp 中断会得到 `Wheel is invalid`"，
其实**只要改名就必然报错**，跟传输是否完整无关（这次 md5 和 zip 校验都是好的，照样报错）。

修法：**永远保持 wheel 的规范文件名**。新增 `install83.sh`：
先在 83 上把 `/tmp/xllm_pd.whl` **改回**规范名（没有重新传 576MB），
再 `docker cp` 进容器（容器内也用规范名），再 `--force-reinstall --no-deps` 安装；
装完逐项核对 `$SP/xllm/xllm`、`glm5_2.py`、`executor.py`、libasio，
并直接在**装好的 ELF** 里 grep 那几条 canonical 路由字符串，最后确认 `--help` 里有 `--pd_route`。

### (4) 附带发现：`bdist_wheel` 会**全量重编 brpc**

`setup.py bdist_wheel` 重新 configure 了一次 cmake，brpc 的 325 个目标全部重编
（实测约 13 分钟）。原因在编译命令行里能直接看到：

```
-DBRPC_REVISION="\|pd-routing-s0s1\|ad98850a8\|2026-09-18T15:01:19+08:00"
```

那个时间戳是 **cmake configure 的时刻**，不是 git 提交时间 —— 所以**每次重新 configure，
BRPC_REVISION 都变，brpc 就整包重编**。

推论：`align_remote_tree.sh` 必须在**打包完全结束之后**再跑，
否则对齐动作会连带触发一次 ~13 分钟的 brpc 重编（外加 ops 门的风险）。
本次因为一直等到 `WHEEL_EXIT=0` 才动，所以没有付出这个代价。

---

## 第 25 轮：**在真机上跑出第一个真 bug**（canonical 静默退化），以及四个环境坑（2026-09-18 16:0x）

### (1) 头条：canonical 路由在真机上**静默退化成 legacy**

第一次把 P/D 拉起来（用了旧 wheel），6 个 rank 全部打出：

```
W mooncake_kv_cache_transfer.cpp:513] The canonical route cannot interpret this
  rank's own published layout: cache tensor declarations are duplicated for
  role 0 group 0; use pd_route=legacy.
```

**这正是第 22 轮预判的那条静默退化路径**（`LOG(WARNING)`，不 fatal，实例照常启动），
`diagnose.sh` 的签名也命中了。根因在**我自己的注册代码**：

`mooncake_kv_cache_transfer.cpp` 的 `register_kv_cache_impl` 里，遍历的是
**每个 layer 的每个 tensor**（本模型 11 个 buffer），对每个都 emplace 一个
`CacheTensorDeclaration`。但 `PeerDirectory::describe`（`cache_directory.cpp:612`）要求
**每个 `(cache_namespace, role, group_id)` 只能有一个 declaration**，并且
`declaration_matches`（`:56`）**故意忽略 layer_id** —— 也就是"一个 declaration 覆盖该族的全部 layer"。
所以 4 层 ⇒ role 0 被声明 4 次 ⇒ 直接判重复。

**修法**：声明前先按 `(namespace, role, group_id)` 去重（`row_bases_` 才是按 layer 的那一份，
本路径留空，`describe_tensor` 在 `row_bases == nullptr` 时有专门分支，安全）。
改的是**远端树的工作区**（不 commit，这样 HEAD 不动、ops 门保持满足），
增量构建只走了 **3 个 ninja 边**（`BUILD_EXIT=0 16:01:22`），**brpc 没有重编**。

### (2) 意外收获：真机日志**独立验证**了第 20 轮的行索引裁定

prefill（S=2）的 kv cache 初始化日志：

```
kv cache capacity: 44.38 GB, blocks: 60593, slot_size: 1152, index_slot_size: 512, indexer_layers: 3
Initializing k cache with shape:      [60593  128 1 512]
Initializing v cache with shape:      [60593  128 1  64]
Initializing indexer cache with shape:[121186 128 1 128]
```

**`121186 = 2 × 60593`** —— index cache 的行数是 **canonical 块数**，而 k/v 的行数是
**逻辑块数**（`60593`）。这和第 20 轮纯从契约推出来的裁定**完全一致**：

| slot | 行号语义 | 行数 |
|---|---|---|
| `key` / `value` | 逻辑块 | 60593 |
| `index` | **canonical 块** | **121186 = 2 × 60593** |

`indexer_layers: 3` 也印证了 `indexer_types=['full','full','full','shared']` 里
`shared` 层**不分配** indexer cache（11 buffer = 4×k + 4×v + 3×index）。
decode 侧（S=1）是 `blocks: 69253, index_slot_size: 256`，进一步说明两边空间不同、必须以 canonical 为桥梁。

### (3) 坑一：4 个同机 rank 共用 `HCCL_IF_BASE_PORT` ⇒ EI0019

```
RuntimeError: createHCCLCommOrigin ... hcclGetRootInfo(&hcclID), error code is 7
Communication_Error_Bind_IP_Port(EI0019): The IP address 11.87.191.83 and port 48439
  have already been bound.
```

我原来给 P 的 4 个 rank 都设了 `HCCL_IF_BASE_PORT=48439`（两个角色分开、但**同角色内没分**）。
单机多 rank 必须**每 rank 一个 base**。改成 `HCCL_IF_BASE_PORT=$((BASE + rank * 200))`，
并且把 base 挪到 **ephemeral 范围(32768-60999)之外**（先扫出容器内 61000-64263 空闲）：
P = 62000/62200/62400/62600，D = 63000/63200。改完 **EI0019 归零**。
另加了启动前的端口自检（`start_workers.sh`），占用就直接报错退出，不再等到几分钟后炸在集合通信里。

### (4) 坑二：我的"就绪"判据是错的

`run_all.sh`/`wait_ready.sh` 原来要求**每个** rank 的日志里都有 `Brpc Server started`。
但只有每个角色的 **master（node_rank 0）** 才起 HTTP/brpc 服务，worker rank 永远不会有这行
⇒ 判据永远不可能满足。已改成只看 `rank_0.log`，并且 `http_58888` 端口作为附加条件。

### (5) 坑三：同机多进程并排进 CANN 算子编译 ⇒ 知识库锁死锁

6 个 rank 全部拉起来后，卡在：decode rank_0 起来了（brpc OK），其余 5 个 **CPU 0%、futex 等待**。
gdb 抓到 prefill rank 0 的栈：

```
#3  acquire_timed (lock=..., timeout=1000000000)      ← timeout=1000 秒 的 Lock.acquire
#51 CannKb::PyInterface::CannKbInit(...)              libcann_kb.so
#53 PythonAdapterManager::InitCannKB()               libop_compile_adapter.so
#55 TbeInitialize()                                   libop_compile_adapter.so
#57 fe::TbeOpStoreAdapter::InitializeInner(...)       libfe.so
#59 fe::OpStoreAdapterManager::InitializeAdapter(...)
```

即 **CANN 的 TBE 算子编译"知识库"(cann_kb) 初始化**里拿一把 1000 秒超时的锁，6 个同机进程
同时进入 ⇒ 互相等。注意第一次运行**曾经**越过这一点（那次是死在后面的 HCCL 端口），
所以这是**竞态**而不是硬阻塞。

对策：`start_workers.sh` 里加 `START_GAP`（默认 15s）**把各 rank 的启动错开**
（`[ "$rank" -lt $((N-1)) ] && sleep "$START_GAP"`）。另写了 `clean_restart.sh`：
杀干净所有 worker + 控制面、**清掉 `/tmp/etcd_pdroute83` 的 etcd 状态**再重启
（残留的实例注册会让下一次启动表现得像代码 bug）。

### (6) 坑四：三个"自己坑自己"的脚本错误（都已修）

| 现象 | 真因 |
|---|---|
| `launch_fix_build.sh` 执行到一半整段消失 | `pkill -f "fix_build_inner.sh"` **匹配到了自己**（脚本正文里就有这个字符串）⇒ 自杀。同理 `pgrep -f "setup.py build"` 也会自匹配（`bash -c` 的正文就是命令行）。改用 `ps -eo comm,args \| awk '$2 ~ /^python/'` 按**可执行名**过滤 |
| `bash: line 8: $2: unbound variable` | 在外层 `bash -c '...'` 里嵌了带**单引号**的 awk 程序 ⇒ 单引号提前闭合，`$2` 被外层 shell 展开。**改成把内层脚本写成文件用 `bash -s` 送进去**（正是本仓库规范说的那条） |
| `check_fix_build2.sh` 完全没有输出 | `sudo docker exec "$C" bash -s <<'INNER'` **少了 `-i`** ⇒ docker 不转发 stdin ⇒ 内层 bash 拿到空脚本。加 `-i` |
| `why_stuck.sh`/`in_ctr.sh` 报 `cannot find env.sh (HERE=/export/home/...)` | 入口脚本经 `bash -s` 送进去时 `$0` 是 `bash`，`dirname` 得到的是当前目录。**这个报错是我的守卫脚本主动报的**（第 25 轮新增），比原来的 `unbound variable` 清楚得多；解法是 `docker exec -i -w <脚本目录>`，子脚本再用 `bash "$HERE/xxx.sh"` 调用（子脚本的 `$0` 就正确了） |

### (7) 状态与下一步

* `BUILD_EXIT=0 16:01:22`；`bdist_wheel` 正在写（wheel 从 15MB 长到 197MB，目标 ~576MB）。
* 之后：`deploy_stage98.sh wheel`（**一次只传一个文件**）→ `install83.sh`（保持规范 wheel 文件名）
  → `clean_restart.sh`（全清 + 错开启动）→ 通了再 `run_trace.sh` + `smoke_long.sh` → `compare_kv.py`。

---

## 第 26 轮：构建耗时拆解（+`strip` 提速结论）与当前唯一拦路虎（2026-09-18 16:17）

### (1) 回答一个关键问题：构建**并没有**每次全编

`SKIP_TEST=1` 从第 19 轮起就一直在用，`setup.py:703` 就是那个 UT 开关：

```python
if "SKIP_TEST" in os.environ:
    logger.info("⏭️ skipping UT because SKIP_TEST is set")
```

最近这一轮（改 `mooncake_kv_cache_transfer.cpp` 一个文件）的真实耗时：

| 阶段 | 时刻 | 耗时 |
|---|---|---|
| 内层脚本启动 | 15:58:46 | — |
| `BUILD_EXIT=0` | 16:01:22 | **2m36s**（重编 1 个 TU + 重链） |
| `WHEEL_EXIT=0` | 16:03:58 | **2m36s**（打包 wheel） |

ninja 全程只报 `[3/3]`。ccache（`CCACHE_DIR=/export/home/shifengmin.3/workspace/.ccache`）
和 `MAX_JOBS=12` 都已在用。

### (2) 真正吃掉 ~13 分钟的是 **cmake 重新 configure**（⇒ brpc 325 个目标全重编）

brpc 把 revision 烤进了编译宏：

```
-DBRPC_REVISION="\|pd-routing-s0s1\|ad98850a8\|2026-09-18T15:01:19+08:00"
```

只要 cmake 重跑且这个值变了，brpc 就整包重编。**所以规矩是：构建前后不动 HEAD**
—— 这也正是 dedup 修复**故意留在远端工作区不 commit** 的原因（HEAD 不动 ⇒ cmake 不重跑
⇒ brpc 不重编，同时 `xllm_ops` 预编译门也保持满足）。

### (3) 结论：剩下的时间不在编译上，而在"大 ELF"上 —— 下次用 `strip`

时间实际花在：① 链接 563MB 的 ELF；② 把 563MB zip 成 576MB 的 wheel（~2.5min）；
③ 576MB 的 scp 98→83（~2-3min）+ pip 安装。三者都被 ELF 里的 **debug info** 主导。

**下次迭代：打包前先 `strip --strip-debug`**（在容器里做，**不用改 CMake、不用重编**），
预期 ELF 从 563MB 降到一两百 MB，链接/打包/传输一起快约 3 倍。
唯一代价是 xllm 自身的 gdb 符号；但本轮所有有用的栈帧都来自 **CANN 库和 CPython**
（`libcann_kb.so` / `libfe.so` / `Python/thread_pthread.h`），所以现在就可以 strip。

### (4) **好消息：dedup 修复被运行期签名确认**

装上含修复的新 wheel（md5 `f87c4008abc0d130098165414fdddd9f`）后：

| 签名 | 修复前 | 修复后 |
|---|---|---|
| `canonical route cannot interpret this rank` | **6** | **0** |
| `Create Mooncake KVCacheTransfer, pd_route=canonical` | 6 | 6 |

即 canonical 路由**不再静默退化成 legacy**。这是本轮验证最核心的产出：
**我在真机上跑出了一个自己的真 bug 并修掉了它。**

### (5) 当前唯一的拦路虎：CANN 算子编译知识库的锁（环境问题，不是路由问题）

16:17:35 的快照：

| 角色 | brpc 就绪 | 日志最后一行 |
|---|---|---|
| DECODE rank 0 | **1**（API 已起） | `Application startup complete.` @16:06:18 |
| DECODE rank 1 | 0 | `get_cache_info success` @16:06:17 |
| PREFILL rank 0..3 | 0/4 | `Successfully connected to xservice` / `register_kv_cache_impl success` @16:07:1x |

* 错误签名：`FATAL` 0、`terminate called` 0、`EI0019` 0（**HCCL 端口修复有效**）。
* etcd 里**仍只有 DECODE 实例**（`XLLM:DECODE:11.87.191.83:29994`），PREFILL 从未注册。
* prefill rank 0 的栈仍在：`acquire_timed(lock, timeout=1000000000)`
  ← `lock_PyThread_acquire_lock` ← … ← `CannKb::PyInterface::CannKbInit`（`libcann_kb.so`）
  ← `PythonAdapterManager::InitCannKB` ← `TbeInitialize` ← `TbeOpStoreAdapter::InitializeInner`
  ← `fe::OpStoreAdapterManager::InitializeAdapter`。

**关键观察：只有"整体第一个启动"的进程（decode rank 0）走过去了，之后启动的全部卡住。**
这与"那把锁被第一个进程长期持有"一致（错开 15s 不够）。

**可观测的预测**：`timeout=1000s`，锁大约在 **16:07:11 + 1000s ≈ 16:23:51** 到期。
到点后要么报错继续、要么真的楔死 —— 这本身就是一条判据（去看 `status.sh` 与 rank 0 日志的 mtime 有没有跳）。

**下一步对这条的三个候选对策**（按代价从低到高）：

1. **观察 16:23:51 那个超时点**（零成本，先做）——看它是超时后自己过去，还是永远楔住。
2. **给每个 rank 独立的编译缓存/KB 目录**（`ASCEND_CACHE_PATH` / `ASCEND_WORK_PATH` 按 rank 分），
   若那把锁是共享文件锁就会被彻底绕开；代价是每个 rank 可能要各自重编算子（慢但无竞争）。
3. **串行预热**：先只起 1 个 rank 让它把算子编译/KB 建好，再起其余 5 个。

（备选终极方案：把 P 和 D 分到两台机器 —— 那是工作日志里**已验证过的两机配方**；
98 目前满卡，需要另找机器。）

### (6) 收尾清单（沿用第 25 轮，补一条）

`trace_remove.sh` 还原插桩 → 把 **dedup 修复提交到 GitHub**（在 local clone 里做同样编辑，
commit + push；目前**只在远端工作区**）→ 清 `PDROUTE_BUILD_ONLY_BYPASS` 等残留
→ 用 `align_remote_tree.sh` 对齐 98 的树（并立刻重钉 ops marker）。

---

## 第 27 轮（16:20–16:55）：所谓「CANN 算子编译知识库锁」是误诊，真因是自定义算子包 ABI 不匹配

### (1) 先推翻上一轮的判断：纳秒，不是微秒

`_threadmodule.c` 的 `acquire_timed(PyThread_type_lock, PyTime_t timeout)` 里 `PyTime_t` 是
**纳秒**。用已知 timeout 的锁做标定（attach 正在阻塞的进程看 frame #3）：

| Python 代码 | gdb 打印的 `timeout` |
|---|---|
| `lock.acquire()` | `-1000000000`（即"无限"哨兵） |
| `lock.acquire(timeout=7)` | `7000000000` |
| `lock.acquire(True, 7)` | `7000000000` |

所以上轮看到的 `+1000000000` = **1 秒**，`-1000000000` = **无限等待**。
上一轮"1000 秒锁 + 16:23:51 到期"的整条推论是把纳秒当成了微秒，**是错的**。

### (2) 更关键的推翻：我一直在调的不是引擎

那 4 个 14 线程进程**不是 prefill 引擎**，是引擎 fork 出来的
`multiprocessing.Manager` 服务进程（父进程已死，被 reparent 到 init，所以 ppid=1）。
在容器里 pip 装 `py-spy 0.4.2` 后拿到 Python 栈才看清：

```
Thread (MainThread): wait (threading.py:331) <- wait (threading.py:629)
  <- serve_forever (multiprocessing/managers.py:176)
  <- _run_server (multiprocessing/managers.py:600)
  <- Manager (multiprocessing/context.py:57)
  <- __init__ (tbe/common/repository_manager/utils/multiprocess_util.py:48)
  <- initialize (tbe/common/repository_manager/route.py:141)
  <- cann_kb_init (tbe/common/repository_manager/interface.py:36)
  <- __call__ (torch/_ops.py:1255)
  <- prepare_quant_weight (xllm/python/kernels_npu/linear.py:62)
  <- process_weights_after_loading (glm5_2.py:441) <- load_weights (glm5_2.py:906)
```

`serve_forever` 里的 `self.stop_event.wait(1)` 正是那个 `+1e9`；其余 9 个线程是
`Condition.wait()`（无限、各自一把新 waiter 锁）= 空闲线程池，完全健康。
gdb 里那串 `CannKbInit` C 栈是 **fork 继承下来的父进程栈**，不是阻塞点。
CANN KB 初始化只是 `torch_npu.npu_format_cast` 的副作用（首次 NPU 格式转换 → TBE/op-store
→ `cann_kb_init` → `multiprocessing.Manager`），**与路由毫无关系**。

### (3) 真因：4 个 prefill 引擎全部 SIGSEGV

僵尸进程的退出码（`/proc/<pid>/stat` 第 52 字段）直接给出死因：

| pid | 启动 | 死因 |
|---|---|---|
| 28528 | 16:06:02 | SIG11 |
| 30838 | 16:06:17 | SIG11 |
| 30927 | 16:06:32 | SIG11 |
| 31017 | 16:06:47 | SIG11 |

间隔正好 15s = 启动错开，即**4 个 prefill 引擎全部段错误**，日志最后一行停在
`xservice_client.cpp:691 Successfully connected to xservice`，没有 FATAL、没有 traceback。

用 gcc 现场编了一个 `LD_PRELOAD` 段错误处理器（`crashwatch.so`，信号栈 + `backtrace_symbols_fd`）
抓到 4 个 rank 完全一致的栈：

```
signal=11 si_addr=0x2
libcust_opapi.so(aclnnSparseFlashAttentionGetWorkspaceSize+0x108)
  <- libtorch_npu.so <- libtorch_python.so <- python 模型前向
  <- PyExecutorImpl::run <- Executor::forward <- LLMWorkerImpl::step_internal
  <- LLMWorkerImpl::step <- WorkerImpl::step_async <- ThreadPool::internal_loop
```

反汇编 +0x108：`str x0, [x24]`，而 x24 来自调用者栈（`ldr x24, [sp, #328]`），
`si_addr=0x2` ⇒ **x24 = 2**，即整数 `attention_mode=2` 落在 `aclOpExecutor**` 出参槽里。
前一条 `cbz x24` 只挡 0，挡不住 2。

为什么只有 prefill 走到这里：`profile_manager.cpp:154-172`，decode 打
`Skipping eager warmup for decode-only instance`，prefill 走 `warmup_for_eager()` →
`run_request(256, 0, 1, ...)` —— 而 `warmup_for_eager()` **在崩溃前一行日志都不打**，
所以日志在 `connected to xservice` 之后戛然而止。

### (4) 根因：镜像自带的自定义算子包与调用方 ABI 不一致

| | 参数个数 | 形态 |
|---|---|---|
| 0801 镜像 `custom_xllm_math`（8-21）的 aclnn 头 | **17** | `scaleValue` 在第 10 位，只有一个 `out` |
| 调用方（torch_npu codegen + xllm python） | 18+ | `scaleValue` 在**第 5 位**，另有 `pre_tokens/next_tokens/attention_mode/return_softmax_lse` |
| 0911 镜像 `glm_next_transformer` 的头 | **23** | 与调用方一致（多 `attentionOut/softmaxMax/softmaxSum`） |

参数整体错位 → 整数落进指针槽 → `str x0,[x24]` 崩在地址 2。
**这是自定义算子包过期，不是路由改动，也不是锁。**

### (5) 换镜像（按你的指示用尾号 0911）

* `quay.io/jd_xllm/xllm-ai:xllm-dev-a3-arm-cann9-20260911`（9-11，17.9GB）**只在 98 上有**；
  已在 83 上 `docker pull` 成功。
* 新容器 `fengmin-pdroute91`，配置照抄旧容器（privileged / `ipc=shareable` / host net /
  shm 64m / 同样 9 个挂载）。旧容器 wheel 已 `docker cp` 救出并装进新容器：
  `xllm_npu_torch2_9_0-0.11.0-cp311-cp311-linux_aarch64.whl`，576,354,918 B，
  md5 `f87c4008abc0d130098165414fdddd9f`。
* **新的环境坑（已修）**：`libcust_opapi.so` 是 xllm 自身的 `DT_NEEDED`；0911 镜像里
  `custom_xllm_math` 整个不存在，三个 vendor 各自只有 `libcust_opapi.so` 且都不在默认
  搜索路径上 → 一启动就是 `error while loading shared libraries: libcust_opapi.so`。
  改法：`env.sh` 读 `opp/vendors/config.ini` 的 `load_priority`，把每个存在 vendor 的
  `op_api/lib` 依次前置到 `LD_LIBRARY_PATH`（0911 下正确解析到 `glm_next_transformer`）。
* 旧容器已 `docker kill`（释放 host 端口）；`in_ctr.sh` 默认容器名改为
  `fengmin-pdroute91`；py-spy 已在新容器重装。

### (6) 换镜像后的结果与**新的拦路虎**

SFA 崩溃**消失**（ABI 对上了），prefill 走到更远处后在 16:47:00 **LOG(FATAL)**：

```
scatter_nd_update.cpp:27] Check failed: get_workspace_size_func_addr != nullptr &&
op_api_func_addr != nullptr aclnnScatterNdUpdateV2 or
aclnnScatterNdUpdateV2GetWorkspaceSize not in libopapi.so, or libopapi.so not found.
```

（同时 decode rank 0 已 `BRPC READY`，rank 1 停在 `get_cache_info success`。）

查证（`xllm/core/kernels/npu/aclnn/pytorch_npu_helper.hpp:230-300, 713-735`）：
查找顺序是 `g_custom_lib_path`（来自 `ASCEND_CUSTOM_OPP_PATH`）→
`g_default_custom_lib_path`（来自 `config.ini` 的 `load_priority`）→ 兜底
`dlopen("libopapi.so")`。而：

* 两个镜像的 `libopapi.so` 都**只有** `aclnnScatterNdUpdate` / `…GetWorkspaceSize`，**没有 V2**；
* 0911 镜像**没有 `custom_xllm_math`**，其 vendor 包不导出 V2；
* 旧容器（0801）的 `custom_xllm_math` **导出 V2**（`nm` 命中 1 处），
  但它的 SFA 是旧的 17 参数 —— **0801 的算子包一半新一半旧**。

结论：`aclnnScatterNdUpdateV2` 和 23 参数 SFA 都必须来自**用当前源码编译出来的自定义算子包**
（`third_party/xllm_ops` → `custom_xllm_math`）。这就是"用 0911 镜像做编译"的落点：
**算子包要按镜像的 CANN 版本重编并安装，不能依赖镜像自带那份。**

### (7) 恢复点（下一步，按顺序）

1. 在 0911 镜像里编译自定义算子包（`third_party/xllm_ops` → `custom_xllm_math`），
   安装到运行时容器的 CANN vendor 目录，并让 `ASCEND_CUSTOM_OPP_PATH` 指向它。
2. 校验该 `libcust_opapi.so` 同时导出 `aclnnScatterNdUpdateV2` **和 23 参数**的
   `aclnnSparseFlashAttentionGetWorkspaceSize`。
3. `env.sh` 需把 `$ASCEND_CUSTOM_OPP_PATH` 下各目录的 `op_api/lib` 排到最前
   （本轮只处理了 config.ini 的 vendor 列表）。
4. `clean_restart.sh` 重新拉起，看 prefill 是否越过 eager warmup；`crashwatch.so`
   （共享目录 `lib/`）与 py-spy 都还在新容器里可用。

本轮最值得记住的一条：**`libcust_opapi.so` 解析到哪一份直接决定算子 ABI** ——
同名库有 3 个 vendor 提供，谁先被 `dlopen` 到就用谁的 ABI，错了就静默地崩在算子内部。

---

## 第 28 轮（17:00–17:35）：算子包缺失、tracer 的 BF16 崩溃、端口抢占，以及**首个真正的路由 bug**

这一轮把「prefill 起不来」这条环境延长线走到了尽头，并且第一次让 canonical 路由**真正开始传字节**（然后立刻暴露出一个真 bug）。

### (1) 0911 镜像根本没有 xllm 自定义算子包

`nm -D` 清点 0911 镜像三个 vendor 的 `libcust_opapi.so`，再加 `libopapi.so`（CANN 内置），
和 wheel 里引用到的 aclnn 名字做差集：

| 来源 | 导出 aclnn 算子数 |
|---|---|
| `glm_next_transformer` | 2 |
| `custom_transformer` | 12 |
| `kpool_transformer` | 2 |
| `libopapi.so`（CANN 内置） | 1130 |
| wheel 引用（去掉 `GetWorkspaceSize`，干净的 103 个） | — |
| **0911 三个 vendor 只覆盖其中 3 个** | |

即 0911 镜像只带了「GLM next transformer」这条线的少量算子，**`custom_xllm_math`（xllm
自己的算子包）整个不存在**（虽然 `vendors/config.ini` 的 `load_priority` 里还写着它）。
所以 `aclnnScatterNdUpdateV2` 找不到、`aclnnSparseFlashAttentionLse` 也找不到。

### (2) 装回算子包：取 98 构建容器里那份，并**让 `glm_next_transformer` 排第一**

98 的 `dcpfp32-build` 容器里有一份 **2026-09-11 16:31 编译安装**的 `custom_xllm_math`
（136 MB，52 个 aclnn 算子，含 `aclnnScatterNdUpdateV2`，kernel 目录是 `ascend910_93`）。
它的 `.xllm_ops_git_head` 记着 `d4a441a60b8b2a42ae2e722e662b21d508d6781c`，
而 wheel 的 `third_party/xllm_ops` pin 是 `b94b873`；两者不一致的原因是构建时把
`XLLM_OPS_GIT_HEAD_CACHED` 钉住了（跳过 ops 预编译），所以 wheel 的 C++ 就是**按
d4a441a6 的头文件**编的 —— 这份包正好匹配。

**关键点：`load_priority` 里 `glm_next_transformer` 必须排在 `custom_xllm_math` 前面。**

| 提供者 | `aclnnSparseFlashAttentionGetWorkspaceSize` 形参个数 |
|---|---|
| 0911 `glm_next_transformer`（8-11） | **23**（含 `preTokens/nextTokens/attentionMode/returnSoftmaxLse/attentionOut/softmaxMax/softmaxSum`） |
| `custom_xllm_math`（d4a441a6，以及 0801 镜像、Aug-11 `.run`） | **17**（`scaleValue` 在第 10 位） |

0911 镜像里的 `libtorch_npu.so`（Aug 7）调的是 **23 参数**那一版；而 `xllm/python/
kernels_npu/sparse_attention.py:426` 已经改成走 `torch.ops.npu.npu_sparse_flash_attention`
（旧的自定义 `xllm_ops::sparse_flash_attention` 那条 C++ 路径已被标为 deprecated），
所以 SFA 的 ABI 由 `libtorch_npu` 决定 ⇒ **SFA 必须解析到 23 参数的那份**，
其余算子（V2、LSE、Laser…）再从 `custom_xllm_math` 兜。

`xllm/core/kernels/npu/aclnn/pytorch_npu_helper.hpp:257-300` 的查找顺序是
`ASCEND_CUSTOM_OPP_PATH` 的每个 vendor → `config.ini` 的 `load_priority` 列表 →
`dlopen("libopapi.so")`，**逐个 vendor 试到找到符号为止**，所以两份包可以共存、各供各的符号。

安装（容器内）：

```bash
V=/usr/local/Ascend/cann-9.0.0/opp/vendors
tar xzf .../lib/xllmops_p98.tgz -C "$V/custom_xllm_math"     # 136 MB, 52 ops, 45 个 kernel 目录
printf 'load_priority=glm_next_transformer,custom_transformer,kpool_transformer,custom_xllm_math\n' > "$V/config.ini"
```

**并且 `env.sh` 里那段「有 `custom_xllm_math/bin/set_env.bash` 就 source 它」被删掉了**：
那个脚本会把 `ASCEND_CUSTOM_OPP_PATH` 设成 `custom_xllm_math`，而 `ASCEND_CUSTOM_OPP_PATH`
是**第一优先级**，等于把 17 参数的旧 SFA 顶到最前面 —— 正是第 27 轮那个段错误的复现路径。

结果：`profile_manager.cpp:1247] Eager warmup completed: tokens=256, latency=71.9 ms`
→ **prefill 第一次越过 eager warmup**，`Brpc Server started`，decode rank 0 报
`Successfully linked instance ... prefill_kv_split_size: 2`（异构 DCP 已互认）。

### (3) 仍然缺的算子：`aclnnSparseFlashAttentionLse`

`xllm/python/layers/sfa_dcp.py:504` 的 DCP SFA 路径调的是
`torch.ops.xllm_ops.sparse_flash_attention_lse(...)`（decode 阶段 `return_softmax_lse=True`），
而这个算子**在 0801 镜像、P98、Aug-11 的 `.run` 里都没有**（三者导出集合几乎相同，都是 52 个）。
全集群唯一一份带它的是 83 上 `xllmops-build` 容器里 **9-16 定向编译**的包
（`nm` 有 `aclnnSparseFlashAttentionLse`，kernel 目录 `ascend910_93/sparse_flash_attention_lse`）。

它是**按需 `dlsym`**（`EXEC_NPU_CMD` 里 `static const auto ...` 首次调用时才解析并 CHECK），
所以 warmup 没触发就不报错。只要 DCP 的 decode 路径跑到就会 FATAL —— 这一条仍然悬着。

### (4) tracer 自己的 bug：`numpy()` 不接受 bfloat16

第一次带 trace 起 worker，prefill 4 个 rank 全部在 `register_kv_cache_impl success` 之后崩：

```
terminate called after throwing an instance of 'pybind11::error_already_set'
  what():  TypeError: Got unsupported ScalarType BFloat16
At: .../xllm/python/_pd_trace.py(168): _row_hashes
```

`_row_hashes` 里 `flat.numpy().tobytes()` 在 BF16 上直接抛 C++ 绑定异常 → 整个 worker
abort（**不是路由问题，是验证工具的问题**）。改法：

```python
def _host_bytes(tensor) -> bytes:
    try:
        return tensor.numpy().tobytes()
    except TypeError:
        import torch
        return tensor.contiguous().view(torch.uint8).numpy().tobytes()
```

刻意不做 `to(float32)`：那会改变我们正要逐字节比对的内容。

### (5) 端口抢占：这个容器的临时端口范围是 **1024–65535**

第二次拉起直接拒绝启动：

```
ERROR: these ports are already in use: P:46100 P:46101 P:46102 P:46103
```

`cat /proc/sys/net/ipv4/ip_local_port_range` = `1024 65535` —— 也就是说**我选的每一个固定端口
都可能被宿主网络命名空间里别的容器的普通外连临时占掉**（`--network=host`）。
这不是「上一个进程没死」（`stop_workers.sh` 已确认 12 个进程全清、端口表也空），
而是**一次性的端口检查本身在有竞态的配置下不可靠**。

`start_workers.sh` 改成**轮询等一个空闲窗口**（`PORT_WAIT` 默认 180s，每 5s 重查），
把「偶发 500 / 起不来」变成「最多等两分钟」。

### (6) 真正的路由 bug：把对端的 manifest 拿去和我们自己的 topology 对等比较

修好 tracer 之后，请求终于走到推送，然后 prefill 4 个 rank 全部：

```
E mooncake_kv_cache_transfer.cpp:936] Cannot assemble the destination instance on the
  canonical route, request_id=chatcmpl-...: the cache layout of peer rank 0 does not match
  the model: layer 0, role 0, group 0: the declared topology differs from the coordinates
  the peer published
F llm_worker_impl.cpp:286] Check failed: kv_transfers.wait() KV cache push failed
```

定位：`cache_directory.cpp:660-676`，`PeerDirectory::describe()` 对 MAIN 命名空间的张量做

```cpp
if (declaration->topology.dp_size != manifest.coordinates.dp_size ||
    declaration->topology.cp_size != manifest.coordinates.cp_size ||
    declaration->topology.tp_size != manifest.coordinates.tp_size ||
    declaration->topology.kv_split_size != manifest.coordinates.kv_split_size) { fail }
```

而 `mooncake_kv_cache_transfer.cpp:928` 传给 `build_route_peer` 的 `declarations_` 是
**本实例**（prefill: cp=2, tp=2, kv_split=2）的声明，`manifest` 却是**对端**（decode:
cp=1, tp=2, kv_split=1）发布出来的坐标 —— 两者按设计就不相等。

**同构部署下它们恰好相等，所以这个检查一直是「零成本」的；异构 DCP 正是要打破它，
于是这条检查把 canonical 路由唯一要服务的场景拒了。**

而这个「相等」本身也不是无害的：`describe_tensor()` 后面所有依赖 `declaration.topology`
的推导都会用错坐标系：

- `KvRedundancy::derive(declaration.topology, ...)` → `redundancy.split()` 应为对端的
  kv_split；用本地的 1 会让 332-354 行的 slice 一致性检查按错的 split 去比
  （`max(kv_split_size,1)` 与 `split()` 一起决定要不要比）。
- `view->local_rank = cp_rank * declaration.topology.tp_size + tp_rank`：tp 不同时直接算错。

**修法（`pd_route_transfer.cpp`）**：新增文件内 helper，把交给 `PeerDirectory::describe`
的声明**按对端坐标重写**；只改 MAIN 家族，只改并行坐标，**model 侧的 group 几何与
`tokens_per_block` 保持我们自己的**——因为两边跑同一个模型，而 `describe_tensor` 仍会拿
它们去和对端发布的 descriptor 对账，真不一致照样拒绝。SPEC_DRAFT 家族保留自己的
topology（manifest 坐标永远描述 MAIN，见 `CacheTensorDeclaration` 的注释）。

```cpp
std::vector<CacheTensorDeclaration> declarations_for_peer(
    const std::vector<CacheTensorDeclaration>& declarations,
    const ParallelCoordinates& coordinates);
```

`build_route_peer` 里逐 rank 用该 rank 的 `manifest->coordinates` 生成，再交给 describe；
`pd_route_transfer.h` 的 `build_route_peer` 文档补上这条规则。这个检查**故意保留**：
它现在的作用变成「谁忘了做这个适配，就在组装点立刻报错」，正是我们踩到的那个坑。

### (7) 重新构建（98 / `fengmin-cann9-20260801`）与验证口径

- 改动同时打进本机 clone 和远端工作树 `/export/home/shifengmin.3/workspace/xllm-dcp-fp32`
  （`ad98850a8` + 未提交改动），**不动 HEAD**，`XLLM_OPS_GIT_HEAD_CACHED` 钉住 ops 门，
  只增量重编 `pd_route_transfer.cpp` + 重链。
- **构建容器是 98 的 `fengmin-cann9-20260801`**，不是 `dcpfp32-build`：后者缺
  `/usr/local/lib/cmake/yalantinglibs/config.cmake` 与 `/usr/local/go`，`setup.py` 会去跑
  `dependencies.sh`，而该脚本在这个容器里因为 `boost1.78-devel`/`msgpack-devel` 找不到而失败
  （还会 `dnf remove` 掉 54 个文件）。**别在 `dcpfp32-build` 里构建。**
- 产物：`dist/xllm_npu_torch2_9_0-0.11.0-cp311-cp311-linux_aarch64.whl`
  = 576,357,358 B，md5 `ac503fe5763fb7b325703736dc7d3f1a`（旧 wheel 是 576,354,918 B /
  `f87c4008abc0d130098165414fdddd9f`）。装到 0911 容器后 ELF md5 从
  `3a65489db9dfd7732a135ccb2ffc791d` 变成 `ee47ec797979cdc0efc9cac322c7cd07`。
- **wheel 文件名不能改**（PEP 427）：`pip` 会自己解析文件名，改成 `xllm_peerdecl.whl`
  直接报 `Invalid wheel filename (wrong number of parts)`；必须先 `cp` 回原名再装。

### (8) 恢复点

下一步就是**用修好的 wheel 重跑 `run_trace.sh`**，看推送能否成功、并落下
PREFILL/DECODE 两侧的 trace，然后 `compare_kv.py` 出逐字节结论。
仍悬着的环境债：`aclnnSparseFlashAttentionLse` 需要一份定向编译的算子包
（源在 `~/work/xllm/third_party/xllm_ops`，`build.sh -n sparse_flash_attention_lse`
需要补上 `third_party/{abseil-cpp,ascend_protobuf,json,makeself,pkg}` 与
`cmake/third_party/build`——后者的 patch 文件在第一次打包时被 `--exclude=./build` 误删过，
正确写法是 `COPYFILE_DISABLE=1 tar czf ... --exclude-vcs --exclude='._*'`，
**不要**排除 `./build`，否则会连 `cmake/third_party/build` 一起去掉，
而且 macOS 的 tar 会塞进 `._*` 苹果双胞胎文件把 AscendC 编译搞崩）。

### (9) 第一次跑通与字节级比对的第一版结论

修好的 wheel 装进 0911 容器后，`run_trace.sh` 一次跑通：

```
http_code=200 time=5.39s   prompt_tokens=3492   completion_tokens=4
trace: PREFILL 4 ranks × 19782 行, DECODE 2 ranks × 2560 行
```

比对（`compare_kv.py --tp-size 2`，prefill 源取**两个 chunk 的并集**）：

| slot | 类 | 总数 | 命中预测位置 | 备注 |
|---|---|---|---|---|
| `key` / `value` | split | 116 | **108** | 每条摘要出现在 **2 个** prefill rank 上 = MLA 单 latent head 在 tp 对上的复制 ✓ |
| `index` | replicated | 87 | 81（并集口径 84） | 48 条在**全部 4 个** prefill rank 上出现 ✓ |

**所有"不匹配"行的 sha256 恰好等于对应形状的全零缓冲区**：

```
key   131072 B -> fa43239bcee7
value  16384 B -> 4fe7b59af6de
index  32768 B -> c35020473aed
```

即**失败项全部是 decode 侧没被写入的行**，且它们的源行**根本不在 prefill 那次 batch 的行集合里**：

* `row 3620+call`：tracer 从 `slot_mapping` 推出的**伪行号**（每步 +1，请求之外的槽位）；
* `row 1`：prefill 的**逻辑块号从 1 开始**（`block_table=[1..14, 128…]`），
  所以 prefill 没有逻辑 0 / canonical 1，decode 的 canonical row 1 无源；
* `index` 的 `row 15..` 是 decode **自己**在上一步之后才写的（`after call0` 非零行 13→14），符合 DSA 逐步写。

⇒ **传输的字节本身是对的**；verdict 的 FAIL 来自「全零行 + 源行不在 prefill batch 内」，
属于比对口径过严 + tracer 伪行号。已把 `block_table` / `slot_min/max/len` / `slot_derived` / `slot_unit`
插桩进 `__meta__`，用于定位最后那个边界块。

### (10) 又一个环境坑：`--shm-size=64m` 被**具名 POSIX 信号量**撑满

连跑第二次 bring-up 时，prefill rank 0/2/3 全死，症状分散得像三个不同的 bug：

```
File "multiprocessing/synchronize.py", line 57, in __init__
  sl = self._semlock = _multiprocessing.SemLock(...)
OSError: [Errno 28] No space left on disk
RuntimeError: SetPrecisionMode:...AclSetCompileopt(...) error code is 500001
F llm_master.cpp:62] Check failed: engine_->init(master_status_)
```

根因：**每个 Python `multiprocessing.Lock`/`Queue` 都是一个住在 `/dev/shm` 的具名信号量**，
而 CANN TBE 并行编译池会创建很多个；**SIGKILL 不会 unlink 它们**，
所以每跑一轮就漏一批，几轮之后 64M 的 `/dev/shm` 满了。

`df -h /dev/shm` 在 reset 前后：`64M 用满` → `0% / 0 个条目`。
修法：`stop_workers.sh` 在所有 worker 都停掉之后
`find /dev/shm -maxdepth 1 \( -name 'sem.*' -o -name 'mp-*' \) -delete`，并打印回收前后用量。

**记住：这个容器里"起不来"的第一件事是看 `/dev/shm`，不是看日志。**

### (11) 插桩定位：伪行号的来源，以及剩下的**唯一**疑点

给 `__meta__` 加上 `block_table` / `slot_min/max/len` / `slot_derived` / `slot_unit` 之后，
run6 的 meta 直接给出答案：

```
DECODE  before call0: block_table=[1..28] len=28 slot_len=1 slot_min=slot_max=3620 slot_unit=128 slot_derived=[28]
PREFILL after  call1: block_table=[1..14] len=14 slot_len=3492 slot_min=256 slot_max=3747 slot_unit=256 slot_derived=[1..14]
```

⇒ decode 的 `slot_mapping` **只有一个槽**（本步新 token，slot=3620 = block28×128+36），
它 `//128` 得到 28，**本来就在 block_table 里**。写入 trace 的 `block 3620` 是 tracer 的 bug：

```python
block_tokens = max(1, int(getattr(metadata, "block_size", 0) or 0)) * split   # ← block_size=0 时得 1！
```

`max(1, 0)` 把"没有 block_size"变成了 **1**，于是 `slot // 1` = 槽位值本身。
prefill 侧同理：`split=2` 让除数变成 2，`slot//2` 产出 `128..1873` 一堆伪块号
（这正是 run4/5 里 prefill `blocks` 出现 128+ 的原因）。
修法：**block_table 非空时只信它**，为空时才用 slot 推导，且缺失 block_size 时回退到 128 而不是 1。

按修好的口径重算 run6：decode 行集 = `block_table`（308 = 28 行 × 11 slot），
K/V 的非零行 **27/28**，唯一全零行是 **row 1**；index 非零 13 行（15..28 是 decode 自己逐步写的）。

### (12) 剩下的唯一疑点：decode 的 row 1（第一个块）

事实：
* decode `block_table=[1..28]`（28 个块），slot 约定是 **1-based**：`slot = block_id*128 + offset`
  （新 token 在位置 3492 = (28-1)*128+36 ⇒ slot 3620）⇒ **block 1 = 位置 0..127**，是序列的第一个块。
* prefill `block_table=[1..14]`，`slot = logical*256 + offset` ⇒ **逻辑 1 = 位置 0..255**，
  即 canonical 2（slice 0 = 位置 0..127）、canonical 3（slice 1 = 位置 128..255）。
* 实测映射（两边一致、108/116 命中）：`decode.row[c] ← prefill 逻辑 c//2, slice c%2`。
  ⇒ decode row 2 拿到的是 **位置 0..127** 的 KV，而 row 2 语义上是位置 128..255;
  decode row 1（位置 0..127）**全零**。

两种解释，必须二选一：

1. **口径对**：canonical id 空间里 0 号块保留不用，序列从 canonical 2 开始，
   decode 的 block 1 是空穴 —— 但 decode 只有 28 个块（1..28），若 1 是空穴则只有 27 个有效块
   = 3456 token < 3492，**差 36 个 token 正好等于最后那个 offset**。
2. **差一个块（真 bug）**：正确的配对应是 `decode (1,2) ← prefill 逻辑 1`、`(3,4) ← 逻辑 2`…
   即 `prefill 逻辑 = (c+1)//2`，而实现用的是 `c//2` —— 偶数块对、奇数块全部错位一格。

**判据（下一次实验）**：让 decode 把它的 KV 行**按语义位置**对一遍：
`decode.row[k]` 应当等于「位置 (k-1)*128.. 的 KV」。prefill 侧位置可从
`slot // 256 = logical-1`、`(slot % 256) // 128 = slice` 直接算出，
于是不需要猜 canonical 定义：把 prefill 的 (位置块, slice) 与 decode 的 row 直接对上即可。
如果 `decode.row[1] == prefill(位置 0..127 的 slice0)` 不成立而 `decode.row[2]` 成立，
就是解释 2，需要把 canonical→row 的映射改成 `(c+1)//2`。

### (13) 与口径无关的硬结论：**decode 只拿到 27 个块的 KV，而请求需要 28 个**

把两边的**位置语义**拉出来对，就不需要再争 canonical 定义：

| | slot 约定 | 第一个块 | 覆盖 |
|---|---|---|---|
| PREFILL | `slot = logical*256 + offset`（`slot_min=256`） | 逻辑 1 = 位置 0..255 | 逻辑 1..14 = 位置 0..3583 |
| DECODE | `slot = block_id*128 + offset`（本步新 token 位置 3492 → slot 3620 = 28*128+36） | 块 1 = 位置 0..127 | 块 1..28 = 位置 0..3583 |

请求 prompt = 3492 token ⇒ **需要 28 个 128-token 块**（块 1..28，最后一块用 36 token）。
而 decode 侧实测：**非零行只有 27 个（row 2..28），row 1 全零** ⇒

* 无论 canonical id 是 0 基还是 1 基，**decode 的 row 1（位置 0..127）是空的**，
  而它有数据的 27 行装的是 prefill 前 27 个 128-token 半块（位置 0..3455）；
* ⇒ **位置 3456..3491（最后 36 个 token）的 KV 没有任何一行接收**；
* ⇒ `decode(块1) 缺数据` 与 `尾部 36 token 缺数据` 二者必居其一，
  取决于映射是「差一个」还是「帧不同」——**无论哪种都是传输侧的真实缺陷，不是比对假阳性**。

判据（口径无关）：decode 的**非零 K/V 行数必须 = ceil(prompt_tokens / 128)**。
本次 27 ≠ 28。

下一步实验（用来二选一，并顺手排除边界噪声）：

1. **把 prompt 长度凑成 256 的整数倍**（例如 3584 token = 14 个逻辑块）再跑一次：
   若尾部差块消失、只剩 row 1 空 ⇒ 是「差一个」的映射问题；
   若仍然差一块 ⇒ 是尾部半块没被规划进 route。
2. 在 `PdRouteTransfer::plan` / binder 里加一行 VLOG，打印
   `(canonical_block -> dst_local_rank, dst_row)` 的配对，直接把映射钉死。
3. 同时给 `compare_kv.py` 的口径加一条**与口径无关的断言**：
   `count(nonzero K/V rows) == ceil(prompt_tokens/128)`，任何映射假设错了都会立刻暴露。

### (14) 定位完成：**目标行整体高了一格（off-by-one）**，证据闭环

用 decode **自己的第一步**当标尺（这一步写的 token 在绝对位置 3492，slot=3620）：

```
decode rank0 before0 -> after0，行内容发生变化的行：
   layer 0/1/2/3 × slot key/value/index  ⇒ 全部是 row 28
```

⇒ **decode 的 row k 覆盖位置 (k-1)*128**：位置 3492 属于块 28（3456..3583）✓ 与 slot 3620 = 28*128+36 自洽。

再把两侧按**位置**对齐（与 canonical 定义无关）：

| 半块 h（0 基，位置 h*128..h*128+127） | 应由谁提供 |
|---|---|
| h = 0..3455/128 = 0..26 | prefill 逻辑 L = h//2 + 1，slice s = h%2 |
| h = 27（位置 3456..3583，prompt 用到 3456..3491） | prefill 逻辑 14，slice 1 |

实测 decode 侧：**row 1 全零**，row 2..28 装的是 prefill 的 `(L = c//2, s = c%2)`，c = 2..28，
即 **h = 0..26 的内容被写进了 row 2..28** —— 而它应该落在 row 1..27。

⇒ **一个 off-by-one 同时解释了两个现象**：
* decode row 1（h=0）空；
* prefill 的最后一个半块（L=14, s=1，h=27）没有任何目标行接收（本该落在 row 28，现在被 h=26 占了）。

**根因**：`canonical_blocks_of_request`（`cache_directory.h:135` 那段注释）把**1 基的逻辑块号**
直接展开成 `canonical = logical * kv_split_size + slice`。prefill 的逻辑块号是 1 基（`block_table=[1..14]`，
`slot_min=256` 说明位置 0 落在逻辑 1），于是展开得到 **2..29**；而 decode 的块号是同一序列的
**1 基半块编号 1..28**（`block_table=[1..28]`）。两者相差 1。
展开式在**同一实例内部**自洽（prefill 自己的 indexer 表也这么展开，所以源侧行号 `c//S` 是对的），
但**跨实例当目标行号用**就错了：目标行应为 `(logical-1)*S + slice + 1 = c - 1`。

**修法方向**（下一轮做，需要重编 + 重跑）：跨实例不再用「各自的 canonical id」当键，
而是**经由共享的逻辑块号 + 半块序号（或绝对位置）换算**到对端的行号；
`KvLayoutIndex` / `ReshardPlanner` 本就是为这件事准备的。修完的验收判据（口径无关）：

```
count(nonzero K/V rows on decode) == ceil(prompt_tokens / 128) == 28
decode row 1 非零，且 == prefill(L=1, s=0) 的摘要
```

---

## 第 29 轮（2026-09-18 晚）：canonical 坐标改成「位置」，off-by-one 落地修掉 + 单元测试 114 全绿

### (15) 定案：canonical 是**位置**，不是地址；三种池子各有一套行布局

第 28 轮 (14) 的结论（目标行整体高一格）在这一轮落到代码里。补上了当时缺的那块拼图：
**池子的行号为什么是 1 基**，以及**同一份 canonical 在不同 family 上落到哪一行**。

三条运行时事实（都在代码里核过，不是推断）：

| 事实 | 出处 | 结论 |
|---|---|---|
| 池子保留第 0 行做 padding | `BlockManagerImpl::BlockManagerImpl`：`// reserve block 0 for padding` + `CHECK_EQ(padding_block_.id(), 0)`；`free_blocks_` 升序发放（`free_blocks_.push_back(total-i-1)`，`allocate` 从尾部取），所以**第一个真实块 id = 1** | 请求的第 `p` 个块 = 池行 `p + 1`；`num_total_blocks() = free_blocks_.size() - 1` |
| `slot = block_id * block_size + offset` | `KVCacheState::cache_slots` | 行号 = 池 id，**1 基** |
| DCP 局部化 `local_row = global_slot / logical_block_size = block_id` | `localize_kv_shard_slots`（`kv_shard_batch_metadata.cpp`） | KV cache 的行 = 池 id = 位置块 + 1 |
| indexer 全序列池按 `row = block_id * dcp_size + slice` 展开 | `expand_kv_shard_indexer_block_table` + `KVShardLayout::globalize` | 全序列副本的行 = 位置 `h` + `dcp_size` |

于是 **canonical 只能定义成「位置」**：`h = (id - 1) * split + slice`（`id` 是池 id，`split` 是该实例的 kv_split）。
把它变成行号按 family 分三种：

```
sequence_scoped（SSM/CONV/LINEAR/EMBEDDING）  row = h            // slot id 本身就是位置
full_sequence_replica（DSA indexer 池）        row = h + kv_split // 每个 slice 一行，整体后移一个保留块
其它（KV cache / MLA latent）                  row = h / split + 1 // 每块一行，第 0 行是 padding
```

**为什么同构时看不出来**：`row = (h + S)/S = h/S + 1` 只在两端 `S` 相同时成立；
`S_P=2, S_D=1` 时 `(h+2)/1 = h+2`，比正确的 `h+1` 高正好一格 —— 就是第 28 轮实测的那一格。
也就是说：**旧的 `canonical = id*S+slice` 是「用源侧坐标当目标行号」，同构下恰好对，异构下必错**。

改动：

* `cache_directory.{h,cpp}`：`canonical_blocks_of_request` 改成 `(id - 1) * factor + offset`，
  并对 `id == 0`（padding 块，没有位置语义）**直接报错**而不是静默平移；头注释重写，把
  「位置而非地址」和三条行布局写清楚。
* `route_binder.{h,cpp}`：新增文件内 `peer_row(view, split, block)`，把 canonical → 池行号的三条规则
  收在一处；`bind` 里 `local_row`/`remote_row` 都走它。注释里点明「同构下旧公式恰好对，异构下必错」。
* `pd_route_transfer.h` / `kv_redundancy.h`：`CanonicalBlock::local_row` 的文档补上
  「这是**位置行**（0 基）；池行是它 +1，因为池子把第 0 行留给了 padding 块」，并说明这个偏移
  属于 peer-dependent 那一侧，所以落在 `RouteBinder` 而不是 `CanonicalBlock`。

### (16) 单元测试跟着改口：114 用例全绿

口径一改，5 个测试文件的 fixture 必须一起改（它们原本**自洽地**编码了旧口径 —— 这正是
`kv_shard_contract_test` 开头那段警告说的「两边用同一个错误公式，谁都发现不了」）：

| 文件 | 改了什么 |
|---|---|
| `kv_shard_contract_test.cpp` | id → canonical 的期望改成 `(id-1)*4+j` 且 id 用池行 `{1,2,4}`；新增「拒绝 padding 块 0」用例；indexer 行断言改成 `h + kv_split`（即运行时展开式） |
| `pd_route_transfer_test.cpp` | fixture `resource_count = 资源数/split + 1`（多出的第 0 行 = padding 行）；`fill_expected` 的行号 `+1`，并给 sequence-scoped family 走 `row = block`（slot 就是行） |
| `pd_route_integration_test.cpp` | 新增 `RowLayout` + `position_of_row()`，把三种行布局写成 fixture 的显式模型；行数按布局算（副本族 = `(位置数/S + 1) * S`）；`canonical_content` 不再用 `CanonicalBlock::local_row` |
| `cache_directory_test.cpp` | 源 `rows=1→2`、目标 `rows=4→5`（给保留行让位） |
| `pd_route_test.cpp` | 三处 golden 字节偏移改成 `block/split + 1`（这些是**独立写死**的期望值，所以它们真能抓错） |

构建 + 运行（同一个 cmake build dir，见 (17)）：

```
kv_shard_contract_test    6/6      pd_route_test            12/12
kv_redundancy_test       12/12     pd_route_transfer_test   14/14
cache_directory_test     23/23     pd_route_integration_test 4/4
reshard_planner_test     43/43     --------------------------------
                                   合计 114 用例，exit=0
```

其中 `pd_route_integration_test` 是**主机端逐字节**跑的 2↔4 异构 DCP 用例（含 MLA/非 MLA、
SSM/CONV、indexer、head class 折叠），它现在按池行布局独立算期望，所以是这次改动真正的回归网。

### (17) 构建环境的两个坑（都能让「只重编一个 .cpp」变成全量或直接失败）

1. **`setup.py` 的 env 会被烧进编译命令**。`scripts/build_support/env.py::set_npu_envs()` 设
   `PYTORCH_INSTALL_PATH` / `LIBTORCH_ROOT` / `PYTORCH_NPU_INSTALL_PATH`，而 CMake 里
   `-I$ENV{PYTORCH_INSTALL_PATH}/include` 是**配置期展开**的。用裸 `ninja`（不经过 setup.py）触发
   重新 configure，得到的命令里会出现 `-I/include`（空值），于是
   **整个 build dir 的 1395 个目标全部失效**并开始重编，而且因为 `LIBTORCH_ROOT` 为空，
   CMake 走 `FetchContent` 分支去**下载 libtorch**（容器没有外网）→ `Build step for libtorch failed`。
   正确做法：先跑一段 python 复刻 setup.py 的 env（`import scripts.build_support.env; set_npu_envs()`，
   把 `os.environ` dump 成 `export`），再 `cmake -S <tree> -B <build dir>` 显式重生成一次。
2. **ops 门要 `XLLM_OPS_GIT_HEAD_CACHED`，值不是 `git ls-tree` 的 pin**。`CMakeLists.txt:74` 比的是
   `git -C third_party/xllm_ops rev-parse HEAD`。xllm_ops **是**注册的子模块（`.gitmodules` 里
   有 `third_party/xllm_ops`），但 **98 构建树里的 `third_party/xllm_ops/.git` 不存在**（子模块
   从未在那棵树里初始化），于是 git 向上找到**父仓库**、打印**父树 HEAD**（本树 `ad98850a8`）；
   本机这份 clone 的 `third_party/xllm_ops/.git` 是 48 字节 gitfile，同一个命令打印的才是
   `b94b873`。不设这个变量 → 每次 configure 都跑 `third_party/xllm_ops/build.sh`（这棵树里没有）→
   `error code: 127` + `Configuring incomplete`。**正确取值 = 在哪棵树里构建，就在那棵树里跑一次
   这条命令取它打印的值**；第 28 轮 RESUME 写的 `b94b873` 是从本机 clone 抄的，在 98 那棵树上
   是错的。

顺带：`stop_workers.sh` 之后 `/dev/shm` 的清理、端口轮询、`custom_xllm_math` 的
`load_priority` 顺序都保持第 28 轮的做法，不再重复。

### (18) 端到端验收：**PASS（字节级）**

重编 wheel（`576,360,599 B` / md5 `864cd27e…`，装完 ELF md5 `873c87ac9d544f119cad5e94bc5572e9`，
旧 `ee47ec79…`；二进制里能 grep 到新错误串 `the reserved padding block` 作为「这版确实进去了」的凭据），
装进 0911 容器 `fengmin-pdroute91`，`run_trace.sh` 跑一次 3492-token 请求，`http_code=200`，
trace 落下（PREFILL 4 rank × 708 行，DECODE 2 rank × 2472 行 —— 修好的 tracer 只 hash
`block_table ∪ block_table*S+j`，比上一轮 19782 行的伪行号集合小一个量级）。

判据 1（口径无关、第 28 轮定的）：**decode 每个 layer 的 K/V 非零行数 = ceil(3492/128) = 28**

```
layer 0 key  : rows=28 nonzero=28 min=1 max=28 all_nonzero=True
layer 0 value: rows=28 nonzero=28 min=1 max=28 all_nonzero=True
... layer 1/2/3 同（8/8 组合全部 1..28 全非零）
```

判据 2：**每一行的字节都来自规则预测的 (prefill rank, prefill row)**。
`compare_kv.py` 的「先按摘要反查真实落点、再和规则比」在 `before call 0`（推送后、decode 自己
第一步之前）这一帧：

```
key    split       total=112 written=112 zero=0 match=112 mismatch=0 absent_in=0
value  split       total=112 written=112 zero=0 match=112 mismatch=0 absent_in=0
index  replicated  total= 84 written= 39 zero=45 match= 84 mismatch=0 absent_in=0
VERDICT: PASS
```

`key/value` 112 = 28 行 × 4 layer，**一条 mismatch / 一条空行都没有**；反查出来的配对是
`(rank,row) = (0,1)、(2,1)、(0,2)、(2,2) …`，正是 `row = h/S + 1`、writer slice `= h % S` 的
新规则（`row 1` 来自 prefill rank 0 row 1 = 逻辑 1 的 slice 0 —— 上一轮那个「空穴」就在这一行）。
index 84 = 28 行 × 3 layer，0 mismatch / 0 absent；45 个 zero 行是**两边都还是零**的
（prefill 自己那 4 个 rank 在该行也是全零，逐行核过：lost=0、phantom=0）。

**顺带推翻了一个之前的臆断**：indexer 池并不是「每个 rank 内容相同」。同一 index 行上，
cp=0 的两个 rank（0,1）与 cp=1 的两个 rank（2,3）持有**不同**的摘要 —— 只有同一 slice 的
副本（tp 对）才逐字节相同。所以 `compare_kv.py` 里那条「摘要必须在全部 4 个 prefill rank 上
出现」的断言是错的期望（`all_ranks=45 ≠ replica_rows=84` 就是它在报 FAIL），已降级为
**信息性统计**：路由按声明把 full-replica family 从**单个 writer rank** 推一份，这个语义没问题，
但「副本天然相同」这个前提在 DCP 下不成立于跨 slice 的副本 —— 这条记在这里，留给后续
判断「indexer 只推一份是否够用」（本次 decode 侧拿到的是 cp=0 那份，逐行与源一致）。

其余 7 个 decode 帧 FAIL 是**口径使然**：`call ≥ 1` 之后 decode 自己往上写了新 token 的那一行
（row 28）和它自己算的 index 行，prefill 的 trace 里当然没有这些字节；`compare_kv.py` 现在
明确把「最早 PASS 的那一帧」当判据，并在输出里写明原因。

### (19) 本轮留下的债（下一轮候选）

1. **位置信息仍然是从块 id 推的**：`canonical = (id - 1) * S + slice` 假设「请求的块 id = 位置块 + 1」。
   池子保留 0 号块 + 升序发放保证了**首个请求**成立；prefix cache 命中（共享块 id 任意）或
   多请求并发（第二个请求的基址不是 1）时会失配。真正的修法是把位置显式带进 `KVTransferMapping`
   （`batch_input_builder` 里 `local_idx` 就是位置，`next_transfer_idx` 是基址），
   让 canonical 由位置而不是 id 推出来。这条现在会**静默**错（不会报错），优先级最高。
2. indexer 跨 slice 副本不同 → 「只推一个 writer」是否够用，需要模型侧确认（见 (18) 末段）。
3. 收尾清理：`scripts/build_support/utils.py` 的 `PDROUTE_BUILD_ONLY_BYPASS`、未跟踪的
   `push_route.{h,cpp}` / `push_route_test.cpp` / `kda_constants.py` / `third_party/dependencies.sh`、
   `trace_remove.sh` 还原；`~/work/pdtrace/` 不进仓库。

### (20) 代码评审补刀：prefix cache / chunked prefill 下位置不能从块 id 推

评审（Claude）指出 (19) 债#1 是真问题，并给了最小修法：`base = index * split`（`index` = `local_ids`
里的位置，恒等于 `p`）。**方向对，但那个式子只在「这一步的列表从位置 0 开始」时才成立**：

* `build_step_transfer_info`（`batch_input_builder.cpp`）里，发给数据面的
  `step_mapping.local_ids` 是**本步**的子序列：`for (local_idx = next_transfer_idx; local_idx < map_end; ++local_idx)`，
  而 `next_transfer_idx` 是传输游标（会被 `remote_shared_num` 压缩到 D 侧已有的共享块数之后）。
  所以列表第 k 项的位置是 `next_transfer_idx + k`，**不是 k** —— 用 `index * split` 会把
  「chunked prefill 的第二块 / 前缀命中之后的第一块」整段平移到位置 0。
* 反过来，原来的 `(id - 1) * split` 在**没有前缀共享**时恰好等于 `next_transfer_idx + k`（因为
  池子按 1 基顺序发放），所以单请求场景看不出问题；前缀命中时 id 是共享前缀所在的行，`id-1`
  与位置无关，**静默错位**。

**本轮做法（＝把「位置」显式带上，而不是从任何东西推）**：

| 位置 | 改动 |
|---|---|
| `common/types.h` | `KVTransferMapping` 新增 `std::vector<uint64_t> local_positions`，与 `local_ids` 一一对齐（**逐项**记录，所以中间丢掉一条 lane 也不会把后面的位置带偏） |
| `batch_input_builder.cpp` | `build_step_transfer_info` 里 push `local_ids` 的同时 push `local_idx`（就是绝对位置） |
| `forward_shared_memory_manager.cpp` | 共享内存序列化（size/write/read ×2）把新字段一起带上，schedule overlap 打开时位置不丢 |
| `cache_directory.{h,cpp}` | `CacheGroupRequest` 增加 `positions`；`canonical = position * split + offset`；**positions 缺失/条数不匹配 → 直接报错**（不再猜），position 溢出也报错 |
| `mooncake_kv_cache_transfer.cpp` | 调用点把 `local_positions` 传进去 |
| `kv_shard_contract_test.cpp` | 改成「位置驱动」：id 用**任意的池行**（37/38/51 模拟前缀命中，60/61 + 位置 7/8 模拟第二块），并覆盖 `id-1` 与 `index` 两种错法；新增「缺 positions 报错」用例 |

这样：单请求（本轮实测 PASS 的那次）行为不变；前缀命中 / chunked prefill 的第二块从
「静默错位」变成「正确」；而**忘了带位置**的生产者会当场报错，不会静默平移。

### (21) 操作坑清单（下次直接照这个躲）

1. **裸 `ninja` 触发 cmake 重新 configure 时，env 必须和 `setup.py` 完全一样**。
   `scripts/build_support/env.py::set_npu_envs()` 设 `PYTORCH_INSTALL_PATH` / `LIBTORCH_ROOT`，
   而 CMake 用 `-I$ENV{PYTORCH_INSTALL_PATH}/include`（配置期展开）。env 不对 → 命令里出现
   `-I/include` → **整个 build dir（1395 目标）失效重编**；`LIBTORCH_ROOT` 为空还会走
   FetchContent **去下载 libtorch**（容器没有外网）→ `Build step for libtorch failed`。
   做法：用 python dump 一遍 `set_npu_envs()` 的 env（`export KEY='VALUE'`）再 source，然后
   `cmake -S "$TREE" -B "$BUILD"` 显式重生成一次（缓存里的 `-D` 参数会复用）。
   **不要在 build 目录里裸跑 ninja 试探。**
2. **`XLLM_OPS_GIT_HEAD_CACHED` 的正确取值** = 在**你实际构建的那棵树**里跑
   `git -C third_party/xllm_ops rev-parse HEAD` 打印出来的值。98 构建树里 xllm_ops 的 `.git`
   缺失，这个命令会向上解析到**父仓库**、打印**父树 HEAD**（本机 clone 里子模块元数据在，同一
   命令给的是 `b94b873` —— 所以★不要★照抄 `git ls-tree` 的 pin）。不设 → 每次 configure 都跑
   `third_party/xllm_ops/build.sh`（这棵树里没有）→ `error code: 127` + `Configuring incomplete`。
3. **工具是两份拷贝**：本机 `~/work/pdtrace/` 和 83 容器里的 `$BASE/tracefiles/` 不是同一个地方，
   `trace_install.sh` 只从 `tracefiles/` 取。改完 tracer **必须**同步过去（本轮就踩了一次：
   容器里跑的是旧版 `_pd_trace.py`，md5 与本地不一致）。跑之前先 `md5sum` 对一遍。
4. **分清脚本该在哪台机器上跑**：`rebuild_peerdecl2.sh` / `deploy_stage98.sh` 依赖本机
   `sudo -n docker`，必须在 **98 host** 上执行（`rrun … jd-node-98 bash -s <`）；在 mac 上直接跑
   只会得到 `sudo: a password is required`。包装器：`r83.sh`（83 容器内）/`r83.sh --host`（83 host）/
   `r98.sh`（98 容器内）。
5. **`rrun` 不转发环境变量**：`STAGE_WHAT=wheel rrun …` 无效（远端拿到默认值 `all`）。要传参就写死
   在脚本里，或另生成一个脚本。
6. **trace 目录是 root 所有**：打包要 `sudo -n tar`，再 `chown` 成自己才能 `scp` 回本机。
7. **验收前先确认「跑的是哪一版二进制」**：装完 wheel 必须
   `md5sum $SP/xllm/xllm` + `strings` 里 grep 一个**本轮新增字符串**（这轮用
   `the reserved padding block`）。否则很容易把旧二进制的结论当成新结论。
8. **`/dev/shm` 只有 64M**：每轮 SIGKILL worker 都会漏具名 POSIX 信号量，撑满后报
   `OSError: [Errno 28]` + `llm_master.cpp:62 Check failed`；`stop_workers.sh` 停完要清，起不来
   第一件事看 `/dev/shm`。
9. **端口随时被抢**：容器 `ip_local_port_range = 1024 65535`，固定端口会被 host 上别的容器占用；
   `start_workers.sh` 用轮询等空闲窗口（`PORT_WAIT` 默认 180s）。
10. **`libcust_opapi.so` 的解析顺序决定算子 ABI**：SFA 必须解析到 `glm_next_transformer`（23 参数，
    匹配 0911 的 `libtorch_npu`）；source 任何把 17 参数旧 SFA 的 vendor 放进
    `ASCEND_CUSTOM_OPP_PATH` 的 `set_env.bash` 都会让 SFA 解析错 → 段错误。
11. **判断「卡住」先分清进程身份**：`multiprocessing.Manager` 子进程的 argv 和引擎完全相同，
   别按 argv 杀错。

## 第 30 轮（2026-09-18 20:00–）：位置显式化的第一次重编，与一次**静默同步失败**造成的「新旧混合构建树」

### (22) 时间线（本轮最贵的一课在第 (25) 节）

| 时刻 | 事件 |
|---|---|
| 19:30 | 把本机改动同步到 98 构建树（**这次同步静默漏了一个文件**，见 (25)） |
| 19:32–19:40 | 重编 `setup.py build` + `bdist_wheel`：`BUILD_EXIT=0`、`WHEEL_EXIT=0`，wheel md5 `49dc1feb5fc36f5a8883d311db70bd4e`（576,367,916 B） |
| 19:45 | 装进 0911 容器：新 ELF md5 `141cbe43c7b3a5f9084b35db5c12fbe7`（旧 `873c87ac…`）；`strings` 能 grep 到本轮新增串 `positions; a block-scoped group has to say where each id`、`reports position` ×1 |
| 19:46–19:51 | 干净重启 + 带 trace 的单请求：**http_code=500**，`Instance is failed and deleted` |
| 19:51:49 | P rank_0：`E mooncake_kv_cache_transfer.cpp:880 Cannot convert the request's cache ids into canonical blocks, cache group 0 supplies 14 block ids but **0 positions**` → `F llm_worker_impl.cpp:286 Check failed: kv_transfers.wait() KV cache push failed` |
| 19:53 | 定位：98 构建树里的 `forward_shared_memory_manager.cpp` 是 **14:08 的旧版**（`grep -c local_positions` = 0），其余 10 个文件 md5 与本机一致 |
| 19:54 | 修好同步脚本（(25)）→ 15 个文件全部 `ok`；重编复测中 |

### (23) 坑：`rrun` 不转发环境变量，`STAGE_WHAT=wheel` 被静默忽略

`STAGE_WHAT=wheel rrun -F … jd-node-98 bash -s < deploy_stage98.sh` 里，那个变量只存在于**本机**
`rrun` 进程的环境里，远端一个都收不到 —— `deploy_stage98.sh` 于是拿默认值 `all`，先去 stage
`libasio.so`（83 上那个文件是 root 所有、15:36 留下的，于是三次 `scp: Permission denied`），
再 stage wheel。表现是「日志看着像失败，其实 wheel 成功了」。

正确写法（无引号嵌套，一次成功）：

```bash
rrun -F ~/work/.ssh-xllm-config jd-node-98 env STAGE_WHAT=wheel bash -s < deploy_stage98.sh
```

### (24) 坑：重编日志在**容器里**，不在 98 host 上

`rebuild_peerdecl2.sh` 是「host 上跑、`docker exec -d` 在容器里编」，所以 `LOG=/tmp/rebuild_peerdecl.log`
写的是**容器**的 /tmp。我在 98 host 上写的等待脚本去 grep host 的 `/tmp/rebuild_peerdecl.log`，
永远看不到 `REBUILD DONE`，白等 30 分钟（`tail` 报 `No such file` 才反应过来；更早一次 `stat`
的输出还和 `date` 的输出混在一起看错了，误以为文件存在）。

修法：把日志写到 **bind-mount 里的路径**（`/export/home/shifengmin.3/workspace/rebuild_peerdecl.log`），
容器内写、host 直接读；host 侧等待脚本 grep 同一个路径。

### (25) 本轮真正的坑：`sync98.sh` 用 argv 传 base64，撞上 `MAX_ARG_STRLEN`

原来的同步脚本对每个文件生成：

```bash
python3 - "$TREE/<rel>" <167 KB 的 base64> <<'PYEOF'
import base64,sys
open(sys.argv[1],'wb').write(base64.b64decode(sys.argv[2]))
PYEOF
```

Linux 单个 argv 的上限是 **128 KiB**（`MAX_ARG_STRLEN`）。于是：

* `types.h`（13 KB → base64 18 KB）、`batch_input_builder.cpp`（66 KB → 90 KB）等 **10 个文件都写成功**；
* `forward_shared_memory_manager.cpp`（**125,503 B** → base64 **167 KB**）**失败**：
  `bash: python3: Argument list too long`；
* 脚本头是 `set -u`（**没有 `-e`**），所以这一行失败**不中断**，后面 4 个文件继续写成功，
  最后还打印了一堆 `git status`；我从「同步脚本跑完了」这个现象判断「同步成功」。

后果非常隐蔽：**构建树里 11 个改动文件有 10 个是新的、1 个是 14:08 的旧的，编译完全通过**，
wheel 正常产出、正常安装、正常启动，只在运行期炸 —— 而且炸出来的现象看起来像「位置逻辑写错了」。
如果本轮没有那条「positions 缺失就报错」的硬检查，它会**静默地把位置全丢**，然后按 D 侧行号
乱搬字节（因为 `local_positions` 空 ⇒ 传不进 canonical，旧行为是猜一个基址）。

### (26) 硬检查的价值：它把「静默错搬」变成了「当场报错」

失败信息精确到「哪个 group、给了几个 id、给了几个 position」：

```
Cannot convert the request's cache ids into canonical blocks, request_id=chatcmpl-…:
cache group 0 supplies 14 block ids but 0 positions; a block-scoped group has to say
where each id sits in the sequence, because its id does not
```

14 正好是 3492 token / 128 token-per-block / 2（本 rank 的 slice）= 14 个块。**这正是
第 29 轮 (20) 想要的性质**：生产者漏带位置时，宁可当场失败，也不要静默平移。

### (27) 修法与防范（已落地）

1. `gen_sync98.py` 改成 **`base64 -d > file <<'B64EOF_…'` 的 heredoc**，payload 走 stdin，
   彻底绕开 argv 限制（`mk_stage83.sh` 一直就是这么做的，同步脚本当初没跟上）。
2. 脚本头改成 **`set -eu`**，任何一步失败立即非零退出。
3. 每个文件写完**立刻 `md5sum` 与本机 md5 比对**，不匹配就打印 `MISMATCH <rel> want=… got=…`
   并最终 `SYNC INCOMPLETE` + `exit 1`。本轮修复后 15 个文件全部 `ok`。
4. 等待/巡检一律读 **bind-mount 路径**上的日志（(24)）。

> 一般化：**「远端脚本返回了」不等于「远端步骤都做了」。** 凡是 base64/大 payload/多步骤的
> 搬运，必须逐项回读校验（md5/行数/关键字），并把 `set -e` 打开。

### (28) 修好同步后的端到端复测：**字节级 PASS**，并且第一次拿到「id ≠ 位置」的运行期铁证

重编（wheel md5 `3dfac6685b3d5e5d7557f8683f0b3f12`，576,369,247 B；装完 ELF
`e8c14a5ef73f72d1f700458bf3990be8`）→ 干净重启 → 单请求 3492 token：

```
http_code=200  prompt_tokens=3492  completion_tokens=4  time=2.81s
VERDICT: PASS
key    split       total=112 written=112 zero=0 match=112 mismatch=0 absent_in=0
value  split       total=112 written=112 zero=0 match=112 mismatch=0 absent_in=0
index  replicated  total= 84 written= 39 zero=45 match=84  mismatch=0 absent_in=0
```

与第 29 轮 (18) 的 PASS 数字**逐项一致** —— 位置显式化对「首个请求」没有行为改变（回归面干净）。

**prefix cache 场景**：同一对 P/D 上发 A、B、B（P 侧 `--enable_prefix_cache=true`）：

| 请求 | cached_tokens | 输出 |
|---|---|---|
| 1 (A) | 3328 | `enujoulette outnumber` |
| 2 (B) | 3584（命中） | `enujoulette outnumber` |
| 3 (B) | 3584（命中） | `enujoulette outnumber` |

`B1 == B2` 为 True，`VERDICT: PASS`。而带 trace 的同款三轮跑，把 **prefill 自己的 block table**
打印出来后，铁证到手（4 个 prefill rank 完全一致）：

```
call=4（请求 A，14 个局部块）blocks=[1, 2, 3, …, 13, 28]
call=5（请求 B，15 个局部块）blocks=[14, 15, 16, …, 28]
call=6（请求 B，前缀命中）    blocks=[14, 15, 16, …, 28]
```

* A 的**第 14 个块 id = 28**（不是 14）—— 池子把之前释放的 28 号行复用给了它，
  `(id-1)*S` 会得到 54，而它真实位置是 13 → canonical 26。
* B 的**第 1 个块 id = 14**（不是 1）—— 前缀命中/分配顺序让它从 14 号行开始，
  `(id-1)*S` 会得到 26，而真实位置是 0 → canonical 0，**整段平移 13 个位置**。

也就是说：老的 `(id - 1)` 规则在这两个请求上**都会静默错位**（A 错在尾块、B 错在整段），
而位置驱动的实现在同一批请求上给出的解码文本与冷启动**逐字节相同**。这是第 29 轮 (20) 那个
「方向对但写法不对」的评审意见所需要的运行期证据 —— 顺带也说明 `index * split` 更错：
B 这一步的列表从位置 0 开始（`next_transfer_idx=0`）时它恰好对，而 A 这种「尾块在 28 号行」的
情形它同样无能为力（列表里第 13 项的位置是 13，不是 28）。

### (29) 复测同时验证的两件事

1. **共享内存这一环确实是位置丢失的唯一出口**：`forward_shared_memory_manager.cpp` 同步过去之后，
   同一批请求从「0 positions 硬报错」直接变成「字节级 PASS」，没有改任何别的文件。
   （P 侧 `--enable_schedule_overlap=false` 也照样走这个序列化器 —— 不要以为关掉 overlap 就没有这一层。）
2. **`grep -c 'the reserved padding block'` 不再是有效的「这版装上了」凭据**（这一版该串为 0）。
   有效的凭据是每轮新增的错误串，例如本轮：
   `positions; a block-scoped group has to say where each id`（×1）、`reports position`（×1），
   加上 `md5sum $SP/xllm/xllm` 的变化（`141cbe43…` → `e8c14a5e…`）。

## 第 31 轮（2026-09-18 20:10–）：再加两个异构场景（P kv4→D kv2、P kv2→D kv4），卡在**连接期的旧分片门禁**

### (30) 请求与配置

在原来那对 P/D 之外再加两个场景，方向相反：

| 场景 | Prefill | Decode | 设备 |
|---|---|---|---|
| `base` | 4 ranks cp2/tp2/kv_split2 | 2 ranks cp1/tp2/kv_split1 | 0-3 / 4-5 |
| `p4d2` | 4 ranks cp4/tp1/kv_split4 | 2 ranks cp1/tp2/kv_split2 | 0-3 / 4-5 |
| `p2d4` | 4 ranks cp2/tp2/kv_split2 | 4 ranks cp1/tp4/kv_split4 | 0-3 / 4-7 |

`env.sh` 里加了 `PD_SCENARIO` 开关（`base|p4d2|p2d4`），`run_trace_p4d2.sh` /
`run_trace_p2d4.sh` 先 `clean_restart.sh reset` 再跑 `run_trace.sh`。

**一个硬约束，先踩到**：本 build 里 **decode 不能用 CP** ——

```
F master.cpp:489] Check failed: !cp_error.has_value()
                   Model-side CP supports only DEFAULT or PREFILL roles
```

（`xllm/core/distributed_runtime/master.cpp:206`，`cp_size > 1` 只允许 DEFAULT/PREFILL。）
所以最初写的「D 4 ranks cp2/tp2/kv2」连启动都过不去，四个 D rank 一起 FATAL。**decode 侧的
kv_split 只能靠 TP 承担**（cp1/tp<kv_split>），上表已经是修正后的配置。

### (31) 两个新几何在**主机端逐字节**都能过（新增 2 个集成用例）

`pd_route_integration_test` 里加了两条与运行期同几何的用例（cp4×tp1 → cp1×tp2、cp2×tp2 → cp1×tp4），
第一次跑 **失败**，但失败信息是**夹具自相矛盾**，不是路由错：

```
layer 0, role 4, group 1: a whole-resource descriptor holds no head axis and 8 local heads;
publish one span per local head instead
```

MLA 下 linear-state 家族是「一个打包 head 的 whole resource」，而夹具把它的 `global_heads` 写成
`kSsmHeads = 8`：tp8 时 `local_heads = 8/8 = 1` 侥幸成立，tp1 时就变成 8 个 local head 与
whole-resource 描述符冲突（`cache_directory.cpp:118` 的检查）。**修夹具**：MLA 下
`linear_key/value_head_count = 1`、`conv/ssm.global_heads = 1`（与 KV 家族
`enable_mla ? 1 : kv_head_count` 的处理保持一致）。改完：

```
kv_shard_contract 6 · kv_redundancy 12 · pd_route 12 · pd_route_transfer 14
pd_route_integration 6 · cache_directory 23 · reshard_planner 43   → 共 116 全绿，NINJA_EXIT=0
```

其中 `pd_route_integration` 6 条包含新的 `RuntimePrefillFourDecodeTwoSlicesReshard` 与
`RuntimePrefillTwoDecodeFourSlicesReshard`（主机端逐字节校验，含 head class 折叠）。

### (32) **运行期卡点**：decode 侧 `link_sessions` 里的旧分片门禁把这两个几何直接拒了

`p4d2` 起来后 P/D 引擎都到了 `Brpc Server started`，但 **HTTP 58888 一直不监听**，
master 日志：

```
I instance_mgr.cpp:1386] Register a new decode instance, instance name : 11.87.191.83:29994
E instance_mgr.cpp:1337] Fail to link instance during registration, op index 0
E instance_mgr.cpp:580]  Fail to register instance: 11.87.191.83:28994
W instance_mgr.cpp:456]  Ignore heartbeat from unknown instance: 11.87.191.83:28994
```

`Fail to link` = master 让 **D** 去 link **P**（`call_link_instance` → `DisaggPDService::LinkInstance`），
D 侧拒绝，拒绝原因在 D 的引擎日志里：

```
E mooncake_transfer_engine.cpp:798] Remote cache layouts cannot cover local destination:
   source CP/KV-split partitions can only collapse into a CP1 destination with
   either KV-split1 or the matching KV-split size
E disagg_pd_scheduler.cpp:1355] Link instance failed, instance_name: 11.87.191.83:28994
```

代码路径（本机源码可读）：

* `MooncakeTransferEngine::link_sessions()` → `ReshardPlanner::select_sources(remote_manifests, *local_manifest, ...)`
* `reshard_planner.cpp::supports_partition_layout()`：

  ```cpp
  if (same_partition_sizes(source, destination)) return true;
  return destination.cp_size == 1 && destination.cp_rank == 0 &&
         (destination.kv_split_size == 1 ||
          (supports_kv_split_topology(source) &&
           source.kv_split_size == destination.kv_split_size));
  ```

  即**目标必须 cp1，且 kv_split ∈ {1, 源的 kv_split}**。于是：

  * `p4d2`：D cp1 ✓、D kv2 ∉ {1,4} → **拒**
  * `p2d4`：D cp1 ✓、D kv4 ∉ {1,2} → **拒**
  * `base`：D cp1 ✓、D kv1 ✓ → 过（所以之前一直能跑）

这是**旧分片模型的假设**（「源的分片只能塌缩成 cp1 且分片数不变」）。canonical 路由本身就是
为了替代这个假设：`RouteBinder` 按 canonical block 自己算 writer，不需要 `select_sources` 的
rank 对齐推导；但**连接期这道门禁在 canonical 模式下也会跑**，所以这两个几何在搬到任何字节之前
就被挡住 —— 主机端路由（(31)）能过、运行期进不去。

**结论**：要跑通 `p4d2`/`p2d4`，需要让 canonical 模式下的连接期选择不再套用旧分片门禁
（`select_sources` 的产物只用来决定哪些远端 rank 开 ACTIVE session / 走 PLAN_ONLY；canonical 的
writer 由 `RouteBinder` 决定）。这是链路校验语义的改动，需单独评审，不在本轮位置修复的范围内。

### (33) 放宽 canonical 模式下的连接期分片门禁（本轮改动）

`link_sessions` 里那次 `select_sources` 的门禁（(32)）挡住的正是 canonical 路由存在的理由，所以在
**canonical 模式下换成一套新的选择**：

| 位置 | 改动 |
|---|---|
| `ReshardPlanner` | 新增 `select_canonical_sources()`：保留「两侧描述同一种 cache」（schema/version、fingerprint、backend、layout_family）与**各自** `supports_kv_split_topology` 的校验，**不再要求** `supports_partition_layout()`（旧规则：目标必须 cp1 且 kv_split ∈ {1, 源的 kv_split}），返回**全部**远端 rank |
| 同上 | 把结构比较抽成 `validate_layout_family()`，`validate_source_instance()`（legacy 路径）改用它 + 原来的分区规则，**行为不变** |
| `MooncakeTransferEngine::link_sessions` | 增加 `bool canonical_route = false` 参数；为真时走新选择，默认值让既有调用与测试行为不变 |
| `MooncakeKVCacheTransferBase::link_clusters` | 传 `canonical_route_` |
| `mooncake_transfer_engine_test` | 新增 `CanonicalLinkAcceptsASplitTheLegacyRuleRefuses`（P cp4/tp1/kv4 → D cp1/tp2/kv2：legacy 拒、canonical 过且 4 个 peer 全 ACTIVE、session 全开）与 `CanonicalLinkStillRejectsADifferentCacheFamily`（fingerprint 不同仍然拒、不开 session） |

为什么是「全部 rank」而不是精确子集：canonical 的 writer 由 `RouteBinder` 按 canonical block 现算，
连接期无法预知某次请求需要哪些 rank；而 PUSH 方向要求**被选的 P rank 拿到 ACTIVE**（D 通过
`set_remote_peer(..., ACTIVE)` 告诉 P「你可以推给我」），所以宁可全开。代价是每个 D rank 与全部
P rank 建 session（本场景 4×2 / 4×4），可接受。

测试与构建：`t98_build_tests.sh` 的目标里加上了 `mooncake_transfer_engine_test`（该套件里的
`*Npu*` 往返用例会 fork 对端进程、需要两块空闲 NPU，在本容器**环境性失败**，已用
`--gtest_filter=-*Npu*` 排除；与路由无关）。**8 个套件 142 全绿**：

```
kv_shard_contract 6 · kv_redundancy 12 · pd_route 12 · pd_route_transfer 14
pd_route_integration 6 · cache_directory 23 · reshard_planner 43 · mooncake_transfer_engine 26
```

> 注意：本改动**尚未编进 wheel**、尚未做端到端复测（`p4d2` / `p2d4` 的目标就是由它打开）。
> 交接给开发机的清单见 `/export/home/shifengmin.3/workspace/handoff/HANDOFF.md`。

## 第 32 轮（2026-09-18 22:50–23:45）：调试循环远端化（4×提速）、ccache 诊断、remote-id 断言修掉、LSE 算子缺口定位

### (34) 循环远端化：改码/编译/部署/分析都在开发机完成

脚本在 98 上：`/export/home/shifengmin.3/workspace/handoff/remote/`
（`remote_loop.sh` 编排 + `env_inner.sh`/`build_inner.sh`/`install_inner.sh`/`run_inner.sh`/`compare_inner.sh`/`archive_inner.sh`
逐段用 `bash -s` 喂进容器，避免嵌套引号）。

```bash
# 在 98 host 上（一次调用即整批）
cd /export/home/shifengmin.3/workspace/handoff/remote
bash remote_loop.sh all ./apply_remote_id_fix.py     # build → deploy → base/p4d2/p2d4
bash remote_loop.sh run p2d4                          # 只重跑一个场景
```

| 环节 | 旧 | 新 | 实测 |
|---|---|---|---|
| 改码 | 本机改 + 同步 22 文件（1 min，曾静默漏文件） | **远端打补丁脚本**（6 处精确替换，逐处断言） | ~5 s |
| 编译 | `setup.py build` | **`ninja xllm` 增量**（不 reconfigure、不拷 python） | **55 s** |
| 出包 | `bdist_wheel` 2m33s | 不出 wheel，直接换 ELF | 0 |
| 送机 | 576 MB scp | **strip 后 111 MB** | ~30 s |
| 安装 | docker cp + `pip install` ~2 min | 覆盖 `$SP/xllm/xllm` + md5/标记串校验 | ~5 s |
| 分析 | trace 取回本机再比 | **容器内跑 `compare_kv.py`** | ~5 s |
| 保全 | 无（失败日志被下一轮截断，曾丢证据） | 每场景归档 `pdroute83/runs/<scenario>-<ts>/{logs,trace}` | ~1 s |

实测：**patch+编译+strip+送机+安装 = 1m26s**；**三场景合计 ≈ 10 min**（旧 ≈ 39 min）。

自坑两条（已修）：`do_run` 忘了加 `ssh 83` 前缀 → `Error: No such container: fengmin-pdroute91`，
三个"1 秒跑完"的假结果；以及事后补的日志归档。

### (35) ccache：装了也接上了，但缓存打满 + 每轮重做无关步骤

```
/usr/bin/ccache + CMAKE_CXX_COMPILER_LAUNCHER=/usr/bin/ccache   （确实在走）
ccache -s: Cacheable 54289/121659 (44.6%) · Hits 16599 (30.6%) · Misses 37690
           Cache size 5.0/5.0 GiB · Cleanups 986        ← 命中率低的真因
```

已把上限提到 60 GB。慢的其余原因是一次性做 `setup.py build`(4m39s) + `bdist_wheel`(2m33s) +
576 MB stage + pip 重装，以及改了广被包含的 `kv_cache_transfer.h`（virtual 化 ⇒ 重编面放大）。

### (36) 第三个拦路虎已修：canonical 不该按本侧 kv_split 跨步目的端 remote id

`apply_remote_id_fix.py`（98 树内 6 处精确替换）：canonical 模式下**跳过** remote-id 的跨步、
覆盖 `CHECK` 与填充循环（canonical 推送只读 `local_ids`+`local_positions`+对端已发布 layout，
`remote_ids` 根本没人读），legacy 路径不变。效果：

* `base` 仍 **字节级 PASS**；
* `p4d2` 不再撞 `batch_input_builder.cpp:378 (7 vs 14)`；
* 两个新场景各推进一层：`p4d2` → `F kv_transfer_completion.cpp:44 futures_.empty()`（待修，路由侧）；
  `p2d4` → **LSE 算子缺失**（见 (37)，4 个 decode rank 都落了 trace，KV 确实推过去了）。

### (37) LSE 算子缺口：确认**没带上**，且可以不重编 ops 修好

| 检查点 | 结果 |
|---|---|
| `xllm_ops` submodule | **有**该算子（121 个匹配文件），并且已有一份 **LSE-only 预编译包**：`third_party/xllm_ops/xllm_ops/build_sfa_lse_nope/install/vendors/custom_xllm_math/`，其 `op_api/lib/libcust_opapi.so` 里 `aclnnSparseFlashAttentionLse` ×3，带 `sparse_flash_attention_lse` 内核目录 |
| 0911 容器实际安装的 `custom_xllm_math` | 45 个 kernel 目录、**LSE 符号 0**、无 lse 内核目录 |
| 四个 vendor | 全部 LSE=0 |

机理（`xllm/core/kernels/npu/aclnn/pytorch_npu_helper.hpp`）：`EXEC_NPU_CMD` → `get_op_api_func_addr()`
**按符号逐个目录搜**：先 `g_custom_lib_path`，再 `g_default_custom_lib_path`（= `opp/vendors/config.ini`
的 `load_priority` 每项拼 `/op_api/lib/`），逐个 `dlopen(<dir>/libcust_opapi.so)` + `dlsym`，命中即返回。
**所以 LSE-only 的库可以独立当一个 vendor 加进去，不影响其它算子。**

修法（不需重编 ops）：

1. 打包 `build_sfa_lse_nope/install/vendors/custom_xllm_math` → `docker cp` 进 0911 容器 `opp/vendors/xllm_sfa_lse/`；
2. 在 `opp/vendors/config.ini` 的 `load_priority` 末尾追加 `xllm_sfa_lse`；
3. 把这步固化进 `pdroute83/env.sh`（现有那段只按 `load_priority` 铺 `LD_LIBRARY_PATH`，没改 `load_priority` 本身）；
4. `bash remote_loop.sh run p2d4` 复跑（~4 min）。

### (38) 恢复点

* 已推：`pd-routing-s0s1` @ `9436311ab`（位置修复 / 文档 / 连接期门禁放宽）。
* 本机 clone 现有**未提交**：引擎 `set_canonical_route` 标志与传递（`mooncake_transfer_engine.{h,cpp}`、
  `mooncake_kv_cache_transfer.{h,cpp}`、`kv_cache_transfer.h`）+ `batch_input_builder.cpp` 的 remote-id 修复
  （从 98 树拉回，md5 `735a8543…` 一致）。
* 容器内已装二进制：ELF `3b0e5411…`（含引擎标志那版）→ 之后 `remote_loop` 换成 strip 版 `d1aea2bc…`；
  wheel `c915dd57…`。**注意：现在容器里是 strip 过的 ELF，功能等价。**
* 单元测试：8 套件 **142 全绿**（含 `mooncake_transfer_engine_test` 26 条 canonical-link 用例；NPU 往返用例
  用 `--gtest_filter=-*Npu*` 排除，属环境性）。
* 下一步顺序：①(37) 装 LSE → 复跑 `p2d4`；②修 `p4d2` 的 `kv_transfer_completion.cpp:44`；
  ③三场景矩阵全 PASS → 提交推送。

## 第 33 轮（2026-09-18 23:33–）：LSE 算子装好、`p2d4` 端到端**字节级 PASS**；`p4d2` 的 `futures_.empty()` 被证明是**二次症状**

### (39) LSE 算子落地：不改 `config.ini`，改用 `ASCEND_CUSTOM_OPP_PATH` 指向一份**旁挂** vendor 树

(37) 里原本计划「装成新 vendor + 追加 `load_priority`」，实际做法换成了更安全的一种：

* 98 构建容器里把 LSE-only 预编译包（`third_party/xllm_ops/xllm_ops/build_sfa_lse_nope/install/vendors/custom_xllm_math`，
  整树 tar 出来 2.8 MB，含 `op_impl/ai_core/tbe/kernel/ascend910_93/sparse_flash_attention_lse/*.o` 与
  `binary_info_config.json`）→ `scp` 到 83 → 解到 **`pdroute83/lse_vendor/`**（bind mount 内，容器内外同路径）；
* `pdroute83/env.sh` 里 `export ASCEND_CUSTOM_OPP_PATH="$BASE/lse_vendor"`（只在目录存在时）；
* **不动** `opp/vendors/`：覆盖已装的 `custom_xllm_math` 会连带丢掉 45 个 kernel（其中就有
  `glm_next_transformer` 依赖的 23 参数 `aclnnSparseFlashAttention`），也不去改 `load_priority`。

为什么这样就够（也是这次特意确认的两点）：

| 事实 | 证据 |
|---|---|
| `ASCEND_CUSTOM_OPP_PATH` 的搜索顺序在 `load_priority` **之前** | `pytorch_npu_helper.hpp:140-215`：`get_custom_lib_path()` 读该变量、每项拼 `/op_api/lib/`，`get_op_api_func_addr()` 先遍历它再遍历 `g_default_custom_lib_path`，**按符号** `dlopen`+`dlsym`，命中即返回 |
| 它不可能遮蔽别的算子 | `nm -D --defined-only lse_vendor/op_api/lib/libcust_opapi.so` 只有 4 个导出：`aclnnSparseFlashAttentionLse{,GetWorkspaceSize}`、`aclnnInnerSparseFlashAttentionLse{,GetWorkspaceSize}`（其余是 libstdc++ 弱符号）；而已装 `custom_xllm_math` 的 `aclnnSparseFlashAttention`（无 Lse）不受影响 |

回滚 = 删掉那一行 export。生效证据：`p2d4` 请求 `http_code=200`，**4 个 decode rank 全部落 trace**
（此前 4 个 rank 都是 `F sparse_flash_attention_lse.cpp:153 ... not found`）。

### (40) `p2d4`（P cp2/tp2/kv2 → D cp1/tp4/kv4）：**K/V 四个 rank 全部字节级一致**

判据取**最早那帧**（`tag=before call=0`，decode 自己还没走一步）：

| 族 | 结果 |
|---|---|
| key / value（split） | 4 个 decode rank 各 **28/28 字节一致**，`mismatch=0 absent_in=0`（28 = 7 块 × 4 层） |
| index（replicated） | 写进去的 **12/12 行**逐字节等于规则预测的源行；`mismatch=0` |

源行分布正是异构重分片该有的样子：D rank3（slice3）的 7 块分别来自 P 的 `2/4/6/8/10/12/14`，
D rank2（slice2）来自 `1/3/5/7/9/11/13`；`discovered split-slot mapping` 里 7 条预测全部命中，
row 0 作为池子保留的 padding 块按约定不参与。

### (41) 顺手修掉比较器的一个**假 mismatch**：目的端 padding 行被算作「搬错了」

上一轮看到的 `index total=21 written=12 zero=9 match=12 mismatch=9` 是**比较器口径错**，不是路由错：

* 目的端 kv_split 4 > 源端 2 时，目的端 index 池**最前面 `S_D` 行**对应 `canonical = -3,-2,-1`，
  根本不是本请求的块，没人写（实测全零）；
* 旧逻辑只问「目的端这一行的摘要在不在源端」，而这些零行的摘要在源端**零行**上找得到，
  于是落进 `mismatch` —— 一个传输**无论怎么做都修不掉**的 mismatch（9 = 3 行 × 3 层）。

改法（`pdtrace/compare_kv.py`，两份都更新：本机 master 与容器内 `pdroute83/tracefiles/compare_kv.py`）：
两个族都先判 `half < 0`（canonical 块为负）→ 记 `abs_oos`（不属于请求），不再参与 match/mismatch。
修完：

* `p2d4` → `key/value 28/28`、`index 12/12`、`mismatch=0 absent_in=0` → **VERDICT: PASS**；
* `base` 用同一版复评**数字不变**（`key/value 112/112`、`index 84/84`、`mismatch=0`）→ 说明判据没有被放松。

> 仍然留给模型侧的问题（第 32 轮就记着，本轮数据更具体了）：index 族在源端**每个 slice 的副本内容不同**
> （`byte-identical on all 4 prefill ranks=0`；P rank0 与 rank2 同一行的摘要不同）。路由是「目的端 canonical 块
> ← 源端同 canonical 块」地各推一行，本次逐字节全对，但「只推一个 writer 的一份副本」是否覆盖 decode
> indexer 需要的全部语义，需要模型侧确认。

### (42) `p4d2` 的 `F kv_transfer_completion.cpp:44` 是**二次症状**，不是记账 bug

新日志（`runs/p4d2-234654`）把因果链摆清楚了 —— prefill rank0（device 0）先在**模型 forward 里**炸：

```
[ASSERT] gather_v3_base.h:137 Assertion `(0 <= val && val < this->gxSize_)' Index 3480 out of range[0 3480)!
[rank0] NPU function error: call aclnnIndexSelect failed, error code is 507035
        Kernel task happen error, retCode=0x31, [vector core exception]
```

设备侧异常让 `forward()` 抛异常 → 栈回退到 `LLMWorkerImpl::step_internal` 时 `KVTransferCompletion`
析构；而它只在 `wait()` **成功后**才 `futures_.clear()`（`folly::collectAll(...).get(60s)` 超时是抛异常），
于是析构里那条 `CHECK(futures_.empty())` 先 abort，把真正的错误盖住。三个要点：

* **不是 60 s 超时**、也不是登记缺失：`forward` 抛异常才是因；上一轮「canonical 推送完成记账没登记」的判断作废；
* **与 LSE 无关**：装上 LSE 后**逐字复现**（同一 `3480 out of range[0 3480)`、同一 device 0）；
* `3480 = 4 × 870`：形状就是「CP 分片边界」（`(cp_rank+1)*chunk` 这种闭区间端点被当索引用），
  也就是**长 prompt × 4 路 CP 切片**才触发。

归因实验（本轮给循环加了 `run <scenario> [route] [repeats]`）：

* **legacy 路由跑不了这个几何** —— 旧门禁 `supports_partition_layout` 要求目的端 kv_split ∈ {1, 源 kv_split}，
  `D kv2` 不在其中，连接期直接拒：表现是「P/D 引擎都 `Brpc Server started`、HTTP 58888 一直不监听、
  请求 `http_code=000`、service 侧只打印 `Ignore heartbeat from unknown instance: 11.87.191.83:28994`」，
  与第 31 轮 §(32) 记录的现象一致。**所以「同一几何跑 legacy」这条路无法用作归因对照**：
  这个几何在本次改动之前**从未被跑过**。
* 可用的对照是把 prompt 缩短：`p4d2` + `SMOKE_REPEATS=20`（≈580 token，仍跨 2 个以上逻辑块）。
  若短 prompt 下逐字节 PASS，则 ① 该几何的**数据面**在 canonical 路由下是正确的；
  ② `3480` 那个设备异常是**长序列 × CP4** 的独立问题（模型侧）。

### (43) 循环的两个小改进

* `remote_loop.sh run <scenario> [route] [repeats]`：`route` 经 `docker exec -e PD_ROUTE=` 注入
  （`env.sh` 用 `${PD_ROUTE:-canonical}`，环境变量优先），`repeats` 经 `-e SMOKE_REPEATS=` 注入
  （`run_trace.sh` 读 `${SMOKE_REPEATS:-120}` 传给 `smoke_long.sh`）；归档名带后缀
  （`p4d2-legacy-…` / `p4d2-r20-…`），两个对照不会互相覆盖。
* `run_one98.sh` + `handoff/remote/scenarios.txt`：一次 rrun 跑一整批（每行 `scenario[:route[:repeats]]`），
  失败也继续跑后面的场景，最后打印 `MATRIX DONE: n failure(s) of m`。

### (44) `p4d2` 的归因结论：**数据面正确**，长 prompt 的失败在模型侧

短 prompt（592 token，`SMOKE_REPEATS=20`）下跑 canonical `p4d2`：

| 族 | 结果（2 个 decode rank 各一份） |
|---|---|
| key / value（split） | **12/12 字节一致**（3 块 × 4 层），`mismatch=0 absent_in=0` |
| index（replicated） | 写进去的 **6/6** 字节一致，另 3 行是目的端 padding 前缀（`abs_oos`） |

→ `VERDICT: PASS`。所以：

1. `P cp4/tp1/kv4 → D cp1/tp2/kv2` 这条几何的**数据面（canonical 路由）是正确的**；
2. 3492-token prompt 下那个 `Index 3480 out of range[0 3480)` 是**长序列 × CP4** 的独立问题
   （`3480 = 4 × 870`，正是 CP 分片边界的形状；同一个长 prompt 在 P cp2 的 `base` / `p2d4` 上都过）；
3. legacy 路由**无法**作为对照（连接期旧门禁拒 `D kv2`，见 (42)），这个几何在本轮之前也从未跑过。

### (45) 踩到的坑：被 kill 的调试轮会在容器里留下**孤儿 worker 树**

现象（第一次跑短 prompt 时）：同一个几何下 `logs/prefill/rank_2.log` 里写的是 `pd_route=legacy`，
其余 rank 都是 `canonical`；链接被 legacy 门禁拒
（`SetCachePeer failed: source and destination CP/KV-split partitions are unsupported`）→ HTTP 58888 不开 →
300 s 超时。机理：

* 我 kill 掉的只是**本机**的 `rrun`/ssh，容器里的 worker 没死；
* 下一轮 `clean_restart.sh reset` 的 `stop_workers.sh` 只匹配它自己那一代，孤儿进程仍占着 rank2 的
  `28996/46102/15304`，于是新一轮的 rank2 起不来，**旧进程（legacy）应答了 RPC**；
* 日志文件因此有两个写者：新进程 `truncate` 后从 0 写，旧进程按自己的 fd offset 继续写，
  中间留下 NUL 空洞 —— `grep` 会报 `binary file matches`，这本身就是「同一路径两个写者」的指纹。

修法：新增 `handoff/remote/hard_reset_inner.sh`（kill 所有 `site-packages/xllm/xllm` + `xllm_master_serving`
+ `etcd --name pdroute83`，清 `logs/*`、`trace/`、`/tmp/etcd_pdroute83`，最后逐端口复查），
并让 `run_inner.sh` 在**每个场景之前**都先跑它。证据：hard reset 前容器里还有
**14 个 xllm 进程 + 1 service + 1 etcd**，清干净后同一几何一次通过。

> 教训：远端跑长任务时，取消必须落到**容器里的进程**上（或下一轮做 hard reset），
> 否则「上一轮的进程」会伪装成「这一轮的 bug」。

### (46) 三场景矩阵（本轮最终证据，00:13–00:22，全绿）

| 场景 | 数字 | 判定 |
|---|---|---|
| `base`（P cp2/tp2/kv2 → D cp1/tp2/kv1，3492 token） | key/value **112/112**、index **84/84**（39 写、45 零） | PASS |
| `p2d4`（P cp2/tp2/kv2 → D cp1/tp4/kv4，3492 token） | key/value **28/28**（4 rank）、index 写进去 **12/12**（9 零、9 abs_oos） | PASS |
| `p4d2`（P cp4/tp1/kv4 → D cp1/tp2/kv2，592 token） | key/value **12/12**（2 rank）、index **6/6**（3 abs_oos） | PASS |

`MATRIX DONE: 0 failure(s) of 3`，三个场景的 `mismatch` 与 `absent_in` 全为 0，
归档在 `pdroute83/runs/{base-001306,p2d4-001552,p4d2-canonical-r20-001909}`。
仍开着的一件事：`p4d2` 的**长 prompt**（3492 token）会撞 prefill forward 的设备异常（模型侧，(42)/(44)）。

### (47) 恢复点

* 代码：本轮**没有改产品代码**（LSE 是环境侧、比较器与循环是调试工具），
  提交只有工作日志；分支 `pd-routing-s0s1`，前缀提交 `17dc81212`。
* 运行环境：`pdroute83/lse_vendor`（LSE 算子）+ `env.sh` 的 `ASCEND_CUSTOM_OPP_PATH`；
  `run_trace.sh` 支持 `SMOKE_REPEATS`；`hard_reset.sh` 已装进容器并接入 `run_inner.sh`。
* 比较器：本机 `~/work/pdtrace/compare_kv.py` 与容器内 `pdroute83/tracefiles/compare_kv.py`
  同 md5 `b8f18e66baeafc287eba93e2ba3f28ff`。
* 下一步候选：`p4d2` 长 prompt 的 `3480` 交给模型侧（CP4 × 长序列的 index 越界）；
  canonical 侧的遗留债见第 32 轮清单（`pull_kv_blocks_canonical` 桩、A1/A3/A4、C1/C2/C3）。

## 第 34 轮（2026-09-19 18:00–）：index 族「replica」的真实语义 —— 只有不切序列的实例才是副本

### (48) 需求（用户给定，作为本轮的设计前提）

* **只开 kv split**：index 池是 **replica**（每个 rank 持有整条序列的 index）。
* **同时开 CP 和 kv split**：index 池是 **切分存的**（每个 rank 只持有自己那一片）。
* 所以路由必须兼容 **prefill = CP + kv split、decode = 仅 kv split** 这种**两侧语义不同**的异构对。

这条把第 33 轮留下的疑问（「源端 index 每片不同是 CP 导致的吗」）钉成了设计输入：
**是**，而且路由必须按「源端分片、目的端副本」来搬。

### (49) 实测：index 池的物理 row 公式（本轮把语义钉死的证据）

tracer 新增 `XLLM_PD_TRACE_SCAN=N`（`_pd_trace.py`）：把**绝对 row 0..N-1** 全哈希一遍，
而不是只哈希从 batch 元数据推出的 row 集合。理由：**用与传输同一个假设构造出来的比较器，
永远只会同意传输**。跑 `p2d4`（P cp2/tp2/kv2 → D cp1/tp4/kv4，3492 token，14 个 logical block），
归档 `pdroute83/runs/p2d4-scan64-183843`：

| 侧 | 元数据 | index 非零 row | key 非零 row |
|---|---|---|---|
| 源 P（rank 0..3） | `has_kv_shard=1`、`kv_split=2`、`kv_split_rank=0,0,1,1` | **0..14**（row 0 = 未拥有 token 被 `clamp_min(0)` 写进来的垃圾；1..14 = 请求的 14 个 block） | 1..14 |
| 目的 D（rank 0..3） | `has_kv_shard=0`、`kv_split=1`、`kv_split_rank=0` | **4..16**（canonical 0..12） | 1..7 |

由此得到两侧的 row 公式（S = 该实例的 kv_split，c = canonical block）：

```
源（CP 实例，index 走 local_slot_mapping，被 owner 掩码裁过）  row = c / S + 1     ← 与 K/V 同 row 空间（紧凑）
目的（无 CP 实例，index 直写 pool slot / 128）                row = c + S         ← 每个 canonical 块一行（副本）
```

四个源 rank 的非零 row 集合**完全相同**，但内容按 DCP group 分成两类（{0,1} 与 {2,3}，
`kv_split_rank` 也是 0,0,1,1）：**同一个 row 在不同 rank 上装的是不同 slice 的 token**
——即「切分存」的物理形态。目的地四个 rank 的 index 内容则**完全一致**（真副本）。

### (50) 用 owner 检查把旧行为钉成可判定的失败

比较器原来的 replicated 分支只问「这一行的字节在源端出现过吗」，一个 writer 的分片和正确的搬运
**同样满足**它 —— 这正是第 33 轮 `p2d4` 判 PASS 而 index 实际是错的机制。
新增 `pdtrace/index_owner_check.py`：按上面的公式要求
`dst[row = c + S_D]` 必须等于**拥有 c 的那个源 rank** 的 `src[row = c / S + 1]`。
在**修复前**的 scan 归档上：

```
source ranks [0,1,2,3] slices [0,0,1,1]
request canonical blocks: 28 (14 logical x 2)
destination rows checked=52 match=0 mismatch=52      ← 52 = 13 行 × 4 rank
  block 0: want 9a694d63 (owner row 1) got cf5b9f3c
  block 1: want ae11b904 (owner row 1) got d6e7a547
destination blocks with no row at all: n=15 [13..27]  ← 一半请求根本没到
```

`got` 的字节正是源端 row 2/3/4 的内容 —— 旧路由把 replica 公式**同时**用在两侧
（源 row = c + S_P、目的 row = c + S_D），并且每个 canonical 块只挑一个 writer，
于是目的地拿到「错位的行 + 一个分片冒充整条序列」，而且 `c ≥ 13` 的块一个都没到。

### (51) 修法：让**实例**决定这族是不是副本（两处 gate，共享一个判定）

* `group_keeps_whole_sequence(topology, group) = group.full_sequence_replica && topology.cp_size <= 1`。
  声明说的是「模型需要池子里有什么」（DSA top-k 要读整条序列）；**只有不切序列的实例（CP=1）才能这样持有**。
* `KvRedundancy::derive`：`full_sequence_replica_` 改用该判定 ⇒ CP 实例上 index 族按
  **split = 配置的 kv_split** 派生（与 K/V 完全一样：`owns(c, slice) = c % split == slice`，
  `replicas_of` = 同 slice 的 TP 副本）。
* `RouteBinder::peer_row`：replica row 公式（`block + kv_split`）只在判定为真时使用，
  否则落到 split 族的 `block / split + 1`。

于是异构对自然成立：**源端从「拥有该 canonical 块的那个 rank」取紧凑 row，
目的地把每个块写进自己的副本 row（`c + kv_split`），而目的地每个 replica rank 都要收全**
（`replicas_of` 本来就 fan-out）。这条路径不需要新的传输语义：两侧的 row 公式各按自己的实例算，
正是 `peer_row(local/remote)` 的既有设计。

单测（98 上跑）：
`KvRedundancyTest.FullSequenceReplicasSplitOnAShardedInstance`、
`PdRouteTest.IndexerPoolIsAShardedSourceAndAReplicaDestination`、
`PdRouteTest.MockIndexTransferCopiesCanonicalBlocksByteForByte` +
`MockIndexTransferIsDiscriminating`（后者把目的端 row 基址挪一行时必须报错，否则测试是空的）。
`kv_redundancy_test` 12/12、`cache_directory_test` 23/23、`pd_route_test` 14/14 通过。

### (52) 环境侧三件事（都影响可复现性，值得写下来）

1. **模型存储掉了**：83 上 `/mnt/cfs/9n-das-admin/llm_models` 的 NFS 挂载在 14:13 消失
   （容器里看到的是本地空目录），cfs 上 `GLM-5.2-W8A8-EcoTech-4layers` 变成 `1080:1080 drwxr-x---`
   （我们读不到，容器 root 能读）。已按 98 的挂载参数把 NFS 挂回来（`11.88.0.24:/cfs/9n-das-admin/llm_models`），
   并**重启容器**让 bind mount 重新指向 NFS —— bind mount 抓的是目录 inode，只重挂宿主机**不会**进容器。
2. **NPU「被抢占」其实是自己的残留**：prefill 在 `load_weights` 抛
   `torch.OutOfMemoryError ... 1.90 GiB free`（61 GiB 卡），`npu-smi` 里 0–9 全是 60+ GiB；
   而 `hard_reset.sh` 杀不干净（它自己报「端口 5389/58889 仍 LISTEN」）。**重启容器后 NPU 0–7 立刻
   `No running processes`** —— 那些显存是我自己上一轮/上上轮留下的 worker 占的。
   结论：**新的 hard reset = 容器 restart（~10 s）**，比 pkill 可靠；应接进 `remote_loop.sh run`。
3. **构建树的 git 是坏的 worktree**：`xllm-dcp-fp32/.git` 指向
   `/export/home/shifengmin.3/workspace/xllm-coding/xllm/.git/worktrees/...`，该主仓已不在，
   所以构建树里 `git status/diff` 全部 `fatal: not a git repository`。
   单测那次输出里看到的 22 files/892 insertions 是**相对它自己那个陈旧 HEAD**的差，
   不代表有未提交改动。结论：**提交只能以本机 clone（`~/work/xllm`，分支 `pd-routing-s0s1`）为准**。

### (53) 本轮改动的文件与恢复点

产品代码（3 个）：`kv_redundancy.h`（新增判定声明 + 文档）、`kv_redundancy.cpp`（判定实现与
`derive` 的 C2 分支）、`route_binder.cpp`（`peer_row` 选公式 + 注释）。
测试（2 个）：`tests/core/framework/kv_cache_transfer/{kv_redundancy_test.cpp,pd_route_test.cpp}`。
工具（本机 + 98 + 83）：`pdtrace/{_pd_trace.py(新增 SCAN 模式),index_owner_check.py,index_fill_pattern.py,
index_rank_classes.py,inspect_dump.py,raw_rows.py}`、`remote/{remote_loop.sh(可传 XLLM_PD_TRACE_SCAN),
apply_index_sharded.py,fix_canonical4.py,unit_index_sharded.sh}`。

修复后的 ELF：98 构建 `1325c61d5d43772cfb93d233a024ce96`（strip 后 111 MB）。

## 第 35 轮（2026-09-19 19:25–）：修好的是**验证链**，不是数据面 —— 三处工具 bug 让「已修好」看起来仍然是坏的

第 34 轮把产品代码修好（`037539cae`）并且用 SCAN 模式实测到 `match=112/mismatch=0`，但三场景矩阵重跑仍然
`MATRIX DONE: 3 failure(s) of 3`。本轮的全部工作是把这条**验收链**修到可信：三处 bug 都在工具里，一处
让判定根本不可见，两处让判定用错了 row 空间。

### (54) 症状：`VERDICT: FAIL` 的签名换了，而且**没有** `OWNER VERDICT` 行

19:15–19:22 三场景串行重跑，每个场景都 FAIL，签名是
`index: rows found=0 byte-identical on all 4 prefill ranks=0`（p2d4 / p4d2）与
`index ... mismatch=81`（base）。K/V 侧 `mismatch=0` 没有变。最关键的一点：日志里**一次都没有出现**
第 34 轮刚接进循环的 `OWNER VERDICT` 行 —— 也就是说，那个专门用来抓「比较器抓不到」的 gate，
在这三次运行里**根本没有执行**，而报告里完全看不出来。

### (55) 工具 bug 1：`compare_inner.sh` 顶上的 `set -e` 把 owner 检查吞掉了

`compare_kv.py` 判 FAIL 时 `exit 1`；脚本第 4 行是 `set -eu`，于是第 9 行那条命令一返回非零，
整个脚本**当场结束**，后面第 21 行起的 owner 检查从未运行。而 `do_run` 用 `|| true` 收输出，
所以既没有报错也没有痕迹。教训很具体：**新加的判定如果挂在旧判定之后，就必须确认旧判定失败时它还会跑**。

修法（`remote/compare_inner.sh`）：去掉 `set -e`，两个检查都跑、各自记 `rc`，最后
`[ "$kv_rc" = 0 ] && [ "$owner_rc" = 0 ]` 作为脚本退出码；`OWNER VERDICT: PASS|FAIL` 一定打印。

### (56) 工具 bug 2：目的端 index 池的 row 空间 = **命令行配置**的 kv_split，不是 metadata 的

把三份归档的 row 空间做了一次普查（`remote/extent83.sh`，只打印每 rank 非零 row 的区间）：

| 场景 | 源 P（分片，compact） | 目的 D（副本） | D 的 metadata.kv_split | D 的 `--kv_split_size` |
|---|---|---|---|---|
| base | K/V + index 1..14 | index 1..28 | 1 | 1 |
| p2d4 | 1..14 | index 写入 **4..31**，trace 只 hash 了 **1..7** | 1 | **4** |
| p4d2 | 1..2 | index 写入 **2..7**，trace 只 hash 了 **1..3** | 1 | **2** |

结论：目的端 index 行 = `logical_block * S_cfg + slice`，`S_cfg` 来自**进程命令行**的 `--kv_split_size`。
decode 侧 metadata 报 1 是因为那一侧的 split 骑在 TP 上（`has_kv_shard=0`），而池子是按配置的 S 分配并写入的。
旧 tracer 用 metadata 的 1 展开，于是在 p2d4 上只看 rows 1..7 —— 真正被填的 4..31 里只观测到 4..7，
比较器于是打印 `rows found=0`，**看起来像路由坏了**，其实是被观测的行选错了。

修法（`pdtrace/_pd_trace.py`）：新增 `_configured_kv_split()`（读 `/proc/self/cmdline`，`kv_split_size`
本来就在 `_CMDLINE_KEYS` 里）；非分片实例的 replica 行集改为
`{base * S_cfg + slice}`（不再与 logical 行求并，避免把 K/V 的行混进 index 的 row 集）；
probe 里记录 `kv_split_cfg` 供下游核对。

单测（新增 `pdtrace/test_pd_trace_rows.py`，纯 python3、不需要 worker）：钉住 6 组断言 ——
p2d4 目的端（split 1..7 / replica 4..31）、base 目的端（两者都是 1..28）、p2d4 分片源端（两者都是 1..14）、
cp1 源端（replica 2..5）、无命令行时回退到 metadata 的 4（replica 4..11）。`ROWS: PASS`。

### (57) 工具 bug 3：比较器的 replicated 分支只会**一种**源端形状

旧公式 `want_row = half + S_P` 假设源端也是「整条序列副本」。但 prefill 开了 CP 时源端是**分片**的：
正确的源行是 `half // S_P + 1`，而且只在该 canonical 块的 owner slice 的 rank 上。这正是第 34 轮
产品修复所确立的契约，比较器没有跟上，于是把**正确的字节**判成 `mismatch`。

修法（`pdtrace/compare_kv.py`）：
* 新增 `prefill_shards_index(records, prefill)`：**从 trace 里读** `has_kv_shard`（模型自己用的那个标志），
  只有在 trace 早于该探针时才回退到声明的 `cp_size > 1`；
* 分片源端走 owner 公式（row + owner slice 双条件），与 `key/value` 的 slice 语义一致；
* **目的端在请求范围内全零 = 丢块**，不再只记 `zero` 而不判负 —— 但只在该行**源端确实写过**时才算丢失
  （新增 `source_rows()` 提供 (rank,row) 非零集合）。否则它是尾部半块的两侧 padding：
  p4d2 短 prompt 的 canonical 5 就是这种（592 token 落在 canonical 4 里，canonical 5 谁都没写）。

离线复跑三份归档的**传输前** dump（`remote/validate83.sh`，强制 `--decode-tag before --decode-call 0`）：

| 场景 | index | K/V | 结论 |
|---|---|---|---|
| base | 84/84 match | 112/112 match | `VERDICT: PASS` |
| p2d4 | 12/12 match | 28/28 match | `VERDICT: PASS` |
| p4d2 | 6/6 match | 12/12 + 4 padding | `VERDICT: PASS` |

（p2d4 / p4d2 这里只有旧 tracer 观测到的行参与评分；补齐观测是 (56) 的事。）

### (58) 工具改进 4：owner 检查的范围改由**目的端自己的 block table** 定义

旧版把范围算成源端的 `block_table_len × S_P`，并且只比「dump 里已经有的行」。旧 tracer 只 hash 了 4..7 时，
它会报「16/16 全对」而**放过 24 个根本没观测到的 canonical 块** —— 一个无法确认的检查比没有检查更危险。

新版（`pdtrace/index_owner_check.py` 重写）：对目的端 `block_table` 的每个条目 × 每个 slice 都要求有行，
按下面几类报，**除 padding 外都判负**：
* `mismatch`：有行但字节不等于 owner 的 `c // S_P + 1`；
* `not written while the owner holds it`：hash 到了但全零，而**源端 owner 行有数据** → 传输丢块；
* `not hashed by the trace`：trace 根本没观测这一行（无法确认，不算通过）；
* `source row absent`：源端 owner 行本身不存在；
* `empty on both sides (padding)`：**两侧都是零** —— 尾部没填满的块。p4d2 的 592 token 落在 canonical 4 里，
  canonical 5 谁都没写过；这类只报告、不判负（否则会把一次正确运行判死）。
源端 dump 取 `after` 的所有 call 求并（chunked prefill 一次 prompt 分多次），目的端取最早的 `before`。

离线结果：

| 归档 | 结果 |
|---|---|
| `p2d4-scan64-183843`（**pre-fix**） | canonical 2 字节不等 + 15 个块 `not written while the owner holds it` → FAIL（能判死旧行为） |
| `p2d4-scan64-184539`（post-fix） | `checked=112 match=112` → PASS |
| `base-191544` | `checked=56 match=56` → PASS |
| `p2d4-191909` / `p4d2-canonical-r20-192203`（旧 tracer） | 已观测的行 16/16、4/4 全 match，但 24 / 4 行 `not hashed` → FAIL（等新 tracer 观测补齐，见 (60)） |

### (59) 本轮改动的文件与恢复点

* 工具：`pdtrace/{_pd_trace.py,compare_kv.py,index_owner_check.py,test_pd_trace_rows.py(新)}`、
  `remote/{compare_inner.sh,matrix35.sh(新),validate83.sh,extent83.sh,rows83.sh,state98.sh,state83.sh,elf83.sh,ports83.sh}`。
  同步到 98 `handoff/remote/` 与 83 `tracefiles/` 的 md5：
  `_pd_trace.py=b625f8f0d5a0520d6c2b468b0b1a953d`、`compare_kv.py=fb3d996187ed0a4a004c664d19994c9a`、
  `index_owner_check.py=f44e4737b8fce4af93c70f199b632011`、`compare_inner.sh=14798f6c7003c78347bbd8585d04bffa`。
* 产品代码：**本轮无改动**。容器内 ELF 仍是第 34 轮那份
  `1325c61d5d43772cfb93d233a024ce96`（111,643,224 字节），marker
  `destination cp_size must be divisible by kv_split_size` 计数 1。
* 另记：`trace_install.sh` 每次 run 都会把 `$BASE/tracefiles/_pd_trace.py` 覆盖安装到
  site-packages 的 `xllm/python/_pd_trace.py`，所以**换了 tracer 不需要重编、只需要重跑**；
  19:37:18 的安装时间戳就是本轮 base 场景用上新 tracer 的证据。

### (60) 在线验收：三场景 `VERDICT: PASS` **且** `OWNER VERDICT: PASS`

`remote/matrix35.sh`（98 上串行、每个场景前 restart 容器）19:36:56–19:46:23 跑完 base / p2d4 / p4d2-canonical-r20，
19:52–19:55:58 单独重跑 p4d2：

| 场景 | 归档（83 `pdroute83/runs/`） | K/V | index | 比较器 | owner 检查 |
|---|---|---|---|---|---|
| base | `base-194004` | 112/112 match | 84/84 match | PASS | PASS（56/56） |
| p2d4 | `p2d4-194329` | 28/28 match | 112/112 match（28 canonical × 4 rank） | PASS | PASS（112/112） |
| p4d2 (r20) | `p4d2-canonical-r20-195558` | 12/12、8/8 + 4 padding | 15/15 | PASS | PASS（10/10，1 padding） |

两点口径说明（都写进比较器/检查的输出里，避免下次误读）：

* 比较器最终的 verdict 取**最早通过的那个 dump** = `before call=0`，即传输刚落下、decode 引擎还没走第一步。
  之后的 dump 会因为 decode 自己写的行而 `absent_in`（base：每个 rank index 3 行 + K/V 4 行/layer 的
  `after call=0` 就是这种），那不是传输的责任。
* p4d2 的第一次在线跑（`p4d2-canonical-r20-194623`）owner 检查报 `not written [5]`：592 token 的 prompt
  让 canonical 5 两侧都是零。这不是丢块而是尾部 padding，于是 owner 检查也对齐成
  「源端 owner 行也没数据 → `empty on both sides (padding)`，只报告不判负」（与 (57) 里比较器的口径一致），
  再单独重跑 p4d2 得 `VERDICT: PASS` + `OWNER VERDICT: PASS`；base / p2d4 无需重跑（新口径对它们是同一范围）。
  **判据强弱没有被削弱**：同一份 pre-fix scan 归档在新口径下仍然 FAIL（canonical 2 字节不等 +
  15 个块 `not written while the owner holds it`）。

### (61) 第 35 轮的结论（写给下一次）

* 产品代码：第 34 轮的修复经三场景字节级验收成立，**本轮没有再改产品代码**；容器里仍是
  `1325c61d…` 那份 ELF。
* 三条验证器口径：
  1. **row 空间要从实例的配置（命令行 `--kv_split_size`）读**，metadata 是模型的视角 —— decode 侧
     那里是 1，按它展开会少观测 24/28 个块，并把「观测不到」误报成路由错误；
  2. 一个判定说「不通过」时，必须能分清 **mismatch / 该写没写 / 没观测到 / 两侧都空(padding)** 四件事，
     否则「无法确认」会被读成通过、「padding」会被读成失败；
  3. 新加的判定挂在旧判定之后时，先确认旧判定失败时它还会执行（`compare_inner.sh` 的 `set -e` 就吃掉了
     第 34 轮刚加进去的 owner 检查，且完全不留痕迹）。
* 仍然开放（模型侧，未动）：p4d2 **长 prompt**（3492 token）prefill forward 的
  `[ASSERT] gather_v3_base.h:137 ... Index 3480 out of range[0 3480)` → `aclnnIndexSelect failed, error code is 507035`。

## 第 36 轮（2026-09-19 20:00–）：新增 **两侧都不开 CP** 的纯 kv-split 异构场景，并把 `3480` 的触发条件钉死

第 35 轮已经证明「prefill 开 CP + kv split、decode 只开 kv split」这条异构对是对的（p2d4）。
本轮按用户要求把 **prefill 的 CP 也关掉**，先跑通「纯 kv split 异构」（两侧都只有 kv split，
只是宽度不同），因为它正是「只开 kv split 时 index 是 replica」这条设计前提的直接检验。

### (62) 新场景 `kv4kv2`：P cp1/tp4/kv4 → D cp1/tp2/kv2

| 场景 | Prefill | Decode | 设备 | 与 `p4d2` 的差别 |
|---|---|---|---|---|
| `p4d2` | 4 ranks **cp4**/tp1/kv4 | 2 ranks cp1/tp2/kv2 | P 0-3 / D 4-5 | — |
| `kv4kv2` | 4 ranks **cp1**/tp4/kv4 | 2 ranks cp1/tp2/kv2 | P 0-3 / D 4-5 | **只有 `P_CP`（4 → 1）** |

所以 `kv4kv2` 与 `p4d2` 构成一对干净的 A/B：rank 数、设备、kv split、prompt 长度全同，唯一变量是
prefill 有没有 CP。

改动（都在 83/98 的运行时脚本里，产品代码零改动）：

* `pdroute83/env.sh`：新增 `kv4kv2)` 分支（`P_NNODES=4 P_CP=1 P_KVSPLIT=4` / `D_NNODES=2 D_CP=1
  D_KVSPLIT=2`，端口与设备沿用默认值）、注释表加一行、`unknown PD_SCENARIO` 的提示补上新名字；
* `pdroute83/run_trace_kv4kv2.sh`：由 `run_trace_p4d2.sh` 生成（同设备分配，便于对照）；
* `handoff/remote/remote_loop.sh` 的 `geometry()` 加 `kv4kv2) echo "1 4 4 1 2 2"`；
* `handoff/remote/scenarios.txt` 加一行 → 标准矩阵入口 `run_one98.sh` 从此覆盖四个场景；
* 应用脚本 `remote/apply_scenario_kv4kv2.py`（带断言的替换，和 `apply_index_sharded.py` 同一套写法）。

两个坑（都值得记）：

1. apply 脚本第一版把 `p2d4)` 分支**替换**掉了 —— 断言只检查「匹配到 1 处」，没检查「结果是增量」，
   于是 `p2d4` 变成 `unknown PD_SCENARIO`。修法：替换写成 `ARM → ARM + NEW_ARM`，并用一个一次性的
   `fix_p2d4.py` 把分支补回来；教训是**增量型替换要断言「结果仍包含原文本」**。
2. `sync_tools.sh` 原来只同步 tracer/比较器，**不同步 `remote_loop.sh`**，于是 98 上跑新场景得到
   `unknown scenario kv4kv2`。已把 `remote_loop.sh` 加入同步清单（它承载场景表，必须跟着工具走）。

### (63) `kv4kv2` 结果：**长 prompt（3492 token）一次全绿**

归档 `runs/kv4kv2-201508`（首次运行，20:15）：

| 族 | 结果 |
|---|---|
| key / value（split） | **56/56 字节一致**（14 logical block × 4 layer），`mismatch=0 absent_in=0` |
| index（replicated） | **84/84 字节一致**（28 canonical × 3 layer），且在**四个 prefill rank 上完全相同**（`present on every one of the 4 prefill ranks=84`） |
| owner 检查 | `checked=56 match=56`，无缺行、无未观测、无 padding、**源端副本无分歧** |
| 判定 | `VERDICT: PASS` + `OWNER VERDICT: PASS` |

这同时把用户给的设计前提**实测确认**了：只开 kv split（无 CP）时，源端 index 池就是
「whole sequence on every rank（`row = c + S`）」的副本 —— 工具按 trace 里的 `has_kv_shard=0`
自己识别出这个形状（输出行 `source indexer pool: whole sequence on every rank`）。

顺带把工具里三处**读起来会误导**的地方改准：

* `index_owner_check.py`：源端是副本实例时不再打印 `source slices [0,0,0,0]`（那只是 DCP rank，
  副本实例的切片在 TP 轴上），改印 `source slices n/a (the source keeps the whole sequence)`；
* 副本源端新增**副本一致性**口径：`want` 取各 rank 的交集，只在部分 rank 命中时单独报
  `source replicas disagree`（判负）—— 一个副本实例在不同 rank 上持有不同字节，本身就是要抓的错，
  而不是「模糊匹配上了就算过」；
* `compare_kv.py` 的提示文案从「a full-sequence pool holds different values per DCP slice」改成
  「a source **that shards the sequence** holds different values per slice」（原文案在副本源端是反的）。

### (64) `kv4kv2` vs `p4d2` 的 A/B：`3480` 的触发条件就是 **prefill 的 CP**

`p4d2` 长 prompt（同一 4 个 prefill rank、同一设备、同一 `kv4`、同一 3492 token）在 prefill forward 崩：

```
[ASSERT] gather_v3_base.h:137 Assertion `(0 <= val && val < this->gxSize_)' Index 3480 out of range[0 3480)!
[rank0] NPU function error: call aclnnIndexSelect failed, error code is 507035
        Kernel task happen error, retCode=0x31, [vector core exception]
```

`kv4kv2` 把 `P_CP` 从 4 改成 1，其余全同，**同一个长 prompt 一次通过**。所以：

* 触发条件是 **prefill 的 CP（cp4）× 长 sequence**，与 kv split、与 canonical 路由、与数据面都无关；
* 代码路径（容器 site-packages，`glm5_2.py` 的 MLA/indexer forward）：只有 `cp_context is not None`
  时才走
  `query_index = cp_context.query_index` + `hidden/q_c/positions.index_select(0, query_index)`；
  `query_index` 由 C++ op `xllm_ops::build_cp_context` 产出（`model_executor/cp_utils.py:109-153`），
  语义是「本 rank 真正持有的 query 行在本地打包 buffer 里的行号」；
* 观测形状：失败张量的 dim0 = 3480、越界值也 = 3480（**恰好等于 size**，闭区间端点被当索引的典型形状），
  即 `query_index` 里出现了一个等于「本地 token 数」的行号（合法范围 0..3479）——与第 33 轮
  `3480 = 4 × 870`「CP 分片边界」的猜测一致，现在有了 A/B 和调用点两重证据。
* 下一步（模型侧）：看 `build_cp_context` 里 `total_real_local` / 段边界（`(cp_rank+1)*chunk`）的计算，
  以及 3492 token 下最后一段的 `query_count` 是否把端点算了进去。

### (65) 四场景矩阵全绿（20:17–20:30），出厂工具复核一致

`handoff/remote/run_one98.sh` + `scenarios.txt`（四行：base / p2d4 / kv4kv2 / p4d2:canonical:20）串行跑，
日志 `/tmp/matrix_round36.log`，结果 **`MATRIX DONE: 0 failure(s) of 4`**（4 个 `VERDICT: PASS` + 4 个
`OWNER VERDICT: PASS`）：

| 场景 | 归档 | K/V | index | 比较器 | owner 检查 |
|---|---|---|---|---|---|
| `base` | `base-202042` | 112/112 | 84/84 | PASS | PASS（56/56） |
| `p2d4` | `p2d4-202406` | 28/28 | 84/84（28 canonical × 3 layer） | PASS | PASS（112/112） |
| `kv4kv2` | `kv4kv2-202700` | 56/56（14 block × 4 layer） | 84/84（四个源 rank 完全一致） | PASS | PASS（56/56） |
| `p4d2` r20 | `p4d2-canonical-r20-202954` | 12/12、8/8 + 4 padding | 15/15（+3 `abs_oos` padding） | PASS | PASS（10/10，1 padding） |

离线用**出厂版本**的检查脚本（`index_owner_check.py=c5942ff8e6a92ed2a33f4ead0d01b50c`、
`compare_kv.py=9c9207adaf9ac11d7ef9fb44ef1c7833`）复核（`remote/validate_final36.sh`）：
pre-fix scan 归档仍 FAIL（canonical 2 字节不等 + 15 块 `not written while the source holds it`），
post-fix scan 112/112，四个最新归档与在线判定逐项一致（含 `kv4kv2` 被识别为
`whole sequence on every rank (row = c + S)`）。

### (66) 复核：prefill 开 CP 时 indexer 按 kv split 切分，「prefill 切分 / decode 不切分」已正确实现

用户问「prefill 开 CP 后 indexer 是否也按 kv split 切分、当前处理是否正确、是不是没处理好 prefill 切分而
decode 不切分的场景」。按仓库 `code-review` skill 走了一遍（含 `custom-code-style.md`），结论：

**确认切分**（代码证据）：

| 层 | 证据 |
|---|---|
| 池大小 | `kv_cache_shape.cpp:387-393`：`index_block_count *= kv_split_size_effective()` |
| 写哪一行 | `npu_paged_attention.py:940`：`local_slot_mapping if has_kv_shard else slot_mapping` |
| 标志来源 | `py_attention_metadata.cpp:275-283`（`has_kv_shard` ≡ batch 带 shard metadata）、`kv_shard_batch_metadata.cpp:145-147`（`kv_split_size = dcp_size()`） |
| 展开表 | `kv_shard_batch_metadata.cpp:129-141`：`logical*dcp + shard` |
| CP 约束 | `npu_paged_attention.py:566-571`（`cp_size % kv_split_size == 0`）、`collective_communicator.cpp:582-583` |

**处理正确**（对已验证形状）：`group_keeps_whole_sequence()`（`kv_redundancy.cpp:46-53`）→
`derive()`（`:87-124`）→ `peer_row()`（`route_binder.cpp:160-180`，且 `:398-401` 两个 peer 各按自己的形状取行）
三层自洽；manifest 的 `kv_split_size` 取 `kv_split_size_effective()`（`mooncake_kv_cache_transfer.cpp:509-512`）
且 `cache_directory.cpp:704-705` 校验 declaration 与 manifest 一致。

**两条仍未验证的形状（review 的主要发现，落档待办）**：

1. **`kv_split_size < cp_size`（`sequence_groups > 1`）**：`KvLayoutIndex::replicas_of`
   （`kv_redundancy.cpp:255-263`）会把同一 slice 的多个 **CP rank** 当序列副本，`writer_of`（`:221-223`）
   只让 replica-0 推。K/V 有 `_materialize_cp_cache`（`npu_paged_attention.py:1029+`，CP all-gather）保证
   同 slice 内容一致，但**索引池只做本 rank 的 scatter 写**，没有任何 materialize。我们只实测过
   `sequence_groups == 1`（此时 slice 内的额外 rank 是 TP 副本）。若该假设不成立，每个 slice 会有一半块
   从不推送。→ 行动：跑一个 `P cp4/tp1/kv2 → D cp1/tp2/kv2` 场景；若不一致，修法是在 `derive` 里
   对 `full_sequence_replica && cp_size > 1` 追加 `split == cp_size` 的约束（明确报错），或让索引池也 materialize。
2. **`layerwise_split_size > 1` + index-page elision**（`platform.h:71-74`）：共享 DSA 层不写索引池，
   路由是否该给这些层推索引行未验证（C1 MixedLayers 欠账）。

另有一个测试缺口：`pd_route_test.cpp:207` 只钉了**分片源端**的不对称对（`:730/766` 断言两侧 row），
**副本源端**（`block + S_P`，即 `kv4kv2` 形状）只在端到端覆盖，缺单测。

### (67) 更正：`xllm_ops::build_cp_context` 的源码就在主树里 —— 我先前两次误判为「只在 .so 注册」

第 66 轮末尾我把 `3480` 的根因分析停在了「op 源码不在我能访问的任何树里」。这是**错的**，
用户提出质疑后逐项复核：

| 事实 | 证据 |
|---|---|
| Python 调用 | `xllm/python/model_executor/cp_utils.py:137` `= torch.ops.xllm_ops.build_cp_context(...)`（docstring `:39/:120` 明确写 "the `xllm_ops::build_cp_context` C++ op and validated by its gtest"） |
| schema | `xllm/core/kernels/npu/npu_ops_library.cpp:829`（与 .so 里 `strings` 到的签名逐字一致） |
| 注册 | 同文件 `:664` `TORCH_LIBRARY(xllm_ops, m)`、`:974` `m.impl("build_cp_context", TORCH_FN(xllm::build_cp_context_npu))`（`TORCH_LIBRARY_IMPL(xllm_ops, CompositeExplicitAutograd, …)`，注释 `:968`：纯 host index math、无 Tensor 入参，故不按 device 派发） |
| 实现 | 同文件 `:504` `build_cp_context_npu(...)`，`TORCH_CHECK(cp_size > 1, …)` 在 `:509` |
| 调用方 | `xllm/python/model_executor/runners/eager.py:88`（`_per_seq_lens_from_metadata` + `self.cp_size`） |
| gtest | `tests/core/kernels/npu/cp_context_builder_test.cpp`（`:96-98` 镜像 `cp_utils.cp_shard_rows`，`:233` `q_local.index_select(0, ctx.query_index)`） |
| V4 的模型侧 plan | `xllm/core/layers/npu_torch/deepseek_v4_cp_context.h`（+ MLU 版 `xllm/core/layers/mlu/deepseek_v4/deepseek_v4_cp_context.{h,cpp}`） |
| 本机 clone = 98 构建树 | `npu_ops_library.cpp=de65c11ac83edab28b11e6c0e0c8f970`、`cp_utils.py=c2b41d63a84379d1d89a78d751a56bdd`、`deepseek_v4_cp_context.h=965270e1c523c5a39dc4461b22cff91a`（三份 md5 相同） |

**误判原因**（三条都是工具/权限问题，不是"源码不在"）：

1. 我用的第一条本机命令是 `timeout 60 grep … build_cp_context` —— macOS 没有 `timeout`，
   命令直接报 `command not found`，等于**根本没执行**；
2. 之后在 98 上做的宽范围 `timeout 90/120 grep -rl`，被 `timeout` 杀掉时**管道里的块缓冲 stdout 一起丢失**，
   我看到的是空输出，就当成"无匹配"；
3. 98 上这些源文件是 `root:root 0640`，不加 `sudo -n` 的 `grep/sed/wc` 静默失败（stderr 被重定向掉了）。

**更正后的分析收窄**（不再是"源码不可得"，而是"算子不可能是越界来源"）：
`build_cp_context_npu` 只为真实行 push（`pos_in_seq < length`），且
`seg_local_base + j < total_local`，故恒有 `0 <= query_index < total_local`；gtest 的用法是先
`cp_shard_rows(global)` 得到 `total_local` 行的本地张量再 `index_select(0, query_index)`。
因此 `dim0 = 3480` 且 `index = 3480` 意味着**被索引的张量比 CP plan 的本地行空间小**，
即 CP plan 与被施加的 buffer 不匹配（未做 CP 分片的全局张量，或 ctx 与 buffer 属于不同 batch），
而不是 packing 循环里的 off-by-one。下一步：在 `glm5_2.py` 的三个 `index_select` 之前断言/打印
`hidden.shape[0] == cp_context.total_local`，并把 `cp_size/cp_rank/q_seq_lens/kv_seq_lens/total_local/
query_index.max()` 记下来。

## 第 37 轮（2026-09-19 21:00–）：`3480` 定位为**预打包的 indexer query**，修在模型侧；padding 场景从必崩到双 gate 全绿

### (68) 触发条件收紧：不是「长 prompt」，是 **prompt 长度不整除 `2 * cp_size`**

把此前所有**绿过**的 CP 场景排一遍，共同点立刻出来：

| 场景 | P 侧 | prompt token | `token % (2*cp_size)` |
|---|---|---|---|
| `base` | cp2 | 3492 | 0 ✓ |
| `p2d4` | cp2 | 3492 | 0 ✓ |
| `p4d2`（短） | cp4 | 592 | 0 ✓ |
| `kv4kv2` | cp1（无 CP） | 3492 | — |
| `p4d2`（长） | cp4 | 3492 | **4 ✗** |

于是做了一个**只改 prompt 长度、不改几何**的判定性实验：`base`（cp2，本来全绿）用 121 repeats
（3521 token，`3521 % 4 = 1`）→ 复现同一个 `aclnnIndexSelect … 507035` 崩溃；同一场景 120 repeats
（3492，整除）→ 仍然绿。**触发条件是 padding 行，不是 CP4/kv4 的形状。**

### (69) 探针：挂在 tracer 通道上（不改构建树、不重编）

`_pd_trace.py` 增加一段 CP plan 探针（只在本轮验证用，不进仓库）：wrap 所有绑定了
`cp_shard_rows / cp_shard_positions / cp_gather_kv / cp_merge_rows` 的模块（`glm5_2` 是
`from … import` 绑定的，只 patch 定义模块会漏掉真正被调用的那个名字）、`Glm52Indexer.select_qli`、
以及后端的 `_update_mla_index_cache`；每次调用记 `in_rows / total_local / cp / restore_n /
restore_max / gathered_rows / values_rows / slots` 并附**调用点**，日志写在 trace 目录（随归档一起留档）。
探针只在 `XLLM_PD_TRACE_DIR` 打开时生效，且 `run_trace.sh` 每次都把 `tracefiles/_pd_trace.py`
覆盖安装进容器，所以迭代探针不需要重编、也不需要动构建树。

两个工具教训：

1. **探针第一版把后端类名猜成 `NpuPagedAttention`**（真名 `NpuPagedAttentionBackend`），
   `AttributeError` 在 warmup 的 `dump()` 里抛出 → worker 当场死掉、HTTP 端口关闭，
   `smoke_long.sh` 只留下 `http_code=000`、trace 目录全空，白跑一轮 7 分钟。现在安装流程整体
   `try/except` + 按内省（遍历模块里的类找 `_update_mla_index_cache`）取类名：**探针绝不能弄死 worker**。
2. **`probe_check.sh` 最初把归档的 probe 日志一起 grep**，于是修复后的检查仍然报 `MISMATCH`
   （命中的是修复前的归档）。verdict 只看 live trace dir，归档只作参考。

### (70) 根因（实测一行即结论）

```
[1789823018.132] cp_gather_kv in_rows=870 total_local=874 cp=4/0 restore_n=3492 restore_max=3495
                 gathered_rows=3480 PLAN_BUFFER_MISMATCH at=glm5_2.py:655:select_qli
```

- 3492 token @ cp4：pad 到 3496、`chunk_len=437`、`total_local=874`；rank0 拥有 chunk0 与 chunk7，
  而 3492..3495 落在 chunk7 尾部 → **rank0 的真实行只有 870**，其它 rank 是 874。
- `Glm52MLAAttention.forward` 的 CP 分支先用 `query_index` 把 `hidden/q_c/positions` 压成 870 行，
  再把它们交给 `select_qli`；而 `select_qli` 里的 `cp_gather_kv` 假定输入是 **`total_local` 行的
  padded 本地布局**：它 `all_gather` 出 `cp_size * 870 = 3480` 行，随后用 `restore_index`
  （按 `cp_size * total_local = 3496` 定义，max = 3495）去索引 → `Index 3480 out of range[0 3480)`。
- 只有 rank0 崩（唯一带 padding 的 rank）；短 prompt 不崩（`592 % 8 == 0`，打包无损）；
  warmup 也不崩（probe 里 `in_rows=64 total_local=64 gathered_rows=256 restore_max=255`），
  所以 worker 能活过 warmup、只在真正的请求上死。
- `kv_transfer_completion.cpp:44 CHECK(futures_.empty())` 与
  `npu_layer_synchronizer.cpp:98 Record event failed: 507035` 都是这次设备端失败的**二次症状**。

### (71) 修复（模型侧，python-only，不需要重编 ELF）

`xllm/python/models/glm5_2.py`：

- CP 分支不再预打包：直接把 **padded 本地** `hidden/q_c/positions` 交给 indexer（与
  `cp_context is None` 分支同一个调用）。`select_qli` 结尾本来就有 `cp_shard_rows(topk)`，
  会把全局 top-k 归还成这个 rank 的真实行。
- 顺手删掉 `cache_hidden` / `cache_positions` 两个参数（唯一调用方就是这里）：它们只为那次
  打包而存在，留着就是把同一个陷阱再埋一遍。

语义核对（为什么这才是对的）：

- `ctx.actual_seq_q = metadata.q_cu_seq_lens[1:]` 是**全局**每请求 q 长度，`restore_index` /
  `shard_index` / `kv_gather_index` 全部按 `total_local` 布局定义 → indexer 的输入必须是 padded
  本地张量；不打包时 q/k/weights 都被 gather 成 `T_real` 行，正好等于 `sum(actual_seq_q)`
  （3492），index 池写入也正好对齐（实测 warmup `values_rows=256 slots=256`）。
- 为什么以前"看起来对"：长度整除 `2 * cp_size` 时每个 rank 的两个 chunk 全真实，
  `query_index` 恰是恒等映射，打包无损 —— 这也是为什么四个绿场景掩盖了这个 bug。
- K/V 那条路（`_fia_prefill_cp`）本来就没打包，`cp_gather_kv(k_3d)` 一直是对的；它只是被
  indexer 抢崩在前面。

### (72) 修复验证（容器内 `glm5_2.py` md5 = 仓库文件 `5c331399f97d77d804a4efd7bfc88c33`）

| 运行 | 修复前 | 修复后 |
|---|---|---|
| `base` r120（cp2，3492，无 padding） | VERDICT + OWNER 全 PASS | VERDICT + OWNER 全 PASS；probe NO-MISMATCH |
| `p4d2` r120（cp4，3492，有 padding） | ❌ `507035`，prefill rank0 崩 | **VERDICT + OWNER 全 PASS**（index 84/84、K/V 56/56、owner 56/56，0 padding 行）|
| `base` r121（cp2，3521，有 padding） | ❌ `507035`，prefill rank0 崩 | **VERDICT + OWNER 全 PASS**；probe NO-MISMATCH |

修复后的 probe 行（rank0）：`select_qli hidden_rows=874 total_local=874`，
`cp_gather_kv in_rows=874 total_local=874 restore_n=3492 restore_max=3495 gathered_rows=3496`
（不再有 `PLAN_BUFFER_MISMATCH`），`_fia_prefill_cp` 的 K/V gather 同样 874 → 3496。

### (73) 工具与场景改动

- 新增场景 `cp4kv2` = P cp4/tp1/kv_split2 → D cp1/tp2/kv_split2：`kv_split` 是 CP 组的真因子，
  `sequence_groups = cp/kv_split = 2`，同一 slice 由两个 CP rank 持有副本 —— 这正是
  「每 slice 单写者」从未被验证过的形状（第 66 轮遗留的复核项 1）。
  `env.sh` 加了 case 臂（`apply_scenario_cp4kv2.py`），`run_trace_cp4kv2.sh` 由 p4d2 的生成，
  `remote_loop.sh` 的几何表加 `cp4kv2) echo "4 1 2 1 2 2"`；五个臂都实测可求值。
- 新脚本：`remote/apply_glm52_cp_fix.py`（幂等、带 `.pre_cpfix` 备份、逐条断言恰好匹配一次）、
  `remote/apply_cp_fix.sh`（在 83 上把补丁打进容器，校验 py_compile + md5）、
  `remote/probe_check.sh`（只看 live trace 的 probe verdict）、
  `remote/probe_cp_runs.sh` / `probe_cp_fix_runs.sh` / `matrix37.sh`。
- md5：`_pd_trace.py=6a2d05b4128f52f75a5bb0ee07c6240c`、
  `remote_loop.sh=f045d34faf62f899be6dffd7194ebdc7`、
  `apply_glm52_cp_fix.py=ebdc882c7008dc91a0224b8758a2e62d`、
  `probe_check.sh=132b8806b314fd0a5735e69d52e4cdfb`、`apply_cp_fix.sh=1f709facd90f03ff6ec5aafdd4b33238`。

### (74) 第 37 轮矩阵：**8 个场景、0 failure**，两个 gate 全绿（21:56–22:20）

`bash matrix37.sh`（98 上，`scenarios.txt` 八行，日志 `/tmp/matrix_round37.log`）：

| 场景 | Prompt | 归档 | compare | owner |
|---|---|---|---|---|
| `base` | 3492 | `base-215908` | PASS（index 84/84，K/V 112/112） | PASS（56/56） |
| `p2d4` | 3492 | `p2d4-220233` | PASS（84/84，28/28） | PASS（112/112） |
| `kv4kv2` | 3492 | `kv4kv2-220527` | PASS（84/84，56/56） | PASS（56/56） |
| `p4d2` | 592（r20） | `p4d2-canonical-r20-220820` | PASS（15/15 +3 padding，8/8 +4 padding） | PASS（10/10 +1 padding） |
| **`p4d2`** | **3492（r120）** | `p4d2-canonical-r120-221119` | **PASS（84/84，56/56）** | **PASS（56/56）** |
| **`base`** | **3521（r121）** | `base-canonical-r121-221413` | **PASS（84/84，56/56）** | **PASS（56/56）** |
| **`cp4kv2`** | 592（r20） | `cp4kv2-canonical-r20-221707` | PASS（15/15 +3 padding，8/8 +4 padding） | PASS（10/10 +1 padding） |
| **`cp4kv2`** | 3492（r120） | `cp4kv2-canonical-r120-222006` | **PASS（84/84，56/56）** | **PASS（56/56）** |

`MATRIX DONE: 0 failure(s) of 8`。

**`cp4kv2`（`kv_split=2 < cp_size=4`，`sequence_groups=2`）结果**：长 prompt 全绿，且 owner 检查的
`source replicas disagree` 计数为 **0** —— 同一 slice 上的两个 CP rank 写出的索引池**字节一致**，
「每 slice 单写者」这条假设在这个形状下成立（第 66 轮遗留复核项 1 的实测答案）。短 prompt 的
`abs_oos`/padding 行与 `p4d2` 完全一致（同一目的端拓扑）。

**流程教训**：矩阵第一次从笔记本以 rrun 前台 ssh 启动，jump host 在 p2d4 的 compare 阶段把会话掐了，
**远端矩阵随之被杀**（但 run 本身已经 archive，只是日志停在 `restarted`）——前台 ssh 的存活期不能当作
远端任务的存活期。改成 98 上 `setsid nohup bash run_one98.sh` 后台跑 + 只读远端日志后，八场景一次跑完。
（`:!` 现象是：`tail -45 | tee` 管道里未刷出的行在进程被杀时一起丢了，日志因此看起来"卡住"。)

### (75) 回归测试：把「不能预打包」钉成 CPU 单测

`tests/python/test_sfa_dcp.py::test_cp_indexer_is_handed_the_padded_local_rows`：用 mock 的
`Glm52MLAAttention` 走到 indexer 调用点（indexer 抛哨兵异常终止 forward），断言
`select_qli` 收到的 `hidden/qr/positions` 行数 == `cp_context.total_local`、且不再传
`cache_hidden`。**判别力实测**（98 构建容器内，同一份测试）：

```
A. 打完补丁的树（5c331399…）  -> 1 passed
B. 恢复 pre-fix 文件（8c4c8a77…）-> 1 failed：assert 4 == 6
   "the indexer must see the padded local layout; a packed query set (4 rows)
    is what crashed the prefill"
C. 再次切回补丁              -> 1 passed
```

（用 `total_local=6 / packed=4` 作为 874/870 的最小同构。）另外在打补丁的树上跑**整个 CPU python
测试目录**：`684 passed, 44 skipped`。

### (76) C++ 侧补齐「副本源端」的字节级单测，并修掉一个本来就在红的测试

第 66 轮复核列出的最后一个验收缺口：`pd_route_test.cpp` 只钉了**分片源端**（`block / S + 1`）
的字节级契约，**副本源端**（`block + S_P`，即 `kv4kv2` 形状）只有端到端覆盖。补上：

* 新增 `run_mock_replica_transfer(bool wrong_source_row)` + 两个用例
  （`MockReplicaTransferCopiesWholeSequenceRows` / `MockReplicaTransferIsDiscriminating`）：
  拓扑就是 `kv4kv2`（P cp1/tp4/kv4 → D cp1/tp2/kv2，组 `full_sequence_replica=true`），
  期望独立推导为「canonical block h 取自源端 `h + S_P` 行、落在目的端 `h + S_D` 行」；
  源端四个 rank **同种一份字节**（这正是端到端 owner 检查测的不变量），所以期望与「路由选中哪个副本」
  无关；`wrong_source_row` 变体按 `h` 读，必须产生 mismatch。
* 顺带修了一个**改动前就在红**的用例：`MockIndexTransferIsDiscriminating` 原本靠
  `remote.explicit_offsets`（把目的端 row 基整体挪一行）制造「错误」，但 `RouteBinder` 后来加了
  「bound region exceeds its cache buffer」的越界保护，于是 `bind` 直接返回 false、
  内层 `EXPECT_TRUE` 失败——测试变成了在测那道保护，而不是测字节比较。改成
  **给源端 rank 种错一份模式**（`mock_pattern(rank + 1, …)`）：绑定仍然合法、结构不变量全过，
  只有字节比较能发现它，判别力回到字节层面。

98 上的结果（打了 `glm5_2.py` 补丁的树）：

```
pd_route_test         16 passed
kv_redundancy_test    12 passed
cache_directory_test  23 passed
```

另外发现构建树 `route_binder.cpp` 与仓库只差一行注释折行（md5 不同、代码相同），已把仓库版本
stage 回树里保持一致（不改变行为）。

### (77) 第 37 轮收尾：layerwise 按决定暂不兼容；下一轮做**异构 TP** 兼容性验证

**决定（用户 2026-09-20）**：`layerwise_split_size > 1`（共享 DSA 层的 index-page elision）
**暂不验证、不做兼容** —— 第 66 轮复核项 2 挂起，不是"已验证没问题"的结论。线索留在此处备查：
`platform.h:71-74` `supports_dsa_indexer_cache_elision()`（"Shared DSA layers reuse the previous
full layer's top-k and never write indexer cache, so those layers skip indexer-page allocation"）
配合 `glm5_2.py` 的 layerwise 分支（只在 `owns_layer_cache` 时调 `select_qli`，也就是只在属主
rank 上写索引池）。真要做时先查：PD 传输是否给被 elide 的层推/收索引行、decode 侧的非 layerwise
路径能否读到它们。

**下一轮目标：异构 TP 兼容性。** 已有矩阵里的 TP 变化都绑在 kv-split reshard 上（`p2d4` 2→4、
`p4d2` 1→2、`kv4kv2` 4→2），所以新增三个把 TP 变化单独拎出来 / 放大的场景。机器只有 8 卡，
两侧 rank 数之和 ≤ 8：

| 场景 | Prefill | Decode | 变的是什么 |
|---|---|---|---|
| `tp2tp1` | 4 ranks cp2/tp2/kv2（0-3） | **1** rank cp1/tp1/kv1（4） | TP 2→1，kv 2→1，decode 退化成单 rank |
| `p4d4` | 4 ranks cp4/tp1/kv4（0-3） | 4 ranks cp1/tp4/kv4（4-7） | TP 1→4，kv 4→4（**只变 TP**） |
| `kv4kv1` | 4 ranks cp1/tp4/kv4（0-3） | **1** rank cp1/tp1/kv1（4） | TP 4→1，kv 4→1，两侧都不开 CP |
| `p2d4`（对照） | cp2/tp2/kv2 | cp1/tp4/kv4 | 第 37 轮矩阵已绿，用来对齐可比性 |

**已经就绪但还没跑**（下一轮直接开跑即可）：

* `env.sh` 三个 case 臂（脚本 `apply_scenario_heterotp.py`，已在 83 应用并实测可求值）；
  `run_trace_{tp2tp1,p4d4,kv4kv1}.sh` 已生成。
* `remote_loop.sh` 几何表三行（`tp2tp1) 2 2 2 1 1 1`、`p4d4) 4 1 4 1 4 4`、`kv4kv1) 1 4 4 1 1 1`），
  md5 `62c74da388ab2d49639231656eaaa7c1`（本机 = 98）。
* `scenarios_heterotp.txt`（四行）+ `matrix_heterotp.sh`：98 上 `setsid nohup` 起，
  日志 `/tmp/matrix_heterotp.log`，避免再被 ssh 掉线杀掉。
* 八个臂全部实测可求值：TP 2→2 / 1→2 / 2→4 / 4→2 / 1→2 / **2→1** / **1→4** / **4→1**。
* **风险点（第一次跑才知道）**：`D_NNODES=1` 从未跑过 —— `start_workers.sh` 的 rank 循环支持它，
  但端口/就绪检查、以及退化成一个 rank 的 DCP 组是否被各 family 正确处理，要靠 `tp2tp1`/`kv4kv1`
  自己暴露。

第 37 轮全部 durable state：本 worklog §(68)–(77) + `RESUME.md`（本机 / 98 `handoff/` /
83 `pdroute83/` 三处同 md5）；已推 `57b856294`、`05b0f972e`。

## 第 38 轮（2026-09-20 11:00–）：异构 TP 要用 Qwen3 验证 —— 域前提更正、混合架构缓存实测、当前卡在 op 包

### (78) 前提更正：GLM-5.2 是 MLA/DSA，KV cache 只有 1 个 head、**不随 TP 切分**

用户在开工前指出这一点，代码确认（`kv_cache_shape.cpp`）：

| 分支 | 形状 | 是否随 `world_size`(=TP) 变 |
|---|---|---|
| `enable_mla()`（GLM-5.2） | `key=[n_blocks, block_size, 1, kv_lora_rank]`、`value=[…, 1, qk_rope_head_dim]` | **不变**（`:316-318`、`:341-344` 直接写死 1） |
| GQA（Qwen 系） | `key=[n_blocks, block_size, local_kv_head_count, head_dim]`，`local_kv_head_count = get_local_head_count(n_kv_heads, world_size)` | **变**（`:320-326`、`:351-357`） |

所以第 37 轮准备的 GLM 异构-TP 场景（`tp2tp1`/`p4d4`/`kv4kv1`）只能验证**rank 几何与副本扇出**
（GLM 的 K/V 在 TP 上是复制的），**无法验证 head-range reshard**；真正要验的是 GQA 模型。

Qwen3.5-0.8B（用户指定，share 里唯一可跑的 Qwen3 系小模型；`Qwen3-VL-*` 是 17G/63G，`Qwen3-Coder-480B`
是 450G，`Qwen3-8B-DFlash-b16`/`Eagle3-*`/`Qwen3EAGLE` 是草稿模型，`Qwen3.5-9B`/`Qwen3.6-27B` 目录为空）：

```
24 层 = 18 linear_attention + 6 full_attention（full_attention_interval=4，full 层 = 3,7,11,15,19,23）
attention : 8 Q heads / 2 KV heads / head_dim 256
linear    : 16 K heads / 16 V heads / head_dim 128 / conv kernel 4
mtp       : 1 层（NUM_SPECULATIVE_TOKENS=0 关掉）  权重前缀 model.language_model.
model_type= qwen3_5 -> 引擎解析成 qwen3_5_text（model_config_utils.cpp:50-62）-> registry 的 qwen3_5/
             qwen3_5_text -> xllm/python/models/qwen3_5.py（NPU 层实现在 layers/npu/qwen3_5/）
```

**TP 约束**（`model_loader/sharding.py:66-75` `gqa_head_split`）：`n_kv_heads >= tp_size` 才按 head 切，
否则每 rank 1 个 KV head、被 `tp_size/n_kv_heads` 个 rank 复制。所以本模型的 attention KV 在
tp1 = 2 heads/rank、tp2 = 1 head/rank、tp4 = 1 head/rank（复制模式）；异构 TP 取 **tp1 ↔ tp2**。

### (79) 实测：Qwen3.5 的缓存张量与 row 契约（来自归档 trace，不是推断）

第一次 smoke（`q35t1t1` = P tp1/kv1 → D tp1/kv1，短 prompt）虽然 forward 失败了，但 tracer 在
`ModelExecution.execute` 入口写出了 batch 的 dump，从归档里读出来的布局与代码完全一致：

```
meta: call=0 is_prefill=True blocks=[1,2] slot_len=256 slot_unit=128 has_kv_shard=0 kv_split_cfg=1
slot=key    shape=[31407, 128, 2, 256]  bf16   row_bytes=131072  layers 3..23  (6 个 full-attention 层)
slot=value  shape=[31407, 128, 2, 256]  bf16   row_bytes=131072  同上
slot=conv   shape=[202, 3, 6144]        bf16   row_bytes=36864   layers 0..22  (18 个 linear 层)
slot=ssm    shape=[202, 16, 128, 128]   fp32   row_bytes=1048576  同上
```

* `key/value` 的第 2 维 = **local KV heads**（tp1 时 2）→ head 切分轴在这里，逐头比对必须按这一维。
* `conv` 的第 2 维 = 3 = `conv_kernel_dim-1`；第 3 维 6144 = `linear_key_head_dim*local_k(16)*2 +
  linear_key_head_dim*local_v(16)` = 4096+2048 → **实测到的就是那个 composite descriptor**。
* `ssm` 的第 1 维 = 16 = `local_linear_value_head_count` → 逐头比对轴在这里。
* 两者都是 **sequence-scoped**（state block，不是 token block），与 `RouteBinder::peer_row` 的
  sequence-scoped 分支对应。

### (80) 路由侧现状（读码结论，待下一轮决策）

`cache_directory.cpp:456-524` `declare_cache_group`：

* **SSM**：`global_head_count = linear_value_head_count`、`sequence_scoped=true`、`full_sequence_replica=false`
  → canonical 路由**可以**服务。
* **CONV**：注释原文 *"Without MLA this role publishes a composite descriptor, which the canonical route
  refuses on purpose (see describe_tensor). Declaring the geometry anyway keeps that refusal in one place."*
  → **非 MLA（= Qwen3.5）的 conv 族目前 canonical 路由拒绝服务**，这正是 C1 MixedLayers 欠账的入口。
  下一轮必须先决定：给 canonical 补上 composite CONV 的 key/value head-range 切分，还是让 conv 走 legacy
  planner（混合路由）。

### (81) 当前阻塞点：镜像里没有任何 vendor 提供 `aclnnMegaGdnPrefill`

首次 smoke 的结论（**模型加载成功，第一次 forward Fatal**）：

```
I llm_engine.cpp:277] Block info, block_size: 128, n_local_kv_heads: 2, head_dim: 256, n_layers: 24, dtype: BFloat16
I llm_engine.cpp:303] Initializing model with ModelArgs: [model_type: qwen3_5_text, ... n_heads: 8, n_kv_heads: 2 ...]
F npu_mega_gdn_prefill.cpp:163] Check failed: get_workspace_size_func_addr != nullptr && op_api_func_addr != nullptr
    aclnnMegaGdnPrefillOp or aclnnMegaGdnPrefillOpGetWorkspaceSize not in libopapi.so, or libopapi.so not found
```

* 镜像的四个 vendor（`custom_transformer` / `custom_xllm_math` / `glm_next_transformer` /
  `kpool_transformer`）的 `op_api/lib/*.so` 里 `strings | grep MegaGdnPrefill` 全为 0，也没有任何
  `*MegaGdn*` 头文件；83/98 的 workspace 里也没有现成的 GDN op 包（只有历次别的 vendor 备份）。
* 引擎侧调用链：`xllm_ops::mega_gdn_prefill`（`npu_ops_library.cpp:917`）→ `EXEC_NPU_CMD(aclnnMegaGdnPrefill)`
  （`xllm_ops/npu_mega_gdn_prefill.cpp:163`）；另有 `aclnnMegaGdnDecode`、`aclnnMegaGdnMtpDecode`。
* 与之前 `aclnnSparseFlashAttentionLse` 的情况同类：需要另找/另建 op 包，再按 `lse_vendor` 的做法用
  `ASCEND_CUSTOM_OPP_PATH` 挂上（env.sh 已有该机制与 vendor 优先级的处理）。
* **结论：Qwen3.5 异构 TP 验证当前卡在 op 包（环境），不是路由。** 归档
  `runs/q35t1t1-canonical-r20-110629` 里只有 `before call=0` 的 dump（96 行 = 24 层 × 4 slot），
  decode 侧没有 dump → 两个 gate 都是 FAIL（预期，请求没走完）。

### (82) 第 38 轮已落地的工具（未跑通）

* 场景臂 `q35t1t1` / `q35t2t1` / `q35t1t2`（脚本 `apply_scenario_q35.py`，md5 `018853cf…`）：
  在臂内覆盖 `MODEL`/`MODEL_NAME` 到 `Qwen3.5-0.8B`，并置 `P_PREFIX_CACHE=false P_CHUNKED_PREFILL=false`
  （state cache 与 chunked prefill 交织会让失败无法归因）。
* `start_workers.sh` 的 prefill 两个 flag 改成环境变量驱动（`${P_PREFIX_CACHE:-true}` /
  `${P_CHUNKED_PREFILL:-true}`，默认值 = 原硬编码值，GLM 场景行为不变）。
* `run_trace_q35{t1t1,t2t1,t1t2}.sh`；`remote_loop.sh` 几何表三行（`1 1 1 1 1 1` / `1 2 1 1 1 1` /
  `1 1 1 1 2 1`，md5 `9d86348c6ec183f5241b271d8dbfebee`）；`q35_smoke.sh`、`verify_scen3.sh`
  （11 个臂全部实测可求值）。
* 工具改造设计（**下一步要做**）：tracer 对 `key/value/ssm` 这类多 head 张量改成按 `(row, head)`
  出记录（head 轴：key/value 是 dim 2、ssm 是 dim 1、conv 是 composite dim 里的两段），并记录
  `tp_size/tp_rank/local_heads`；比较器与 owner check 按 `(canonical block, 全局 head)` 比对，
  `head_base` 规则 = sharded 时 `tp_rank*local_heads`、replicated 时 `tp_rank % global_kv_heads`；
  conv/ssm 还要按 state block / checkpoint stride 建 row 契约。

**暂停点（用户指示）**：smoke 结束后存档、暂停，等压缩上下文后继续。下一轮顺序：
(a) 拿到 `aclnnMegaGdnPrefill`/`Decode`(/`MtpDecode`) 的 op 包并挂上 → 重跑 `q35t1t1`（短 prompt）；
(b) 用 trace 确认 conv/ssm/key/value 的真实 row 契约（本轮的形状表是起点）；
(c) 实现上面的逐头比对工具；
(d) 跑 `q35t2t1` / `q35t1t2`，并同时决定 CONV composite descriptor 在 canonical 路由下的走法。

---

## 第 39 轮：Qwen3.5 异构 TP 逐头验收（(a)-(d) 全部完成）

### (83) 环境阻塞解除：`aclnnMegaGdnPrefill` op 包

* 83 的暂存 ops 树里**本来就有** `mega_gdn_prefill_op`，之前构建失败的真因是源码包缺
  `cmake/third_party/build/modules/patch/{protobuf_25.1_change_version,protobuf-hide_absl_symbols}.patch`
  （`gmake ascend_protobuf_build_xllm-patch Error 1`；更早的 LSE 构建死在同一个坑）。从本机补 `cmake/`
  （`cp -rn`，只补缺）后一次构建成功。
* 装成独立 vendor `custom_extra_xllm_math`（`aclnnMegaGdnPrefillOp`/`Decode`/`MtpDecode`，kernel 覆盖
  `ascend910_93`）。`build.sh -n` 的 `update_vendor_config` 会**前插** `load_priority`，于是把
  `config.ini` 重写为 `glm_next_transformer,custom_transformer,custom_xllm_math,kpool_transformer,custom_extra_xllm_math`
  ——新 vendor 放最后，`libcust_opapi.so` 的解析顺序与 GLM 场景完全一致。
* 效果：prefill 不再 Fatal，trace 正常落盘（PREFILL 868~2984 行、DECODE 750 行量级）。

### (84) 逐头验收工具（tracer + 比较器）

* `_pd_trace.py`：新增 `_HEAD_AXIS`（key/value = dim 2、ssm = dim 1、conv = 复合段）、
  `_head_layout`（conv 的 head 几何从**同层 ssm** 推：`heads=ssm.shape[1]`、`head_dim=ssm.shape[2]`；
  旧版按 `width%3` 把 6144 feature 算成 2048 个假 head）、`_linear_state_slot_ids`/`_linear_rows`
  （conv 的 row = linear state slot id；ssm 的 row = `slot*stride + r`，`stride = ssm.shape[0]//conv.shape[0]`）。
  每个 dump 记录 `heads`/`head_sha256` 与 `linear_state_ids`/`linear_state_rows`/`linear_ssm_stride`。
  **根因修正**：旧 tracer 用 K/V block table 当 conv/ssm 的 row，行空间根本不对，所以旧比较器把
  每一条都判成 missing。
* `compare_heads.py`：按 `(canonical block, 全局 head)` 判三件事——放置、覆盖、零值；`head_base`
  = sharded 时 `rank*(G/tp)`、否则 `rank%G`；源端索引对**所有** prefill call 取并集（旧版只取 call 0，
  把后面的 block 全判 missing）；decode meta 缺 `linear_state_rows` 时 conv/ssm 标 **ungraded** 而不是
  静默 PASS。单测 `test_compare_heads.py` 8/8，其中 5 个是必须 FAIL 的鉴别用例（head 区间互换、
  丢 rank、目标全零、无逐头摘要、linear 行空间未知）。

### (85) canonical 拒绝 COMPOSITE 的根因与修复

* 现象（两个 rank 相同）：
  `mooncake_kv_cache_transfer.cpp:538] The canonical route cannot interpret this rank's own published layout:
  layer 0, role 3, group 5: a composite descriptor interleaves several logical tensors in one physical ...`
  role 3 = CONV，group 5 = linear group；随后 `llm_worker_impl.cpp:286] Check failed: kv_transfers.wait() KV cache push failed`。
  这就是 canonical 路由在 Qwen3.5 上的**唯一**阻塞点（不是 GDN 算子）。
* 设计：`HeadGeometry` 里原来只有单一 head 轴，现在补 `runs`（`HeadRunGeometry{physical_offset_bytes,
  head_bytes, repeat_count, physical_stride_bytes}`）与 `unit_stride_bytes`。
  `derive_composite_head_geometry` 按 `logical_tensor` 分组（首现顺序 = 物理打包顺序），每个组件一条 run；
  校验：组件数 ≥2、每组件恰好 `local_heads` 个 span 且全局 head 区间**完全一致**（同一 head class、
  同 `head_bytes`）、组件内 span 的 logical 偏移按全局 head 递增、physical 偏移按 local head 连续、
  owner 一致、**组件之间 repeat/stride 一致**、**物理区间从 0 起无缝铺满一行**、总覆盖 ==
  `resource_stride_bytes`；并要求 `units == 1`（state 行在资源**内部**，不能再叠 sub-unit 循环）。
  `unit_stride_bytes = resource_stride_bytes`。
* 数据面（`route_binder`）：`PeerCacheView` 加 `head_runs` + `unit_stride_bytes`；bind 先逐 run 比对两侧
  的 run 数 / `head_bytes` / `repeat_count`，再出 region：
  `base + run.offset + (edge.head_begin - 该侧 rank.head_begin) * run.head_bytes + unit*unit_stride + repeat*run.stride`。
  单 run 族（K/V、SSM、MLA 整资源）的 `derived_head_geometry` 给的就是 `{0, head_bytes, repeat=1}` 且
  `unit_stride = local_heads*head_bytes`（整资源路径 `= head_bytes`，此时 `local_heads==1`），与旧公式
  `local_unit_stride = local_rank.local_head_count * head_bytes` **逐字节等价**。
* 关键点：同一 head 区间在两个 peer 上的偏移**本来就不同**——tp1 的 rank 持有全部 4 个 head，区间
  `[2,4)` 要从组件内第 2 个 head 起读；tp2 的 rank 只有 heads `[2,4)` 并且它们就是它自己组件的
  第 0/1 个 head，偏移为 0。这正是 head reshard 的实质。

### (86) 单测（98 上 `cache_directory_test 27 / pd_route_test 18 / kv_redundancy_test 12` 全绿）

* `cache_directory_test`：+3 接受/拒绝（复合描述符读出三条 run 并逐条断言物理偏移/repeat/stride、
  组件 head 数不同拒绝、组件落在不同 head class 拒绝）+2 硬化（组件之间 stride 不一致拒绝、
  组件偏移整体前移 32 字节——总量不变因此**只有铺满检查能看到**——拒绝）。
* `pd_route_test`：+2（TP1→TP2 的复合 bind 逐 run 断言 + 双胞胎：dst rank 0 与 rank 1 的偏移必须不同；
  peer 之间 run 数/重复次数/unit stride 不一致拒绝）。`make_view` 现在按 descriptor 的方式发布
  run 与 unit stride，既有用例的语义不变。
* 这一轮踩到的**测试自身**的坑，值得记下来：第一版把「head class 偏移」也加到了 destination 上，
  于是 `BinderWalksEveryRunOfACompositeRow` 报 remote 期望 864 实得 832。**实现是对的，期望是错的**：
  `edge.head_begin - rank.head_begin` 在 destination 侧本来就是 0（那是它自己 class 的起点）。
  这也说明这个用例确实有鉴别力。

### (87) 端到端：三条 Qwen3.5 臂全部 PASS

部署不走 wheel：`remote_loop.sh build`（ninja xllm + `strip`，563MB → 111MB）+ `deploy`
（98 → 83，`docker cp` 换入 site-packages 的 `xllm`），每次核对 md5。最终部署件 md5 `3ad29504bb18e981024ba0270bcd587c`。

| 臂 | 几何 | http | compare_heads |
|---|---|---|---|
| `q35t1t1` | P cp1/tp1/kv1 → D cp1/tp1/kv1 | 200（3492 token） | **HEAD VERDICT: PASS** |
| `q35t2t1` | P cp1/tp2/kv1 → D cp1/tp1/kv1 | 200 | **HEAD VERDICT: PASS** |
| `q35t1t2` | P cp1/tp1/kv1 → D cp1/tp2/kv1 | 200 | **HEAD VERDICT: PASS** |

三条臂的逐族数字（`rows` 是目标侧被 dump 的行数，`graded` 是 `(block, global head)` 对数）：

* conv：`graded=288 match=288 misplaced=0 missing=0 zero=0 | source pairs=16 not delivered=0 unexpected=0`
* ssm：同上（288/288）
* key/value：`graded=336 match=336 misplaced=0 missing=0 zero=0 | source pairs=56 not delivered=0 unexpected=0`

`q35t1t2` 的目标侧宽度 `conv/ssm dest widths=r0:8h,r1:8h`、`key/value r0:1h,r1:1h`
（tp2 下 conv 的 16 个 head 被两个 rank 各 8 个、K/V 的 2 个 head 各 1 个）；`q35t2t1` 的
`conv/ssm r0:16h`、`key/value r0:2h`。archive：
`runs/q35t1t1-canonical-r20-121734`、`runs/q35t2t1-canonical-r20-121948`、`runs/q35t1t2-canonical-r20-122152`。
另外行级 `compare_kv.py` 在这三条上都报 `VERDICT: PASS`（key/value `match=168 mismatch=0 absent_in=0`），
conv/ssm 它归到 `other/written=18/36 zero=0`——**行级比较器看不见 head reshard，逐头才是判据**。

### (88) 结论与边界

* **S3-4 / C1（MixedLayers 的 CONV 复合描述符）阻塞解除**：Qwen3.5 的混合 cache（全注意力 GQA KV
  + linear-attention 的 conv/ssm 状态）现在都能走 canonical 路由，且**同构与双向异构 TP**（tp2→tp1、
  tp1→tp2）逐头逐字节全部命中。head 轴口径两侧一致、`units==1`、铺满检查同时成立。
* 复合描述符的能力边界（写进代码注释）：**各组件必须暴露同一 head 区间**——Qwen3.5 的
  `linear_key_head_count == linear_value_head_count == 16` 成立；`NK≠NV` 的模型会被明确拒绝
  （错误信息点名「同一 head interval」）。`units != 1`（资源内含多个 sub-unit）也拒绝，因为
  composite 的 repeat 已经在资源内部。
* 工具口径：`remote_loop.sh run` 对 Qwen3.5 会因 `index_owner_check` 找不到 indexer dump
  （`no dumps found`）而记 `OWNER VERDICT: FAIL` → 整条 run 记 FAIL。**这是工具口径而非路由问题**；
  Qwen3.5 臂统一走 `q35_run98.sh`（跑一遍 + 逐头判定）。
* 运维坑（重要）：长任务挂在 ssh 会话下会被**静默截断**——本地拿到 `rc=0`、远端脚本却在中途被切断
  （build 成功、deploy 成功，但后续 3 个臂从未开始；日志停在 deploy 之后）。现在一律
  `setsid nohup bash <driver> > /tmp/<driver>.log 2>&1 < /dev/null &` 完全脱离会话，再用短探针读日志。
* legacy 对照（`q35t1t1-legacy-r20-122624`，同一个几何、同一个逐头比较器）：http 200，
  **conv 288/288、ssm 288/288 全 match**，K/V `match=312 misplaced=24（全是 block ±1）not delivered=4`
  → **HEAD VERDICT: FAIL**。两条结论：
  1. **纠正上一轮的判断**：「legacy 只搬 K/V、decode 侧 conv/ssm 全零（`ssm zero=486/504`）」是
     **旧 tracer 行空间的产物**——它拿 K/V block table 当 conv/ssm 的行，而那些行本来就没人写。
     行空间改对之后可见：在 tp1→tp1 这种同构对上，legacy 的数据面**确实把 linear-state 的 slot 行搬过去了**
     （逐头逐字节命中）。
  2. 更重要的：同一个几何、同一个比较器，**legacy FAIL、canonical PASS**，说明逐头判定不是
     「永远 PASS」的空判据。legacy 那 24 个 `(block ±1, head)` 错位与 4 个未交付的**归属**
     （legacy 的行映射本就与 canonical 不同，还是 legacy 自己的缺陷）不足以由这一个 run 判定——
     逐头比较器的行映射是按 canonical 写的。如实记为「未定」。
* 比较器口径再确认（避免把「两边都零」当成 match）：`compare_heads.py` 把目标侧逐头摘要等于
  `zero_digest` 的算 `zero` 而非 `match`，源端索引只收非零摘要，`match` 必须是目标摘要等于某个源摘要
  → `match=288 zero=0` 的含义是「目标真的持有源字节」。canonical 三条臂的 PASS 因此不是空判据。

