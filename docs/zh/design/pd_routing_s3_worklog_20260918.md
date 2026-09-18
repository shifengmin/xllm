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
| S3-2 | **适配器**：`CacheTensorManifest`/`ParallelCoordinates` → `BufferDirectoryEntry` / `PeerCacheView`，字段映射见 handover §7.4 | host 单测：用**真实 `describe_cache_tensor`** 造 manifest，再与手算期望逐字段比对 | ☐ |
| S3-3 | **host 集成测试**：manifest → 适配器 → `PdRouteTable::build` → `RouteBinder::bind` → memcpy 端到端 | 目标场景逐字节正确（T5 的"真实 manifest"版本） | ☐ |
| S3-4 | **数据面切换**：`transfer(edges, canonical_blocks, opcode)` 统一入口，先 PULL 后 PUSH；用 `--pd_route=legacy\|canonical` 开关，默认 legacy | 编译通过 + 单测；运行时行为未验证（见约束） | ☐ |
| S3-5 | **规范逻辑地址层**：`logical_offset` 改规范块坐标，`bind` 只做物理换算 | `S_P = S_D` 等价锚点 + `S_P ≠ S_D` 折叠用例 | ☐ |

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
