# PD 路由重构 · 交接说明（S0/S1 已落地，S2 待做）

## 0. 一句话现状

按「KV 冗余模型」重写 PD 传输路由的**前两步已实现并在开发机上验证通过**：删掉了两套死代码，
新增了零依赖的 `(h, t, c)` 纯算术层 + 10 个穷举单测。**下一步是 S2：新增 `PdRouteTable` /
`RouteBinder` / `BufferDirectory`，与旧计划器并行跑逐边比对。**

- 日期：2026-09-17
- 本地基线：`200939593`（detached HEAD）
- 存档分支：`pd-routing-s0s1`（本地，未 push）

---

## 1. 背景阅读顺序（必读，按序）

| # | 文档 | 作用 |
|---|---|---|
| 1 | `kv_redundancy_model_and_pd_routing_20260917.md` | **概念模型**：`D`/`G` 冗余定义、`(h,t,c)` 三索引、C1/C2 约束、路由算法、§7 规范逻辑地址空间 |
| 2 | `pd_routing_simplification_20260917.md` | **现状盘点**：三套 rank 配对实现（两套死代码）、18 类特例分支、按模型的简化映射 |
| 3 | `pd_route_verification_plan_glm53flash_20260917.md` | **方案 review（F1–F9）+ 验收设计**：GLM 5.3 flash 场景解析、T1–T6 单测与 mock 用例 |
| 4 | `pd_transfer_redesign_proposal_20260917.md` | **重构方案**：分层、接口草案、删除清单、S0–S6 阶段；§8.1 是本次实施进展 |

四份都在 `docs/zh/design/`。

---

## 2. 核心结论（跳过细节也要记住的）

### 2.1 模型

`kv_split` 的唯一目的是**消除 KV 冗余**。由此定义：

```
Hl  = max(G / TP, 1)      每 rank 本地 head 数
D_tp= max(TP / G, 1)      TP 冗余度
Hc  = TP / D_tp           head 类数，恒有 Hc × Hl == G
D   = CP × D_tp           复合冗余度
S_eff                     本组实际切分宽度
N_rep = D / S_eff         残余冗余度
```

三个正交索引：`h`（head 类）、`t`（序列片）、`c`（冗余副本号）。
**路由 = head 路由 ⊗ block 路由**：`h` 走全局 head 区间求交，`t` 走块号取模，`c` 在源侧取 0（唯一写者）、
在目的侧全遍历（副本都要填）。

### 2.2 `S_eff` 派生规则（本次最关键的修正，F1）

`S` 是**实例级**配置，而 `D` 依赖**每个缓存组**的 `G`。按实例级判 `S ≤ D` 会把 GLM5-next 的
KDA 缓存（`G=64, TP=8 ⇒ D=1`，而实例 `S=4`）误拒。

```
sequence_scoped        => S_eff = 1     无块维可切
D == 1                 => S_eff = 1     无冗余可消
S <= D 且 D % S == 0   => S_eff = S
其余                    => 报错          不做静默降级
```

这条规则**推导出**了现在硬编码的白名单 `is_kv_split_cache_block_type`（`block/block.h:75-88`），
改造后该函数应降级为断言 `EXPECT_EQ(is_kv_split_cache_block_type(t), S_eff(group) > 1)`。

### 2.3 GLM 5.3 flash 目标场景

模型 = `glm5_next`（`xllm/python/models/glm5_next.py`），混合架构 45 层。
场景：prefill 8 卡 `TP8 + DCP4` → decode 8 卡 `DP4 + TP2 + DCP2`。

| group | `G` | scope | P: `D`/`S_eff`/`N_rep` | D: `D`/`S_eff`/`N_rep` | DCP |
|---|---|---|---|---|---|
| MLA latent | **1** | BLOCK | 8 / **4** / 2 | 2 / **2** / 1 | ✅ |
| indexer kPool | **1** | BLOCK | 8 / **4** / 2 | 2 / **2** / 1 | ✅（**S_eff 待实测，见 §6.1**） |
| KDA conv / ssm | **64** | SEQUENCE | 1 / **1** / 1 | 1 / **1** / 1 | ✗ |

**场景可解的必要条件是 `G = 1`**（P 需 `max(8/G,1) ≥ 4` 且被 4 整除 ⇒ `G∈{1,2}`；D 需 `max(2/G,1) ≥ 2` ⇒ `G=1`）。
换成 GQA（`G ≥ 4`）在模型上就不成立，应在建链前被拒 —— 已有负例单测固定。

其他已确认事实：`qk_rope_head_dim = 0` ⇒ **MLA 没有 VALUE 张量**（numel=0 被 `add_tensor` 跳过）；
`KEY/VALUE/INDEX/*_SCALE` 同属 `group_id = cache_group_id(BlockType::KV) = 0`。

---

## 3. 已落地的代码（本次 change set）

### 3.1 新增（762 行）

| 文件 | 行数 | 内容 |
|---|---|---|
| `xllm/core/framework/kv_cache_transfer/kv_redundancy.h` | 190 | `KvTopology` / `GroupTopology` / `KvRedundancy` / `KvLayoutIndex` / `CanonicalBlock` |
| `xllm/core/framework/kv_cache_transfer/kv_redundancy.cpp` | 222 | 同上实现；**零非标准库依赖** |
| `tests/core/framework/kv_cache_transfer/kv_redundancy_test.cpp` | 350 | 10 个用例 |

关键 API：

```cpp
static bool KvRedundancy::derive(const KvTopology&, const GroupTopology&,
                                 KvRedundancy* out, std::string* error);
KvLayoutIndex(topology, redundancy);          // rank <-> (h,t,c)
  head_begin/head_end(head_class)
  head_class_of(tp) / slice_of(cp,tp) / replica_of(cp,tp)
  writer_of(dp, h, t, int32_t* rank)          // c == 0 的唯一写者
  replicas_of(dp, h, t, std::vector<int32_t>*)// N_rep 个副本
CanonicalBlock(tokens_per_block, split);      // local_row = canonical / split
```

### 3.2 删除（-323 行）

| 项 | 位置 |
|---|---|
| `push_route.{h,cpp}` + `push_route_test.cpp` + 两处 CMake 条目 | 全树生产调用点为 0 |
| 基类 `KVCacheTransfer::merge_kv_blocks` 实现（108 行 modulo 路由） | `kv_cache_transfer.cpp:266-373`，改为**纯虚** |
| `mooncake_kv_cache_transfer.cpp` 里只 include 不调用的 `push_route.h` | 第 30 行 |

两处 CMakeLists 已加入 `cc_library(kv_redundancy)` 与 `cc_test(kv_redundancy_test)`，并把
`kv_cache_transfer` 的 DEPS 从 `:push_route` 换成 `:kv_redundancy`。

---

## 4. 验证证据（开发机 jd-node-98）

### 4.1 环境

- 工作树：`~/workspace/xllm-pdroute`（`xllm-dcp-fp32` 的完整副本，**原树未被改动**）
- 容器：`quay.io/jd_xllm/xllm-ai:xllm-dev-a3-arm-cann9-20260911`
  （`cmake 3.27.9` / `ninja 1.11.1` / `gtest 1.14.0`，含 `/usr/local/Ascend/cann-9.0.0`、`python3.11.15`）
- 工具脚本：`~/pdroute_tools/`（`pdroute_container_manual.py` 是手动编译驱动）

### 4.2 结果

| 项 | 结果 |
|---|---|
| `kv_redundancy.cpp` 用**项目真实编译命令**（取自 `compile_commands.json`）编译 | ✅ |
| `kv_redundancy_test.cpp` 编译 + 链接 vcpkg gtest | ✅ |
| 运行 | ✅ **10 tests / 3 suites 全 PASSED** |
| 两处 CMakeLists 语法（真实 cmake 3.27.9 + 桩宏 include 整文件） | ✅ `syntax OK`，新条目字段正确 |

派生值实测与文档表格逐项吻合：

```
P MLA latent (TP8,S4)   Hl=1  D_tp=8 Hc=1 D=8 split=4 N_rep=2
D MLA latent (TP2,S2)   Hl=1  D_tp=2 Hc=1 D=2 split=2 N_rep=1
P KDA (TP8,S4,seq)      Hl=8  D_tp=1 Hc=8 D=1 split=1 N_rep=1
D KDA (TP2,S2,seq)      Hl=32 D_tp=1 Hc=2 D=1 split=1 N_rep=1
counter G=4 TP8 S4      REJECT（D=2 < 4）
counter G=2 TP8 S3      REJECT（3 ∤ 4）
```

### 4.3 复现命令

```bash
# 在 jd-node-98 上；绕开需要整树 vcpkg 重装的 reconfigure，用真实 flags 手动编译链接
sudo docker run --rm --privileged \
  -v ~/workspace/xllm-pdroute:/export/home/shifengmin.3/workspace/xllm-dcp-fp32 \
  -v ~/pdroute_tools/pdroute_container_manual.py:/tmp/manual.py:ro \
  --entrypoint bash \
  quay.io/jd_xllm/xllm-ai:xllm-dev-a3-arm-cann9-20260911 \
  -c 'cd /export/home/shifengmin.3/workspace/xllm-dcp-fp32 && python3 /tmp/manual.py'
```

---

## 5. 开发机上的坑（省下重复踩的时间）

1. **整机没有 ninja**。`pip3 install --user ninja` 可装到 `~/.local/bin/ninja`（已装）。
2. **系统 cmake 3.22 < 项目要求的 3.26**。`pip3 install --user "cmake>=3.26,<4"` 装 3.31.10（已装）。
   **但容器内自带 cmake 3.27.9，才是与原 `build/` 匹配的版本。**
3. **`build/` 目录是在容器里配置的**，缓存里引用 `/usr/local/python3.11.15`、`/usr/local/Ascend/cann-9.0.0`
   —— 这些在宿主机上**不存在**。所以宿主机上 `ninja` 必然失败，必须进容器。
4. **换路径会导致 vcpkg 全量重装**：`build/` 与 `_deps/vcpkg-src` 里的状态与绝对路径绑定，
   复制到新路径后 `vcpkg install` 会从 242 个 port 重新开始（实测启动后即放弃，并**破坏
   `vcpkg_installed/arm64-linux/{include,lib}`**，需从原树拷回）。
   → 规避办法：把副本**挂载到原路径**再进容器（`-v <copy>:/export/home/shifengmin.3/workspace/xllm-dcp-fp32`）。
5. **`xllm_ops` precompile gate**：`third_party/xllm_ops/build.sh` 在本 checkout 中缺失，
   而 gate 比较 `git rev-parse HEAD`，必然触发并失败。
   → `export XLLM_OPS_GIT_HEAD_CACHED=$(git -C third_party/xllm_ops rev-parse HEAD)`；
   容器里还要先 `git config --global --add safe.directory '*'`（否则 git 返回空）。
6. **`VCPKG_FORCE_SYSTEM_BINARIES=1`** 必须设，否则 vcpkg 在 aarch64 上拒绝 bootstrap。
7. **容器必须 `--privileged`**，否则 `ninja: fatal: posix_spawn: Operation not permitted`。
8. **`<torch_npu/torch_npu.h>` 在三个可用镜像里都无法满足**
   （实际在 `torch_npu/include/torch_npu/csrc/libs/torch_npu.h`）。
   → 任何包含 `platform/stream.h` 的生产 TU 都编不过。**这是环境问题，不是本次改动引入的**
   （已做对照实验：用同一套 flags 编译 `git show HEAD:` 的改动前同名文件，失败信息逐字相同）。
9. 批量 `sed` 改 build 目录时**务必用 `grep -rIl`（大写 I 跳过二进制）** —— 否则会把
   `libopencv_core4.a` 这类静态库改坏。

---

## 6. 已知未决问题（S2/S3 前需要定音）

### 6.1 index cache 是「切分」还是「复制」？（影响 §2.3 的 indexer `S_eff`）

`init_index_cache_shape`（`kv_cache_shape.cpp:388-403`）在 `supports_dsa_indexer_cache_sharding() && S > 1`
时把 **index 张量行数 × S**；同一条件下 `kv_cache_estimation.cpp:96-99` 又把 **index 每 token 字节数 × S**。
两处同时生效 ⇒ index 行数 = `n_blocks × S` = 全部规范块数，指向「**每个 rank 保留全序列 index**」（复制）。

若成立，**indexer 的 `S_eff` 应为 1 而非 4**，且行↔规范块映射与 KV 不同（KV 用 `row = canonical / S`，
INDEX 用 `row = canonical`）。

**定音探针**：目标配置（TP8 + DCP4）下单次启动，打印 `index_cache_shape()` 与 `k_cache_shape()` 的行数比、
`kv_cache_cap.n_blocks()`、`num_indexer_layers()`。

### 6.2 kPool packed 宽度

`cache_head_dim = index_kpool_compress ? index_head_dim*2 + 1 : index_head_dim`。
`glm5_next` 代码默认 `false`（宽度 128），但紧邻注释与 `deepseek_v2_attention.cpp:56` 的
`use_kpool_indexer_ = has_indexer_ && args.index_kpool_compress()` 表明**真实 checkpoint 预期为 true**（宽度 257）。
mock 应对两种各出一例。

### 6.3 kv_split 的划分发生在哪一层

`has_rank_preserving_kv_groups`（`disagg_pd_scheduler.cpp:162-173`）在 GLM5-next 的分组集合（KV + LINEAR）上
**恒为 true** ⇒ `rank_local_mapping = true` ⇒ `filter_kv_split_infos` 里 `if (rank_local_mapping) continue`
使 remap **被整体跳过**。若探针确认，则说明 **kv_split 的序列划分是在 D 侧分配 block id 时就完成的
（rank-preserving 契约）**，而不是在传输层做的 —— 这直接决定 S3 应在**传输层**还是**分配层**引入 canonical block。

### 6.4 其他（详见 review 文档）

- `B_token` 暂定取 `options_.block_size()`（= 现有 `CacheTensorManifest::block_token_capacity`），
  需确认两侧 `--block_size` 配置相同即成立；
- F7 `RouteBinder::bind` 应按 `t` 预分桶（`O(#blocks)` 而非 `O(#edges × #blocks)`）；
- F8 边表按 `(本侧拓扑, 对侧拓扑)` 缓存，而非按 peer；
- F9 `bind_regions` 的 `repeat_count / local_stride / remote_stride`（块内 token 维压缩）**必须保留**。

---

## 7. 下一步：S2 的具体任务

按 `pd_transfer_redesign_proposal_20260917.md` §3.3/§3.4 与 `pd_route_verification_plan_glm53flash_20260917.md`
第三部分（T3/T4/T5）：

1. **`PdRouteTable`**：`build(src_topology, src_group, dst_topology, dst_group, edges)`。
   `RouteEdge` 只含 **DP 组内局部 rank**（F3）：`{src_local_rank, dst_local_rank, head_begin, head_end, src_slice, dst_slice}`；
   全局 rank = `dp * (CP*TP) + local_rank`，DP 配对由 `TransferKVInfo.dp_rank` 给出。
2. **`RouteBinder`** + **`BufferDirectory`**：规范块 → 物理 `(buf_id, offset, length)`，
   按 `t` 预分桶，`explicit_offsets` 支持 XTensor。
3. **与旧计划器并行比对**：同构/异构配置下，新边表与原 `select_sources` 的 ACTIVE 集合**逐边一致**；
   `S_P = S_D` 时 `RouteBinder` 输出与 `bind_outgoing_regions` **逐字节一致**。
4. **目标场景 golden**：MLA 16 条边（写者只有 P rank 0–3）、KDA 32 条边（P 侧 8 个 rank 全参与）。
5. **host mock 端到端（T5）**：真实内存 + 本地 memcpy 代替 RDMA，验证「边表 + 绑定 + 搬运」合起来字节正确；
   含"故意打乱一条边必须校验失败"的判别性反例。

**顺序原则**：S2 必须"新旧并行 + 逐边逐字节比对"通过后才进 S3；
S3（规范逻辑地址）之前**不要**同时改 `block_size` 语义（那是独立的 S6）。

---

## 8. 交接清单

| 项 | 位置 |
|---|---|
| 本地改动 | `xllm` 仓库分支 **`pd-routing-s0s1`**（本地，未 push）。见 `git log -1 --format=%H pd-routing-s0s1`；subject：`refactor(kv_cache_transfer): add KV redundancy layer, drop dead routing code` |
| 四份设计文档 + 本文 | `docs/zh/design/{kv_redundancy_model_and_pd_routing,pd_routing_simplification,pd_route_verification_plan_glm53flash,pd_transfer_redesign_proposal,pd_routing_handover}_20260917.md`（已随该 commit 一并提交） |
| 开发机工作树（含改动 + 可手动编译） | jd-node-98 `~/workspace/xllm-pdroute` |
| 开发机验证脚本 | jd-node-98 `~/pdroute_tools/` |
| 原树（**未被改动，勿动**） | jd-node-98 `~/workspace/xllm-dcp-fp32`（含 5 个与本工作无关的未提交文件） |
| 容器镜像 | `quay.io/jd_xllm/xllm-ai:xllm-dev-a3-arm-cann9-20260911` |

### 8.1 恢复现场

```bash
cd <xllm 仓库>
git checkout pd-routing-s0s1          # 从 detached HEAD 切过来
git log --oneline -1                  # refactor(kv_cache_transfer): add KV redundancy layer, ...
```

- 该分支基于 detached HEAD `200939593` 建立，**未 push**；
- 工作树里 `third_party/{Mooncake,xllm_atb_layers,xllm_ops}` 的改动是**仓库原有的**，与本工作无关，未被提交；
- `docs/zh/design/` 下其他人的文档仍是 untracked，本 commit 只加入了上面列出的 5 份。

### 8.2 提交后已复核

`clang-format` pre-commit 钩子（`v20.1.6`）在首次提交时重排了 `kv_redundancy_test.cpp`
（因为我在最后一次 clang-format 之后又改过该文件）。已重新格式化并再次同步到开发机，**重跑容器内单测仍 10/10 PASSED**，
文件 sha256 本地与开发机一致（`847b32b6bf27f3ad…`）。
