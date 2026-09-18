# PD 路由重构 · 交接说明（S0/S1/S2 已落地，下一步 S3）

## 0. 一句话现状

按「KV 冗余模型」重写 PD 传输路由的**前三步已实现并在开发机容器内验证通过**：删掉两套死代码，
新增零依赖的 `(h, t, c)` 纯算术层（L1）与**规范块边表 + 绑定层（L2/L3）**，
host 侧 23 个用例全绿。**但新层目前没有任何生产调用点**：`PdRouteTable` / `RouteBinder`
只被单测引用，生产仍是旧计划器。**下一步是 S3：接入并切换数据面**（先 PULL 后 PUSH），
同时引入规范逻辑地址层；S2 原定的"与旧计划器逐边比对"已按 §7.1 取消。

- 日期：2026-09-17（S0/S1）→ 2026-09-18（S2）
- 基线：`200939593`
- 存档分支：`pd-routing-s0s1` = `9605a7c6a`（S0/S1）+ `abfcdc422`（S2），**已 push 到
  `origin`（`git@github.com:shifengmin/xllm.git`）**

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
full_sequence_replica  => S_eff = 1     语义性全序列复制（显式声明，S2 新增）
D == 1                 => S_eff = 1     无冗余可消
S <= D 且 D % S == 0   => S_eff = S
其余                    => 报错          不做静默降级
```

**S2 修正**：`S_eff` **不能**由 `G/TP/CP/S` 推导出来。indexer kPool（`G=1`）按规则会得到
`S_eff=4`，但它必须整序列复制（top-k 读全序列），所以由 `GroupTopology::full_sequence_replica`
**声明**。因此原先设想的
`EXPECT_EQ(is_kv_split_cache_block_type(t), S_eff(group) > 1)`
（MLA latent 与 indexer 同 `BlockType::KV`、同 `group_id`，却一个 4 一个 1）**作废**。

### 2.3 GLM 5.3 flash 目标场景

模型 = `glm5_next`（`xllm/python/models/glm5_next.py`），混合架构 45 层。
场景：prefill 8 卡 `TP8 + DCP4` → decode 8 卡 `DP4 + TP2 + DCP2`。

| group | `G` | scope | P: `D`/`S_eff`/`N_rep` | D: `D`/`S_eff`/`N_rep` | DCP |
|---|---|---|---|---|---|
| MLA latent | **1** | BLOCK | 8 / **4** / 2 | 2 / **2** / 1 | ✅ |
| indexer kPool | **1** | BLOCK | 8 / **1** / 8 | 2 / **1** / 2 | ✗（**全序列复制，`full_sequence_replica`，见 §6.1**） |
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

### 4.4 S2 的验证证据（2026-09-18）

| 项 | 结果 |
|---|---|
| `kv_redundancy_test`（L1，含新增的 `full_sequence_replica` 用例） | ✅ **11 tests / 3 suites 全 PASSED** |
| `pd_route_test`（L2/L3：T3 golden + 拓扑矩阵不变量 + T4 字节 golden + T5 mock） | ✅ **12 tests 全 PASSED** |
| MLA golden：表本体 4 条、DP 展开 16 条；写者仅 local rank 0–3 | ✅ |
| KDA golden：表本体 8 条、DP 展开 32 条；P 侧 8 个 rank 全参与 | ✅ |
| indexer：`full_sequence_replica` ⇒ 1 源 rank → 2 个目的副本（对照：不声明时是 4 条 MLA 形状） | ✅ |
| 拓扑矩阵：`G ∈ {1..64} × TP ∈ {1,2,4,8} × split ∈ {1,2,4,8} × {block, sequence}` 全枚举，`validate` 覆盖不变量全成立 | ✅ |
| T4：`local_row = b/4`、`remote_row = b/2` 与文档表格逐行一致；发散方向、`explicit_offsets`、checkpoint 3 子单元均逐字节正确 | ✅ |
| T5：真实内存 + memcpy 端到端字节校验，且"故意把远端行基点错位"必须被校验抓住 | ✅ |
| clang-format **20.1.6**（pre-commit 缓存里的真版本）后复跑 | ✅ 仍全绿 |

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
   **2026-09-18 更新：已可绕过。** 建 `/tmp/torch_npu_shim/torch_npu/torch_npu.h`
   （`#pragma once` + `#include "<真实路径>"`），在真实 flags 后加 `-I/tmp/torch_npu_shim`，
   则 `kv_cache_transfer.cpp` 与 `mooncake_kv_cache_transfer.cpp` 都能编过（实测 rc=0）。
   脚本：`~/pdroute_tools/probe_compile.py`（见 S3 工作日志）。
9. 批量 `sed` 改 build 目录时**务必用 `grep -rIl`（大写 I 跳过二进制）** —— 否则会把
   `libopencv_core4.a` 这类静态库改坏。
10. **CMake 重配置在这棵树里已经不可能成功**（2026-09-18 复核）：
    - `vcpkg install` 的 `detect_compiler` 会认为在跨编译，去找 `aarch64-linux-gnu-gcc`，镜像里只有 `/usr/bin/gcc`；
    - 加 `-DVCPKG_MANIFEST_INSTALL=OFF` 可跳过 vcpkg，但下一关是
      `FETCHCONTENT_SOURCE_DIR_LIBTORCH = /export/home/shifengmin.3/.dcplab/libtorch-src`（主机路径，未挂进容器）；
    - 只要动到任何 `CMakeLists.txt`，`ninja` 就会触发重配置 → 失败 → `build.ninja` 不更新（旧文件仍在）。
    - ⇒ **新目标不要指望 `ninja`**，用 §5.11 的手编回路。
11. **手编回路**（`~/pdroute_tools/s2_host_test.py`）：`compile_commands.json` 取真 flags（库源用
    `reshard_planner.cpp` 的、测试源用 `reshard_planner_test.cpp` 的）编译，再**最小链接**
    `libgtest_main.a + libgtest.a` 即可 —— 新层只用标准库，不需要 `libcommon.a`。
12. **21:58 那棵预编译树不可靠**：`xllm/core/common/libcommon.a` 是坏档案（`malformed archive`），
    镜像里也没有 `-lcust_opapi`，所以**任何依赖旧 planner 的测试在这棵树里链接不起来**（这也是
    §7.1 取消旧路径比对的附带原因）。新层刻意不依赖这些库。

---

## 6. 已知未决问题（S2/S3 前需要定音）

### 6.1 index cache 是「切分」还是「复制」？（影响 §2.3 的 indexer `S_eff`）

`init_index_cache_shape`（`kv_cache_shape.cpp:388-403`）在 `supports_dsa_indexer_cache_sharding() && S > 1`
时把 **index 张量行数 × S**；同一条件下 `kv_cache_estimation.cpp:96-99` 又把 **index 每 token 字节数 × S**。
两处同时生效 ⇒ index 行数 = `n_blocks × S` = 全部规范块数，指向「**每个 rank 保留全序列 index**」（复制）。

若成立，**indexer 的 `S_eff` 应为 1 而非 4**，且行↔规范块映射与 KV 不同（KV 用 `row = canonical / S`，
INDEX 用 `row = canonical`）。

**S2 的处理**：建模上已按"复制"落地 —— `full_sequence_replica = true` ⇒ `S_eff=1`、`N_rep=D`，行映射仍是
统一公式 `row = canonical / S_eff`（退化为 `row = canonical`，正好对上 index 行数 `= n_blocks × S`）。
目标场景下 indexer 的边表因此是 1 个源 rank 扇出到 `N_rep` 个目的 rank（P 侧 8、D 侧 2）。

**✅ 2026-09-18 已实测确认（无需再跑探针）**：生产 DCP4 日志
`kv_cache_shape.cpp:195 Initializing indexer cache with shape: [26052 128 1 257]`，同实例 `blocks: 6513`、`kv_split=4`，
`26052 = 6513 × 4` = 全部规范块 ⇒ **index cache 是复制**，`S_eff=1` 成立。
来源：`~/work/glm5-next-dcp-session/DCP4容量实测-中断存档.md`。

### 6.2 kPool packed 宽度

`cache_head_dim = index_kpool_compress ? index_head_dim*2 + 1 : index_head_dim`。
`glm5_next` 代码默认 `false`（宽度 128），但紧邻注释与 `deepseek_v2_attention.cpp:56` 的
`use_kpool_indexer_ = has_indexer_ && args.index_kpool_compress()` 表明**真实 checkpoint 预期为 true**（宽度 257）。
mock 应对两种各出一例。

**✅ 2026-09-18 已实测：生产实例用的是 257**（上文 `[26052 128 1 257]` 的最后一维）。mock/测试以 257 为主用例。

### 6.3 kv_split 的划分发生在哪一层

`has_rank_preserving_kv_groups`（`disagg_pd_scheduler.cpp:162-173`）在 GLM5-next 的分组集合（KV + LINEAR）上
**恒为 true** ⇒ `rank_local_mapping = true` ⇒ `filter_kv_split_infos` 里 `if (rank_local_mapping) continue`
使 remap **被整体跳过**。若探针确认，则说明 **kv_split 的序列划分是在 D 侧分配 block id 时就完成的
（rank-preserving 契约）**，而不是在传输层做的 —— 这直接决定 S3 应在**传输层**还是**分配层**引入 canonical block。

**S2 期间的旁证（读码，非探针）**：MLA 的描述符来自 `describe_replicated_tensor`（`enable_mla` 分支），内容是
"整行一个 span + `owner_tp_rank=0`"，**描述符里根本没有块身份**。既然 manifest 无法表达"哪些规范块归哪个 rank"，
该划分只可能来自调度侧分配的 block id ⇒ 倾向"canonical block 应引入在**契约层/分配层**"。

**✅ 2026-09-18 已确认**（`_resume/DCP×PD兼容性与linear-cache静态审查-20260915.md`）：
`rank_local_mapping = kv_split_size > 1 && has_rank_preserving_kv_groups(resp)`，对 GLM5-next 的普通 KV 组恒真
⇒ `filter_kv_split_infos` 的 remap 整体跳过 ⇒ **划分确实在 D 侧分配 block id 时完成**。S3 按"契约层"设计。

### 6.5 旧 PD 路径支持哪些 kv_split 形状（S3 的范围约束）

同一份兼容性审查（只读代码 + 实测）给出：

| 组合 | 旧路径 | 失败模式 |
|---|---|---|
| P 开(k>1) / D 不开(1) | ❌ | planner 放行，但块映射层 `kv_cache_transfer.cpp` 的 size mismatch ⇒ `CHECK(kv_transfers.wait())` **abort** |
| P 不开(1) / D 开(k>1) | ❌ | 建链期 `supports_partition_layout` 皆假 ⇒ `invalid` |
| 两端都开、**k 相同** | ✅ **唯一可用** | 强制 `same_partition`（cp_rank + kv_split_rank 全等），D 只把 rank 对齐的 P 设为 ACTIVE |
| 两端都开、k 不同 | ❌ | `supports_partition_pair` 假 ⇒ 建链期报错 |

**前置条件：`kv_split == cp_size × tp_size == world / dp`。** 目标场景正好在此范围之外 —— 这正是重构要打开的新形状。

> **⚠️ 2026-09-18 更正（工作日志第 5 轮）**：目标场景的准确描述是 **`cp_size = 4`（PCP 4）+ `tp_size = 8`（world 32，dp 1），
> `kv_split_size` 未设置 ⇒ effective = `cp_size` = 4**。理由：`world / kv_split = 8 = tp_size` 正是归档里"DCP 组为
> `global_rank % (world/kv_split)` 的 strided 组"（= 固定 tp、变动 cp），而 `ContextParallelTopology` 只允许
> "DCP 划分 PCP"或"DCP 覆盖整个 DP-local 域"两种形状，`cp=1 + S=4` 会直接 `CHECK` 崩溃。
> 因此 `slice = dcp_rank = cp_rank`，**不是** `tp % 4` —— S2 的 `KvLayoutIndex::slice_of` 需要按此修正（详见工作日志第 5 轮）。
> 旧前提 `kv_split == cp*tp`（这里 4 ≠ 32）依然不成立，所以"目标形状在旧路径支持范围之外"的结论不变。⇒ **S3-4 不能只"换一条算路"**，必须同时替换
`filter_kv_split_infos` / `rotate_dst_rank` 那套 "`S == TP` 且两侧 rank 1:1 对齐" 的隐含前提；
另外 `fingerprint` 不含 `kv_split`，跨实例的 `kv_split` / `B_token` 一致性必须由新校验兜住（S5）。

### 6.4 其他（详见 review 文档）

- `B_token` 暂定取 `options_.block_size()`（= 现有 `CacheTensorManifest::block_token_capacity`），
  需确认两侧 `--block_size` 配置相同即成立；
- F7 `RouteBinder::bind` 按 `t` 预分桶 ✅（已实现）；
- F8 边表按 `(本侧拓扑, 对侧拓扑)` 缓存，而非按 peer（S3 接入时实现）；
- F9 块内 token / checkpoint 行的压缩 ✅：`RouteBinder` 用 `units_per_resource` + 每 rank 的
  `local_head_count` 计算子单元步长，输出再合并相邻区间（因此目标场景下"整行搬运"是一条 region）。

---

## 7. S2 已完成（2026-09-18），下一步是 S3

原计划 5 项的去向：

| # | 原计划 | 结果 |
|---|---|---|
| 1 | `PdRouteTable`（DP 组内局部 rank 的边表） | ✅ 已实现并跑绿 |
| 2 | `RouteBinder` + `BufferDirectory`（规范块 → 物理字节区间） | ✅ 已实现并跑绿 |
| 3 | 与旧计划器并行、逐边/逐字节比对 | ❌ **取消**：旧路径在目标配置下不可表达（见 §7.1），验收改为 golden + mock |
| 4 | 目标场景 golden（MLA / KDA） | ✅ T3 全绿（表本体 4 / 8 条，DP 展开 16 / 32 条） |
| 5 | host mock 端到端（T5，含判别性反例） | ✅ 全绿 |

### 7.1 为什么取消"与旧计划器比对"

MLA 走 `describe_replicated_tensor`（整行、`owner_tp_rank=0`、REPLICATED），而 `select_sources` 的去重是
"`same_partition` + `tp_rank == owner_tp_rank`"。在 `TP8 + DCP4` 下：用运行时真实的 `kv_split_rank = rank % S`
⇒ 除 dst rank 0 外 **0 个写者**；用 fallback `rank/(world/kv)` ⇒ 每个目的 rank **2 个写者**。两条路都过不了
覆盖率校验 —— 目标场景本来就是旧路径不支持的形状，所以"S2 必须新旧逐边一致"在目标场景上不可执行。
旧路径支持的形状（`G ≥ TP` 且 `S=1`、`G < TP` 且 `S=1`、sequence-scoped）仍可比对，属可选补充。

### 7.2 新增/改动文件

| 文件 | 内容 |
|---|---|
| `xllm/core/framework/kv_cache_transfer/pd_route_table.{h,cpp}` | `RouteEdge` + `build`（`head_pairs ⊗ block_map`）+ `validate`（逐 `(目的 rank, 源片, head)` 覆盖不变量 + 源侧唯一写者） |
| `xllm/core/framework/kv_cache_transfer/route_binder.{h,cpp}` | `RouteRegion` / `BufferDirectoryEntry` / `BufferDirectory` / `PeerCacheView` + `bind` |
| `tests/core/framework/kv_cache_transfer/pd_route_test.cpp` | 12 个用例：T3 golden（MLA / KDA / indexer）、拓扑矩阵不变量、T4 字节 golden、T5 mock |
| `xllm/core/framework/kv_cache_transfer/kv_redundancy.{h,cpp}` | `GroupTopology::full_sequence_replica` + 派生规则第 2 行 |
| 两处 `CMakeLists.txt` | `cc_library(pd_route_table)` / `cc_library(route_binder)` / `cc_test(pd_route_test)` |

### 7.3 验证方式（全部在 jd-node-98 容器内，本机不能构建）

`~/pdroute_tools/s2_host_test.py`：从 `compile_commands.json` 取真实 flags 手编 4 个 TU，再用最小链接
（`libgtest_main.a` + `libgtest.a`）成两个测试二进制并运行。结果 `kv_redundancy_test` **11/11 PASSED**、
`pd_route_test` **12/12 PASSED**（clang-format 20.1.6 之后复跑仍全绿）。详见 §4.4 与 §5.10。

**顺序原则不变**：S3 之前不要同时改 `block_size` 语义（那是独立的 S6）。

### 7.4 未完成清单与下一个会话的起点

**最重要的前提**：S2 只交付了 L1/L2/L3 与 host 验证，`PdRouteTable` / `RouteBinder` 在
`xllm/` 下的**生产调用点为 0**（`grep -rn "PdRouteTable\|RouteBinder" xllm | grep -v kv_cache_transfer/` 为空）。
所以"端到端 PD 测试"目前**不可能**跑：既没有接线，目标模型也还不支持 PD 分离。

| 剩余 | 内容 | 前置 |
|---|---|---|
| **S3** | ① 写 manifest → `PeerCacheView` 适配器；② 统一入口 `KVCacheTransfer::transfer(edges, canonical_blocks, opcode)`，先切 PULL 再切 PUSH；③ 引入规范逻辑地址层（`logical_offset` 改规范块坐标，`bind` 只做物理换算） | S2 ✅ |
| **S4** | 删旧路径（§7.1 的 D1/D2/D3）、删 `rank_local_mapping`、`SetCachePeer` 去掉 mode/plan、建链收敛 | S3 |
| **S5** | 门禁提前：`PdTopo` → 完整 `KvTopology`，`KvRedundancy::derive` 在建链前报 C1/C2 与 `B_token` 不一致 | S3 |
| **S6（可选、独立）** | 解除 `block_size × kv_split_size` 绑定（`llm_engine.cpp:653`），触及 BlockManager / prefix cache 哈希 | S3 之后独立评估 |
| **探针** | §6.1 index 行数比、§6.2 kPool 打包宽度（128 / 257）、§6.3 `filter_kv_split_infos` 是否真被跳过 | 需要真实实例 |
| **T6** | 真实 PD 逐 `(block, group)` 字节/校验和验收 | **GLM5.3flash 尚不支持 PD 分离** |

**S3 进展（2026-09-18，详见 `pd_routing_s3_worklog_20260918.md` 与 proposal §8.1）**：

- ① **已完成**：`cache_directory.{h,cpp}` + `cache_directory_test.cpp`（19/19）与
  `pd_route_integration_test.cpp`（4/4，真实 `describe_cache_tensor` + memcpy 逐字节），给出两条实现约束——
  **COMPOSITE（CONV）描述符不在"每条边一个 head 区间"的表达范围内**（只在非 MLA 实例可达；适配器显式拒绝，
  S3-4 需要决定走旧 planner 还是给边表加 per-component 偏移）；**整资源描述符只在本地 1 个 head 时可路由**
  （`H_l == 1`，head 身份取自 rank）。MLA 实例下的 SSM/CONV 正是整资源形态，因此这条准入规则是链路能否建立的前提。
- **S3-3 已完成**：链路 `真实张量 → describe_cache_tensor → manifest → PeerDirectory → PdRouteTable → bind → memcpy`
  在 4 个场景（MLA kv4→kv4/kv4→kv2/kv2→kv4、非 MLA 头分片 cp4/tp8/kv4→cp4/tp4/kv2）逐字节正确。
- 探针全部关闭（§6.1/6.2/6.3 见 §6），S3-0（torch_npu include shim）也已完成，生产 TU 可编译验证。
- **S3-5 前置阻塞已定位（第 5 轮，必须先修）**：物理切片 = `ContextParallelTopology::dcp_rank`（NPU 侧
  `qwen_dcp_attention.cpp` 用 `dcp_group.rank()` 构造 `KVShardLayout`），而 S2 的
  `KvLayoutIndex::slice_of = (cp*D_tp + tp%D_tp) % S_eff` 是另一套分组。pilot 下正确值是 `slice = cp_rank`
  （`cp_size=4`），写者是 `(cp=s, tp=0)`。**不修就会静默搬错块**，因此 S3-4 接线前必须完成：给
  `KvRedundancy::derive` 加 C3（DCP 形状）校验 + 重写 `slice_of`/`replica_of`/`writers_of` + 重算 S2 的
  rank golden（不变量与边表规模不变）。修法与测试清单见工作日志第 5 轮。
- 未决：`coordinates.kv_split_rank`（= `dcp_rank`）与 `slice_of` 的对账即上述修正；`bind` 的 `local_rank` 校验
  只覆盖 MAIN 命名空间。
- ② 数据面切换与 ③ 规范逻辑地址层**未开始**（③ 的阻塞项已定案，先做 ③ 再做 ②）。

**下一个会话的第一件事（建议顺序）**：

1. 复跑一次 S2 的 host 测试，确认环境仍然可用（命令见 §7.3）。若 `s2_host_test.py` 不在，
   用 §5.11 的规则重建（`compile_commands.json` 取 flags + 最小链接 gtest）。
2. ~~写适配器~~ **✅ 已完成（2026-09-18，`cache_directory.{h,cpp}`，18/18 单测）**。落地时的字段对应关系：

    | L3 字段 | 来源 |
    |---|---|
    | `buffer_id` | `CacheTensorManifest::mooncake_buffer_id` |
    | `resource_count` / `resource_stride_bytes` / `buffer_bytes` | 同名字段 |
    | `explicit_offsets` | `CacheTensorManifest::explicit_resource_offsets` |
    | `units_per_resource` | BLOCK 组取 `block_token_capacity`；SEQUENCE 组取 `physical_rows_per_resource` |
    | `topology.{cp,tp,kv_split}_size`、`tokens_per_block` | `ParallelCoordinates` / `options_.block_size()` |
    | `group.global_head_count` | MLA/indexer 为 1；SSM 取 `linear_value_head_count`；KV 取 `kv_head_count`。**非 MLA 实例的 CONV 无法表达**（见下） |
    | `group.head_bytes` | 由描述符的 `bytes_per_region` 推出（整资源 span 则 `resource_stride_bytes / units`）；声明值非 0 时必须相符 |
    | `group.sequence_scoped` | `CacheResourceScope::SEQUENCE`，且必须与声明一致 |
    | `group.full_sequence_replica` | **只有 indexer kPool 声明 `true`**；MLA latent 声明 `false`（实测 KV 行数 `6513 = 26052/4` ⇒ `S_eff=4`）。此处原表格把二者混为一谈，已更正 |
    | `row_offsets` | 仅 `explicit_offsets` 时填，来自 `GlobalXTensor` 的页基点；非页映射张量给出基点即报错 |

    **落地时新发现的两条约束**（详见 proposal §8.1 与工作日志第 3、4 轮）：
    - `describe_conv` 的 **COMPOSITE 描述符**（`conv_key_a` / `conv_key_b` / `conv_value` 混在一行）不落在
      "每条边一个 head 区间"的表达范围内，适配器**显式拒绝**；S3-4 需决定 COMPOSITE 组继续走旧 planner
      还是给边表加 per-component 字节偏移。该分支**只在非 MLA 实例可达**（`enable_mla == true` 时先命中
      replicated 分支）。
    - **整资源（whole-resource）描述符只在本地 1 个 head 时可路由**（`H_l == 1`）：它不带 head 轴，
      head 身份只能由 rank 推出。MLA 实例下 SSM/CONV 也是整资源 span，所以这条规则决定链路能否建立。
3. 然后按 S3 → S4 → S5 推进；每一步都保持 `S_P = S_D` 的退化配置作为等价锚点（已在
   `pd_route_integration_test.cpp` 里固化为第一个场景）。

**已锁定的决策（不要再翻）**：
- `S_eff` 不匹配时**报错**，不静默退 1；语义性全序列保留必须显式声明（`sequence_scoped` / `full_sequence_replica`）；
- `full_sequence_replica` 承载 indexer kPool（同组内 MLA latent 与 indexer 的 `S_eff` 不同）；
- 取消"与旧 ReshardPlanner 逐边比对"（原因见 §7.1），S2 验收 = golden + mock；
- F1 的 `EXPECT_EQ(is_kv_split_cache_block_type(t), S_eff > 1)` 断言作废。

**环境禁忌（详见 §5.10–5.12）**：不要指望 `ninja`/cmake 重配置；不要用 21:58 预编译树里的
`libcommon.a`（坏档案）与 `-lcust_opapi`（镜像里没有）；新层只依赖标准库 + gtest，正是为了绕开这些。

## 8. 交接清单

| 项 | 位置 |
|---|---|
| S0/S1（已提交） | `9605a7c6a`，subject：`refactor(kv_cache_transfer): add KV redundancy layer, drop dead routing code` |
| S2（已提交并 push） | `abfcdc422`，subject：`refactor: resolve pd transfers through canonical block edges.`（14 files, +2013/-53） |
| 分支 | **`pd-routing-s0s1`**，已 push 到 `origin` = `git@github.com:shifengmin/xllm.git`（upstream 已设）。Review 只见 S2 一笔：`https://github.com/shifengmin/xllm/compare/9605a7c6a...abfcdc422` |
| 四份设计文档 + 本文 | `docs/zh/design/{kv_redundancy_model_and_pd_routing,pd_routing_simplification,pd_route_verification_plan_glm53flash,pd_transfer_redesign_proposal,pd_routing_handover}_20260917.md`（已随该 commit 一并提交） |
| 开发机工作树（含改动 + 可手动编译） | jd-node-98 `~/workspace/xllm-pdroute` |
| 开发机验证脚本 | jd-node-98 `~/pdroute_tools/`（S2 用 `s2_host_test.py`） |
| 原树（**未被改动，勿动**） | jd-node-98 `~/workspace/xllm-dcp-fp32`（含 5 个与本工作无关的未提交文件） |
| 容器镜像 | `quay.io/jd_xllm/xllm-ai:xllm-dev-a3-arm-cann9-20260911` |

### 8.1 恢复现场

```bash
cd <xllm 仓库>
git fetch origin pd-routing-s0s1      # 若换机器/换 clone
git checkout pd-routing-s0s1
git log --oneline -2                  # abfcdc422 (S2) -> 9605a7c6a (S0/S1) -> 200939593
```

- 该分支基于 detached HEAD `200939593` 建立，**未 push**；
- 工作树里 `third_party/{Mooncake,xllm_atb_layers,xllm_ops}` 的改动是**仓库原有的**，与本工作无关，未被提交；
- `docs/zh/design/` 下其他人的文档仍是 untracked，本 commit 只加入了上面列出的 5 份。

### 8.2 提交后已复核

`clang-format` pre-commit 钩子（`v20.1.6`）在首次提交时重排了 `kv_redundancy_test.cpp`
（因为我在最后一次 clang-format 之后又改过该文件）。已重新格式化并再次同步到开发机，**重跑容器内单测仍 10/10 PASSED**，
文件 sha256 本地与开发机一致（`847b32b6bf27f3ad…`）。
