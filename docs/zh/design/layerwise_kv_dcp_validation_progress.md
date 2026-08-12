# Layerwise KV DCP 验证进度

## 验证目标

- 分支：`codex/feat/layerwise-kv-scratch`
- 构建：`python setup.py build --device npu`
- 单机拓扑：16 个 NPU rank，DP=2，attention TP=8，decode DCP=2
- 运行模式：ATB eager、关闭 graph、单机混布启动
- 验收：构建成功，所有 rank 完成 layerwise KV 分配并进入可服务状态，执行真实请求验证 prefill 和 decode

## 环境与初始状态

- 日期：2026-08-12（UTC）
- 硬件：16 张 Ascend 910，初始检查均为 `Health=OK`，无运行中的 NPU 进程
- 当前提交：`f0b65fa7a feat: replace layerwise null KV with shared scratch cache.`
- 分支前置提交：`029979ea9`、`00cf8ef19`
- 测试模型默认路径：`/export/home/models/GLM-5-final-w8a8`
- 参考脚本：`tools/start_decode_dcp_kv_validation.sh`
- 工作区初始已有未跟踪脚本 `tools/start_decode_dcp_kv_validation.sh`，以及 `third_party/torch_npu_ops` 子模块状态变化；均保留，不主动覆盖或回退。

## 进度记录

### 0. 上下文恢复记录

- 2026-08-12：首次上下文压缩后已重新阅读本文档，并复核工作区、验证脚本、16 张 NPU、模型路径及构建修复状态；继续执行第三次完整 NPU 构建。
- 2026-08-12：第二次上下文压缩后重新阅读全文并复核原始响应日志。DCP2 与 DCP 关闭基线的事实题、算术题和技术题语义一致；两条 2504-token 并发请求均为 HTTP 200。此前唯一 HTTP 500 来自 13004-token 测试输入超过 4096 限制，属于预期拒绝。构建与验证修复已分别提交为 `702f1cd23` 和 `d3718eb59`，16 张 NPU 当前均健康且无运行进程；继续 review/profile host metadata 热路径。
- 2026-08-12：第三次上下文压缩后重新阅读全文并复核当前未提交 diff。完整 NPU 构建已由独立 build agent 在当前快照上通过；继续检查 prepare stream/compute stream 同步、metadata 生命周期及 decode/spec-verify/空 batch 分支，测试 agent 正在执行最终功能与性能回归。
- 2026-08-12：第四次上下文压缩后重新阅读全文。冻结优化代码的完整 NPU 构建已通过；测试 agent 已完成 11/11 placement/config/mapping、12/12 KV transfer/Mooncake、WORLD16 DP2/TP8/DCP2 长请求及语义回归，服务期间全 rank 新增异常为 0，停机后 16 张 NPU 均无进程。继续完成代码审查和优化前后性能量化，尚未将功能回归等同于性能优化完成。
- 2026-08-12：第五次上下文压缩后重新阅读全文并复核工作区。原始功能/语义基线与第一轮优化冻结快照的完整构建、聚焦测试、WORLD16 回归和全 rank 日志审计均保持通过；继续完成 prepare/forward 边界、流同步、tensor 生命周期和全执行分支审查，并量化优化前后 host metadata 开销后再提交。
- 2026-08-12：第六次上下文压缩后重新阅读全文并复核未提交文件。第一轮优化的构建和功能冻结结果保持有效，当前仍未把功能通过等同于性能优化完成；继续分块审查全部调用链，并以相同 DCP2 配置量化优化前后 host metadata 开销，完成后再形成独立 `perf:` 提交。
- 2026-08-12：第七次上下文压缩后重新阅读全文并复核分支与未提交文件。完整构建、聚焦测试、WORLD16 DP2/TP8/DCP2 长请求、语义和全 rank 审计结果保持有效；本轮继续闭环异步流与 tensor 生命周期审查，并使用同一 DCP2 工作负载量化优化前后 host metadata 开销，所有门禁通过后再提交第一轮 `perf:` 优化。
- 2026-08-12：第八次上下文压缩后重新阅读全文并复核工作区及子 agent 状态。A/B profile 已完成，prefill/decode metadata 均值分别减少 `99.923%`/`99.865%`；最终代码已移除临时打点并包含 `EMPTY` 占位输入的 dummy metadata 修复。当前由 build agent 执行最终完整 NPU 构建，主 agent 同步完成最终 diff 审查，构建通过后再由 test agent 执行最终聚焦测试、WORLD16 语义回归和全 rank/设备释放审计。
- 2026-08-12：第九次上下文压缩后重新阅读全文并复核最终构建、功能、语义、性能和提交状态。原验收项均已完成；用户新增要求是在相同单机 DP2/TP8 拓扑与相同 decode 工作负载下对比开启 DCP2 和关闭 DCP（DCP1）的 TPOT，当前交由既有测试 agent 执行，完成后补记原始日志、统计口径和结论。
- 2026-08-12：第十次上下文压缩后重新阅读全文并复核 DCP1/DCP2 TPOT、最终提交和工作区。DCP2 的 rank 0/device 0 `msprof` 动态采集与导出已完成；本轮按有效 decode layer invocation 归一化设备算子和 DCP 通信，拆解 DCP2 的 slot mapping、KV broadcast 及 owner/non-owner 搬运开销，并以无 profiler 的 DCP1/DCP2 TPOT A/B 作为端到端边界。

### 1. 仓库与机器预检

- [x] 阅读项目自定义代码规范。
- [x] 确认分支及 layerwise DCP 提交范围。
- [x] 确认单机有 16 张空闲、健康的 Ascend 910。
- [x] 核对模型、运行脚本和历史日志。
- [x] 执行 NPU 构建。
- [x] 执行 DCP 配置、placement、mapping 和 KV transfer 聚焦测试。
- [x] 执行 DP2/TP8/DCP2 启动验证。
- [x] 执行真实请求，覆盖 prefill 和 decode。
- [x] 与 DCP 关闭基线完成语义一致性对比。
- [x] 停止测试进程并确认设备释放。
- [x] Review 并 profile host 侧 metadata 准备瓶颈。
- [x] 优化 metadata 准备，回归构建和功能并量化收益。
- [x] 使用相同 decode 工作负载对比开启与关闭 DCP 的 TPOT。

## 问题记录

1. 初次访问工作区时执行沙箱挂载失败，报错为 `bwrap: Can't find source path ...: Permission denied`。用户恢复完整权限后已解除；该问题与代码无关。

2. 发现 2026-08-11 的旧验证日志：旧二进制曾在 16 个 rank 上完成 78 层 KV 分配，但进程随后被外部终止并产生 Python `resource_tracker` 提示。旧日志不作为验收证据，本轮重新构建和启动。

3. 首次执行 `python setup.py build --device npu`：TileLang 和 `xllm_ops` 预编译完成，主工程在 Ninja `511/540` 处以 `subcommand failed` 退出。`setup.py` 使用 `-j640 --verbose` 产生约百万 token 并行输出，真正失败行未保留在终端尾部。低并发重放后定位到 `third_party/xllm_atb_layers/models/deepseekv2/operation/sparse_latent_attention.cpp:2481`：`addCacheGather` lambda 的宏失败路径返回 `atb::Status`（`int32_t`），成功路径返回 `atb::ErrorType`，自动返回类型推导不一致。修复为显式 `-> atb::Status`，不改变返回值语义。

4. 修复后的低并发重放已成功编译 `sparse_latent_attention.cpp` 并生成 `libxllm_atb_layers.a`。诊断命令直接调用 Ninja，未经过 `setup.py` 的 NPU 环境配置，最终链接缺少 CANN 库搜索路径并报告 `-lascendcl/-lhccl/-lllm_datadist` 等找不到；这是诊断 shell 环境问题。下一步重新执行用户指定的 `python setup.py build --device npu`，以其完整环境完成最终链接。

5. 第二次完整构建仍在最终链接失败，确认不是诊断 shell 特例。`scripts/build_support/env.py::set_npu_envs()` 只设置 `LD_LIBRARY_PATH`，没有设置链接器使用的 `LIBRARY_PATH`。缺失的 CANN/DataDist 库均位于 `${NPU_TOOLKIT_HOME}/lib64`，`libms_tools_ext.so` 位于 `${NPU_TOOLKIT_HOME}/tools/mstx/lib64`。已在 NPU 环境初始化中将这两个存在的目录去重加入 `LIBRARY_PATH`，使 `python setup.py build --device npu` 不依赖外部 shell 预配置。

6. 第三次执行用户指定的完整命令 `python setup.py build --device npu` 成功，退出码为 0。CMake 重新配置后，`xllm`、`export_module` 和 `all_tests` 目标均通过 Ninja 校验；构建日志保存于 `build/dcp-kv-validation/command-logs/build_npu_attempt3.log`。

7. 首次聚焦 CTest 命令在进入测试体前失败：为独立测试初始化 NPU Python runtime 设置的 `LD_PRELOAD` 同时作用于管道中的 `tee`，而 shell 当时未包含 PyTorch 动态库路径，`tee` 报 `libtorch_python.so` 找不到并退出 127。该问题属于测试命令环境配置；补齐 `torch/lib` 和 `torch_npu/lib` 后重跑。

8. 聚焦测试复验通过：
   - `DecodeDcpLayerPlacementTest.*`、`ConfigJsonTest.ParallelConfigReadsDecodeDcpSize`、`TestMappingNPU.DecodeDcpSplitsEachAttentionTpGroupContiguously` 共 11/11 通过，日志为 `build/dcp-kv-validation/command-logs/focused_ctest.log`。
   - `MooncakeTransferEngineServiceTest.*`、`KVCacheTransferTest.*`、`MooncakeKVCacheTransferDefaultTest.*` 共 12/12 通过，其中直接覆盖 `LayerwisePushUsesDestinationOwnerAndKeepsDraftFull`，日志为 `build/dcp-kv-validation/command-logs/kv_transfer_ctest.log`。

9. 首次 16 rank 启动已完成权重加载、KV 初始化和 HTTP 服务启动，但验证脚本仍等待源码中已不存在的 `[KVCacheDCP]` 日志标记。为避免健康实例在 600 秒后被误判并停止，中断了前台等待脚本；其 `setsid` 服务进程继续运行。随后将 readiness 改为所有 rank 的权重和 LlmDataDist 初始化完成，并要求 rank 0 报 `Application startup complete`。

10. 验证脚本的 `status` 一度报告 `ALIVE=32/16`。最初误以为 `/proc` 恢复扫描重复计数，后续停机验证确认这些是实际存在的 multiprocessing 关联进程：rank 主进程退出后，15 个子进程被 reparent 到 PID 1 并继续持有 NPU context，导致 1-15 号卡 HBM 未释放。脚本改为分别报告 `ALIVE_RANKS` 和 `ALIVE_ASSOCIATED_PROCESSES`，预检和停止流程继续匹配并清理全部关联进程。

11. 真实请求验证：
    - 短请求 HTTP 200，`prompt_tokens=36`、`completion_tokens=32`，rank 0 记录 TTFT 7.541 秒、总时延 17.616 秒；日志为 `build/dcp-kv-validation/command-logs/e2e_chat_request_1.log`。
    - 首次长请求测试输入误生成 13004 tokens，超过 4096 限制并被服务正确拒绝为 HTTP 500；这是测试数据问题，不是推理故障。
    - 修正后并发两条长请求，每条 `prompt_tokens=2504`、`completion_tokens=16`，均 HTTP 200，客户端耗时约 5.4 秒。两个请求的 prefill token 总数 5008，超过单个 replica 的 `max_tokens_per_batch=4096`，且同时完成，验证 DP2 两个 replica 实际参与；日志为 `build/dcp-kv-validation/command-logs/e2e_dp2_long_context.log`。

12. 用户追加要求：功能验证后优化 host 侧 metadata 准备开销。每一轮必须先 review 并通过 profile 打点确认瓶颈，再修改代码并回归构建/功能，最后量化性能收益。

13. 性能优化前增加语义一致性验收。DCP2 layerwise 开启时使用 `temperature=0`、`max_tokens=256` 并发执行三条可判定问答：
    - 事实题最终答案为 `Paris`，`finish_reason=stop`。
    - 算术题明确计算 `17 * 6 + 5 = 107`，最终答案为 `107`，`finish_reason=stop`。
    - 技术题明确说明 KV cache 避免重复计算历史 K/V，prefill 并行处理 prompt 并填充 cache，decode 逐 token 读取并追加 KV；虽因显式推理较长在 256 tokens 截断，但核心结论完整正确。
    - 完整响应保存在 `build/dcp-kv-validation/command-logs/semantic_dcp2_full.jsonl`。

14. DCP2 语义样本完成后已停止全部 16 个服务进程，`npu-smi` 确认无运行进程。验证脚本新增 `DCP_KV_LAYERWISE_KV` 开关，允许使用完全相同的启动参数运行 `DCP_SIZE=1`、layerwise 关闭基线。

15. DCP 关闭语义基线使用 WORLD=16、DP=2、attention TP=8、`DCP_SIZE=1`、`LAYERWISE_KV=false`，二进制 SHA 与 DCP2 完全一致。三条请求均 HTTP 200：
    - 事实题最终答案同为 `Paris`。
    - 算术题计算过程和最终答案同为 `107`。
    - 技术题同样说明 KV cache 避免重复计算历史 K/V，prefill 处理完整 prompt 并填充 cache，decode 逐 token 读取并追加 KV。
    - DCP2 与关闭基线的结论和主要推理结构一致，只存在措辞和执行时延差异，满足非字符级语义一致验收。基线完整响应位于 `build/dcp-kv-baseline/command-logs/semantic_dcp1_full.jsonl`。

16. DCP2 和 DCP 关闭实例均稳定复现 multiprocessing 关联子进程不响应 TERM 的问题，说明它与 DCP 功能无关。验证脚本现分别统计 rank 主进程与全部关联进程；主 rank 全部退出后会立即 KILL 精确匹配相同 binary/master 地址的残留进程，并等待确认清理完成，避免显存泄漏到下一轮测试。

17. 第一轮 host metadata review 定位到三个同源问题：同一 batch 的 decode metadata 在 78 层重复构造；每层新建 3 个 host vector、执行 3 次 CPU→NPU tensor 转换并分配 2 个 device buffer；layerwise prefill 每层重复读取 block table 并构造 history slots。该设计的 metadata 完全是 step 级数据，已开始改为 Worker prepare stream 在模型 `forward` 前一次准备，并由所有 layer 共享；layer forward 内不再允许 metadata 构造、H2D/D2H 或临时 buffer 分配。

18. 第一轮优化冻结快照已通过完整构建和功能回归：`python setup.py build --device npu` 退出码 0，`xllm` 70/70、`export_module` 4/4、`all_tests` 37/37；placement/config/mapping 11/11、KV transfer/Mooncake 12/12；WORLD16 DP2/TP8/DCP2 两条并发 2504-token 请求均 HTTP 200，语义题输出 `Paris`、`107`，技术题正确描述 KV cache、prefill、decode。服务期间全 rank 新增 `ERROR/FATAL/traceback` 为 0，停机后 rank 和关联进程均为 0。构建日志为 `perf_round1_final_build_npu.log`，功能日志使用 `perf_round1_reviewed_*` 前缀。性能收益仍待量化。

19. 第一轮 host metadata A/B profile 已完成。基线使用独立 worktree `xllm-dcp-perf-baseline-d3718` 的 `d3718eb59`，只增加逐层函数计时并按 78 层聚合；优化侧在 worker prepare 函数计时。两侧均为 WORLD16、DP2、TP8、DCP2、layerwise=true、ATB eager、graph=false、schedule-overlap=false，先 warmup 1 条，再执行 5 条相同的 `19 prompt + 96 completion` 请求，并在所有 16 个 rank 上按 warmup 后日志 offset 取样。每侧得到 80 个 prefill 样本和 7600 个 decode 样本：
    - baseline 二进制 SHA `993d0a61994a90ecf8769898983e355dbf75eda6db3edf654f3aed89166824c1`；prefill 均值 `101760.637 us`、p50 `99701 us`、p95 `117626 us`，decode 均值 `102083.787 us`、p50 `101809 us`、p95 `113916 us`。
    - optimized profile 二进制 SHA `d1305fcdc2e6085ec6ccc5725a40df103531780bfb0073f345cfec6413124465`；prefill 均值 `78.300 us`、p50 `73 us`、p95 `118 us`，decode 均值 `137.564 us`、p50 `135 us`、p95 `159 us`。
    - prefill metadata 均值减少 `99.923%`（`1299.62x`），decode metadata 均值减少 `99.865%`（`742.08x`）。客户端 5 请求均值由 `15.201734 s` 降至 `7.552816 s`，中位数由 `15.328810 s` 降至 `7.417507 s`；业务时延只作为辅助结果，metadata 直接计时作为优化收益主证据。
    - 原始日志为 `perf_round1_profile_baseline_{profile_lines,raw_elapsed_us,stats}.tsv/log`、`perf_round1_profile_optimized_{profile_lines,raw_elapsed_us,stats}.tsv/log` 和两侧 `requests.jsonl`。两侧 profile 二进制每 step 各保留一条 INFO 计时日志，最终源码已移除该临时打点，因此最终运行时开销不高于上述优化侧结果。

20. 完整代码审查确认 step 级 tensor 由 `ForwardInput` 持有，prepare stream 通过 `metadata_ready_event` 与 compute stream 建立依赖；普通非 overlap 路径在输入析构前同步默认流，overlap/spec no-sync 路径由 retained input 保持生命周期，graph 模式被 DCP eager 检查拒绝；同一 step 的共享 buffer 只在同一 ATB compute stream 上按层顺序复用，不存在并发覆盖。审查同时发现一个真实边界回归：真正的 `EMPTY` 输入会在 worker prepare 提前返回，但 NPU 模型随后仍可能用占位 token 进入 prefill forward，导致共享 metadata 未初始化。已在提前返回前准备 dummy DCP 输入，并让 `_mtp` 模型与 layer 侧禁用 DCP metadata 的判断保持一致。最终源码中的 layer forward 只绑定已准备的 tensor，不再构造 DCP host vector、执行 DCP metadata H2D/D2H 或分配 DCP 临时 device buffer；临时 profile 日志也已移除。

21. 最终快照完成全部门禁并形成提交：
    - 最终机器可读汇总为 `perf_round1_final_review_validation.log`，其中 `VALIDATION_EXIT_CODE=0`。
    - 完整执行 `python setup.py build --device npu`，退出码 0；`xllm`、`export_module`、`all_tests` 分别完成 `[3/3]`、`[2/2]`、`[32/32]`，日志为 `perf_round1_final_review_build_npu.log`。最终二进制 SHA-256 为 `282bee5d822cea28c49a71d06efe9eeee9ec72e06356fe08ace1635df7367d19`。
    - placement/config/mapping 11/11、KV transfer/Mooncake 12/12 通过，日志分别为 `perf_round1_final_review_focused_ctest.log`、`perf_round1_final_review_kv_transfer_ctest.log`。
    - WORLD16、DP2、attention TP8、DCP2、layerwise=true 启动后 16/16 rank 就绪。两条并发长请求均 HTTP 200，每条 `2504 prompt + 16 completion`；合计 5008 个 prefill token 超过单 replica 的 4096 上限，且两条请求在同一 15.304 秒窗口内完成，确认两个 DP replica 均参与。
    - 最终语义回归中事实题为 `Paris`、算术题为 `107`，技术题正确说明 KV cache 避免历史 K/V 重算、prefill 并行填充 cache、decode 逐 token 读取和追加 cache；与优化前 DCP1/DCP2 基线语义一致。响应保存在 `perf_round1_final_review_semantic_dcp2_full.jsonl`。
    - 服务窗口全 16 rank 新增 `ERROR/FATAL/traceback` 为 0。停机脚本因 `/proc/56334/cmdline` 消失竞态返回 1，随后只按最终 binary 路径和 `--master_node_addr=11.87.191.98:22998` 精确执行 TERM/KILL；最终 `ALIVE_RANKS=0/16`、关联进程为 0、匹配进程为 0、NPU process table 全空且 16 卡 HBM 回到空闲基线，记录于 `perf_round1_final_review_cleanup_note.log`。
    - 优化源码提交为 `cb2e88bb6 perf: prepare shared dcp metadata before forward.`；pre-commit clang-format 检查通过。

22. DCP 开关 TPOT A/B 已完成。两侧使用同一最终二进制 SHA-256 `282bee5d822cea28c49a71d06efe9eeee9ec72e06356fe08ace1635df7367d19`，均为 WORLD16、DP2、attention TP8、ATB eager、graph=false、schedule-overlap=false、prefix-cache=false；唯一功能差异是开启侧使用 `DCP_SIZE=2`、layerwise=true，关闭侧使用 `DCP_SIZE=1`、layerwise=false。每档负载先执行 1 组双并发 warmup，再执行 5 组双并发正式请求，共 10 个有效样本；请求均为 `temperature=0`、`max_tokens=256`，全部 HTTP 200、实际生成 256 tokens，并使用服务端 `(total_latency - ttft) / (generated_tokens - 1)` 的 `avg tpot` 作为主口径。p95 使用 nearest-rank 定义。

    | Prompt tokens | 配置 | 样本数 | TPOT mean | TPOT p50 | TPOT p95 | DCP2 相对 DCP1 |
    | ---: | --- | ---: | ---: | ---: | ---: | ---: |
    | 26 | DCP1 / layerwise=false | 10 | `53.960 ms` | `54.400 ms` | `56.700 ms` | 基线 |
    | 26 | DCP2 / layerwise=true | 10 | `68.650 ms` | `68.100 ms` | `71.400 ms` | `+14.690 ms` / `+27.22%` |
    | 2504 | DCP1 / layerwise=false | 10 | `57.860 ms` | `57.750 ms` | `62.400 ms` | 基线 |
    | 2504 | DCP2 / layerwise=true | 10 | `78.410 ms` | `78.900 ms` | `82.300 ms` | `+20.550 ms` / `+35.52%` |

    - 结果表明，在当前单机 DP2/TP8、双并发、256-token decode 工作负载下，开启 DCP2 没有降低 TPOT，反而因额外 KV 分片通信与同步使短/长 prompt 的平均 TPOT 分别增加 `27.22%`/`35.52%`。DCP 的主要收益是提高每个 attention TP group 的有效 KV 容量，不能将前述 host metadata 开销下降直接解释为 DCP 端到端 TPOT 收益。
    - DCP1 原始请求和服务端指标分别为 `tpot_dcp_ab_dcp1_measured_requests.jsonl`、`tpot_dcp_ab_long_dcp1_measured_requests.jsonl`、`tpot_dcp_ab_dcp1_short_server_metrics.log`、`tpot_dcp_ab_dcp1_long_server_metrics.log`；DCP2 对应文件使用相同文件名并将 `dcp1` 替换为 `dcp2`，均位于 `build/dcp-kv-validation/command-logs/`。
    - 两侧服务窗口全 rank 新增 `ERROR/FATAL/Traceback` 均为 0。停止后两侧均为 `ALIVE_RANKS=0/16`、关联进程 0，两个 master 地址的匹配进程为 0，`npu-smi` process table 显示所有 NPU 均无运行进程。

23. DCP2 `msprof` 动态采集完成并拆解设备侧 overhead。采集对象为 rank 0/device 0，配置与最终 DCP2 长 prompt 测试一致；明确有效请求为 `2504 prompt + 256 completion`、HTTP 200，响应记录在 `msprof_dcp2_profiled_request_valid.json`。原始 profile 位于 `build/dcp-kv-validation/msprof/dcp2_rank0/PROF_000001_20260812091014694_00136187GLDOGDQH/`，已导出 task、op、HCCL 和 communication analyzer 数据。
    - `msprof` 使该请求的 TPOT 上升到 `103.7 ms`，而无 profiler 的同负载 DCP2 TPOT 均值为 `78.410 ms`；因此端到端 DCP 增量继续使用无 profiler A/B 的 `+20.550 ms/token`，`msprof` 只用于内部归因，不能把 profiled TPOT 当成正常性能。
    - 导出窗口包含 `18,445` 次完整 DCP decode layer invocation；rank 0 其中 owner 层 `9,222` 次、non-owner 层 `9,223` 次。窗口首尾没有完整覆盖全部请求，故所有数据按 layer invocation 归一化，再按模型 `78` 层折算单 token 开销。

    | DCP2 新增设备工作 | Profile 证据 | 折算单 token | 新增设备工作占比 |
    | --- | --- | ---: | ---: |
    | logical top-k 到 physical slot 映射 | 每层 5 个 `Gather32I32Kernel` 加 2 cast + mul + add + cast | `16.391 ms` | `70.56%` |
    | 两次 DCP broadcast | 2048 个 INT32 top-k 与 `2048 x 576` BF16 selected KV | `3.807 ms` | `16.39%` |
    | owner/non-owner KV pack/unpack | owner gather+concat、全层 copy、non-owner split+scatter | `3.033 ms` | `13.06%` |
    | 合计 | 仅统计源码与调用数可严格识别的 DCP2 新增设备工作 | `23.231 ms` | `100%` |

    - 最大单项是 block-table 行展开。当前每层先把一个 query 的 block-table row 从 `[1, num_blocks]` 按 2048 个 top-k entry 重复 gather 成 `[2048, num_blocks]`，再执行一次 batched gather 得到 physical block。profile 中 `num_blocks=20/21/22` 的行展开单次分别约 `79.414/156.406/156.313 us`；按窗口分布折算，行展开约 `11.246 ms/token`，后续 physical-block gather 约 `1.639 ms/token`。两项合计 `12.885 ms/token`，约为无 profiler DCP2-DCP1 长 prompt TPOT 差值的 `62.7%`。
    - slot mapping 的其余开销为：两个 logical-position LUT gather 约 `2.065 ms/token`，top-k compact gather 约 `0.902 ms/token`，int32/fp32 cast 及 mul/add 约 `0.538 ms/token`。整个 mapping 链约 `16.391 ms/token`，是 DCP2 的首要瓶颈。
    - DCP 专属通信不是 profile 中累计更大的 reduce-scatter/all-gather；那些属于原有 attention/FFN TP 路径。与源码 `ATTN_DECODE_DCP` 严格对应的是 group size 2 的两次 `hcom_broadcast_`：top-k payload 为 `8 KiB/layer`，约 `0.687 ms/token`；selected KV payload 为 `2.25 MiB/layer`，约 `3.120 ms/token`。通信与后续 attention 存在数据依赖并在同一执行序列中，当前没有可利用的关键路径重叠。
    - owner/non-owner 搬运合计约 `3.033 ms/token`：owner 的两次 cache gather 和 concat 约 `0.992 ms/token`，top-k 与 selected-cache 的全层 copy 约 `0.924 ms/token`，non-owner split 和 scratch scatter 约 `1.117 ms/token`。
    - 上述新增设备工作合计略高于无 profiler 的净 TPOT 差值，是因为 DCP2 同时让 non-owner 跳过了 layer-local indexer/top-k 计算，存在抵消收益；此外 `msprof` 自身显著扰动执行时延。profile 中 `LightningIndexer` 只有约一半 layer invocation，验证了 owner-only 计算路径。
    - 已优化的 host metadata 不是当前主要矛盾：decode metadata 直接计时均值仅 `137.564 us/step`，相比 `20.550 ms/token` 的 DCP2 净增量不足 `0.7%`。
    - 优化顺序：第一优先级是消除 `[batch, num_blocks] -> [packed_topk, num_blocks]` 的 block-table 行展开，可用 flattened block-table gather 加 forward 前准备的 row base offset，或用单个 NPU kernel 融合 logical block/offset/physical slot 解析；第二优先级是进一步融合 top-k compact、两个 LUT gather 和 slot arithmetic；第三优先级才是压缩 selected-KV broadcast payload或融合 gather/copy/split/scatter。源码确认每层 top-k 由该层 indexer 生成，因此不能把整个 slot 解析错误地前移到模型 forward 之前，只有与 layer 无关的 row-base 等静态 metadata 可以前移。

## 上下文恢复检查点

上下文压缩后从本节恢复：构建、功能、语义、资源释放和第一轮 host metadata 优化均已完成；最终优化提交为 `cb2e88bb6`，构建与验证修复提交为 `702f1cd23`、`d3718eb59`。DCP1/DCP2 TPOT A/B 表明短/长 prompt 平均 TPOT 分别增加 `27.22%`/`35.52%`。最新 DCP2 `msprof` 拆解显示，新增设备工作约 `23.231 ms/token`：slot mapping `16.391 ms`、两次 broadcast `3.807 ms`、owner/non-owner KV pack/unpack `3.033 ms`；其中 block-table 行展开与后续 physical-block gather 合计约 `12.885 ms/token`，是下一轮优化的第一目标。profile 只用于归因，端到端收益必须继续用无 profiler 的相同负载 A/B 验证。
