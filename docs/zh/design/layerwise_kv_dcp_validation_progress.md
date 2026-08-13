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
- 2026-08-12：第十一次上下文压缩后重新阅读全文并复核 slot mapping 融合草稿。旧草稿的 multi-query NPU 单测失败，定位为 TileLang 标量 GM 写回落在错误执行域；改为显式 `T.Scope("V")`、双 vector lane 任务切分和 int32 UB 写回后，生成 Ascend-C 只保留 AIV 路径且无 fp32/cast，3/3 聚焦用例通过。继续执行完整 NPU 构建、WORLD16 功能和性能回归。
- 2026-08-12：第十二次上下文压缩后重新阅读全文并复核工作区。当前存在尚未由主 agent 完整审查和验收的 fused slot-mapping 实现及单测改动；聚焦 NPU 单测已有 3/3 通过记录，但完整 `python setup.py build --device npu`、WORLD16 DP2/TP8/DCP2 功能与语义回归、DCP1/DCP2 无 profiler TPOT A/B 和必要的 msprof 复核仍未完成。因用户通知当前环境被占用，已立即中止 build/test 子 agent，保留全部源码、日志和未提交状态，暂停任何 NPU 编译、运行及性能采集，等待用户明确通知后从代码审查继续。
- 2026-08-12：恢复执行后以用户确认的冻结点为准：第二轮 fused slot-mapping 快照已完成用户指定的 `python setup.py build --device npu`，`BUILD_EXIT_CODE=0`，`xllm`、`export_module`、`all_tests` 均通过且新增 wrapper 测试目标成功链接。最终二进制 SHA-256 为 `8cbf81421a00df2c4b9556ec2b1e7d3e5473606b267c6c81c439f81e57086d81`，构建日志为 `perf_round2_fused_slot_mapping_build_npu.log`；当前进入聚焦测试、WORLD16 DP2/TP8/DCP2 功能语义回归及同二进制 DCP1/DCP2 TPOT A/B。
- 2026-08-12：第十三次上下文压缩后重新阅读全文并复核源码、日志和子 agent 状态。`3edefc2b...` 二进制上的 wrapper 3/3 与 int32 slot guard 4/4 仍有效，但随后 WORLD16 首请求已暴露 ATB plugin 错误拒绝图级非零 workspace 的集成 bug；源码已删除该错误约束并新增非零 workspace 回归用例。由于修复发生在该二进制构建之后，当前候选必须重新执行完整 `python setup.py build --device npu`、聚焦测试和 WORLD16 回归，不能沿用旧二进制作为最终结果；测试 agent 正在清理失败现场并回报准确状态。
- 2026-08-12：确认测试停滞来自失败后的孤儿进程，而不是有效 NPU 工作：精确匹配 master `11.87.191.98:22798` 的 16 个 rank 均已 reparent 到 PID 1，并在 `futex_wait_queue_me` 睡眠约 69 分钟；所有日志停在 layer 0 fatal，`npu-smi` 当时无对应 context。TERM 后仍有 15 个不退出，已按 binary 与 master 地址精确 KILL，最终残留为 0。清理后新的外部作业立即占用 16 卡约 `60 GiB/卡`，PID 为 `1907403` 至 `1916675`，不属于本 workspace；不清理未知 context，最新源码完整构建继续执行，设备回归等待外部作业释放。
- 2026-08-12：外部作业释放后复检 16 卡均健康、process table 全空、HBM 回到约 `2.9-3.2 GiB/卡` 基线。包含 ATB graph workspace 修复的最新源码已完成完整 `python setup.py build --device npu`，`BUILD_EXIT_CODE=0`；日志明确重新编译 `topk_logical_to_physical_slots_operation.cpp` 和 TileLang wrapper，并重新链接新增测试。候选 xLLM SHA-256 为 `95913444d0d10e39c42eaebaefe6f8f912b729b09ee7615b0106bc231adc1672`，构建日志为 `perf_round2_workspace_fix_build_npu.log`；开始最终聚焦测试和 WORLD16 回归。
- 2026-08-12：准备启动最终测试时，外部作业再次占用全部 16 卡约 `60 GiB/卡`，`npu-smi` PID 为 `1972614` 至 `1976269`，这些 PID 在当前容器 `/proc` 中不可见，且 workspace 无 xLLM 进程。最新构建结果保持有效；不清理未知外部 context，聚焦 NPU 测试与 WORLD16 等待设备再次释放后继续。
- 2026-08-12：build agent 最终确认 workspace 修复快照的完整构建全部完成：主 `xllm` `[16/16]`、`export_module` `[2/2]`、`all_tests` `[63/63]`，wrapper 二进制包含新增 `AtbOperationAcceptsGraphWorkspaceArguments` 用例；构建日志未发现 FAILED/fatal/build stopped，agent 未修改源码或文档。除 xLLM SHA 外，wrapper test SHA-256 为 `be9a20a1157ffa54ea8704e5488b7fb1b23b535d689967a9a59a922b62f98fa3`。
- 2026-08-12：用户再次通知环境空闲后即时复检，16 卡均健康、HBM 回到约 `2.9-3.2 GiB/卡`、process table 全空，workspace 无残留 xLLM；候选 xLLM 与 wrapper SHA 保持 `95913444...`、`be9a20a1...`。正式启动最终 wrapper、WORLD16 功能语义、清理审计和 TPOT A/B。
- 2026-08-12：上述空闲窗口内测试 agent 未成功重启，wrapper/WORLD16 实际尚未启动；随后外部作业重新占用 16 卡约 `56 GiB/卡`，PID 为 `2020376` 至 `2022355`，当前容器 `/proc` 均不可见。复核 workspace 没有 xLLM、wrapper 或验证脚本进程，确认本轮未制造该占用。保持冻结二进制与构建结论，下一次设备空闲后直接启动测试命令。
- 2026-08-12：收到用户立即启动通知后，空闲 preflight 与 wrapper 在同一命令中连续执行，避免空窗。最新 workspace 修复二进制上的 `topk_logical_to_physical_slots_wrapper_test` 4/4 通过，`TEST_EXIT_CODE=0`；新增非零 ATB graph workspace 回归用例通过，原有 multi-query、dummy 和 slot `16,777,217` 用例保持通过。日志为 `perf_round2_workspace_fix_wrapper_test.log`，开始 WORLD16 DP2/TP8/DCP2 集成回归。
- 2026-08-12：测试前设备复检发现新的外部环境占用：16 卡均有本容器 `/proc` 不可见的 NPU context，HBM 约 `56 GiB/卡`，`npu-smi` 显示 PID `1737203` 至 `1738332` 且报 `Get pid name failed`。确认本容器没有 xLLM、构建或测试进程，测试 agent 未启动新任务且未清理未知 context；证据为 `perf_round2_fused_slot_mapping_pretest_npu_recheck.log`。窄测仍保留 3/3、`CODEX_EXIT_CODE=0` 的有效结果；WORLD16 和 TPOT 等待外部占用释放后继续。
- 2026-08-12：静态收尾发现旧 fp32 实现遗留的启动容量上限仍将 DCP cache 限制在 `2^24` slots，虽然融合 kernel 已全程 int32 且聚焦单测证明 slot `16,777,217` 正确。该限制会让新语义在系统层不可用，已直接改为按 int32 最大 slot 地址校验，并增加“超过 fp32 mantissa 仍允许、超过 int32 范围才拒绝”的 CPU 单测。由于源码快照变化，外部 NPU 占用释放后需重新执行完整构建和全部设备回归，不能沿用上一个二进制作为最终结果。
- 2026-08-12：尝试直接用现有 Ninja 目录编译更新后的 CPU 聚焦测试时，CMake 自动重配置在进入源码编译前因当前 shell 未设置 `NPU_HOME_PATH` 退出；这是绕过 `setup.py` 的环境问题，与此前诊断现象一致，日志为 `perf_round2_int32_slot_guard_test_build.log`。最终构建只接受用户指定的 `python setup.py build --device npu` 结果。
- 2026-08-12：int32 容量修复后的最终源码已重新完成用户指定的 `python setup.py build --device npu`，`BUILD_EXIT_CODE=0`，最终二进制 SHA-256 为 `3edefc2b479ce3e5c58ccea8a3a0b1009206df82705b1a76867bb992ec56e59e`，日志为 `perf_round2_fused_slot_mapping_final_build_npu.log`。新增 slot 容量 CPU 单测 4/4 通过：DCP1 跳过限制、int32 边界允许、超过 fp32 mantissa 允许、超过 int32 范围拒绝；日志为 `perf_round2_int32_slot_guard_test.log`。NPU/WORLD16 仍等待外部作业释放设备。
- 2026-08-12：外部 NPU 作业已释放，最终二进制上的 fused wrapper NPU 聚焦测试重新执行 3/3 通过，`TEST_EXIT_CODE=0`；覆盖多 query/多 block-table row、block 边界和 compact 顺序、dummy entry 以及 slot `16,777,217`，日志为 `perf_round2_final_wrapper_test.log`。测试后 16 卡 process table 全空，开始 WORLD16 DP2/TP8/DCP2 回归。
- 2026-08-12：第十四次上下文压缩后重新阅读全文并复核最终源码、日志和提交状态。融合实现与 int32 边界修复已分别形成主仓本地提交 `115e16921`、`9e5680f9f`，ATB 子仓对应提交为 `23b40c6`、`9cd2088`。同一源码提交已完成 WORLD16 DP2/TP8/DCP2 功能语义和 TPOT A/B；临时 review 草稿把 TileLang 动态符号误写为当前安装版本不存在的 `T.dynamic`，该次构建按预期失败，源码改回已验证的 `T.symbolic` 后重新执行完整构建并通过。当前先按用户要求推送两个仓库，随后只剩优化后 DCP2 `msprof` 复采和 overhead 重拆解。

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

24. 第二轮优化实现单个 TileLang Ascend 算子融合 logical top-k 到 physical KV slot 的完整映射。输入为 layer-local top-k、共享 block table、forward 前准备的 packed gather index 和 query block-table row；kernel 内以 int32 语义完成 compact、logical block/offset 解析、二维 block-table 间接读取和 physical slot 计算，输出直接供 KV gather 使用。ATB graph 原有 5 个 gather、2 次 int32/fp32 cast、mul、add和最终 cast 被替换为 1 个自定义节点；同时删除按 `max_position_embeddings` 创建并逐层绑定的两个 LUT，不新增 forward 内 metadata、H2D 或 device buffer 准备。
    - 首版 kernel 的单 entry 用例通过，但 multi-query 用例失败。生成 Ascend-C 显示直接 GM scalar 写回被错误放入 cube guard；修复为 `T.Scope("V")` 中按 `(cid, vid)` 划分 32 个 vector task，并先写 int32 UB 后用 `copy_ub_to_gm<int>` 输出。修复后 codegen 只保留 AIV 路径，所有位置与 slot 运算均为整数，无 `float`、`Cast`、`Mul` 或 `Add` 浮点指令。
    - 聚焦 NPU 单测最终为 4/4 通过：除多 query/不同 block-table row、block 边界、compact 顺序、dummy entry 和 slot `16,777,217` 外，还直接覆盖 ATB graph 传入非零 workspace 参数。日志为 `perf_round2_workspace_fix_wrapper_test.log`；int32 slot guard 4/4 通过，日志为 `perf_round2_int32_slot_guard_test.log`。

25. 最终提交源码重新执行用户指定的 `python setup.py build --device npu` 并通过，`BUILD_EXIT_CODE=0`，日志为 `perf_round2_fused_slot_mapping_build_npu_latest.log`；`xllm`、`export_module`、`all_tests` 和新增 wrapper 测试目标均成功链接。最终 `xllm` SHA-256 为 `ed8a7d564c7e81c2e4b9463370d5ddb77e5574355cf5fe3bc6ab09923045dfc2`，wrapper test SHA-256 为 `635315556745afc1047cadfa64d0953975559524357e547e749b857c6459d827`。此前临时将 API 改成不存在的 `T.dynamic` 导致的失败日志为 `perf_round2_final_review_build_npu.log`，该草稿未保留在最终源码中。

26. 首次 WORLD16 集成回归没有通过，且已确认是本轮 ATB plugin 接入 bug，不是语义波动：第一条请求约 46 秒后服务主动断开，16 个 rank 同时在 layer 0 的融合 mapping plugin 返回 status 1；ATB 日志精确指向 owner 的 node 25 和 non-owner 的 node 11。`atb::Status` 中 1 是 `ERROR_INVALID_PARAM`，代码审查发现 plugin `Execute` 错误地拒绝 `workspace_size != 0`。图 runner 会传入图级 workspace 参数，即使该 plugin 在 `Setup` 中声明自身需要 0 字节，因此 standalone wrapper 测试通过而 ATB graph 必然失败。已删除该无效约束，并新增 `AtbOperationAcceptsGraphWorkspaceArguments` 回归用例，直接以非零 workspace size 调用 `OperationInfra::Execute`；修复后 wrapper 4/4 和 WORLD16 集成均通过。

27. 融合优化后的 WORLD16 DP2/TP8/DCP2 功能、语义、TPOT 和清理回归完成：
    - DCP2 事实题、算术题和技术题均为 HTTP 200，结论分别为 `Paris`、`107`，以及 KV cache 在 prefill 中为完整 prompt 生成并保存 K/V、decode 中只计算新 token 并追加 K/V；日志为 `perf_round2_workspace_fix_semantic_dcp2_full.jsonl` 和 `perf_round2_workspace_fix_semantic_dcp2_completion.jsonl`。
    - 双并发长上下文请求均为 HTTP 200，每条 `3133 prompt + 16 completion`，两条在约 `3.59 s` 的同一窗口完成，日志为 `perf_round2_workspace_fix_dp2_long_context.jsonl`。ready 之后的全 rank 增量异常审计为空；启动早期完整日志仍保留历史 Python traceback，因此不将完整文件误报为全程零异常。
    - 相同 2504-token prompt、双并发、256-token decode 的无 profiler A/B 中，DCP1 平均 TPOT 为 `57.400 ms`，DCP2 为 `66.290 ms`，DCP2 仍有 `+8.890 ms`、`+15.49%` 开销；相比优化前 DCP2 的 `78.410 ms`，降低 `12.120 ms`、`15.46%`，DCP2 相对 DCP1 的净 overhead 从 `20.550 ms` 降到 `8.890 ms`，减少 `56.74%`。汇总为 `perf_round2_workspace_fix_tpot_ab_summary.log`。
    - 停止脚本因 `/proc` 消失竞态返回 1，随后按 binary/master 精确清理，`FINAL_LEFT=0`；最终 `npu-smi` process table 无运行进程。该阶段优化后 DCP2 `msprof` 尚待复采，结果记录在下一项。

28. 融合优化后的 DCP2 `msprof` 复采和 overhead 拆解完成。采集对象为 rank 0/device 0，原始 profile 位于 `build/dcp-kv-validation/msprof/optimized_dcp2_rank0/PROF_000001_20260812195243978_00030306PRAGMDHA/`；有效请求为 `3338 prompt + 256 completion`、HTTP 200，记录在 `perf_round2_optimized_msprof_direct_profiled_request.json`。导出窗口完整包含 `19,968 = 78 x 256` 次 decode layer invocation，其中 owner/non-owner 各 `9,984` 次。
    - 旧 mapping 链的 5 个 gather、`[packed_topk, num_blocks]` block-table 行展开和 fp32 cast/mul/add/cast 索引 hack 已从 DCP layer 邻接序列中完全消失。新 `topk_logical_to_physical_slots_kernel__bs128` 恰好执行 `19,968` 次，累计 `362.924 ms`，均值 `18.175 us/layer`、p50 `18.240 us`、p95 `19.280 us`，折算 `1.418 ms/token`。
    - 优化前 logical top-k 到 physical slot mapping 为 `16.391 ms/token`；融合后降至 `1.418 ms/token`，减少 `14.973 ms/token`、`91.35%`。

    | 优化后 DCP2 可识别设备工作 | 折算单 token | Profile 分布或组成 |
    | --- | ---: | --- |
    | fused logical-to-physical mapping | `1.418 ms` | `18.175 us/layer` mean，`18.240 us` p50，`19.280 us` p95 |
    | INT32 top-k broadcast | `3.054 ms` | `39.154 us/layer` mean，`5.440 us` p50，`90.301 us` p95；存在 profiler/HCCL 同步长尾 |
    | BF16 selected-KV broadcast | `3.658 ms` | `46.896 us/layer` mean，`40.420 us` p50，`67.021 us` p95 |
    | owner/non-owner pack、copy、unpack、scatter | `2.959 ms` | owner gather `0.723`、concat `0.290`、全层 copy `0.783`、non-owner split `0.282`、scatter `0.881 ms/token` |
    | 合计 | `11.088 ms` | 仅统计源码和每层邻接序列可严格识别的 DCP2 新增设备工作 |

    - 优化前同口径合计为 `23.231 ms/token`，优化后减少 `12.143 ms/token`、`52.27%`。该变化与无 profiler 的 DCP2 TPOT 从 `78.410 ms` 降到 `66.290 ms`、改善 `12.120 ms/token` 基本一致；DCP2 相对 DCP1 的净 TPOT overhead 从 `20.550 ms` 降到 `8.890 ms`，减少 `56.74%`。
    - 本次 `msprof` 请求的服务端 TPOT 为 `176.6 ms`，明显高于无 profiler DCP2 的 `66.290 ms`，不能作为端到端性能口径。尤其 INT32 broadcast 的 p50 仅 `5.440 us/layer`，但少量同步长尾最高达到 `8.476 ms/layer`，将累计值拉高到 `3.054 ms/token`；因此通信项用于定位依赖和剩余优化方向，不应被解释为稳定的纯链路带宽成本。
    - 优化后的主要剩余 DCP 工作已从 slot mapping 转为两次同步通信。下一轮候选方向是取消 selected-KV broadcast，改为把 decode Q 路由到 layer owner，由 owner 使用本地完整 KV 执行 attention，再把 attention output 路由回原 rank；实施前必须先确认 Q/output shape、TP 语义、owner 轮转、batch 路由和 collective 配对，不能只按 payload 大小替换通信。

29. 已完成“Q 路由到 layer owner、owner 执行 attention、attention output 路由回原 rank”的方案调研和初步设计，结论是语义可行，建议先实现可验证原型，再由 profile 决定是否继续做自定义 root gather/scatter。
    - `ATTN_DECODE_DCP` 不是跨 DP replica 的通信组，而是在每个 attention TP group 内按连续 rank 划分的子组。当前 TP8/DCP2 中的 DCP group 形如 `[0, 1]`、`[2, 3]`；组内 rank 处理同一批请求和相同 block table，但各自持有不同 attention head 的 Q/权重分片。因此 owner 可以收齐组内 Q head shard，使用本层唯一的完整 KV cache 计算所有 head 的 sparse attention，再按原 head shard 把结果返回；各 rank 随后继续使用本地 V projection 和 O projection 权重。
    - 正确切分点是 MLA Q reproject/rope 之后、`SparseFlashAttention` 之前汇聚 Q；在 `SparseFlashAttention` 之后、V reproject 之前返回 `intermediate_self_attention`。不能在 O projection 之后才返回，否则会改变现有 TP 权重分片以及 O projection 后续 reduce-scatter/all-reduce 的语义。
    - GLM-5 配置为 64 attention heads、TP8，因此每 rank 有 8 heads；`q_nope` 的 latent width 为 512，`q_rope` width 为 64，SFA 输出 latent width 为 512。单 token、单 rank 的 BF16 Q payload 为 `8 x (512 + 64) x 2 = 9 KiB`，DCP2 owner 汇聚后的总 Q 为 `18 KiB`；单 rank需要返回的 SFA 输出为 `8 x 512 x 2 = 8 KiB`，两 rank 合计 `16 KiB`。当前 selected-KV broadcast 为 `2048 x (512 + 64) x 2 = 2.25 MiB/layer`，新路径通信 payload 约缩小 68 倍。
    - 新路径删除普通 decode 层的 selected-KV broadcast、logical-to-physical slot mapping、owner cache gather/concat、selected-cache copy、non-owner split/scratch scatter，以及为这些操作准备的 `packed_gather_indices`、`packed_query_block_rows`、`selected_cache_buffer`。owner 直接让 SFA 使用原始 logical top-k、真实 block table 和本地 KV cache；non-owner 不再执行 SFA，也不需要本层 scratch KV 中存在被选中的历史行。
    - 对存在 DSA `index_share` 的模型，index 仍必须广播，但只在 `outputTopk=true` 的 producer 层广播。producer 将 owner 生成的 top-k 同步到 DCP group 后，各 rank 的模型级 `prev_topk_indices` 才都有效；后续 `skipTopk=true` consumer 层的 owner 可能因 `layer_id % dcp_size` 轮换而变成另一个 rank，它从本 rank 的 `in_shared_topk_indices` 读取该 index。普通非共享层的 top-k 只供本层 owner 使用，不广播；consumer 层也不重复广播。当前 GLM-5 配置未启用跨层 index share，但实现必须保留该语义。
    - 推荐第一阶段使用 ATB 已有 collective 验证数据流：分别或打包 `q_nope/q_rope` 后在 DCP group 上 `AllGather`，仅 owner 执行合并 head 后的 SFA；owner 将完整 SFA 输出 `Broadcast` 给组内 rank，各 rank按 DCP local rank 提取自己的 head shard，再执行本地 V/O projection。ATB 当前公开了 `AllGatherParam` 和 `BroadcastParam`，没有对应的 ATB collective `ScatterParam`；CANN HCCL 底层虽提供 `HcclScatter` 和 `HcclBatchSendRecv`，第一版不应先承担自定义通信 operation 的 stream、workspace、错误传播和生命周期风险。
    - AllGather 的输出是 rank-major `[dcp, token, local_heads, dim]`，SFA 需要 token-major `[token, dcp * local_heads, dim]`。当 batch/token 数大于 1 或进入 expanded speculative verify 时，不能直接 reshape，否则会把 token 与 rank 交错；必须先转置为 `[token, dcp, local_heads, dim]` 再合并 head。返回方向相反：owner SFA 输出 `[token, dcp * local_heads, 512]`，先恢复为 `[token, dcp, local_heads, 512]`，各 rank 再选择自己的 DCP shard。后续若切到 `HcclScatter`，root send buffer 还需转换为 rank-major `[dcp, token, local_heads, 512]`。
    - 第一阶段可保留现有 `MlaPreprocessV2` 在所有 rank 上生成本地 Q。该融合节点还会让 non-owner 向共享 scratch cache 写当前 token，但新 attention 路径不再读取该 scratch；语义正确后可再拆出 Q-only/non-owner specialization，消除这次无用写入。owner 仍必须写本层当前 token KV，并仅由 owner 运行 indexer/index-cache 更新。
    - 性能收益不能只按通信字节推断。当前 SFA 在 profile 中约 `93.2 us/layer`；DCP2 owner 一次计算 16 heads 可能比原先每 rank 并行计算 8 heads 更慢，若近似线性翻倍，额外串行计算上界约为 `78 x 93.2 us = 7.27 ms/token`。新路径预计移除当前可识别 DCP 工作中的大部分 `11.088 ms/token`，但会新增 Q collective、output collective、布局变换和 owner 扩展 SFA；净收益必须由无 profiler TPOT A/B 和优化后 `msprof` 共同确认。
    - 第二阶段只在第一阶段 profile 证明 collective 固定延迟或冗余复制仍是主要瓶颈时实施：增加 ATB custom communication operation，用 `HcclBatchSendRecv` 把各 rank 的 Q 只发往 root、用 `HcclScatter` 把各 head shard 的 output 只发回目标 rank，并用融合 pack/unpack kernel完成 rank-major/token-major/head-major 布局转换。当前 CANN 的 `hccl.h` 提供 `HcclScatter` 和 `HcclBatchSendRecv`，但未提供可直接调用的 `HcclGather`，不能假设存在对称 gather API。DCP size 大于 2 时还要单独评估 owner 集中计算的关键路径扩张；owner 按层轮换能平衡整模型设备负载，但不能缩短单层 owner 的串行时延。
    - 实现门禁：先增加纯 shape/路由测试，覆盖 batch>1、DCP owner 轮换、`outputTopk -> skipTopk` 跨 owner、普通 decode、expanded speculative verify 和 dummy/empty batch；再完成 wrapper/ATB graph 聚焦测试、用户指定完整 NPU 构建、WORLD16 DP2/TP8/DCP2 语义回归、DCP1/DCP2 TPOT A/B，最后复采 `msprof` 验证 selected-KV broadcast、slot mapping 和 scratch gather/scatter 确实消失。

30. 已完成 Q 路由原型的 ATB 图级 contract 核验，第一阶段可以直接实施，但必须按以下精确布局和角色分图，不能将通信原语的第 0 维语义当作 token 维。
    - `MlaPreprocessV2` 的两个 Q 输出分别为 `q_nope: [T, H_local, 512]`、`q_rope: [T, H_local, 64]`；`SparseFlashAttention` 使用 `TND` 输入并将输出 shape 逐维继承为 `intermediate_self_attention: [T, H, 512]`。该 ATB custom SFA operation 只从 query tensor shape 推导输出，并没有将 `selfAttentionParam.headNum` 传给 ACLNN SFA。因此 owner 可直接运行 `H = DCP * H_local` 的 SFA，**不**应为该节点另行把 `selfAttentionParam.headNum` 扩大。该参数仍服务于 Q 的旧预处理/split 路径，不能全局修改后复用到 V/O projection。
    - DCP head 顺序与上述 slice 假设已由权重 loader 核对：DeepSeek V32 的 `q_b_proj` 先按 `[H_local, qk_nope_head_dim + qk_rope_head_dim, -1]` view，再对 local TP rank 取得连续 shard；因此 DCP group rank `r` 的 Q 对应全局 head 区间 `[r * H_local, (r + 1) * H_local)`。output 返回后用同一 `r` 的连续 Slice 是语义正确的。该结论只适用于当前 attention TP 的 contiguous head sharding；若以后引入 head permutation、交错 TP 或非连续专家布局，必须改为显式 head index tensor，不能复用固定 offset。
    - 为避免两个小 collective，首版将 Q 在 feature 维拼为 `dcp_q_packed_local: [T, H_local, 576]`。ATB `AllGather` 会在第一维形成 rank-stacked 输出 `dcp_q_packed_rank_major: [DCP, T, H_local, 576]`；随后以 `Transpose(1, 0, 2, 3)` materialize 成 `[T, DCP, H_local, 576]`，再把第 1、2 维解释为 `[T, DCP * H_local, 576]` 并按 feature 维 split 回 owner 的 `q_nope/q_rope`。不能省略 transpose：rank-major 的连续内存直接 reshape 会在 `T > 1` 时把 rank 0 的后续 token 排到 rank 1 的首 token 前，batch decode 和 expanded speculative verify 都会产生错误 head-token 配对。
    - 所有 DCP rank 都必须以相同顺序入图并执行 `AllGather(Q)` 与 `Broadcast(SFA output)`。owner 图执行：`MLA preprocess -> [indexer, 条件 top-k broadcast] -> pack Q -> AllGather -> layout reorder -> Split Q -> SFA(local KV, logical top-k, block table) -> Broadcast`。non-owner 图执行：`MLA preprocess -> [仅条件 top-k receive] -> pack Q -> AllGather -> layout reorder -> receive Broadcast`；它不建 indexer、不建 SFA、不读取 selected KV，也不做 slot mapping。
    - `AllGather` 之后的 layout reorder 只在 owner 图消费：non-owner 仍需把相同的 `AllGather(Q)` collective 入图，但其 rank-major 输出可以直接作为通信完成后的临时输入，不执行 transpose、merge-head 或 Q split，随后进入 output broadcast。这样既保持 collective 顺序一致，又避免 non-owner 做无用布局计算。实现时必须给该临时 tensor 一个稳定的 intermediate 名称，不能让 ATB 因无消费者而回收 collective output。
    - output broadcast 的首版采用与当前 selected-KV 一致的 in-place ATB `Broadcast` pattern：owner 的 SFA 结果先作为 `dcp_attention_output`；non-owner 先将新的外部 device buffer `in_dcp_attention_output_buffer` copy 到同名 internal tensor；所有 rank 再对该 internal tensor broadcast。完成后每张卡以 `Slice(offset=[0, dcp_rank * H_local, 0], size=[T, H_local, 512])` 取回本 rank 的 `intermediate_self_attention`，其后 `AddEinReprojVNode` 和 O projection 保持字节级原路径和本地权重切分。
    - 新路径的 host/variant-pack 改动必须和图改动同一个提交完成：删除 `NpuDecodeDcpInput::{selected_cache_buffer, packed_gather_indices, packed_query_block_rows}`，删除 worker 中 `visible_lengths/query_block_rows/packed_*` 的构造和 3 次对应 device tensor 准备，删除 ATB `decode_dcp` 中 selected-cache 与 slot-mapping 的输入及其 variant-pack 绑定。保留 `expanded_query_cu_seq_lens`，因为它不是 slot mapping 元数据，而是 expanded speculative verify 的实际 SFA query-length contract。`topk_buffer` 不能无条件删除：它应重命名为 `topk_receive_buffer`，且仅在 `outputTopk=true` 的 index-share producer 层分配和绑定，供 non-owner 接收 owner 的 index broadcast；普通层和 `skipTopk` consumer 不分配。新增 `attention_output_buffer`，shape 为 `[T, DCP * H_local, 512]`、BF16；它只为 non-owner 的 broadcast receive 提供外部存储，但为保持 graph input contract 简单可在全 rank 分配。GLM-5、DCP2、单 token 下仅为 `1 * 16 * 512 * 2 = 16 KiB`，远小于删除的 selected-KV buffer。
    - 首版继续让所有 rank 执行 `MlaPreprocessV2`。该算子把当前 token 写入 `in_k_cache/in_k_rope_cache`，owner 的写入必需，non-owner 写入其不再被 attention 消费的 scratch cache，语义上安全但仍有额外工作。因此首版必须保留现有 non-owner shared scratch KV allocation 和 KV cache ABI，不能因 selected-KV 路径已删除而同时删除 scratch cache。只有在首版语义和性能验证后，才能引入 Q-only fused preprocess 或为该 ACLNN op 增加 suppress-cache-write 参数；届时才能重新评估删除 shared scratch、cache capacity accounting 和 KV cache transfer 的改动。不能通过跳过整个节点来优化，因为它同时生成各 rank 必需的本地 Q。
    - 预期首版的通信流量不是最终下界：AllGather 会把 `DCP * Q` 复制到每张卡，Broadcast 也把完整 `DCP * O` 复制到每张卡，DCP2 单 token 合计约 34 KiB/rank（Q packed 18 KiB + SFA output 16 KiB），仍显著低于旧 selected-KV 的 2.25 MiB/layer。只有当 profile 显示这两个 collective 或布局 materialization 仍占主导时，第二阶段才实现 root-only `HcclBatchSendRecv + HcclScatter`；其数据布局分别应为 Q 的 rank-major `[DCP, T, H_local, 576]` 和 output 的 rank-major `[DCP, T, H_local, 512]`，且 custom operation 必须保证 DCP group 内 collective 调用顺序、ATB stream、workspace 生命周期和失败传播均与既有 `BroadcastParam` 一致。

31. 用户已确认验证便利性优先，首版正式选用 ATB `AllGather(Q) + Broadcast(attention output)`，不在本轮引入自定义 `HcclBatchSendRecv/HcclScatter`。当前进入代码实现阶段。
    - 本轮目标是原子替换 decode DCP 的 selected-KV 路径：所有 rank 生成并 pack 本地 Q、执行 DCP AllGather；仅 owner 重排 Q 并用本地完整 KV 执行 SFA；完整 attention output 在 DCP group 内 Broadcast，各 rank slice 自己的 head shard后继续本地 V/O projection。
    - 本轮必须同时删除旧 logical-slot mapping、selected-cache gather/broadcast/scatter 及对应 host metadata/device buffer 准备；保留 expanded speculative verify 所需的 `expanded_query_cu_seq_lens`、所有 rank 的 `MlaPreprocessV2` 和 non-owner shared scratch KV ABI。
    - `index_share` contract 不变：只在 `outputTopk=true` producer 层广播 index，普通层 top-k 留在 owner，`skipTopk=true` consumer 不重复广播。
    - 本轮门禁为：完成代码编写、按项目规范做静态 review、执行用户指定的 `python setup.py build --device npu` 并修复全部编译问题。由于机器上存在其他进程，暂不启动功能、语义、TPOT 或 `msprof` 测试；待环境允许后再从 shape/路由聚焦测试和 WORLD16 回归继续。

32. 验证版代码主链已完成，第一轮静态 review 通过，正在进入完整 NPU 编译阶段。
    - ATB decode DCP 图已改为：所有 rank 执行本地 Q pack 和 DCP AllGather；owner 将 rank-major Q 转为 token-major、合并 DCP head shard并执行唯一一次 SFA；随后完整 SFA output Broadcast，各 rank按 `decodeDcpInfo.rank * H_local` slice 自己的 head shard，再继续原有本地 V/O projection。
    - Q AllGather 与 output Broadcast 在所有 rank 上顺序一致，并复用现有 `AddDapEventsBeforeComm/AfterComm` 围栏。只有 `outputTopk=true` producer 在 Q AllGather 前执行 index Broadcast；普通层 owner 直接使用本地 top-k，`skipTopk=true` consumer 直接使用 `in_shared_topk_indices`。
    - xLLM/ATB input ABI 已删除 `selected_cache_buffer`、`packed_gather_indices`、`packed_query_block_rows` 和无条件 `topk_buffer`，旧 logical-slot mapping、selected-cache gather/broadcast/split/scratch scatter 已从 decode 图完全断开。保留的外部 device input 为完整 output broadcast 的 `attention_output_buffer`，模型存在跨层 index share 时再分配一个可复用的 `topk_receive_buffer`。
    - worker 不再构造 `visible_lengths/query_block_rows/valid_topk_counts/packed_*` host vector，也不再执行对应 CPU→NPU metadata tensor 转换。expanded speculative verify 的 `[1..T]` 累计 query length 改为直接在 NPU 上 `arange`；普通 decode 不准备该 tensor。
    - review 修复了一个潜在 host 侧反优化：不在每次 forward 构造遍历全部 78 层的 `DsaTopkSharePlan`，改为用模型配置 O(1) 判断是否需要共享 receive buffer。已核对 decoder layer 与 latent attention 两层 tensor map 的条件输入完全一致，owner/non-owner collective 次序一致，`git diff --check` 通过。

33. `AllGather(Q) + owner SFA + Broadcast(output)` 验证版已完成代码、静态 review 和用户指定完整 NPU 构建，本轮按要求不执行运行测试。
    - 最终 review 核对了 fused decode Q 的实际 shape：`MlaPreprocessV2` 输出 `[T, H_local, 512]` 和 `[T, H_local, 64]`；ATB AllGather 增加 rank 维后，owner 的 `Transpose(1,0,2,3) -> merge DCP/head -> feature Split` 与 batch decode、expanded speculative verify 的 token/head 顺序一致。SFA 输出按 DCP local rank 连续 slice，保留原本每 rank 的 V/O projection 和后续 TP 通信语义。
    - 每层 collective 顺序在 owner/non-owner 间一致：共享 index producer 为 `top-k Broadcast -> Q AllGather -> output Broadcast`；普通层和共享 index consumer 为 `Q AllGather -> output Broadcast`。consumer 从前一 producer 已同步到本 rank 的 `in_shared_topk_indices` 读取 index，不重复广播。
    - review 删除了已经失效的 `decodeDcpBlockSize` xLLM→ATB ABI 和 fused slot-mapping block-size 白名单，避免新路径仍被旧实现约束；保留 owner 直接读取 paged KV 所需的 ND cache layout 检查、`index_topk <= 2048` 约束和 attention TP/DCP 整除关系检查。
    - `python setup.py build --device npu` 首轮完整构建通过，日志为 `build/dcp-kv-validation/build_allgather_broadcast_npu_20260813_01.log`，`BUILD_EXIT_CODE=0`。最终清理后又执行一次独立增量全构建，日志为 `build/dcp-kv-validation/build_allgather_broadcast_npu_20260813_02.log`，`BUILD_EXIT_CODE=0`；当前源码重新编译了 ATB sparse attention，并成功链接 `xllm`、`export_module/xllm_export` 和 `all_tests` 全部测试目标。
    - 实现已形成两个本地提交：ATB 子仓为 `4f8377c perf: route decode attention through layer owner.`，主仓为 `a63dfed07 perf: replace selected kv decode broadcasts.`。主仓提交同时固定上述 ATB 子仓指针；既有 `third_party/torch_npu_ops` 用户状态未修改、未纳入提交。
    - 提交钩子对两个 xLLM 文件执行了纯格式化调整，因此又对精确提交源码执行第三次 `python setup.py build --device npu`。日志为 `build/dcp-kv-validation/build_allgather_broadcast_npu_20260813_03.log`，`BUILD_EXIT_CODE=0`；格式化后的 worker、NPU decoder wrapper 以及 ATB 代码均完成编译，`xllm`、`export_module/xllm_export` 和 `all_tests` 全部成功链接。
    - 本轮没有运行任何测试二进制，也没有启动 WORLD16、TPOT 或 `msprof`，因此这里只确认代码 review 和编译通过，尚不能声明运行期语义或性能通过。机器可用后的顺序是：shape/路由与 ATB graph 聚焦测试 -> WORLD16 DP2/TP8/DCP2 对比 DCP1 的语义回归 -> DCP1/DCP2 TPOT A/B -> `msprof` 复采并确认 selected-KV broadcast、slot mapping 和 scratch scatter 从 decode profile 消失。

34. 机器空闲后已完成第二轮独立 review 和 `AllGather(Q) + owner SFA + Broadcast(output)` 的 WORLD16 功能回归。
    - 独立 review 对主仓 `a63dfed07` 和 ATB `4f8377c` 检查了 worker buffer、xLLM/ATB 两级可选输入 ABI、Q pack/AllGather/transpose/merge/split、owner SFA、output broadcast/slice、owner 与 KV placement、普通/expanded/dummy token shape 以及所有 rank 的 collective 顺序。Critical、Important、Minor 问题均为 0，未发现确定性的数据流或 shape bug。
    - 静态确认 worker 的 output receive buffer 为 `[T, DCP * H_local, kv_lora_rank]`，共享 index receive buffer 为 `[T, 1, index_topk]`、`int32`；owner 将 `[DCP, T, H_local, F]` 转为 `[T, DCP * H_local, F]` 后运行 SFA，broadcast 完整 output 后每个 rank 连续切回 `[T, H_local, kv_lora_rank]` 并继续本地 V/O projection。owner 与 KV placement 均使用 `layer_id % DCP_size`，broadcast root 使用 DCP 组内 rank。
    - 聚焦 CTest 共 12/12 通过，覆盖 decode DCP placement、int32 slot 边界、配置解析和 attention TP 内连续 DCP group mapping；日志为 `build/dcp-kv-validation/command-logs/owner_attention_focused_ctest.log`，`FOCUSED_CTEST_EXIT_CODE=0`。
    - WORLD16 使用 `/export/home/models/GLM-5-final-w8a8`，配置为 `WORLD=16, DP=2, attention TP=8, EP=16, DCP=2, layerwise_kv=true`，精确二进制 SHA-256 为 `3e1147a578f7677a0e156c8175457c2ec3fe2b465cc6d2fcc58f996b929670a5`。16/16 rank 完成权重和 LlmDataDist 初始化，rank 0 报 `Application startup complete`，启动日志为 `owner_attention_world16_start.log`，`OWNER_ATTENTION_WORLD16_START_EXIT_CODE=0`。
    - 首个真实请求返回 HTTP 200，完成 10 prompt + 16 completion tokens，无 collective deadlock。它动态证明 non-owner 即使不消费 AllGather 输出，ATB 运行时仍执行该通信节点，所有 78 层的 `Q AllGather -> output Broadcast` 能够配对完成；日志为 `owner_attention_world16_short_request.jsonl`。该请求包含冷启动图构建，不作为性能数据。
    - 语义请求均为 HTTP 200：事实题结论为 Paris，算术题结论为 107；KV cache 技术题正确说明 prefill 写入历史 K/V、decode 读取历史 K/V 并追加当前 token K/V，长输出触及显式 completion token 上限但核心语义正确。原始响应位于 `owner_attention_world16_semantic.jsonl` 和 `owner_attention_world16_technical_complete.jsonl`。
    - DP2 并发验证同时提交两条 `prompt_tokens=3134, completion_tokens=16` 请求，均为 HTTP 200，客户端耗时分别为 3.602 秒和 3.601 秒，总 prompt tokens 为 6268，超过单 replica 的 `max_tokens_per_batch=4096` 且同时完成，确认两个 DP shard 实际参与；日志为 `owner_attention_world16_long_pair.jsonl`。
    - readiness 之后记录了全 rank 日志 offset，请求窗口新增 Traceback/ERROR/FATAL 为 0。启动阶段出现 272 行同一 Python multiprocessing forkserver signal-handler `TypeError`，但没有 ERROR/FATAL，且未阻止 16/16 rank ready 或任何请求；当前归类为非阻断启动告警，不是 DCP 数据流错误。分类与请求期审计分别见 `owner_attention_world16_startup_audit_summary.log`、`owner_attention_world16_post_request_audit.log`。
    - 服务停止返回 `OWNER_ATTENTION_WORLD16_STOP_EXIT_CODE=0`；脚本因部分关联进程未在 TERM 窗口内退出而执行精确强制清理，随后 workspace 相关进程为 0，`npu-smi` 的 16 卡 process table 均为空。
    - 本轮动态覆盖的是 GLM-5 当前普通 decode 路径和 owner 轮换下的 collective 执行。expanded speculative decode、空 DP shard、`outputTopk -> skipTopk` 跨 owner 的专门请求，以及 fused/non-fused MLA 两种配置尚未分别动态覆盖；它们已通过静态 shape/顺序审查，但后续不能描述为实跑通过。新路径的 DCP1/DCP2 TPOT A/B 与 `msprof` 也尚未执行。

35. 新 owner-attention 路径的同机 DCP1/DCP2 TPOT A/B 已完成。
    - 两侧使用同一二进制 SHA-256 `3e1147a578f7677a0e156c8175457c2ec3fe2b465cc6d2fcc58f996b929670a5`，均为 WORLD16、DP2、attention TP8、ATB eager、graph=false、schedule-overlap=false、prefix-cache=false；DCP2 为 `decode_dcp_size=2, layerwise=true`，DCP1 为 `decode_dcp_size=1, layerwise=false`。每侧先执行 1 组双并发 warmup，再执行 5 组双并发正式请求；每条正式请求均为 `2504 prompt + 256 completion`、HTTP 200。

    | 配置 | 样本数 | TPOT mean | TPOT p50 | TPOT p95 |
    | --- | ---: | ---: | ---: | ---: |
    | DCP1 / layerwise=false | 10 | `54.430 ms` | `54.250 ms` | `56.600 ms` |
    | DCP2 / layerwise=true | 10 | `59.970 ms` | `59.850 ms` | `65.400 ms` |

    - 新路径下 DCP2 相对同机 DCP1 的 mean TPOT overhead 为 `+5.540 ms/token`、`+10.18%`。上一版 fused-slot DCP2 在相同 2504-token、双并发、256-token 口径下为 `66.290 ms`，因此新 owner-attention DCP2 再降低 `6.320 ms/token`、`9.53%`；DCP2 相对 DCP1 的净 overhead 从上一轮 `8.890 ms` 降到本轮 `5.540 ms`，减少 `3.350 ms`、`37.68%`。跨轮绝对值会受运行窗口波动影响，主要结论以本轮同二进制 DCP1/DCP2 A/B 为准。
    - DCP2 原始请求、服务端指标和统计分别为 `owner_attention_tpot_dcp2_requests_2504.jsonl`、`owner_attention_tpot_dcp2_server_metrics.log`、`owner_attention_tpot_dcp2_stats.log`；DCP1 对应文件将文件名中的 `dcp2` 替换为 `dcp1`。两侧 readiness 后请求窗口新增 Traceback/ERROR/FATAL 均为 0。
    - 两侧停止均为 `STOPPED=FORCED`、退出码 0；精确 master/binary 关联进程已清零。DCP1 清理后出现外部 NPU context `1621831..1621846`，这些 PID 在当前容器 `/proc` 中不可见且不匹配本 workspace。用户确认它们是稳定的低显存背景负载，因此本轮保留、不纳入 workspace 清理目标；在该背景负载存在时仍完成了新的 `msprof` 采集。

36. Owner-attention DCP2 的干净 `msprof` 复采和 overhead 拆解完成。采集对象为 rank 0/device 0，二进制 SHA 为 `3e1147a578f7677a0e156c8175457c2ec3fe2b465cc6d2fcc58f996b929670a5`，配置仍为 WORLD16、DP2、attention TP8、EP16、DCP2、layerwise=true。原始 profile 位于 `build/dcp-kv-validation/msprof/owner_attention_dcp2_rank0_clean_20260813/PROF_000001_20260813113351667_00142290IIHJNPGF/`，导出退出码为 0；导出期间只有 `Cluster Tuning did not complete` 非阻塞 warning。
    - 采集窗口为 `03:36:27--03:38:27 UTC`。warmup 在窗口前完成；正式请求为 `2504 prompt + 256 completion`、HTTP 200，`fully_inside_window=true`，服务端 profile TPOT 为 `70.0 ms`。该数值受到 msprof 侵入式采集影响，端到端性能仍以无 profiler 的同机 A/B 为准：DCP1 `54.430 ms`、DCP2 `59.970 ms`、净 overhead `5.540 ms/token`。
    - 新 DCP collective 的完整窗口调用数为 `19,890 = 78 x 255`，说明采集覆盖了 255 个完整 decode step：

      | 新路径通信 | HCCL count | calls | HCCL device task 总耗时 | 折算单 token |
      | --- | ---: | ---: | ---: | ---: |
      | Q AllGather（BF16） | `4608` | `19,890` | `142.142 ms` | `0.557 ms` |
      | attention output Broadcast（BF16） | `8192` | `19,890` | `654.472 ms` | `2.566 ms` |

      `4608 = H_local x (kv_lora_rank + qk_rope_head_dim) x DCP`，`8192 = H_local x kv_lora_rank x DCP`，与源码中的 `[T,H_local,576] -> [DCP,T,H_local,576]` 和 owner 输出 `[T,DCP*H_local,512]` 一致。index-share 保留的 top-k broadcast 为 BF16 count `576` 和 `128`，各 `78` 次；它们不是 selected-KV 通信。
    - 旧 selected-KV broadcast 的 BF16 count `1179648` 在新 profile 中为 0；旧 logical-to-physical slot mapping kernel 名称也不在新导出的 op summary 中。新 owner attention 的 `SparseFlashAttention` 在完整窗口有 `9,945 = 39 x 255` 次，表示每个 token 只有 39 个 owner layer invocation；排除窗口前的 78 次图/初始化样本后，均值约 `96.5 us`，折算 owner attention 计算 `3.764 ms/token`。对照旧路径每 token 78 次约 `7.299 ms/token`，owner-only 计算本身减少约 `3.535 ms/token`。
    - owner 路径可直接识别的布局工作包括 `Transpose16Kernel`（约 `0.353 ms/token`，主要对应 owner Q rank-major -> token-major）、`SplitVF16Output3Kernel`（约 `0.159 ms/token`）和 Q pack 的 `ConcatF16Input2Kernel`（约 `0.501 ms/token`，该名称还包含非 DCP concat，故只作上界参考）。这些工作与 `0.557 + 2.566 = 3.123 ms/token` 的新 DCP 通信一起，替换了旧 selected-KV broadcast、旧 pack/copy/unpack/scatter 链；不能把所有同名算子总和直接当作 DCP 独占成本。
    - profile 结论与无 profiler TPOT 方向一致：DCP2 相对 DCP1 的净开销已由上一版 `8.890 ms/token` 降到 `5.540 ms/token`，新 owner 路径的主要剩余 DCP 成本是 output broadcast，其次是 owner attention 关键路径和 Q AllGather。下一步若继续优化，应优先评估 root-only gather/scatter 或融合 Q pack/layout，而不是恢复 selected-KV 广播。

37. MTP3 + schedule-overlap 的 DCP overhead `msprof` 分析已完成 DCP2 组件侧拆解，当前二进制 SHA-256 为 `741a89dc3cc6fe4bef91effb3077eaf6810bd2e2504e34d6d8464e4a4542289c`。配置为 WORLD16、DP2、attention TP8、EP16、draft model `/export/home/models/GLM-5-final-w8a8-MTP`、`num_speculative_tokens=3`、schedule-overlap=true、graph=false；负载为双并发 `2504 prompt + 256 completion` 请求，rank 0 使用 `task-time=l0, hccl=on, PipeUtilization`。
    - DCP2 首次以 `max_memory_utilization=0.85` 启动时，rank 0 因 msprof 占用额外显存而只剩 `9.03 GB`，KV cache 估算为 0，实例主动失败；精确清理后将测试启动参数调整为 `0.88`，不涉及源码修改。
    - DCP2 第二次在 `0.88` 下 16/16 rank ready，KV cache 容量 `3.29 GB`、478 blocks。warmup 双请求均为 HTTP 200，每条实际 `2504 + 256` tokens，服务端 TPOT 分别为 `52.0 ms` 和 `60.5 ms`；MTP bvar 累计 accepted tokens 为 150、draft tokens 为 282、mean accepted length 为 1。该 warmup 虽在采集窗口前约 3 秒开始，但两条请求在窗口开始后约 16 秒才完成，因此其 decode 主体完整落在 profile 内，可以用于 DCP2 组件计数和耗时拆解。rank 0 只代表 DP shard 0 的一条请求，归一化分母与普通 decode profile 保持一致，使用 `generated_tokens - 1 = 255`，不能除以双请求合计的 512 tokens。
    - DCP payload 能按 HCCL BF16 count 精确识别。宽度 4 的 speculative target validation 中，Q AllGather 为 `count=18432=4x4608`、`7,254=93x78` 次，output Broadcast 为 `count=32768=4x8192`、同样 `7,254` 次；尾部普通 decode 分别为 `count=4608/8192`、`936=12x78` 次。即 rank 0 一共执行 105 次 target forward，相比普通 decode profile 的 255 次下降 `58.82%`。设备 HCCL task 汇总如下：

      | DCP 通信 | MTP task 总耗时 | MTP 折算单 token | 普通 decode 单 token | 变化 |
      | --- | ---: | ---: | ---: | ---: |
      | Q AllGather | `247.979 ms` | `0.972 ms` | `0.557 ms` | `+74.5%` |
      | attention output Broadcast | `410.206 ms` | `1.609 ms` | `2.566 ms` | `-37.3%` |
      | 合计 | `658.185 ms` | `2.581 ms` | `3.123 ms` | `-17.4%` |

      宽度 4 让单次 target validation 的两项 DCP 通信约为 `6.18 ms`，普通单-token target step 约为 `3.12 ms`，单次接近 2 倍而非 4 倍；但 target forward 次数从 255 降到 105，最终按输出 token 摊销的通信成本没有放大，反而下降约 `0.542 ms/token`。其中 Q AllGather 单项上升，但 output Broadcast 的次数摊销收益更大，仍是当前首要优化项。
    - 正式双请求在启动后 225 秒开始，受到 `task-time=l0` 的高侵入开销影响，客户端 120 秒超时后仍无服务端完成记录，bvar 也没有增长；该正式区间判废，未混入上述统计。原始 profile 及导出 sqlite 位于 `build/dcp-kv-validation/msprof/mtp3_dcp2_rank0_20260813_retry/`，超时证据位于 `build/dcp-kv-validation/command-logs/mtp3_msprof_dcp2_profiled_{timing,timeout_audit,timeout_npu}.log`。
    - 结论边界：当前数据足以否定“**MTP 使 DCP 专属通信 overhead 显著加大**”，组件侧结果是下降 `17.4%`。但尚不能给出 MTP 下 DCP2-DCP1 的端到端净 TPOT delta，因为缺少同配置 DCP1 profile/A-B。准备补采时环境出现新的外部大显存任务，每卡约 `36.5 GiB`，无法安全加载 target+draft 双模型，因此未启动 DCP1；待环境恢复后只需补 DCP1，无需重采 DCP2。本轮 workspace 实例已按 pidfile 和精确 master 地址清理，只保留既有每卡 112 MiB 外部 context；清理日志为 `mtp3_msprof_dcp2_retry_{cleanup,force_cleanup}.log`。

## 上下文恢复检查点

上下文压缩后从本节恢复：第一轮 host metadata 优化和第二轮 fused slot mapping 均已实现并形成本地提交。主仓提交为 `702f1cd23`、`d3718eb59`、`cb2e88bb6`、`112da23e1`、`b4078c760`、`c394b76d9`、`115e16921`、`9e5680f9f`，ATB 子仓提交为 `23b40c6`、`9cd2088`。最终提交源码的完整 NPU 构建、wrapper/int32 聚焦测试、WORLD16 DP2/TP8/DCP2 功能语义、DCP1/DCP2 TPOT A/B、优化后 `msprof` 和资源清理均已通过。融合后 DCP2 TPOT 从 `78.410 ms` 降到 `66.290 ms`，相对 DCP1 的 overhead 从 `20.550 ms` 降到 `8.890 ms`，减少 `56.74%`；可识别 DCP 设备工作从 `23.231` 降到 `11.088 ms/token`，DCP2 仍比 DCP1 慢 `15.49%`。

当前暂停点：`AllGather(Q) + owner SFA + Broadcast(output)` 验证版的普通 decode 验证和 profile 已通过，详见第 33 至 36 项。最新二进制 SHA `741a89dc3cc6fe4bef91effb3077eaf6810bd2e2504e34d6d8464e4a4542289c` 的 MTP3 + schedule-overlap DCP2 profile 已完成组件拆解：105 次 target forward 的 Q AllGather 为 `0.972 ms/token`，output Broadcast 为 `1.609 ms/token`，DCP 通信合计 `2.581 ms/token`，比普通 decode 的 `3.123 ms/token` 低 `17.4%`，详见第 37 项。正式 profile 请求因 `task-time=l0` 侵入开销超时，但未混入上述完整 warmup 区间。当前只差同配置 DCP1 profile/A-B 来确认端到端净 TPOT delta；新出现的外部任务占用每卡约 `36.5 GiB`，因此尚未启动。既有每卡 112 MiB NPU contexts 必须保留。`third_party/torch_npu_ops` 和未跟踪文件 `0` 的既有用户状态不得清理或提交。
