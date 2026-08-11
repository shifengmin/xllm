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
- [ ] Review 并 profile host 侧 metadata 准备瓶颈。
- [ ] 优化 metadata 准备，回归构建和功能并量化收益。

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

## 上下文恢复检查点

上下文压缩后从本节恢复：完整 NPU 构建成功，两组聚焦测试分别 11/11、12/12 通过。单机 WORLD=16、DP=2、attention TP=8、DCP=2 已完成短请求、两个并发 2504-token 长请求及三条语义问答；DCP 关闭基线用相同三条请求复验，`Paris`、`107` 和 KV cache/prefill/decode 技术结论均一致。所有服务及关联进程已停止。下一步将已验证的构建和运行 bugfix 形成本地 commit，然后 review/profile host metadata 瓶颈；每个有量化进展的优化阶段形成独立本地 `perf:` commit。
