# 功能边界、实现证据与成熟工具对比

文档核对日期：2026-10-30。实现基线为 [`bae4194`](https://github.com/Nicksxs/llm-hardware-calculator/commit/bae419487a5be54f3c4e198bfe1835cbba5d3998)；本文整理已有实现与 10 月 16–17 日记录，不新增功能或实测结果。

## 1. 项目解决什么问题

输入一个本地 GGUF 文件和 GPU/RAM/context 预算，输出可追溯到每个张量的存储清单、受支持的模型结构、单独显卡与 CPU 的放置方案，以及容量判断所用的假设。核心路径是：

`GGUF 随机读取 → 目录与整数边界校验 → 张量字节核算 → 架构/形状校验 → 固定后端放置 → GPU/RAM 预算报告`

MoonBit 实现解析、结构提取、分组与规划；C 只桥接 POSIX 随机文件读取，Python 用于独立对照和测量。本项目不内嵌推理引擎。原来的参数量/BPW 公式入口仍保留为没有模型文件时的粗估模式，与真实文件路径分开。

## 2. 已实现能力及证据入口

| 能力 | 实现与可核查证据 | 当前边界 |
| --- | --- | --- |
| 真实模型文件解析 | [parse.mbt](../src/gguf/parse.mbt)、[格式与 ReaderAt API](gguf.md)：解析 GGUF 元数据与张量目录，检查短读、UInt64 溢出、已知跨度越界和重叠。 | 单个小端 v2/v3 文件；不读取/解码权重值；不支持分片合并、大端或嵌套数组。未知类型保留为不完整清单，不能继续规划。 |
| 张量级字节核算 | [layout.mbt](../src/gguf/layout.mbt)、[profile.mbt](../src/model/profile.mbt)：按每个张量的实际类型、维度、量化块长度及块存储字节核算，分别统计 embedding、output 和非均匀 block。 | 精确对象是受支持类型的序列化 payload；不等同于驱动、计算缓冲、对齐开销和进程显存峰值。 |
| 结构与形状校验 | [profile.mbt](../src/model/profile.mbt)、[适配规则](artifact-planning.md)：从文件元数据提取结构，再与张量名称、维度、必需成员交叉检查。 | 仅 dense、full-attention 的 `llama`/`qwen2` 子集；额外张量或不支持特征直接报错，不静默忽略。Qwen2 目前只有合成测试，没有真实硬件加载记录。 |
| 单 GPU + CPU 放置 | [plan.mbt](../src/model/plan.mbt)：按固定 llama.cpp 版本的 output-first 规则评估指定卸载数，或搜索满足 GPU/RAM 预算的最大卸载数；保留 CPU embedding，并核算 tied-output 的 GPU 副本。 | 一张独立显卡与普通 CPU buffer；不是多 GPU、统一内存或通用异构调度器。返回可容纳的方案不代表速度最优。 |
| KV 与容量报告 | [规划说明](artifact-planning.md)：区分请求 context 与 256-token 对齐后的分配 context，输出 GPU/RAM、steady/loading 和额外预留。 | 一个序列；运行时和加载预留仍是策略参数，未完成校准。量化 KV 只作为已披露的容量场景，运行兼容性未验证。 |
| 独立对照与后端验证 | [验证总表](validation.md)、[CPU 复现](backend-validation.md)、[CUDA 复现](gpu-validation.md)。 | 下节分别列出证据强度；不能由单元测试数或小模型通过率推断大模型性能。 |

### 已有测量能支持什么结论

- **目录正确性：** 两个官方小模型 `stories260K.gguf` 和 `stories15M-q4_0.gguf`，共 105 个张量描述符，与固定版本官方 Python GGUFReader 的名称、维度、类型、元素数、payload 字节和偏移逐项一致。[原始报告](../examples/validation/gguf-comparison.json)
- **大偏移处理：** 实际稀疏文件逻辑长度 4,294,967,460 字节，解析读取 109 字节。它验证大于 4 GiB 的文件偏移与目录读取，不是加载了一个 4 GiB 语言模型。大于 `2^53` 的偏移另由虚拟 ReaderAt 测试覆盖。[验证说明](validation.md)
- **CPU 加载：** 两个小模型、四次运行，模型 payload 和分配 KV 的预期值落在后端两位小数 MiB 日志的舍入区间内；独立记录峰值 RSS，不把 RSS 当作张量字节。[原始报告](../examples/validation/cpu-runtime.json)
- **CUDA 放置：** RTX 5060 Ti 上七个用例，包括五个官方小模型用例和两个明确标记的衍生 tied-output fixture。逐层 CPU/CUDA0 放置和模型/KV 分配对照通过；不是七个不同真实大模型。[原始报告](../examples/validation/gpu/gpu-runtime-run1.json)
- **保留反例与限制：** CUDA 的 `ngl=0` 虽将权重/KV 放在 CPU，仍记录 17.69 MiB CUDA compute buffer。设备采样包含其他进程，进程 VRAM 峰值未取得；不能据此声称零 GPU 总占用或普遍内存误差百分比。

这些记录没有建立 27B 容量结论、真实大模型吞吐基准、输出质量结论或普遍后端兼容保证。本次文档更新没有重新运行这些测量。

## 3. 与成熟工具如何比较

比较针对具体组件，不把某个命令的边界推广到整个上游项目。下表是功能和输入/输出对照，**不是性能排行榜**，没有进行同条件耗时或内存开销竞赛。

核查来源：

- **A：Python / Hugging Face Accelerate `estimate-memory`**，固定文档版本 [90b233c](https://github.com/huggingface/accelerate/blob/90b233c86202a5ded4bf66407e75222c5fb007fc/docs/source/usage_guides/model_size_estimator.md)。通过配置在 meta device 上构建模型，报告不同 dtype 下最大层、总大小和 Adam 训练估算，不需下载完整权重。
- **B：Python / llama.cpp GGUFReader**，固定版本 [88aaf32 的 reader 源码](https://github.com/ggml-org/llama.cpp/blob/88aaf325a7026ac1127d0897394b0a3d6413dc2a/gguf-py/gguf/gguf_reader.py)。提供 GGUF 元数据、张量描述符和基于 NumPy memmap 的数据视图，是本项目目录对照的参考。
- **C：C++ / llama.cpp loader 与 llama-bench**，同一固定版本的 [模型加载/设备选择](https://github.com/ggml-org/llama.cpp/blob/88aaf325a7026ac1127d0897394b0a3d6413dc2a/src/llama-model.cpp) 和 [benchmark 文档](https://github.com/ggml-org/llama.cpp/blob/88aaf325a7026ac1127d0897394b0a3d6413dc2a/tools/llama-bench/README.md)。实际加载执行；benchmark 可重复测量 prompt processing / token generation，报告 tokens/s、标准差及结构化结果。该指标不包含 tokenization 和 sampling 时间。

| 比较维度 | 本项目 MoonBit 路径 | A：estimate-memory | B：GGUFReader | C：loader / llama-bench |
| --- | --- | --- | --- | --- |
| 输入基础 | 本地 GGUF 目录及用户容量预算；另有手动粗估入口 | 模型配置、模型库和目标 dtype | 本地 GGUF 文件 | 模型文件、后端设备与运行参数 |
| 实际混合量化张量清单 | 按文件中逐张量类型/形状核算 | 上述命令估算模型/dtype，不读取本地 GGUF 混合量化目录 | 提供实际描述符、类型和字节数 | 读取模型并按后端规则分配 |
| 运行前放置解释 | 输出逐张量 CPU/GPU 位置、重复副本及预算理由；只覆盖固定单 GPU 策略 | 上述命令主要报告大小；不把这一边界等同于 Accelerate 整库能力 | Reader 本身不做设备容量规划 | 有实际设备选择与分配机制，并可输出加载日志 |
| 架构与多设备范围 | dense llama/qwen2 子集；无多 GPU/MoE 规划 | 随所选模型库及实现变化；本文不声称整个 Accelerate 不支持多设备或 MoE | 文件格式读取与架构执行支持是两回事 | benchmark 提供 split-mode、tensor-split、device、CPU MoE 等选项；可用性取决于架构与后端，不能理解为任意组合均支持 |
| 实际推理速度 | 未建立真实大模型性能基准 | 此命令不是推理速度测试 | Reader 不是推理引擎 | 实际执行并统计 pp/tg 吞吐及重复测量分布 |
| 在本项目中的用途 | 可审查的预先容量账本与 MoonBit 原生实现 | 确定概念级估算与文件级核算的差别 | 逐项检查解析输出 | 核对实际放置、组件分配，并作为后续性能测量工具 |

项目的现有工程价值在于：将有界二进制解析、类型/形状校验、非均匀层核算、指定卸载评估和独立证据串成可审查的 MoonBit 路径。它不替代上述成熟库，也没有证明比它们更快、更准确或架构覆盖更广。原生语言实现和无外部 Mooncakes 依赖是实现属性，不单独构成功能或性能优势。

## 4. 尚未完成的功能边界

| 缺口 | 需要补上的工程与验证 |
| --- | --- |
| 多 GPU 放置 | 每张卡独立预算、张量/层分配、跨设备副本与通信约束，以及实际多卡日志验证。当前单卡搜索不能充当多卡算法。 |
| 通用异构与统一内存 | 不同 buffer/backend 的分配语义、共享内存与实际回退行为。目前仅固定独立显卡 + CPU 方案。 |
| MoE 架构 | 专家张量、共享专家/路由器、驻留与激活边界等适配；不能只用激活参数量代表全部常驻权重。当前 profile 主动拒绝 expert 特征。 |
| 真实性能评估 | 代表性受支持真实模型的固定条件 pp/tg 吞吐、内存观测和预测误差分析；当前 tiny fixture 的分配检查不替代这一项。 |
| 总内存预测校准 | 将权重、KV、计算缓冲、驱动与加载阶段分别观测，再检验运行时/峰值预留；当前容量预算仍有未校准项。 |

若继续性能验证，至少应保存模型与二进制哈希、硬件/驱动、后端版本、卸载层数、context、KV dtype、batch、线程数、预热和重复次数。先独立对照张量 payload 与后端组件日志，再报告可取得的进程/设备内存，最后记录 pp/tg 吞吐；不可取得的峰值保留为空并说明原因。这是一份后续验证要求，不是已经执行的结果或交付承诺。
