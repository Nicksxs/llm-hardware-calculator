# LLM Hardware Calculator｜项目规划参考（AI 辅助草稿）

**立项阶段，2026 年 10 月 7 日。** 目前仅有项目文档与初步调研；MoonBit 工具链未安装，核心和 CLI 未实现，未运行测试或硬件实测。

**本文为 AI 辅助生成的规划参考，不能作为正式申报书提交。** 报名表要求申报书由申请人撰写（不要使用 AI 编写），采用 Markdown 格式且一页以内；仓库还需至少 10 个有效 commits，不得以无意义拆分、空提交或重复提交凑数。当前尚未提交报名。

项目拟帮助本地大模型用户理解权重、KV 缓存、运行时预留与硬件容量的关系，给出可追溯的内存预算和部署情景。独立 MoonBit core 承担校验、计算与解释；Native CLI 提供文本和 JSON，计划以公式、边界及报告一致性测试验证。

**Basic 仅需五项：** `params/quant/vram/ram/context`。内部采用显式结构代理，展示层数、KV 头数、头维度、来源、选择依据及敏感性情景；它是透明假设，不是自动识别模型。默认 FP16 KV。Advanced 可覆盖结构参数，选择 FP16、Q8_0、Q4_0 KV；量化 KV 展示内存情景，不保证后端与模型支持该组合。

首期覆盖单张独立 GPU、dense decoder-only、full attention、MHA/GQA、单序列；权重支持 FP16、BF16、Q8_0、Q6_K、Q5_K_M、Q4_K_M、IQ4_XS/NL。比较 **27B 参数、16 GiB VRAM、64 GiB RAM** 下四种重点量化与 **8K/16K/32K/64K** 上下文，展示代理变化如何影响结果，不宣称已有可运行或实测结论。

首期必须支持完整 GPU 驻留、**GPU+CPU OFFLOAD、CPU ONLY、INSUFFICIENT**。按公开的层均匀近似寻找分配方案，分别核算 GPU 与 RAM，不把二者相加。报告区分稳态驻留与加载峰值，列出 KV 放置、预留、暂存和副本假设；稳态满足不等于必然能加载，内存满足也不代表速度达标。

方法依据 [GGML 量化块](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-common.h)和 [llama.cpp KV 分配](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-kv-cache.cpp)。权重计入 scale 与混合精度，KV 按层、KV 头、维度、容量和精度估算。层大小、缓冲、padding 与碎片带来误差；敏感性范围不是实测置信区间，目前没有准确率或固定误差界限。完整公式与来源见 [README](../README.md#method-and-evidence)。MoE、MLA、混合或滑动注意力、多模态、多 GPU 暂不纳入。

所有代理与量化预设将版本化保存，报告保留原始输入和中间结果，便于第三方复核。对接近容量边界的方案明确显示余量与假设，同时提供结构变化后的对照结果。后续硬件验证会记录模型文件、后端版本、启动配置和分项占用，将理论预测与观察结果分别标注。

已完成范围和源码调研；后续依次固定工具链，实现代理、计算、分配和 CLI，补齐测试与可复现对照，再公开发布。后期可选 MoonBit JS 静态 Web，首期交付以 core、CLI、JSON、测试与透明方法为主。
