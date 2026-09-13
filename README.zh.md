# MLX

<p align="center">
  <a href="README.md">English</a> · <b>简体中文</b>
</p>

[**快速上手**](#快速上手) | [**安装指南**](#安装指南) |
[**官方文档**](https://ml-explore.github.io/mlx/build/html/index.html) |
[**实战范例**](#实战范例)

[![CircleCI](https://circleci.com/gh/ml-explore/mlx.svg?style=svg)](https://circleci.com/gh/ml-explore/mlx)

MLX 是由 Apple 机器学习研究团队 (Apple machine learning research) 推出的面向 Apple 芯片的高性能机器学习阵列计算框架 (Array framework)。

MLX 的核心特性包括：

- **熟悉且一致的 API (Familiar APIs)**：MLX 提供了与 NumPy 深度对齐的 Python API。同时还提供了全功能的 C++、[C](https://github.com/ml-explore/mlx-c) 和 [Swift](https://github.com/ml-explore/mlx-swift/) 原生 API，其设计风格高度镜像对齐 Python API。此外，MLX 提供了 `mlx.nn` 和 `mlx.optimizers` 等高阶模块，其 API 风格高度贴近 PyTorch，极大简化了复杂模型的构建门槛。

- **可组合的函数变换 (Composable function transformations)**：MLX 原生支持可任意组合的函数变换技术，包括自动微分 (Automatic Differentiation)、自动向量化 (Automatic Vectorization) 以及计算图优化 (Computation Graph Optimization)。

- **惰性计算 (Lazy computation)**：MLX 采用惰性计算机制。仅在真正需要时才会实例化并物化阵列计算结果。

- **动态计算图构建 (Dynamic graph construction)**：MLX 的计算图采用动态构建机制。动态变更函数入参的维度形状 (Shapes) 不会触发缓慢的重新编译，使得模型调试变得极为简单直观。

- **跨设备支持 (Multi-device)**：各类算子操作均可在受支持的硬件设备上无缝执行（当前支持 CPU 与 GPU）。

- **统一内存架构 (Unified memory)**：MLX 与其他传统框架最为显著的核心差异在于其*统一内存模型 (Unified Memory Model)*。MLX 中的阵列直接驻留在共享内存中，在不同受支持的设备类型之间执行算子时无需进行任何冗余的数据搬运与内存拷贝。

MLX 由机器学习研究者专为机器学习研究者倾力打造。该框架旨在保持极致易用性的同时，兼具模型训练与部署的高吞吐效率。框架本身在概念架构上追求极简，力求让研究人员能够轻而易举地扩展和改进 MLX，从而以极快的速度探索与验证全新的前沿想法。

MLX 的架构设计灵感汲取自 [NumPy](https://numpy.org/doc/stable/index.html)、[PyTorch](https://pytorch.org/)、[Jax](https://github.com/google/jax) 以及 [ArrayFire](https://arrayfire.org/) 等经典科学计算与深度学习框架。

## 实战范例

[MLX 范例仓库 (mlx-examples)](https://github.com/ml-explore/mlx-examples) 提供了丰富全面的实战用例，包括：

- [Transformer 语言模型](https://github.com/ml-explore/mlx-examples/tree/main/transformer_lm)预训练与训练。
- 基于 [LLaMA](https://github.com/ml-explore/mlx-examples/tree/main/llms/llama) 的大规模文本生成以及基于 [LoRA](https://github.com/ml-explore/mlx-examples/tree/main/lora) 的高效微调。
- 基于 [Stable Diffusion](https://github.com/ml-explore/mlx-examples/tree/main/stable_diffusion) 的文本到图像生成。
- 基于 [OpenAI Whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper) 的自动语音识别 (ASR)。

## 快速上手

请参阅官方文档中的[快速入门指南 (Quickstart Guide)](https://ml-explore.github.io/mlx/build/html/usage/quick_start.html)。

## 安装指南

MLX 已发布至 [PyPI](https://pypi.org/project/mlx/)。在 macOS 环境下安装 MLX，请执行：

```bash
pip install mlx
```

在 Linux 环境下安装支持 CUDA 后端的软件包：

```bash
pip install mlx[cuda]
```

在 Linux 环境下安装仅支持 CPU 的轻量软件包：

```bash
pip install mlx[cpu]
```

如需从源码编译 C++ 和 Python API，请参阅[安装与源码编译文档](https://ml-explore.github.io/mlx/build/html/install.html#)。

## 参与贡献

请查阅[贡献指南 (Contributing Guide)](https://github.com/ml-explore/mlx/tree/main/CONTRIBUTING.md) 获取参与 MLX 社区贡献的完整指引。更多关于从源码构建与运行单元测试的细节，请参考[官方说明文档](https://ml-explore.github.io/mlx/build/html/install.html)。

我们对所有[社区贡献者](https://github.com/ml-explore/mlx/tree/main/ACKNOWLEDGMENTS.md#Individual-Contributors)深表感谢！如果您向 MLX 贡献了代码或文档并希望被致谢鸣谢，请在您的 Pull Request 中将您的姓名添加到致谢名单中。

## 引用 MLX

MLX 软件套件最初由 Awni Hannun、Jagrit Digani、Angelos Katharopoulos 和 Ronan Collobert 以同等贡献共同研发。如果您在学术研究或项目中使用了 MLX 并希望予以引用，请使用如下 BibTeX 条目：

```text
@software{mlx2023,
  author = {Awni Hannun and Jagrit Digani and Angelos Katharopoulos and Ronan Collobert},
  title = {{MLX}: Efficient and flexible machine learning on Apple silicon},
  url = {https://github.com/ml-explore},
  version = {0.0},
  year = {2023},
  }
```

---

> 💡 **文档维护说明**：本中文文档由社区志愿者（@JasonYeYuhe）翻译维护，最后同步更新于 2026年9月13日。如发现内容与官方英文原版存在差异或新特性滞后，欢迎提交 PR 共同完善！
