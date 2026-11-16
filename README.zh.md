# llm.c

<p align="center">
  <a href="README.md">English</a> · <b>简体中文</b>
</p>

无需 245MB 的 PyTorch 或 107MB 的 cPython，采用极简、纯粹的 C/CUDA 实现大语言模型（LLM）。当前重点关注预训练阶段，尤其是复现 [GPT-2](https://github.com/openai/gpt-2) 与 [GPT-3](https://arxiv.org/abs/2005.14165) 系列小规格模型，同时在 [train_gpt2.py](train_gpt2.py) 中提供了并行的 PyTorch 参考实现（该文件基于作者早期的 [nanoGPT](https://github.com/karpathy/nanoGPT) 项目微调而来）。目前，llm.c 的运行速度比 PyTorch Nightly 略快约 7%。除了位于 [train_gpt2.cu](train_gpt2.cu) 中的前沿主线代码外，还在单一文件 [train_gpt2.c](train_gpt2.c) 中以约 1,000 行清晰规范的代码提供了 CPU fp32 参考实现。本项目旨在专注于维护纯 C 与 CUDA 代码。非常欢迎向其他编程语言或独立仓库的移植实现，建议在独立仓库中进行，作者乐意在下方的“知名衍生分支 (notable forks)”章节中提供链接。开发者协调交流主要在 [Discussions](https://github.com/karpathy/llm.c/discussions) 以及 Discord 上进行（包括 [Zero to Hero](https://discord.gg/3zy8kqD9Cp) Discord 的 `#llmc` 频道，或 [GPU MODE](https://discord.gg/gpumode) Discord 的 `#llmdotc` 频道）。

## 快速上手 (Quick Start)

了解当前 llm.c 代码库的最佳切入点是复现 GPT-2 (124M) 模型。[Discussion #481](https://github.com/karpathy/llm.c/discussions/481) 详细记录了完整步骤。支持在 llm.c 以及并行的 PyTorch 实现中复现 GPT-2 与 GPT-3 系列的其他模型规格，详情请参阅 [scripts 说明文档](scripts/README.md)。

调试技巧：当执行 `make` 命令编译二进制文件时，可将 `-O3` 替换为 `-g`，以便在常用 IDE（如 VS Code）中进行单步源码调试。

## 快速上手（单 GPU，仅限 fp32）

如果不进行多节点分布式训练、暂不涉及混合精度，且希望学习 CUDA 编程，早期归档的 fp32 文件会很有参考价值。这些文件是 llm.c 早期发展历史中保存并冻结的代码快照，逻辑更简练、可移植性更强，也更易于理解。运行单 GPU fp32 代码的方法如下：

```bash
chmod u+x ./dev/download_starter_pack.sh
./dev/download_starter_pack.sh
make train_gpt2fp32cu
./train_gpt2fp32cu
```

`download_starter_pack.sh` 脚本可快速下载启动所需的系列 `.bin` 文件，包含：1）以 fp32 及 bfloat16 格式保存的 GPT-2 124M 模型权重；2）用于单元测试的“调试状态”（包含小批量数据及其对应的目标激活值和梯度）；3）GPT-2 分词器（Tokenizer）；4）经分词处理后的 [tinyshakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt) 数据集。此外，也可以不运行 shell 脚本，通过以下 Python 命令手动重新生成这些工件：

```bash
pip install -r requirements.txt
python dev/data/tinyshakespeare.py
python train_gpt2.py
```

## 快速上手 (CPU)

适合本地缺乏 GPU 硬件资源的环境。虽然在 CPU 上无法开展大规模训练，但依然可以直观体验 llm.c 的训练流程。与上述 fp32 版本类似，CPU 版本是 llm.c 早期纯 C 参考实现的代码快照。例如，可以基于 GPT-2 small (124M) 在莎士比亚风格文本上进行微调：

```bash
chmod u+x ./dev/download_starter_pack.sh
./dev/download_starter_pack.sh
make train_gpt2
OMP_NUM_THREADS=8 ./train_gpt2
```

如果不想运行 Starter Pack 脚本，如前所述，可依次执行 `python dev/data/tinyshakespeare.py` 与 `python train_gpt2.py` 生成完全相同的 `.bin` 文件与工件。

上述流程执行步骤包括：(1) 下载已分词的 [tinyshakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt) 数据集与 GPT-2 (124M) 预训练权重；(2) 在 C 代码中初始化权重，并使用 AdamW 优化器在 tinyshakespeare 上微调训练 40 个步长（batch size 为 4，上下文长度为 64）；(3) 评估验证集损失并采样生成文本片段。除非拥有性能强劲的 CPU 并适当调大启动命令中的 OMP 线程数，否则在 CPU 上训练 LLM 速度较为有限，但非常适合用作教学与参考演示。在配备 Apple Silicon M3 Max 的 MacBook Pro 上的输出如下所示：

```
[GPT-2]
max_seq_len: 1024
vocab_size: 50257
num_layers: 12
num_heads: 12
channels: 768
num_parameters: 124439808
train dataset num_batches: 1192
val dataset num_batches: 128
num_activations: 73323776
val loss 5.252026
step 0: train loss 5.356189 (took 1452.121000 ms)
step 1: train loss 4.301069 (took 1288.673000 ms)
step 2: train loss 4.623322 (took 1369.394000 ms)
step 3: train loss 4.600470 (took 1290.761000 ms)
... (trunctated) ...
step 39: train loss 3.970751 (took 1323.779000 ms)
val loss 4.107781
generating:
---
Come Running Away,
Greater conquer
With the Imperial blood
the heaviest host of the gods
into this wondrous world beyond.
I will not back thee, for how sweet after birth
Netflix against repounder,
will not
flourish against the earlocks of
Allay
---
```

## 数据集 (Datasets)

位于 `/dev/data/(dataset).py` 的数据处理脚本负责下载、分词并将 Token 保存为 C 代码可直接读取的 `.bin` 二进制文件。例如运行：

```bash
python dev/data/tinyshakespeare.py
```

该命令将下载并分词 [tinyshakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt) 数据集，输出信息如下：

```
writing 32,768 tokens to ./dev/data/tinyshakespeare/tiny_shakespeare_val.bin
writing 305,260 tokens to ./dev/data/tinyshakespeare/tiny_shakespeare_train.bin
```

生成的 `.bin` 文件包含 1024 字节的文件头信息，紧接着是以 uint16 格式存储的 Token 连续流（对应 GPT-2 分词器的 Token ID）。更多数据集可在 `/dev/data` 中获取。

## 测试 (Test)

代码库包含用于验证 C 代码与 PyTorch 参考实现严格一致的单元测试。以 CPU 环境为例，编译并运行测试：

```bash
make test_gpt2
./test_gpt2
```

该测试将加载由 `train_gpt2.py` 生成的 `gpt2_124M_debug_state.bin` 文件，执行前向传播，比对 Logits 与 Loss 与 PyTorch 的一致性，随后使用 Adam 运行 10 次迭代训练并验证损失值完全对齐。测试 GPU 版本的命令如下：

```bash
# fp32 测试（不支持 cuDNN）
make test_gpt2cu PRECISION=FP32 && ./test_gpt2cu
# 混合精度 cuDNN 测试
make test_gpt2cu USE_CUDNN=1 && ./test_gpt2cu
```

该测试将同时覆盖 fp32 路径与混合精度路径。测试通过后将输出 `overall okay: 1`。

## 教程 (Tutorial)

在 [doc/layernorm/layernorm.md](doc/layernorm/layernorm.md) 中提供了一份教程，循序渐进地讲解了如何在 C 语言中实现 GPT-2 模型的 LayerNorm 层，是理解模型各层在底层 C 语言中如何实现的优秀起点。

**Flash Attention**：自 2024 年 5 月 1 日起，项目集成了基于 cuDNN 的 Flash Attention。由于引入 cuDNN 会将编译耗时从数秒增加到约一分钟，且该代码路径较为前沿，因此默认未开启。编译时可通过如下命令显式启用：

```bash
make train_gpt2cu USE_CUDNN=1
```

启用后将链接 cuDNN 并运行。系统需预先安装 cuDNN 环境。可参考 [cuDNN 官方安装指南](https://developer.nvidia.com/cudnn) 使用 apt-get 安装。以 Ubuntu 22.04 与 CUDA 12.x 为例，最小化依赖安装命令如下：

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install libcudnn9-dev-cuda-12
```

此外还需要 [cuDNN frontend](https://github.com/NVIDIA/cudnn-frontend/tree/main)（纯头文件库，直接 clone 到本地即可）。Makefile 默认在用户主目录或当前目录下寻找该库；若放置在其他路径，可在 `make` 命令行中追加 `CUDNN_FRONTEND_PATH=/path/to/your/cudnn-frontend/include`。

## 多 GPU 分布式训练 (Multi-GPU Training)

确保系统已安装 MPI 与 NCCL。以 Linux 系统为例：

```bash
sudo apt install openmpi-bin openmpi-doc libopenmpi-dev
```

NCCL 安装请参考 [NVIDIA 官方网站指引](https://developer.nvidia.com/nccl/nccl-download)（如使用网络安装器）。安装完成后执行：

```bash
make train_gpt2cu
mpirun -np <GPU数量> ./train_gpt2cu
```

或者直接运行 `./scripts/` 目录下的自动化运行脚本。

## 多节点分布式训练 (Multi-Node Training)

请确保已按照[多 GPU 训练](#多-gpu-分布式训练-multi-gpu-training)章节完成 NCCL 的安装。

当前支持以下 3 种多节点训练启动方式：
1) 使用 OpenMPI 交换 NCCL ID 并初始化 NCCL，详见 `./scripts/multi_node/run_gpt2_124M_mpi.sh` 脚本。
2) 使用共享文件系统（Shared File System）初始化 NCCL，详见 `./scripts/multi_node/run_gpt2_124M_fs.sbatch` 脚本。
3) 使用 TCP 套接字（TCP Sockets）初始化 NCCL，详见 `./scripts/multi_node/run_gpt2_124M_tcp.sbatch` 脚本。

注意事项：
* 若在 Slurm 集群环境中运行且 Slurm 不支持 PMIx（鉴于 `slurm-wlm` 已停止支持 PMIx，这较为常见），可采用方式 (2) 共享文件系统或方式 (3) TCP 套接字。可通过运行 `srun --mpi=list` 检查输出中是否包含 `pmix`。
* 若未配置 Slurm，可直接通过 `mpirun` 使用方式 (1) MPI 启动多节点训练。

这 3 种方式各有适配场景，可根据具体集群环境灵活选用。

## 实验与超参数扫描 (Experiments / Sweeps)

以在配备 4 块 GPU 的服务器上针对 TinyStories 进行学习率超参数网格搜索为例，创建并运行 shell 脚本 `sweep.sh`（执行前确保赋予执行权限 `chmod u+x sweep.sh`）：

```bash
#!/bin/bash

learning_rates=(3e-5 1e-4 3e-4 1e-3)

for i in {0..3}; do
    export CUDA_VISIBLE_DEVICES=$i
    screen -dmS "tr$i" bash -c "./train_gpt2cu -i data/TinyStories -v 250 -s 250 -g 144 -l ${learning_rates[$i]} -o stories$i.log"
done

# 可通过以下命令统一关闭所有后台 screen 会话：
# screen -ls | grep -E "tr[0-3]" | cut -d. -f1 | xargs -I {} screen -X -S {} quit
```

该示例启动 4 个独立的 screen 后台会话，以不同的学习率并行训练，并将损失信息输出至 `stories$i.log` 日志文件中，后续可在 Python 中进行可视化分析。解析与绘制日志曲线的示例参见 [dev/vislog.ipynb](dev/vislog.ipynb)。

## 项目设计哲学 (Repo)

关于本项目的设计理念与定位：

首先，希望 `llm.c` 成为一个优秀的教育教学平台。例如项目中的 `dev/cuda` 目录汇集了全套手工编写且注释详尽的神经网络算子算核（Kernel）库，从最基础直接的实现演进至高度优化的高性能算核。若有探索不同权衡取向的新算核，非常欢迎提交贡献。

其次，也追求让 `llm.c` 具备极致性能，可直接用于实际网络训练。例如支持复现完整的 GPT-2 (1.6B) 预训练流程。这要求整合顶尖算核库，包括 cuBLAS、cuBLASLt、CUTLASS、cuDNN 等。这种设计兼具教学价值，能为算核优化建立权威基准（例如评估手写算核达到了 cuBLAS 速度的 80% 等），开发者既可以选择追求极速运行，也可以自由“插拔组合”感兴趣的手写算核。

项目的一大核心约束是保持根目录下主线 `llm.c` 的简洁与高可读性。如果某个 PR 带来了 2% 的性能微调，但代价是增加 500 行晦涩复杂的 C 代码或引入小众第三方依赖，可能会被拒绝，因为这违背了代码简洁性原则。相比之下，将用于矩阵乘法的 cuBLAS 设为主训练循环的默认选择则是明确的：它大幅提升了主线速度，代码仅需单行且语义清晰，属于通用成熟依赖。与此同时，可以在 `dev/cuda` 中保留并演进与 cuBLAS 竞争的手写算子实现。

总而言之，根目录的核心代码保持极高的代码纯粹性与低复杂度；而 `dev/` 目录则可作为开发试验场，用于构建算核库、共享教育演示代码，并允许局部存在一定的工程复杂度。

## 知名衍生分支 (Notable Forks)

- **AMD 硬件支持**
  - [llm.c](https://github.com/anthonix/llm.c)（由 @[anthonix](https://github.com/anthonix) 开发）：支持 AMD 显卡设备（如 Radeon 7900 XTX）

- **C# 实现**
  - [llm.cs](https://github.com/azret/llm.cs)（由 @[azret](https://github.com/azret) 开发）：本项目的 C# 移植版本
  - [Llm.cs](https://github.com/nietras/Llm.cs)（由 @[nietras](https://github.com/nietras) 开发）：注重跨平台开箱即用的 C# 移植版本，克隆即可运行 ✅

- **CUDA C++**
  - [llm.cpp](https://github.com/gevtushenko/llm.c)（由 @[gevtushenko](https://github.com/gevtushenko) 开发）：基于 [CUDA C++ 核心库 (CCCL)](https://github.com/NVIDIA/cccl) 的移植版本
     - [GPU MODE Discord 社区讲座](https://discord.gg/cudamode)中包含了关于此分支的[专题分享视频](https://www.youtube.com/watch?v=WiB_3Csfj_Q)

- **C++/CUDA**
  - [llm.cpp](https://github.com/zhangpiu/llm.cpp/tree/master/llmcpp)（由 @[zhangpiu](https://github.com/zhangpiu) 开发）：基于 [Eigen](https://gitlab.com/libeigen/eigen) 的移植版本，支持 CPU 与 CUDA

- **WebGPU C++**
  - [gpu.cpp](https://github.com/AnswerDotAI/gpu.cpp)（由 @[austinvhuang](https://github.com/austinvhuang) 开发）：基于原生 WebGPU 的 C++ 可移植 GPU 计算库，旨在打造通用库的同时将 llm.c 算核移植至 WGSL

- **C++ 极简头文件库**
  - [llm.cpp](https://github.com/GaoYusong/llm.cpp)（由 @[GaoYusong](https://github.com/GaoYusong) 开发）：包含单头文件 [tinytorch.hpp](https://github.com/GaoYusong/llm.cpp/blob/main/tinytorch.hpp) 库的 C++ 移植版本

- **Go 语言实现**
  - [llm.go](https://github.com/joshcarp/llm.go)（由 @[joshcarp](https://github.com/joshcarp) 开发）：本项目的 Go 语言移植版本

- **Java 语言实现**
  - [llm.java](https://github.com/harryjackson/llm.java)（由 @[harryjackson](https://github.com/harryjackson) 开发）：本项目的 Java 语言移植版本

- **Metal (Apple Silicon)**
  - [llm.metal](https://github.com/regrettable-username/llm.metal)（由 @[regrettable-username](https://github.com/regrettable-username) 开发）：基于原生 C / Metal 着色语言（MSL）实现的 LLM 训练

- **Mojo 实现**
  - [llm.🔥](https://github.com/dorjeduck/llm.mojo)（由 @[dorjeduck](https://github.com/dorjeduck) 开发）：本项目的 Mojo 语言移植版本

- **OpenCL 实现**
  - [llm.c](https://github.com/krrishnarraj/llm.c)（由 @[krrishnarraj](https://github.com/krrishnarraj) 开发）：本项目的 OpenCL 移植版本

- **Rust 实现**
  - [llm.rs](https://github.com/yijunyu/llm.rs)（由 @[Yijun Yu](https://github.com/yijunyu) 开发）：以同等高性能为目标的 Rust 重写版本
  - [llm.rs](https://github.com/ToJen/llm.rs)（由 @[ToJen](https://github.com/ToJen) 开发）：本项目的 Rust 语言移植版本

- **Swift 实现**
  - [llm.swift](https://github.com/otabuzzman/llm.swift)（由 @[otabuzzman](https://github.com/otabuzzman) 开发）：本项目的 Swift 语言移植版本

- **Zig 实现**
  - [llm.zig](https://github.com/Saimirbaci/llm.zig)（由 @[saimirbaci](https://github.com/Saimirbaci) 开发）：本项目的 Zig 语言移植版本

- **Habana Gaudi2**
  - [llm.tpc](https://github.com/abhilash1910/llm.tpc)（由 @[abhilash1910](https://github.com/abhilash1910) 开发）：面向 Habana Gaudi2 硬件平台的移植版本

- **Nim 实现**
  - [llm.nim](https://github.com/Vindaar/llm.nim)（由 @[Vindaar](https://github.com/Vindaar) 开发）：本项目的 Nim 语言移植版本

## 社区讨论与参与 (Discussions)

开发者组织与交流通道：

- 遇到具体代码问题与缺陷？请提交至 [Issues](https://github.com/karpathy/llm.c/issues)。
- 准备了功能贡献或修复代码？欢迎提交 [Pull Request](https://github.com/karpathy/llm.c/pulls)。
- 探讨交流代码库用法、提出疑问等？欢迎访问 [Discussions](https://github.com/karpathy/llm.c/discussions)。
- 需要更快速的实时交流？欢迎加入 [Zero to Hero Discord 社区](https://discord.gg/3zy8kqD9Cp)的 `#llmc` 专属频道。

## 开源许可证 (License)

遵循 [MIT 许可证](LICENSE) 开源发布。

---

> **翻译说明（Translation Disclaimer）**
> 本文档由社区志愿者（[@JasonYeYuhe](https://github.com/JasonYeYuhe)）人工翻译并维护，旨在降低中文开发者的阅读门槛。最后同步日期：2026年11月16日。若译文与英文原版存在任何出入或滞后，请以 [英文原版 (README.md)](README.md) 为准。欢迎通过 PR 协助改进译文！
