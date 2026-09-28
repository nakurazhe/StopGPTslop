# StopGPTslop

[English](README.md) | **简体中文**

清除 GPT Image 2 生成图上的那些痕迹——锐度过高、随机光点、边缘不自然、鳞片状纹路——同时保住画面本身。

本工具是专门针对 GPT Image 2 训练的。它处理的是该模型特有的问题，而非通用的 AI 图像瑕疵，因此用在其他生成器的输出上大概不会有效。

```
输入 → FLUX.2-VAE 编码 → z + α · R(z) → FLUX.2-VAE 解码 → 输出
```

`R` 是一个很小的残差网络（0.48M 参数），它预测图像隐空间表示上的一个修正量。`α` 缩放这个修正量，也是你唯一需要调的参数。改动它不涉及任何重新训练。

## 快速开始

用 [uv](https://docs.astral.sh/uv/)，无需手动配置环境：

```bash
git clone https://github.com/nakurazhe/StopGPTslop.git
cd StopGPTslop
uv run webui.py                                  # 网页界面，会自动打开浏览器
uv run modeling.py -i ./images -o ./cleaned      # 批量处理
```

或者用 pip：

```bash
pip install -e .
python webui.py
```

首次运行会从 HuggingFace 下载 VAE（约 320 MB）并缓存。离线环境可以单独下载后用 `--vae /path/to/FLUX.2-VAE` 指定本地目录。

推荐使用 CUDA 显卡；也能跑在 CPU 上（`--device cpu`），耗时约为 10 倍。

### 显卡驱动与 CUDA 版本

锁定的 PyTorch 构建对应 **CUDA 12.6**，需要 NVIDIA 驱动 **525 及以上**。确认显卡确实被用上了：

```bash
uv run python -c "import torch; print(torch.cuda.is_available())"
```

如果你有 NVIDIA 显卡但这里输出 `False`，通常是驱动版本低于该 CUDA 构建。把 `pyproject.toml` 里的 index URL 改成与驱动匹配的版本（`cu118`、`cu124`、`cu128` 等）再执行 `uv sync`。注意这种情况下 PyTorch 会**静默退回 CPU**，所以装完值得查一次。

## 网页界面

```bash
uv run webui.py                 # 然后打开 http://127.0.0.1:7860
```

拖入、点选或粘贴一张图片，拖动中缝即可对比处理前后。移动强度滑块，结果会立即更新。可以只下载处理结果，也可以下载并列对比图。

界面的语言（English / 简体中文）和明暗主题都跟随系统，也都能在顶栏手动切换。

## 命令行

```bash
uv run modeling.py -i INPUT -o OUTPUT             # 单个文件或整个目录
uv run modeling.py -i ./in -o ./out --alpha 0.25  # 更温和
uv run modeling.py -i ./in -o ./out --recursive   # 包含子目录
uv run modeling.py -i ./in -o ./out --side-by-side
```

已有输出的图片会被跳过，所以中断后直接重跑即可继续。用 `--overwrite` 强制重做。完整选项见 `--help`。

## 强度

| α | 档位 | 效果 |
|---|---|---|
| 0.25 | 保守 | 细节基本不动，轻度清理 |
| **0.5** | **平衡（默认）** | **较好的折中** |
| 0.75 | 强力 | 明显更干净，伴随一定纹理损失 |
| 1.0 | 激进 | 清理最强，可见的变软与颗粒感 |
| 0 | — | 不清理；用作参照 |

**怎么选**：从 0.5 起步。仍有残留就往上调；画面发软或纹理开始糊掉就往下调。在你觉得合适的取值中，选最低的那个。

有一点值得知道：**不要用"痕迹是否彻底消失"来判断效果。** 即便只清掉一部分，观感通常也已大幅改善；而为了追最后那点残留把强度推高，代价是真实细节。请按画面看起来如何来判断，而不是按去掉了多少。

## 性能

较新的显卡上约 **0.7 秒处理一张 1.5 MP 的图**，显存占用约 3 GB。这些时间几乎全部花在 VAE 上，修正网络本身的开销可以忽略。

推理路径已经调优过——VAE 会被编译，图像编码使用快速预设——两者都默认开启。批量处理时命令行会自动启用编译；长期运行的服务请显式传 `--compile on`。

## 已知局限

**部分细节无法挽回。** 这些痕迹与真实纹理处在同一尺度上，且在空间上互相重叠，因此清除它们总要付出一些真实细节的代价。在少部分图像上，两者耦合得足够紧，以至于没有令人满意的强度取值：清理够了就会明显变软，想保住细节就得留下痕迹。这是方法本身的性质，调参解决不了。

**最佳强度因图而异。** 同样是从 0.75 降到 0.5，有的图几乎没变化，有的图差别很大。目前没有可靠的自动选取办法，所以对一批相似的图，值得先用 `--side-by-side` 试几个取值。

**每张图都会过一遍 VAE。** 影响非常小，但严格来说输出在未改动区域也不是输入的逐像素副本。

## 权重

`weights/latent_residual.pt`（1.9 MB）。查看其元信息：

```python
import torch
print(torch.load("weights/latent_residual.pt", weights_only=False)["meta"])
```

## 许可

采用 [PolyForm Noncommercial License 1.0.0](https://polyformproject.org/licenses/noncommercial/1.0.0) 发布——可自由用于、修改和分享于非商业目的。商业使用请联系版权持有者。以 [LICENSE](LICENSE) 文件及其链接的许可证原文为准，本节仅为说明。

本工具所依赖的 FLUX.2-VAE 模型由 Black Forest Labs 以 Apache License 2.0 单独发布，在运行时下载，并未随本项目重新分发。
