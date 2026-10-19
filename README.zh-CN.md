# HammerTranslator

[English](README.md)

HammerTranslator V1 将英文 PDF 解析、翻译并回填为简体中文 PDF，尽量保持原始页数、页面尺寸和文本边界框。完整流程依次为 MinerU 解析、受保护内容翻译、固定布局回填（Reflow）和最终 PDF 验证。

当前 V1 适合单用户运行。Apple Silicon Mac 是本地开发环境；Linux/NVIDIA GPU 命令行环境已在 a6000 完成实际部署与端到端测试。面向公网的多用户服务和富文本样式重建尚未纳入 V1。

## 在 Mac 上安装

在仓库根目录执行：

```bash
conda activate hammertrans
python -m pip install -r requirements/macos.txt
npm --prefix formula_runtime ci --ignore-scripts --no-audit --no-fund
```

按 [字体说明](assets/fonts/README.md)放置 `SimHei.ttf`、`SimSun.ttf` 和 `SimKai.ttf`。将 `.env.example` 复制为本地 `.env`，只填写实际使用的翻译服务凭据。字体二进制文件和 `.env` 不提交到 Git。

## 环境检查

```bash
hammer-doctor --profile local_mac --backend pipeline \
  --provider deepseek --require-provider-key
hammer-smoke
```

`hammer-doctor` 检查依赖、凭据是否存在、字体校验、MathJax 和运行目录，不打印密钥值。`hammer-smoke` 是离线的 Translation→Reflow 测试，不会调用 MinerU 或翻译 API。

在 Linux/GPU 环境中，将预检参数改为 `--profile linux_gpu`。详细依赖和安装方式见 [Linux/GPU 说明](docs/LINUX_GPU_RUNTIME.md)。

## 运行完整流程

以下命令会实际调用 MinerU 和所选翻译服务，因此可能产生 API 费用：

```bash
conda run --no-capture-output -n hammertrans \
  hammer-translator \
  --jobs-root var/jobs \
  run input.pdf \
  --run \
  --environment-profile local_mac \
  --parser-backend pipeline \
  --provider deepseek
```

在 Linux/GPU 环境中使用 `--environment-profile linux_gpu --parser-device cuda`，并先用 `nvidia-smi` 确认目标 GPU 有足够空闲显存。需要固定某张卡时，可在命令前设置 `CUDA_VISIBLE_DEVICES=GPU编号`。`pipeline` 是当前常用的解析后端；只有需要视觉语言模型路径时才选择 `vlm-engine`。

命令开始时会打印任务 ID（下文记为 `JOB_ID`）。`--jobs-root` 决定任务存储位置，命令在不同工作目录执行时，相对路径可能指向不同位置；正式运行建议使用绝对路径。每次 `run` 都会创建新任务，不会覆盖旧任务。

要先测试前两页，在 `run` 命令末尾加上 `--parser-start-page 0 --parser-end-page 1`；完整论文测试请省略这两个参数。页码从 0 开始。

## 查看状态和结果

查看任务状态、事件，或恢复可重试的失败任务：

```bash
hammer-translator --jobs-root var/jobs status JOB_ID
hammer-translator --jobs-root var/jobs events JOB_ID
hammer-translator --jobs-root var/jobs resume JOB_ID --run
```

主要产物位于 `<jobs-root>/<JOB_ID>/`：

| 路径 | 内容 |
| --- | --- |
| `input/source.pdf` | 本任务保存的原始 PDF 副本 |
| `parse/` | MinerU 输出及适配后的 JSON |
| `translation/` | 译文、布局规划和阶段记录 |
| `output/translated.pdf` | 回填后的 PDF |
| `output/reflow_feedback.json` | 回填结果与待复核信息 |
| `output/validation_report.json` | 最终 PDF 验证报告 |
| `job.json`、`events.jsonl` | 任务状态与进度事件 |

`completed` 表示全部阶段成功且没有警告；`completed_with_warnings` 表示已生成 PDF，但仍有待复核内容。V1 会在原位置保留无法安全处理的内容，不会依据人工回复自动修改 PDF。参考文献、独立公式中的英文说明等内容按当前策略保留原文。验收时应同时查看验证报告和 PDF 的实际视觉效果。

## 更多文档

- [Pipeline 阶段和产物契约](PIPELINE_V1.md)
- [运行路径与环境变量](docs/RUNTIME_CONFIGURATION.md)
- [Linux/GPU 安装与预检](docs/LINUX_GPU_RUNTIME.md)
- [发布与迁移检查清单](docs/RELEASE_CHECKLIST.md)
- [翻译阶段设计记录](TRANSLATION_REWRITE_GUIDE.md)

仓库根目录的旧 Python 脚本仍作为兼容入口保留；新部署优先使用安装后的 `hammer-*` 命令。
