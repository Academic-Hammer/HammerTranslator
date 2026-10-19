# HammerTranslator

[English](README.md)

HammerTranslator 将英文科研 PDF 转换为简体中文 PDF。它使用 MinerU 解析文档，通过大语言模型服务翻译可处理的文字，再将译文回填到原页面布局中，最后验证生成的 PDF。页面尺寸和文字边界框保持不变；无法安全回填的内容可以保留原文，并标记为待复核。

V1 是**单用户命令行程序**，支持 Apple Silicon macOS 和配备 NVIDIA GPU 的 Linux。当前版本不包含网页前端，也不能直接作为面向公网的多用户服务部署。

## 运行前准备

- Git、Conda、带 npm 的 Node.js（部署时使用过 Node.js 22）。
- Conda 环境中的 Python 3.12。软件包声明支持 Python 3.11–3.13，已测试的基线版本是 3.12。
- Apple Silicon Mac，或装有可用 NVIDIA 驱动和 GPU 的 `x86_64` Linux 服务器。本项目不会安装 GPU 驱动。
- 翻译服务的 API key（下文以 DeepSeek 为例）。真实翻译会请求服务商 API，可能产生费用。
- 下载安装两款 [SIL OFL 授权的默认字体](hammer_translator/resources/fonts/README.md)所需的网络连接。字体二进制文件单独安装，避免增大 Git 仓库和 Python wheel。
- 下载 MinerU 模型所需的磁盘空间与网络连接。首次解析可能因下载模型而更慢。

## 从源码安装

克隆仓库，并在仓库根目录执行后续命令：

```bash
git clone https://github.com/Academic-Hammer/HammerTranslator.git
cd HammerTranslator
conda create -n hammertrans python=3.12 -y
conda activate hammertrans
```

根据设备类型，**只安装其中一组**依赖：

```bash
# Apple Silicon macOS
python -m pip install -r requirements/macos.txt

# 或：配备 NVIDIA GPU 的 Linux
python -m pip install -r requirements/linux-gpu.txt
```

这两组依赖都会以可编辑模式安装当前源码。接着安装公式渲染使用的本地 MathJax：

```bash
npm --prefix formula_runtime ci --ignore-scripts --no-audit --no-fund
hammer-fonts install
hammer-fonts check
```

`hammer-fonts install` 会从作者的官方仓库下载[霞鹜文楷 Medium](https://github.com/lxgw/LxgwWenKai/blob/7ebd0771fda4285299670f83c398f32f32c7bef8/fonts/TTF/LXGWWenKai-Medium.ttf)和[霞鹜文楷 Regular](https://github.com/lxgw/LxgwWenKai/blob/7ebd0771fda4285299670f83c398f32f32c7bef8/fonts/TTF/LXGWWenKai-Regular.ttf)，校验固定版本的文件大小及 SHA-256，并安装至 `~/.hammer-translator/fonts/`。授权和清单见[字体说明](hammer_translator/resources/fonts/README.md)。该命令可重复执行，已校验通过的文件不会再次下载。

若服务器无法访问 GitHub，可在其他设备下载这两个文件，保持原文件名并传到服务器同一个目录，然后运行 `hammer-fonts install --from-dir /path/to/downloaded-fonts`。离线安装仍会校验文件大小和 SHA-256，并一同安装授权说明。

升级已有部署时，如果 `~/.hammer-translator/fonts/` 中仍是旧字体，请保留该目录供旧任务使用，先将 `HAMMERTRANSLATOR_FONTS_DIR` 指向新的空目录，再执行上述字体安装与检查命令。后续翻译和回填渲染也要保持相同的设置。

如果要在 Linux 上安装非可编辑版本，请参考 [Linux/GPU 部署说明](docs/LINUX_GPU_RUNTIME.md)中的 wheel 安装方式。锁定的解析依赖仍需与服务器驱动及 Python/CUDA 运行时兼容。

## 配置 API key

复制环境变量示例文件，再只在本机编辑 `.env`：

```bash
cp .env.example .env
```

默认使用 DeepSeek 时，在 `.env` 中填写 `TRANSLATION_API_KEY`；选择 `openai` 时填写 `OPENAI_API_KEY`。服务地址、模型以及变量名在 [translation_config.yaml](translation_config.yaml) 中配置，不要把密钥明文写入 YAML。进程中已有的环境变量优先于 `.env`。**不要提交 `.env` 或任何真实密钥。**

默认使用霞鹜文楷 Medium 排标题和小标题，Regular 排正文和图注。若已有其他合法授权字体，可通过 `HAMMERTRANSLATOR_FONTS_DIR` 指定自定义字体集；旧字体的兼容方式见[说明](assets/fonts/README.md)。更换字体会改变排版测量结果，使用自定义字体时请检查生成的 PDF。

从本仓库运行时，任务默认保存在 `var/jobs/`，密钥文件默认是仓库根目录的 `.env`。需要将运行数据放在其他位置时，可设置 `HAMMERTRANSLATOR_HOME` 或按[运行路径说明](docs/RUNTIME_CONFIGURATION.md)分别指定。执行 `run`、`status` 和 `resume` 时须使用同一个任务目录。

## 检查安装情况

按运行设备选择预检命令：

```bash
# macOS
hammer-doctor --profile local_mac --backend pipeline \
  --provider deepseek --require-provider-key

# Linux/NVIDIA
hammer-doctor --profile linux_gpu --backend pipeline \
  --provider deepseek --require-provider-key
```

预检会检查软件包和解析依赖、服务配置、字体哈希、MathJax、运行目录是否可写；不会打印密钥。Linux 预检还会检查 CUDA。随后可运行离线冒烟测试，验证翻译与回填模块的基本调用；它**不会**运行 MinerU，也不会调用翻译 API：

```bash
hammer-smoke
```

出现预检错误时应先修复再处理真实文档。依赖安装成功不代表 MinerU 模型已缓存完毕。

## 翻译 PDF

在仓库根目录运行完整的“解析 → 翻译 → 回填 → 验证”流程：

```bash
# macOS
hammer-translator --jobs-root var/jobs run /path/to/input.pdf \
  --run --environment-profile local_mac \
  --parser-backend pipeline --provider deepseek

# Linux/NVIDIA
hammer-translator --jobs-root var/jobs run /path/to/input.pdf \
  --run --environment-profile linux_gpu \
  --parser-backend pipeline --parser-device cuda --provider deepseek
```

输入必须是 PDF。`--run` 会实际调用 MinerU 和翻译服务；`--simulate` 只测试流程契约，结果不能作为正式译文发布。推荐默认使用 `pipeline` 解析后端；`vlm-engine` 和 `hybrid-engine` 需要满足各自的模型及运行条件，资源消耗可能明显更高。Linux 上可在命令前设置 `CUDA_VISIBLE_DEVICES=GPU编号` 选择 GPU。若只想先解析前两页，可在 `run` 命令末尾加 `--parser-start-page 0 --parser-end-page 1`（页码从 0 开始，包含结束页）；处理完整 PDF 时不要加这两个参数。

命令开始时会显示任务 ID，并实时输出进度。使用该 ID 和同一个任务目录查看或恢复任务：

```bash
hammer-translator --jobs-root var/jobs status JOB_ID
hammer-translator --jobs-root var/jobs events JOB_ID
hammer-translator --jobs-root var/jobs resume JOB_ID --run
```

`resume` 会重试失败任务，不重复已完成的阶段。每次 `run` 都会新建任务。如果后续命令可能在不同工作目录执行，建议为 `--jobs-root` 使用绝对路径。

## 获取和检查结果

任务成功后，译后 PDF 位于 `var/jobs/JOB_ID/output/translated.pdf`（或指定的任务目录下）。同一任务目录还保存输入 PDF 副本、解析及翻译产物、`output/reflow_feedback.json`、`output/validation_report.json`、`job.json` 和 `events.jsonl`。

`completed_with_warnings` 表示已生成 PDF，但仍有内容需要复核。默认情况下，无法安全处理的区域会保留原文，而不是强行塞入固定边界框。V1 没有“人工回复后自动改写 PDF”的闭环；参考文献、独立公式中的英文说明等也可能按规则保留原文。交付前请检查验证报告和 PDF 实际视觉效果；V1 尚未完整还原加粗、斜体等富文本样式。

## 进一步了解

- [运行路径与环境变量](docs/RUNTIME_CONFIGURATION.md)
- [Linux/NVIDIA 部署细节](docs/LINUX_GPU_RUNTIME.md)
- [Pipeline 阶段与产物契约](PIPELINE_V1.md)
- [依赖分层](requirements/README.md)

仓库根目录的旧 Python 脚本仍作为兼容入口保留；新安装建议使用上文的 `hammer-*` 命令。
