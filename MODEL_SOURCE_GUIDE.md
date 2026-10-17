# MinerU 模型源配置说明

## 🔧 配置优先级（从高到低）

MinerU 按以下优先级确定使用哪个模型源：

1. **环境变量** `MINERU_MODEL_SOURCE` （最高优先级）
2. **配置文件** `~/mineru.json` 中的 `model-source` 字段
3. **自动检测** `auto` （默认，按顺序尝试 huggingface → modelscope）

## 🌐 支持的模型源

- **`local`** - 使用本地已下载的模型（推荐，避免网络问题）
- **`huggingface`** - 从 HuggingFace 下载模型
- **`modelscope`** - 从 ModelScope 下载模型（国内推荐）
- **`auto`** - 自动检测（仅用于命令行参数）

## 🚀 使用方法

### 方法1: 环境变量（最高优先级）

```bash
# 设置环境变量
export MINERU_MODEL_SOURCE=modelscope

# 运行脚本（会使用环境变量中的设置）
python batch_parse_pdfs.py

# 临时设置（单次使用）
MINERU_MODEL_SOURCE=modelscope python batch_parse_pdfs.py
```

### 方法2: 配置文件

编辑 `~/mineru.json`:

```json
{
    "model-source": "modelscope",
    "models-dir": {
        "pipeline": "/path/to/models",
        "vlm": ""
    }
}
```

### 方法3: 使用本地模型（推荐）

```bash
# 设置为本地模式
export MINERU_MODEL_SOURCE=local

# 或在配置文件中设置
# "model-source": "local"
```

## 🇨🇳 国内网络环境推荐配置

### 方案A: 使用本地模型（最稳定）
```bash
export MINERU_MODEL_SOURCE=local
# 配置文件需要设置好本地模型路径
```

### 方案B: 使用 ModelScope
```bash
export MINERU_MODEL_SOURCE=modelscope
# 或者在配置文件中设置
```

### 方案C: 使用 HuggingFace 镜像
```bash
export MINERU_MODEL_SOURCE=huggingface
export HF_ENDPOINT=https://hf-mirror.com
```

## ⚠️ 注意事项

1. **环境变量优先级最高**: 如果设置了环境变量，配置文件的设置会被忽略
2. **本地模型需要配置路径**: 使用 `local` 时，必须在配置文件中设置正确的 `models-dir`
3. **网络问题**: 国内建议使用 `modelscope` 或本地模型，避免连接超时

## 🔍 诊断和测试

### 检查当前配置
```bash
# 检查环境变量
echo $MINERU_MODEL_SOURCE

# 检查配置文件
cat ~/mineru.json | grep model-source

# 测试模型源
python -c "from mineru.utils.models_download_utils import resolve_model_source; print('当前模型源:', resolve_model_source())"
```

### 重置环境变量
```bash
# 取消环境变量设置（恢复使用配置文件）
unset MINERU_MODEL_SOURCE

# 设置为新的值
export MINERU_MODEL_SOURCE=local
```

## 📝 快速参考

| 配置方式 | 命令 | 优先级 | 适用场景 |
|---------|------|--------|----------|
| 环境变量 | `export MINERU_MODEL_SOURCE=local` | 最高 | 临时切换、测试 |
| 配置文件 | `"model-source": "local"` | 中 | 永久配置 |
| 自动检测 | 默认 | 最低 | 初次使用 |

## 🛠️ 故障排除

**Q: 设置了配置文件但还是从网络下载？**
A: 检查是否设置了 `MINERU_MODEL_SOURCE` 环境变量，它的优先级更高

**Q: 如何临时测试不同的模型源？**
A: 使用环境变量单次设置：
```bash
MINERU_MODEL_SOURCE=modelscope python your_script.py
```

**Q: 本地模型配置正确但还是报错？**
A: 检查 `~/mineru.json` 中的 `models-dir` 路径是否正确，以及 `model-source` 是否设置为 `local`