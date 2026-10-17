# MinerU 模型手动下载指南

## 推荐方案：使用git clone下载完整模型仓库

### 方案一：使用Git直接克隆（推荐）

```bash
# 创建模型存储目录
mkdir -p ~/mineru_models

# 下载Pipeline模型
cd ~/mineru_models
# HuggingFace镜像
git clone https://huggingface.co/opendatalab/PDF-Extract-Kit-1.0 pipeline_models

# 如果HuggingFace下载慢，可以使用ModelScope
# git clone https://www.modelscope.cn/OpenDataLab/PDF-Extract-Kit-1.0 pipeline_models

# 下载VLM模型（可选，如果需要高级功能）
git clone https://huggingface.co/opendatalab/MinerU2.5-Pro-2605-1.2B vlm_models
# 或者
# git clone https://www.modelscope.cn/OpenDataLab/MinerU2.5-Pro-2605-1.2B vlm_models
```

### 方案二：使用ModelScope CLI

```bash
# 安装ModelScope CLI
pip install modelscope

# 下载Pipeline模型
python -c "from modelscope import snapshot_download; snapshot_download('OpenDataLab/PDF-Extract-Kit-1.0', cache_dir='~/mineru_models/pipeline_models')"

# 下载VLM模型
python -c "from modelscope import snapshot_download; snapshot_download('OpenDataLab/MinerU2.5-Pro-2605-1.2B', cache_dir='~/mineru_models/vlm_models')"
```

### 方案三：使用HuggingFace CLI

```bash
# 安装HuggingFace CLI
pip install huggingface_hub

# 下载Pipeline模型
huggingface-cli download opendatalab/PDF-Extract-Kit-1.0 --local-dir ~/mineru_models/pipeline_models

# 下载VLM模型  
huggingface-cli download opendatalab/MinerU2.5-Pro-2605-1.2B --local-dir ~/mineru_models/vlm_models
```

## 配置mineru.json

下载完成后，更新你的mineru.json配置文件：

```json
{
    "models-dir": {
        "pipeline": "/home/你的用户名/mineru_models/pipeline_models",
        "vlm": "/home/你的用户名/mineru_models/vlm_models"
    },
    "model-source": "local",
    "config_version": "1.3.2"
}
```

### 更新配置的具体步骤：

```bash
# 编辑配置文件
nano ~/mineru.json

# 或者直接替换（修改实际路径）
sed -i 's|"pipeline": ""|"pipeline": "/home/'$USER'/mineru_models/pipeline_models"|' ~/mineru.json
sed -i 's|"vlm": ""|"vlm": "/home/'$USER'/mineru_models/vlm_models"|' ~/mineru.json
sed -i 's|"model-source": "modelscope"|"model-source": "local"|' ~/mineru.json
```

## 验证安装

```bash
# 测试模型是否正确配置
python -c "from mineru.utils.models_download_utils import get_local_models_dir; import json; print(json.dumps(get_local_models_dir(), indent=2))"

# 运行一个简单的PDF解析测试
mineru-pdf test.pdf
```

## 如果下载仍然失败

可以尝试以下加速方案：

### 1. 使用国内镜像
```bash
# ModelScope通常国内访问更快
export MINERU_MODEL_SOURCE=modelscope

# 设置HuggingFace镜像（如果使用HuggingFace）
export HF_ENDPOINT=https://hf-mirror.com
```

### 2. 分模型手动下载
如果完整仓库下载失败，可以单独下载每个子模型：

```
Pipeline模型包含的子模型：
1. PP-DocLayoutV2 (布局检测)
2. unimernet_hf_small_2503 (数学公式识别)  
3. pp_formulanet_plus_m (数学公式识别)
4. paddleocr_torch (OCR)
5. slanet-plus.onnx (表格识别)
6. unet.onnx (表格结构识别)
7. paddle_table_cls (表格分类)
```

## 预计存储空间
- Pipeline模型：约2-3GB
- VLM模型：约5-8GB

## 常见问题

**Q: 下载过程中断怎么办？**
A: 使用git clone或者huggingface-cli download都支持断点续传

**Q: 如何选择下载源？**
A: 国内用户推荐ModelScope，国外用户推荐HuggingFace

**Q: 只需要Pipeline模型吗？**
A: 是的，基本功能只需要Pipeline模型，VLM模型用于更高级的AI处理功能

**Q: 下载后可以移动模型文件夹吗？**
A: 可以，移动后需要更新mineru.json中的路径
