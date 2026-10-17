#!/usr/bin/env python3
"""
批量解析 PDF 脚本 - 直接使用 MinerU Python API
模型只加载一次，然后批量处理所有 PDF 文件，保持目录结构

GPU配置说明:
- 当前设置: GPU 1
- 可用空闲GPU: 0, 1, 2, 5, 6, 7
- 修改方法: 更改下面的 os.environ['CUDA_VISIBLE_DEVICES'] = 'X'
"""

import os
import sys
from pathlib import Path

# ========== GPU配置 ==========
# 设置使用的GPU序号（推荐: 0, 1, 2, 5, 6, 7 都是空闲GPU）
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# ========== 公式模型配置 ==========
# 使用完整的 pp_formulanet_plus_m 模型（支持中文公式）
os.environ['MINERU_FORMULA_CH_SUPPORT'] = 'true'

from loguru import logger

# 添加 MinerU 到路径
mineru_path = Path("/data/dingjingrao/pdftranslator/MinerU")
if str(mineru_path) not in sys.path:
    sys.path.insert(0, str(mineru_path))

# 配置
ROOT_DIR = Path(__file__).parent.resolve()
SOURCE_DIR = ROOT_DIR / "demo"
OUTPUT_DIR = ROOT_DIR / "demo-output-vlm"
BACKEND = "pipeline"  # 使用pipeline后端，稳定可靠
METHOD = "auto"
LANG = "ch"
FORMULA_ENABLE = True  # 启用公式识别（使用 pp_formulanet_plus_m 模型）
TABLE_ENABLE = True


def collect_pdfs(source_dir: Path) -> list[tuple[Path, Path]]:
    """收集所有 PDF 文件及其相对路径"""
    pdf_files = []
    for pdf_path in source_dir.rglob("*.pdf"):
        relative_path = pdf_path.relative_to(source_dir)
        pdf_files.append((pdf_path, relative_path.parent))
    return sorted(pdf_files)


def process_pdf_direct(pdf_path: Path, relative_parent: Path) -> bool:
    """直接调用 MinerU Python API 处理 PDF"""
    try:
        from mineru.cli.common import do_parse

        # 创建输出目录，保持相对路径结构
        # 注意：do_parse会在output_dir下创建 pdf_file_name/parse_method/ 的结构
        output_subdir = OUTPUT_DIR / relative_parent
        output_subdir.mkdir(parents=True, exist_ok=True)

        # 显示相对路径，方便用户追踪
        rel_path_str = str(relative_parent / pdf_path.name)
        logger.info(f"📄 开始处理: {rel_path_str}")

        # 读取 PDF 文件为字节数据
        with open(pdf_path, 'rb') as f:
            pdf_bytes = f.read()

        # 直接调用解析函数
        do_parse(
            output_dir=str(output_subdir),
            pdf_file_names=[pdf_path.stem],  # 使用去掉后缀的文件名
            pdf_bytes_list=[pdf_bytes],
            p_lang_list=[LANG],
            backend=BACKEND,
            parse_method=METHOD,
            formula_enable=FORMULA_ENABLE,
            table_enable=TABLE_ENABLE,
        )

        logger.info(f"✅ 完成处理: {rel_path_str}")
        return True

    except Exception as e:
        logger.error(f"❌ 处理失败: {relative_parent / pdf_path.name}, 错误: {e}")
        import traceback
        logger.error(traceback.format_exc())  # Changed to error to show full traceback
        return False


def batch_process_pdfs():
    """批量处理所有 PDF 文件"""
    logger.info("=" * 60)
    logger.info("MinerU 批量 PDF 解析工具（直接 Python API）")
    logger.info(f"🎮 GPU设置: CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
    logger.info(f"源目录: {SOURCE_DIR}")
    logger.info(f"输出目录: {OUTPUT_DIR}")
    logger.info(f"后端模式: {BACKEND}")
    logger.info("=" * 60)

    # 1. 收集所有 PDF 文件
    logger.info("📂 正在扫描 PDF 文件...")
    pdf_list = collect_pdfs(SOURCE_DIR)

    if not pdf_list:
        logger.warning("❌ 未找到任何 PDF 文件")
        return

    logger.info(f"✅ 找到 {len(pdf_list)} 个 PDF 文件")

    # 统计每个子目录的文件数
    dir_stats = {}
    for _, relative_parent in pdf_list:
        dir_key = str(relative_parent)
        dir_stats[dir_key] = dir_stats.get(dir_key, 0) + 1

    for dir_path, count in sorted(dir_stats.items()):
        logger.info(f"  - {dir_path}: {count} 个文件")

    # 2. 批量处理 PDF
    logger.info("\n📚 开始批量处理 PDF...")
    logger.info("=" * 60)

    success_count = 0
    fail_count = 0

    for pdf_path, relative_parent in pdf_list:
        success = process_pdf_direct(pdf_path, relative_parent)
        if success:
            success_count += 1
        else:
            fail_count += 1

    # 3. 总结
    logger.info("\n" + "=" * 60)
    logger.info("📊 处理完成统计:")
    logger.info(f"  ✅ 成功: {success_count} 个")
    logger.info(f"  ❌ 失败: {fail_count} 个")
    logger.info(f"  📁 输出目录: {OUTPUT_DIR}")
    logger.info("=" * 60)


def main():
    """主函数"""
    # 配置日志
    logger.remove()
    logger.add(
        sys.stderr,
        level="INFO",
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>"
    )

    # 显示GPU配置
    gpu_id = os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')
    logger.info(f"🎮 GPU配置: CUDA_VISIBLE_DEVICES={gpu_id}")

    # 验证GPU是否可用
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used,memory.total', '--format=csv,noheader'],
                              capture_output=True, text=True)
        if result.returncode == 0:
            logger.info("🖥️  当前GPU状态:")
            for line in result.stdout.strip().split('\n'):
                if line:
                    parts = [p.strip() for p in line.split(',')]
                    if parts[0] == gpu_id:
                        logger.info(f"  GPU {parts[0]}: {parts[1]} / {parts[2]} - 🎯 将使用此GPU")
                    else:
                        logger.info(f"  GPU {parts[0]}: {parts[1]} / {parts[2]}")
    except:
        logger.debug("无法获取GPU状态信息")

    # 检查源目录
    if not SOURCE_DIR.exists():
        logger.error(f"❌ 源目录不存在: {SOURCE_DIR}")
        sys.exit(1)

    # 创建输出目录
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 运行批量处理
    try:
        batch_process_pdfs()
    except KeyboardInterrupt:
        logger.warning("\n⚠️ 用户中断")
        sys.exit(130)
    except Exception as e:
        logger.error(f"\n❌ 程序异常: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
