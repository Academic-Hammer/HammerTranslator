#!/usr/bin/env python3
"""
将 origin.json 格式转换为 content_list.json 格式

用法:
    python convert_origin_to_content_list.py <input_origin.json> <output_content_list.json>
"""

import json
import sys
import os
from typing import Any, Dict, List


def convert_bbox(origin_bbox: List[float]) -> List[int]:
    """
    坐标转换：简单取整
    """
    return [int(coord) for coord in origin_bbox]


def map_type(origin_type: str) -> str:
    """
    类型映射：heading/paragraph/list 统一映射为 text
    其他类型保持不变
    """
    type_mapping = {
        'heading': 'text',
        'paragraph': 'text',
        'list': 'text',
        'image': 'image',
        'caption': 'caption'
    }
    return type_mapping.get(origin_type, origin_type)


def get_text_level(origin_item: Dict[str, Any]) -> int:
    """
    获取文本级别：heading类型设置为1，其他为None
    """
    if origin_item.get('type') == 'heading':
        return 1
    return None


def convert_origin_to_content_list(origin_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    将 origin.json 格式转换为 content_list.json 格式
    """
    content_list = []

    for logical_para_id, kid in enumerate(origin_data.get('kids', [])):

        # 创建基础转换项
        converted_item = {}

        # 添加递增的 logical_para_id
        converted_item['logical_para_id'] = logical_para_id

        # 基础字段映射
        if 'page number' in kid:
            converted_item['page_idx'] = kid['page number'] - 1  # 转为0-based索引

        if 'id' in kid:
            converted_item['id'] = kid['id']  # 保留原始 id

        if 'content' in kid:
            converted_item['text'] = kid['content']

        if 'bounding box' in kid:
            converted_item['bbox'] = convert_bbox(kid['bounding box'])

        # 类型映射
        origin_type = kid.get('type', 'text')
        converted_item['type'] = map_type(origin_type)

        # 文本级别
        text_level = get_text_level(kid)
        if text_level is not None:
            converted_item['text_level'] = text_level

        # image特殊处理
        if origin_type == 'image':
            if 'source' in kid:
                converted_item['img_path'] = kid['source']
            if 'alt_source' in kid:
                converted_item['alt_source'] = kid['alt_source']

        # caption特殊处理
        if origin_type == 'caption':
            if 'linked content id' in kid:
                converted_item['linked_content_id'] = kid['linked content id']

        # 保留所有其他字段（按原样保留）
        for key, value in kid.items():
            if key not in ['page number', 'id', 'content', 'bounding box', 'type',
                          'source', 'alt_source', 'linked content id']:
                converted_item[key] = value

        content_list.append(converted_item)

    return content_list


def main():
    if len(sys.argv) != 3:
        print("用法: python convert_origin_to_content_list.py <input_origin.json> <output_content_list.json>")
        print("示例: python convert_origin_to_content_list.py input.json output.json")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        print(f"错误: 输入文件不存在: {input_file}")
        sys.exit(1)

    try:
        # 读取origin.json
        print(f"读取输入文件: {input_file}")
        with open(input_file, 'r', encoding='utf-8') as f:
            origin_data = json.load(f)

        # 转换格式
        print("开始转换...")
        content_list = convert_origin_to_content_list(origin_data)

        # 写入content_list.json
        print(f"写入输出文件: {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(content_list, f, ensure_ascii=False, indent=2)

        # 输出统计信息
        print(f"\n转换完成！")
        print(f"输入项数: {len(origin_data.get('kids', []))}")
        print(f"输出项数: {len(content_list)}")

        # 类型统计
        type_count = {}
        for item in content_list:
            item_type = item.get('type', 'unknown')
            type_count[item_type] = type_count.get(item_type, 0) + 1

        print(f"\n输出类型分布:")
        for item_type, count in sorted(type_count.items()):
            print(f"  {item_type}: {count}")

    except json.JSONDecodeError as e:
        print(f"JSON解析错误: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"转换过程中发生错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
