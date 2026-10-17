#!/usr/bin/env python3
"""独立调用 DeepSeek，读取脚本同目录 .env 中的 TRANSLATION_API_KEY。

安装依赖：python3 -m pip install openai python-dotenv
使用示例：python3 call_deepseek.py "你好，请介绍一下自己"
指定模型：python3 call_deepseek.py --model deepseek-v4-pro "解释一下机器学习"
"""

import argparse
import sys
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt", nargs="?", help="发送给模型的问题；不填则交互输入")
    parser.add_argument("--model", default="deepseek-flash", help="模型名称")
    parser.add_argument("--thinking", action="store_true", help="开启深度思考（默认关闭以减少等待）")
    parser.add_argument("--timeout", type=float, default=60.0, help="网络读取超时秒数（默认 60）")
    parser.add_argument(
        "--env-file", type=Path, default=Path(__file__).resolve().with_name(".env"),
        help="密钥文件路径（默认为脚本同目录的 .env）",
    )
    parser.add_argument("--system", default="You are a helpful assistant.", help="系统提示词")
    args = parser.parse_args()
    if not 0 < args.timeout < float("inf"):
        parser.error("--timeout 必须是有限正数")

    try:
        from dotenv import dotenv_values
        from httpx import Timeout
        from openai import APIConnectionError, APIStatusError, OpenAI
    except ImportError:
        print("缺少依赖，请运行：python3 -m pip install openai python-dotenv", file=sys.stderr)
        return 1

    env_path = args.env_file.expanduser()
    if not env_path.is_file():
        print(f"找不到密钥文件：{env_path}", file=sys.stderr)
        return 1
    try:
        config = dotenv_values(env_path, encoding="utf-8-sig", interpolate=False)
    except (OSError, UnicodeError):
        print("无法读取密钥文件，请检查权限与 UTF-8 编码。", file=sys.stderr)
        return 1
    api_key = (config.get("TRANSLATION_API_KEY") or config.get("DEEPSEEK_API_KEY") or "").strip()
    if not api_key:
        print("请在 .env 中设置 TRANSLATION_API_KEY 或 DEEPSEEK_API_KEY。", file=sys.stderr)
        return 1

    prompt = args.prompt if args.prompt is not None else input("请输入问题：")
    if not prompt.strip():
        parser.error("问题不能为空")

    print(f"正在请求 DeepSeek（{args.model}），等待回复…", file=sys.stderr, flush=True)
    started = time.monotonic()
    received_text = False
    thinking_reported = False
    try:
        with OpenAI(api_key=api_key, base_url="https://api.deepseek.com",
                    timeout=Timeout(args.timeout, connect=10.0), max_retries=0) as client:
            response = client.chat.completions.create(
                model=args.model,
                messages=[
                    {"role": "system", "content": args.system},
                    {"role": "user", "content": prompt},
                ],
                stream=True,
                extra_body={"thinking": {"type": "enabled" if args.thinking else "disabled"}},
            )
            with response:
                for chunk in response:
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta
                    if getattr(delta, "reasoning_content", None) and not thinking_reported:
                        print("模型正在思考，等待正文…", file=sys.stderr, flush=True)
                        thinking_reported = True
                    if delta.content:
                        print(delta.content, end="", flush=True)
                        received_text = True
            if received_text:
                print(flush=True)
    except APIConnectionError:
        print("连接 DeepSeek 失败或请求超时，请检查网络后重试。", file=sys.stderr)
        return 1
    except APIStatusError as exc:
        # 不打印原始异常或响应，避免意外暴露凭据。
        print(f"DeepSeek 请求失败（HTTP {exc.status_code}），请检查密钥、余额及模型名称。", file=sys.stderr)
        return 1

    if not received_text:
        print("模型未返回文本，请重试。", file=sys.stderr)
        return 1
    print(f"完成，用时 {time.monotonic() - started:.1f} 秒。", file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyboardInterrupt, EOFError):
        print("\n已取消。", file=sys.stderr)
        raise SystemExit(130)
