#!/usr/bin/env python3
"""
Complete Translation & Rendering Script
========================================
1. Read content_list.json from folder
2. Translate content
3. Backfill translation to JSON
4. Render translated PDF

Usage:
    python translate_and_backfill.py /path/to/folder
    python translate_and_backfill.py /path/to/folder --pdf /path/to/original.pdf
"""

import json
import os
import sys
import yaml
import copy
import argparse
import traceback
from pathlib import Path
from typing import Dict, List
from datetime import datetime

# Setup paths
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR / "Colayout-pure" / "scripts"))
sys.path.insert(0, str(BASE_DIR / "Colayout-pure" / "tools"))
sys.path.insert(0, str(BASE_DIR))
CONFIG = BASE_DIR / "Colayout-pure" / "config.yaml"

def load_config(config_path: str = CONFIG ) -> Dict:
    """Load API configuration."""
    config_file = Path(config_path)
    if not config_file.exists():
        print(f"❌ Config not found: {config_path}")
        sys.exit(1)

    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # Expand environment variables
    def expand_env(value):
        if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
            return os.environ.get(value[2:-1], value)
        return value

    return expand_env(config)


def process_folder(
    folder_path: Path,
    config: Dict,
    target_lang: str = "zh",
    pdf_path: Path = None,
    output_json: Path = None,
    output_pdf: Path = None
) -> bool:
    """
    Complete process: Translate → Backfill → Render PDF
    """
    try:
        start_time = datetime.now()
        print(f"\n{'='*70}")
        print(f"🚀 Processing: {folder_path.name}")
        print(f"{'='*70}")

        # 1. Find input files
        content_files = list(folder_path.rglob("*content_list.json"))

        if not content_files:
            print(f"❌ content_list.json not found in: {folder_path}")
            return False

        for content_file in content_files:

            print(f"\n📂 Step 1: Loading content")
            print(f"   Reading: {content_file}")

            with open(content_file, 'r', encoding='utf-8') as f:
                original_data = json.load(f)

            print(f"   Found {len(original_data)} items")

            # Auto-detect PDF path if not provided
            if pdf_path is None:
                # Get the directory containing content_list.json
                content_dir = content_file.parent

                # Find *_origin.pdf in the same directory as content_list.json
                origin_pdfs = list(content_dir.glob("*_origin.pdf"))

                if origin_pdfs:
                    pdf_path = origin_pdfs[0]
                    print(f"   Auto-detected PDF: {pdf_path}")
                else:
                    print(f"⚠️  PDF not found, translation only mode")
                    pdf_path = None

            # 2. Setup API clients and agents
            print(f"\n🔧 Step 2: Setting up API clients")

            from tools.api_client import APIClient
            from scripts.PlannerAgent.PlannerAgent import PlannerAgent
            from scripts.TranslationAgent.TranslationAgent import TranslationAgent
            from scripts.ReflowAgent import ReflowAgent

            # Translation client
            trans_api = config['translation']['api']
            translation_client = APIClient(
                api_key=trans_api['api_key'],
                base_url=trans_api['base_url'],
                model=trans_api['model']
            )

            # Reflow client
            reflow_api = config['reflow']['api']
            reflow_client = APIClient(
                api_key=reflow_api['api_key'],
                base_url=reflow_api['base_url'],
                model=reflow_api['model']
            )

            # Planner
            planner = PlannerAgent(target_lang=target_lang)

            # Translation agent
            translation_agent = TranslationAgent(translation_client, target_lang=target_lang, planner=planner)

            print(f"   ✅ Translation model: {trans_api['model']}")
            print(f"   ✅ Reflow model: {reflow_api['model']}")

            # 3. Translate content
            print(f"\n🌍 Step 3: Translating content")

            work_data = copy.deepcopy(original_data)
            translated_data = translation_agent.process_documents(work_data, str(pdf_path) if pdf_path else None)

            # 4. Save translated JSON
            if output_json is None:
                output_json = folder_path / "content_list_translated.json"

            output_json.parent.mkdir(parents=True, exist_ok=True)
            with open(output_json, 'w', encoding='utf-8') as f:
                json.dump(translated_data, f, ensure_ascii=False, indent=2)

            print(f"\n💾 Step 4: Saved translated JSON")
            print(f"   Output: {output_json}")

            # 5. Render PDF (if PDF path provided)
            if pdf_path:
                print(f"\n📄 Step 5: Rendering translated PDF")

                if output_pdf is None:
                    output_pdf = folder_path / "translated.pdf"

                enable_rich_text_style = config.get('reflow', {}).get('enable_rich_text_style', False)

                reflow_agent = ReflowAgent(
                    reflow_client,
                    target_lang=target_lang,
                    planner=planner,
                    enable_rich_text_style=enable_rich_text_style
                )

                stats = reflow_agent.run_reflow_task_with_data(
                    translated_data,
                    str(pdf_path),
                    str(output_pdf),
                    enable_rewrite=True,
                    max_rounds=4
                )

                print(f"\n✅ Step 6: Rendered PDF")
                print(f"   Output: {output_pdf}")
                print(f"   Stats: {stats}")

                # Save planner log
                planner_json = output_pdf.parent / f"{output_pdf.stem}_planner.json"
                with open(planner_json, 'w', encoding='utf-8') as f:
                    json.dump(planner.export_summary(), f, ensure_ascii=False, indent=2)
                print(f"   Planner log: {planner_json}")

            elapsed = (datetime.now() - start_time).total_seconds()
            print(f"\n{'='*70}")
            print(f"✅ Completed in {elapsed:.0f}s")
            print(f"{'='*70}")

            return True

    except Exception as e:
        print(f"\n❌ Error: {e}")
        traceback.print_exc()
        return False


def legacy_main():
    parser = argparse.ArgumentParser(
        description="Translate content_list.json and render PDF",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python translate_and_backfill.py /path/to/folder
    python translate_and_backfill.py /path/to/folder --pdf original.pdf
    python translate_and_backfill.py /path/to/folder --output-json translated.json --output-pdf result.pdf
        """
    )
    parser.add_argument("input", help="Folder containing content_list.json")
    parser.add_argument("--pdf", help="Original PDF path (auto-detect if not specified)")
    parser.add_argument("--output-json", help="Output JSON path", default=None)
    parser.add_argument("--output-pdf", help="Output PDF path", default=None)
    parser.add_argument("--config", help="Config file path", default=CONFIG)
    parser.add_argument("--target-lang", help="Target language", default="zh")
    args = parser.parse_args()

    print("=" * 70)
    print("🌍 Complete Translation & Rendering Pipeline")
    print("=" * 70)

    # Load config
    config = load_config(args.config)

    # Process folder
    input_folder = Path(args.input)
    pdf_path = Path(args.pdf) if args.pdf else None
    output_json = Path(args.output_json) if args.output_json else None
    output_pdf = Path(args.output_pdf) if args.output_pdf else None

    success = process_folder(
        input_folder,
        config,
        args.target_lang,
        pdf_path,
        output_json,
        output_pdf
    )

    if success:
        print("\n✅ All tasks completed successfully!")
        return 0
    else:
        print("\n❌ Processing failed!")
        return 1


def main():
    """Stop the retired entry point before it can call a remote API."""

    print(
        "This legacy entry point is retired. Use translate_pdfs.py with "
        "--check or --dry-run. No API request was made.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())
