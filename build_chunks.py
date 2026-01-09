#!/usr/bin/env python3
"""
Build Chunks Script

将清洗后的 JSONL 文件转换为适合 RAG 的语义块。

工作流程：
1. 加载清洗后的 JSONL 文件
2. 按卷、页码、行号排序
3. 按 (work_title, chapter_title) 分组（硬边界）
4. 对每个章节组应用 KantTextSplitter
5. 保存到 data/chunks/
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from itertools import groupby
from collections import defaultdict

from loguru import logger
from tqdm import tqdm

# 处理相对导入
try:
    from src.database.chunker import KantTextSplitter, TextChunk
except ImportError:
    # 添加项目根目录到路径
    project_root = Path(__file__).parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from src.database.chunker import KantTextSplitter, TextChunk


def load_jsonl(file_path: Path) -> List[Dict[str, Any]]:
    """加载 JSONL 文件
    
    Args:
        file_path: JSONL 文件路径
        
    Returns:
        文档列表
    """
    docs = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    doc = json.loads(line)
                    docs.append(doc)
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse line {line_num} in {file_path}: {e}")
        logger.info(f"Loaded {len(docs)} documents from {file_path}")
    except Exception as e:
        logger.error(f"Failed to load {file_path}: {e}")
    
    return docs


def sort_documents(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """对文档进行排序
    
    排序键：(volume, page_start, line_start)
    
    Args:
        docs: 文档列表
        
    Returns:
        排序后的文档列表
    """
    def sort_key(doc: Dict[str, Any]) -> Tuple[str, int, int]:
        metadata = doc.get('metadata', {})
        volume = metadata.get('volume', 'unknown')
        page_start = metadata.get('page_start') or metadata.get('page', 0)
        
        # 解析 line_range (格式: "1-15" 或 "1")
        line_range = metadata.get('line_range', '0')
        if '-' in line_range:
            line_start = int(line_range.split('-')[0])
        else:
            try:
                line_start = int(line_range)
            except ValueError:
                line_start = 0
        
        return (volume, page_start, line_start)
    
    return sorted(docs, key=sort_key)


def group_by_chapter(docs: List[Dict[str, Any]]) -> List[Tuple[Tuple[str, Optional[str]], List[Dict[str, Any]]]]:
    """按章节分组文档
    
    Args:
        docs: 已排序的文档列表
        
    Returns:
        [(group_key, docs), ...] 列表，其中 group_key = (work_title, chapter_title)
    """
    def group_key(doc: Dict[str, Any]) -> Tuple[str, Optional[str]]:
        metadata = doc.get('metadata', {})
        work_title = metadata.get('work_title', 'Unknown Work')
        chapter_title = metadata.get('chapter_title')
        return (work_title, chapter_title)
    
    groups = []
    for key, group_docs in groupby(docs, key=group_key):
        groups.append((key, list(group_docs)))
    
    logger.info(f"Grouped {len(docs)} documents into {len(groups)} chapter groups")
    return groups


def process_volume(
    volume: str,
    input_file: Path,
    output_dir: Path,
    splitter: KantTextSplitter
) -> Dict[str, int]:
    """处理单个卷
    
    Args:
        volume: 卷号（如 "aa01"）
        input_file: 输入 JSONL 文件路径
        output_dir: 输出目录
        splitter: 文本切分器
        
    Returns:
        统计信息字典
    """
    logger.info(f"Processing volume: {volume}")
    
    # 加载文档
    docs = load_jsonl(input_file)
    if not docs:
        logger.warning(f"No documents found in {input_file}")
        return {"chunks": 0, "groups": 0}
    
    # 排序
    sorted_docs = sort_documents(docs)
    
    # 分组
    chapter_groups = group_by_chapter(sorted_docs)
    
    # 处理每个章节组
    all_chunks = []
    for group_key, group_docs in tqdm(chapter_groups, desc=f"Chunking {volume}"):
        try:
            chunks = splitter.process_chapter_group(group_docs, group_key)
            all_chunks.extend(chunks)
        except Exception as e:
            logger.error(f"Failed to process chapter group {group_key} in {volume}: {e}")
            continue
    
    # 保存结果
    output_file = output_dir / f"{volume}.jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    chunk_count = 0
    with open(output_file, 'w', encoding='utf-8') as f:
        for chunk in all_chunks:
            output_obj = {
                "id": chunk.id,
                "content": chunk.content,
                "metadata": chunk.metadata
            }
            f.write(json.dumps(output_obj, ensure_ascii=False) + '\n')
            chunk_count += 1
    
    logger.info(f"Saved {chunk_count} chunks to {output_file}")
    
    return {
        "chunks": chunk_count,
        "groups": len(chapter_groups)
    }


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Build semantic chunks from cleaned JSONL files'
    )
    parser.add_argument(
        '--input-dir',
        type=str,
        default='data/cleaned',
        help='Input directory containing cleaned JSONL files (default: data/cleaned)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/chunks',
        help='Output directory for chunk JSONL files (default: data/chunks)'
    )
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=1000,
        help='Chunk size in tokens (default: 1000)'
    )
    parser.add_argument(
        '--chunk-overlap',
        type=int,
        default=200,
        help='Chunk overlap in tokens (default: 200)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='text-embedding-3-small',
        help='Tokenizer model name (default: text-embedding-3-small)'
    )
    parser.add_argument(
        '--volume',
        type=str,
        default=None,
        help='Process only a specific volume (e.g., aa01). If not specified, process all volumes.'
    )
    
    args = parser.parse_args()
    
    # 设置路径
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    if not input_dir.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        return
    
    # 初始化切分器
    splitter = KantTextSplitter(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        model_name=args.model
    )
    
    # 查找所有 JSONL 文件
    if args.volume:
        jsonl_files = [input_dir / f"{args.volume}.jsonl"]
        jsonl_files = [f for f in jsonl_files if f.exists()]
    else:
        jsonl_files = sorted(input_dir.glob("*.jsonl"))
    
    if not jsonl_files:
        logger.warning(f"No JSONL files found in {input_dir}")
        return
    
    logger.info(f"Found {len(jsonl_files)} JSONL files to process")
    
    # 处理每个文件
    total_stats = defaultdict(int)
    for jsonl_file in jsonl_files:
        volume = jsonl_file.stem  # 例如 "aa01"
        try:
            stats = process_volume(volume, jsonl_file, output_dir, splitter)
            total_stats["chunks"] += stats["chunks"]
            total_stats["groups"] += stats["groups"]
            total_stats["volumes"] += 1
        except Exception as e:
            logger.error(f"Failed to process {jsonl_file}: {e}")
            continue
    
    # 输出统计信息
    logger.info("=" * 60)
    logger.info("Processing complete!")
    logger.info(f"  Volumes processed: {total_stats['volumes']}")
    logger.info(f"  Chapter groups: {total_stats['groups']}")
    logger.info(f"  Total chunks: {total_stats['chunks']}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
