#!/usr/bin/env python3
"""
Kantian Semantic Chunking Module

Two-stage pipeline for transforming page-based cleaned text into semantically
complete, token-sized chunks suitable for RAG, while preserving precise
"Akademie-Ausgabe" (AA) citations.

Phase 1: Logical Unit Reconstruction (Stitching)
- Group by work_title + chapter_title
- Concatenate content with hyphenation handling
- Build coordinate mapping system

Phase 2: Sliding Window Chunking (Cutting)
- Token-aware recursive splitting
- Map chunks back to original page/line metadata
- Enrich metadata with citation information
"""

import bisect
import hashlib
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict

from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class SourceBlock:
    """原始数据块：从 JSONL 加载的单个记录"""
    id: str
    content: str
    metadata: Dict[str, Any]
    start_offset: int  # 在重构文本中的起始字符位置
    end_offset: int    # 在重构文本中的结束字符位置


@dataclass
class TextChunk:
    """切分后的文本块"""
    id: str
    content: str
    metadata: Dict[str, Any]


# ============================================================================
# Coordinate Mapping System
# ============================================================================

class CoordinateMapper:
    """坐标映射器：将字符偏移量映射回原始页码/行号"""
    
    def __init__(self, source_blocks: List[SourceBlock]):
        """初始化映射器
        
        Args:
            source_blocks: 源数据块列表，按偏移量排序
        """
        self.source_blocks = sorted(source_blocks, key=lambda b: b.start_offset)
        self.start_positions = [b.start_offset for b in self.source_blocks]
    
    def find_intersecting_blocks(
        self, 
        chunk_start: int, 
        chunk_end: int
    ) -> List[SourceBlock]:
        """查找与给定区间相交的所有源块
        
        Args:
            chunk_start: 块的起始字符位置
            chunk_end: 块的结束字符位置
            
        Returns:
            相交的源块列表
        """
        # 使用 bisect 找到第一个可能相交的块
        # 查找第一个 start_offset >= chunk_start 的位置
        left_idx = bisect.bisect_left(self.start_positions, chunk_start)
        
        # 如果 left_idx > 0，前一个块可能也相交
        if left_idx > 0:
            left_idx -= 1
        
        intersecting = []
        for i in range(left_idx, len(self.source_blocks)):
            block = self.source_blocks[i]
            # 检查是否相交：block 的区间与 [chunk_start, chunk_end) 有重叠
            if block.start_offset < chunk_end and block.end_offset > chunk_start:
                intersecting.append(block)
            # 如果 block 的起始位置已经超过 chunk_end，后续块也不会相交
            elif block.start_offset >= chunk_end:
                break
        
        return intersecting
    
    def aggregate_metadata(
        self, 
        intersecting_blocks: List[SourceBlock],
        chunk_start: int,
        chunk_end: int
    ) -> Dict[str, Any]:
        """聚合相交块的元数据
        
        Args:
            intersecting_blocks: 相交的源块列表
            chunk_start: 块的起始字符位置
            chunk_end: 块的结束字符位置
            
        Returns:
            聚合后的元数据字典
        """
        if not intersecting_blocks:
            return {}
        
        # 收集所有页码
        pages = set()
        page_char_counts: Dict[int, int] = defaultdict(int)
        
        # 计算每个页面贡献的字符数（用于确定 primary_page）
        for block in intersecting_blocks:
            page = block.metadata.get('page_start') or block.metadata.get('page', 1)
            pages.add(page)
            
            # 计算该块与 chunk 的重叠长度
            overlap_start = max(chunk_start, block.start_offset)
            overlap_end = min(chunk_end, block.end_offset)
            overlap_length = max(0, overlap_end - overlap_start)
            page_char_counts[page] += overlap_length
        
        # 确定主要页码（贡献字符数最多的页面）
        primary_page = max(page_char_counts.items(), key=lambda x: x[1])[0] if page_char_counts else None
        
        # 生成页码范围字符串
        sorted_pages = sorted(pages)
        if len(sorted_pages) == 1:
            pages_str = str(sorted_pages[0])
        else:
            pages_str = f"{sorted_pages[0]}-{sorted_pages[-1]}"
        
        # 使用第一个和最后一个块的元数据作为基础
        first_block = intersecting_blocks[0]
        last_block = intersecting_blocks[-1]
        base_metadata = first_block.metadata.copy()
        
        # 合并元数据
        aggregated = {
            **base_metadata,
            "pages": pages_str,
            "primary_page": primary_page,
            "page_start": sorted_pages[0],
            "page_end": sorted_pages[-1],
            "source_id_start": first_block.id,
            "source_id_end": last_block.id,
        }
        
        return aggregated


# ============================================================================
# Hyphenation Handler
# ============================================================================

class HyphenationHandler:
    """连字符处理器：处理 18 世纪德语排版中的行尾连字符"""
    
    @staticmethod
    def merge_blocks(prev_content: str, next_content: str) -> str:
        """合并两个文本块，处理连字符
        
        Args:
            prev_content: 前一个块的内容
            next_content: 下一个块的内容
            
        Returns:
            合并后的文本
        """
        prev_stripped = prev_content.rstrip()
        next_stripped = next_content.lstrip()
        
        # 检查是否以连字符结尾，且下一行以小写字母开头
        if prev_stripped.endswith('-') and next_stripped and next_stripped[0].islower():
            # 软连字符：删除连字符，无缝拼接
            return prev_stripped.rstrip('-') + next_stripped
        else:
            # 其他情况：用空格连接
            return prev_stripped + " " + next_stripped


# ============================================================================
# Main Chunker Class
# ============================================================================

class KantTextSplitter:
    """康德文本切分器：使用 Token 感知的递归切分，保留精确的学术引用"""
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        model_name: str = "text-embedding-3-small"
    ):
        """初始化切分器
        
        Args:
            chunk_size: 块大小（Token 数）
            chunk_overlap: 重叠大小（Token 数）
            model_name: 用于 Token 计数的模型名称（或编码名称，如 "cl100k_base"）
        """
        # 康德特定的分隔符优先级：分号在德语长句中很重要
        separators = ["\n\n", "\n", "; ", ". ", ", ", " "]
        
        # 使用 Token 感知的切分器
        try:
            self.splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                model_name=model_name,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=separators
            )
        except Exception as e:
            logger.warning(f"Failed to initialize tiktoken encoder with {model_name}, falling back to cl100k_base: {e}")
            # 回退到 cl100k_base 编码
            self.splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                model_name="cl100k_base",
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=separators
            )
        
        self.hyphenation_handler = HyphenationHandler()
        self.chunk_overlap = chunk_overlap  # 保存重叠值以便后续使用
        logger.info(f"Initialized KantTextSplitter: chunk_size={chunk_size}, overlap={chunk_overlap}, model={model_name}")
    
    def process_chapter_group(
        self,
        chapter_docs: List[Dict[str, Any]],
        group_key: Tuple[str, Optional[str]]
    ) -> List[TextChunk]:
        """处理单个章节组的数据
        
        Args:
            chapter_docs: 该章节的所有文档（已排序）
            group_key: (work_title, chapter_title) 元组
            
        Returns:
            切分后的文本块列表
        """
        if not chapter_docs:
            return []
        
        work_title, chapter_title = group_key
        logger.debug(f"Processing chapter group: work={work_title[:50]}..., chapter={chapter_title}")
        
        # Phase 1: 重构（Stitching）
        full_text, source_blocks = self._reconstruct_text(chapter_docs)
        
        if not full_text.strip():
            logger.warning(f"Empty text for chapter group: {group_key}")
            return []
        
        # Phase 2: 切分（Cutting）
        raw_chunks = self.splitter.split_text(full_text)
        
        # Phase 3: 元数据映射
        mapper = CoordinateMapper(source_blocks)
        final_chunks = []
        
        current_pos = 0
        for i, chunk_text in enumerate(raw_chunks):
            # 在 full_text 中找到 chunk_text 的精确位置
            # 处理可能的空白字符差异
            abs_start = full_text.find(chunk_text, current_pos)
            if abs_start == -1:
                # 如果找不到精确匹配，尝试模糊匹配（去除空白字符）
                chunk_normalized = ' '.join(chunk_text.split())
                full_normalized = ' '.join(full_text[current_pos:].split())
                rel_pos = full_normalized.find(chunk_normalized)
                if rel_pos != -1:
                    # 近似计算：基于字符比例
                    abs_start = current_pos + int(rel_pos * len(full_text[current_pos:]) / len(full_normalized))
                else:
                    logger.warning(f"Could not locate chunk {i} in full_text, skipping")
                    continue
            
            abs_end = abs_start + len(chunk_text)
            
            # 查询相交的源块
            intersecting_blocks = mapper.find_intersecting_blocks(abs_start, abs_end)
            
            if not intersecting_blocks:
                logger.warning(f"No intersecting blocks found for chunk {i}, skipping")
                continue
            
            # 聚合元数据
            aggregated_metadata = mapper.aggregate_metadata(intersecting_blocks, abs_start, abs_end)
            
            # 生成块 ID
            chunk_id = self._generate_chunk_id(aggregated_metadata, chunk_text, i)
            
            final_chunks.append(TextChunk(
                id=chunk_id,
                content=chunk_text,
                metadata=aggregated_metadata
            ))
            
            # 更新搜索位置（考虑重叠）
            current_pos = abs_start + max(1, len(chunk_text) - self.chunk_overlap)
        
        logger.info(f"Generated {len(final_chunks)} chunks for chapter group: {group_key}")
        return final_chunks
    
    def _reconstruct_text(
        self,
        docs: List[Dict[str, Any]]
    ) -> Tuple[str, List[SourceBlock]]:
        """重构文本：将多个文档拼接成连续文本，同时构建坐标映射
        
        Args:
            docs: 文档列表（已排序）
            
        Returns:
            (full_text, source_blocks): 重构后的完整文本和源块列表
        """
        if not docs:
            return "", []
        
        # 同时构建 full_text 和 source_blocks，确保偏移量准确
        full_text_parts = []
        source_blocks = []
        current_offset = 0
        
        prev_content = None
        prev_doc = None
        
        for i, doc in enumerate(docs):
            content = doc.get('content', '').strip()
            if not content:
                continue
            
            if i == 0:
                # 第一个块
                start_offset = current_offset
                end_offset = start_offset + len(content)
                
                source_block = SourceBlock(
                    id=doc.get('id', ''),
                    content=content,
                    metadata=doc.get('metadata', {}),
                    start_offset=start_offset,
                    end_offset=end_offset
                )
                source_blocks.append(source_block)
                full_text_parts.append(content)
                current_offset = end_offset
                prev_content = content
                prev_doc = doc
            else:
                # 后续块：处理连字符
                merged = self.hyphenation_handler.merge_blocks(prev_content, content)
                
                # 计算新增的文本部分
                # merged 是完整合并后的文本，我们需要找出新增部分
                prev_len = len(prev_content.rstrip())
                new_text = merged[prev_len:]
                
                # 如果发生了连字符合并（merged 不等于简单拼接）
                if merged != prev_content.rstrip() + " " + content.lstrip():
                    # 更新前一个块：扩展其结束位置以包含合并后的文本
                    # 但为了保持映射准确性，我们保持前一个块的原始内容
                    # 只更新其结束位置
                    if source_blocks:
                        # 前一个块现在延伸到合并后的位置
                        source_blocks[-1].end_offset = current_offset + len(merged) - len(new_text)
                        current_offset = source_blocks[-1].end_offset
                
                # 添加新文本部分
                if new_text:
                    start_offset = current_offset
                    end_offset = start_offset + len(new_text)
                    
                    source_block = SourceBlock(
                        id=doc.get('id', ''),
                        content=new_text,  # 使用实际添加到 full_text 的文本
                        metadata=doc.get('metadata', {}),
                        start_offset=start_offset,
                        end_offset=end_offset
                    )
                    source_blocks.append(source_block)
                    full_text_parts.append(new_text)
                    current_offset = end_offset
                    prev_content = merged  # 更新为合并后的完整文本
                else:
                    # 没有新文本（全部合并到前一个块）
                    prev_content = merged
                prev_doc = doc
        
        full_text = ''.join(full_text_parts)
        
        # 验证并修正偏移量：确保 source_blocks 的偏移量与 full_text 一致
        verified_blocks = []
        verified_offset = 0
        
        for block in source_blocks:
            # 重新计算偏移量，确保与 full_text 中的实际位置一致
            block_text = block.content
            found_pos = full_text.find(block_text, verified_offset)
            
            if found_pos != -1:
                start_offset = found_pos
                end_offset = start_offset + len(block_text)
            else:
                # 如果找不到（可能因为连字符合并），使用连续位置
                start_offset = verified_offset
                end_offset = start_offset + len(block_text)
            
            verified_block = SourceBlock(
                id=block.id,
                content=block.content,
                metadata=block.metadata,
                start_offset=start_offset,
                end_offset=end_offset
            )
            verified_blocks.append(verified_block)
            verified_offset = end_offset
        
        return full_text, verified_blocks
    
    def _generate_chunk_id(
        self,
        metadata: Dict[str, Any],
        content: str,
        chunk_index: int
    ) -> str:
        """生成块 ID
        
        Args:
            metadata: 元数据字典
            content: 块内容
            chunk_index: 块索引
            
        Returns:
            唯一 ID 字符串
        """
        volume = metadata.get('volume', 'unknown')
        pages = metadata.get('pages', '?')
        primary_page = metadata.get('primary_page', '?')
        work_title = metadata.get('work_title', '')
        chapter_title = metadata.get('chapter_title') or ''
        
        # 生成基于内容的短哈希
        content_hash = hashlib.md5(content[:100].encode('utf-8')).hexdigest()[:8]
        
        # 构建 ID
        id_parts = [
            volume,
            f"p{pages}",
            f"chunk{chunk_index:03d}",
            content_hash
        ]
        
        return "_".join(id_parts)
