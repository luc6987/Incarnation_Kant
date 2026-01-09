#!/usr/bin/env python3
"""
HTML to Semantic Stream Converter

三阶段 HTML 清洗管道：
1. Atomic Extraction: HTML DOM -> Text Atoms
2. Semantic Reconstruction: Atoms -> Continuous Stream (处理连字符)
3. Normalization: 清洗和标准化
"""

import json
import re
import hashlib
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from typing import Iterator, Optional, Dict, List
from collections import defaultdict

from bs4 import BeautifulSoup, Tag
from loguru import logger
from tqdm import tqdm
from html import unescape
import sys
from pathlib import Path

# 处理相对导入和绝对导入
try:
    from .toc_parser import TocParser
except ImportError:
    # 添加项目根目录到路径
    project_root = Path(__file__).parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from src.database.toc_parser import TocParser


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class TextAtom:
    """文本原子：从 HTML 中提取的原始文本单元"""
    volume: str  # e.g., "aa01"
    page: int    # e.g., 1
    line: int    # e.g., 1
    text: str    # 原始提取的文本


@dataclass
class SemanticBlock:
    """语义块：重构后的连续文本块"""
    id: str      # 唯一标识符
    content: str # 重构后的连续文本
    metadata: dict  # 包含 work_id, page_start, page_end, line_range 等


class HyphenationType(Enum):
    """连字符类型"""
    SOFT = "soft"      # 软连字符：删除连字符，无缝拼接
    HARD = "hard"      # 硬连字符：保留连字符
    NONE = "none"      # 自然换行：插入空格


# ============================================================================
# Stage 1: Atomic Extraction
# ============================================================================

class AtomicExtractor:
    """阶段一：原子提取 - 将 HTML DOM 拓扑降维为离散的文本原子"""
    
    def __init__(self):
        """初始化提取器"""
        # 锚点模式：name="z\d+"
        self.anchor_pattern = re.compile(r'^z(\d+)$')
    
    def extract(self, html_path: Path) -> Iterator[TextAtom]:
        """从单个 HTML 文件提取文本原子（生成器模式）
        
        Args:
            html_path: HTML 文件路径
            
        Yields:
            TextAtom: 文本原子
        """
        try:
            # 尝试多种编码
            encodings = ['utf-8', 'latin-1', 'iso-8859-1']
            soup = None
            
            for encoding in encodings:
                try:
                    with open(html_path, 'r', encoding=encoding, errors='ignore') as f:
                        soup = BeautifulSoup(f, 'lxml')
                    break
                except (UnicodeDecodeError, Exception) as e:
                    if encoding == encodings[-1]:
                        raise
                    continue
            
            if soup is None:
                logger.error(f"Failed to parse {html_path}: Could not decode with any encoding")
                return
            
            # 提取元数据（卷号和页码）
            volume, page = self._extract_meta(soup, html_path)
            
            # 遍历所有 <tr> 行
            for row in soup.find_all('tr'):
                atom = self._process_row(row, volume, page)
                if atom:
                    yield atom
                    
        except Exception as e:
            logger.error(f"Failed to parse {html_path}: {e}")
    
    def _extract_meta(self, soup: BeautifulSoup, html_path: Path) -> tuple[str, int]:
        """提取卷号和页码
        
        Args:
            soup: BeautifulSoup 对象
            html_path: HTML 文件路径
            
        Returns:
            (volume, page): 卷号和页码
        """
        # 从文件路径提取卷号：data/raw/kant/aa01/001.html -> aa01
        path_parts = html_path.parts
        volume = None
        for part in path_parts:
            if part.startswith('aa') and part[2:].isdigit():
                volume = part
                break
        
        if not volume:
            # 尝试从 HTML title 提取
            title = soup.find('title')
            if title:
                title_text = title.get_text()
                # 尝试匹配 "AA I" 或 "aa01" 等格式
                match = re.search(r'AA\s*([IVX]+|\d+)', title_text, re.IGNORECASE)
                if match:
                    volume = match.group(1).lower()
                else:
                    # 默认使用目录名
                    volume = html_path.parent.name if html_path.parent.name != 'kant' else 'unknown'
        
        # 从文件名提取页码：001.html -> 1
        page = 1
        filename = html_path.stem
        page_match = re.search(r'(\d+)', filename)
        if page_match:
            page = int(page_match.group(1))
        
        return volume or 'unknown', page
    
    def _process_row(self, row: Tag, vol: str, page: int) -> Optional[TextAtom]:
        """处理单个 <tr> 行，使用拓扑定位提取文本
        
        Args:
            row: BeautifulSoup Tag 对象（<tr>）
            vol: 卷号
            page: 页码
            
        Returns:
            TextAtom 或 None（如果该行不包含有效文本）
        """
        # 1. 找到锚点所在的 <a> 标签
        anchor = row.find('a', attrs={'name': self.anchor_pattern})
        if not anchor:
            return None
        
        # 2. 获取锚点的父 <td>
        current_td = anchor.find_parent('td')
        if not current_td:
            return None
        
        # 3. 找到紧邻的下一个 <td> (存放文本的地方)
        text_td = current_td.find_next_sibling('td')
        if not text_td:
            return None
        
        # 4. 获取所有文本，忽略内部的具体标签 (h2, p, font, b...)
        # separator=' ' 保证标签间有空格，避免 <p>End</p><p>Start</p> 变成 EndStart
        raw_text = text_td.get_text(separator=' ', strip=True)
        
        # 5. 关键：再次执行 .strip()，确保原子文本无首尾空格
        # 防止后续 Stage 2 拼接时的粘连问题
        raw_text = raw_text.strip()
        
        # 如果文本为空，跳过
        if not raw_text:
            return None
        
        # 6. 提取行号
        anchor_name = anchor.get('name', '')
        line_match = self.anchor_pattern.match(anchor_name)
        line_num = int(line_match.group(1)) if line_match else 0
        
        return TextAtom(volume=vol, page=page, line=line_num, text=raw_text)


# ============================================================================
# Stage 2: Semantic Reconstruction
# ============================================================================

class SemanticReconstructor:
    """阶段二：语义重构 - 解决德语语料核心痛点（断行连字符）"""
    
    def __init__(self):
        """初始化重构器"""
        self.pending_hyphen: Optional[str] = None  # 跨页连字符状态
        self.current_block_lines: List[TextAtom] = []  # 当前块的原子列表
        self.current_volume: Optional[str] = None
        self.current_page_start: Optional[int] = None
        self.current_line_start: Optional[int] = None
    
    def process_stream(self, atom_stream: Iterator[TextAtom]) -> Iterator[SemanticBlock]:
        """处理原子流，输出语义块流（生成器模式）
        
        将连续的原子合并成连续的文本流，在页面边界处分割块。
        每个块保留其起始和结束的行号、页码信息。
        
        Args:
            atom_stream: 文本原子流
            
        Yields:
            SemanticBlock: 语义块
        """
        prev_atom: Optional[TextAtom] = None
        accumulated_text = ""  # 累积的连续文本
        block_start_atom: Optional[TextAtom] = None  # 当前块的起始原子
        last_atom: Optional[TextAtom] = None  # 当前块的最后一个原子
        
        for atom in atom_stream:
            # 检测卷切换
            if self.current_volume is None:
                # 初始化第一个卷
                self.current_volume = atom.volume
                block_start_atom = atom
                last_atom = atom
                accumulated_text = atom.text
                prev_atom = atom
                continue
            elif atom.volume != self.current_volume:
                # 卷切换：输出当前卷的最后一个块
                if accumulated_text and block_start_atom and last_atom:
                    yield self._create_block_from_text(
                        accumulated_text,
                        block_start_atom.page,
                        last_atom.page,
                        block_start_atom.line,
                        last_atom.line
                    )
                # 开始新卷
                self.current_volume = atom.volume
                block_start_atom = atom
                last_atom = atom
                accumulated_text = atom.text
                self.pending_hyphen = None
                prev_atom = atom
                continue
            
            # 处理当前原子（同一卷内）
            if prev_atom:
                # 检测页面切换
                page_changed = atom.page != prev_atom.page
                
                if page_changed:
                    # 页面切换：输出当前页的块
                    if accumulated_text and block_start_atom and last_atom:
                        yield self._create_block_from_text(
                            accumulated_text,
                            block_start_atom.page,
                            last_atom.page,
                            block_start_atom.line,
                            last_atom.line
                        )
                    # 处理跨页拼接
                    self._handle_page_boundary(prev_atom, atom)
                    # 开始新页的块
                    block_start_atom = atom
                    accumulated_text = atom.text
                else:
                    # 同一页内：将新原子合并到累积文本中
                    # 检测连字符类型
                    hyphen_type = self._detect_hyphenation(
                        accumulated_text.rstrip(),
                        atom.text.lstrip()
                    )
                    # 合并
                    accumulated_text = self._merge_lines(
                        accumulated_text.rstrip(),
                        atom.text.lstrip(),
                        hyphen_type
                    )
            else:
                # 第一个原子
                accumulated_text = atom.text
                block_start_atom = atom
            
            # 更新最后一个原子
            last_atom = atom
            prev_atom = atom
        
        # 输出最后一个块
        if accumulated_text and block_start_atom and last_atom:
            yield self._create_block_from_text(
                accumulated_text,
                block_start_atom.page,
                last_atom.page,
                block_start_atom.line,
                last_atom.line
            )
    
    def _handle_page_boundary(self, prev_atom: TextAtom, next_atom: TextAtom) -> None:
        """处理页面边界
        
        Args:
            prev_atom: 上一页的最后一个原子
            next_atom: 下一页的第一个原子
        """
        # 如果上一页最后一行以连字符结尾，保存状态
        if prev_atom.text.rstrip().endswith('-'):
            self.pending_hyphen = prev_atom.text
        else:
            self.pending_hyphen = None
    
    def _merge_atoms(self, prev: TextAtom, curr: TextAtom) -> str:
        """合并两个原子
        
        Args:
            prev: 前一个原子
            curr: 当前原子
            
        Returns:
            合并后的文本
        """
        prev_text = prev.text.rstrip()
        curr_text = curr.text.lstrip()
        
        # 检测连字符类型
        hyphen_type = self._detect_hyphenation(prev_text, curr_text)
        
        # 合并文本
        merged = self._merge_lines(prev_text, curr_text, hyphen_type)
        
        return merged
    
    def _detect_hyphenation(self, current: str, next: str) -> HyphenationType:
        """检测连字符类型
        
        Args:
            current: 当前行文本
            next: 下一行文本
            
        Returns:
            HyphenationType: 连字符类型
        """
        current_stripped = current.rstrip()
        
        # 检查是否以连字符结尾
        if current_stripped.endswith('-'):
            # 检查下一行首字母
            next_stripped = next.lstrip()
            if next_stripped and next_stripped[0].islower():
                # Case A: 软连字符 - 下一行小写
                return HyphenationType.SOFT
            else:
                # Case B: 硬连字符 - 下一行大写或空
                return HyphenationType.HARD
        
        # Case C: 自然换行
        return HyphenationType.NONE
    
    def _merge_lines(self, line1: str, line2: str, hyphen_type: HyphenationType) -> str:
        """合并两行文本，根据连字符类型明确控制空格插入
        
        Args:
            line1: 第一行文本
            line2: 第二行文本
            hyphen_type: 连字符类型
            
        Returns:
            合并后的文本
        """
        # 前提：line1 和 line2 已在 Stage 1 执行过 .strip()，无首尾空格
        # 但为了安全，再次处理
        line1 = line1.rstrip()
        line2 = line2.lstrip()
        
        if hyphen_type == HyphenationType.SOFT:
            # Case A: 软连字符 - 删除连字符，无缝拼接
            return line1.rstrip('-') + line2
        elif hyphen_type == HyphenationType.HARD:
            # Case B: 硬连字符 - 保留连字符，通常连字符后不需要额外空格
            return line1 + line2
        else:
            # Case C: 自然换行 - 必须明确插入空格，防止粘连
            return line1 + " " + line2
    
    def _create_block_from_text(
        self,
        content: str,
        page_start: int,
        page_end: int,
        line_start: int,
        line_end: int
    ) -> SemanticBlock:
        """从文本创建语义块
        
        Args:
            content: 块的内容文本
            page_start: 起始页码
            page_end: 结束页码
            line_start: 起始行号
            line_end: 结束行号
            
        Returns:
            SemanticBlock: 语义块
        """
        metadata = {
            "work_id": self.current_volume,
            "page_start": page_start,
            "page_end": page_end,
            "line_range": f"{line_start}-{line_end}",
            "language": "de",
            "author": "Kant"
        }
        
        # 生成唯一 ID
        block_id = self._generate_id(metadata, content)
        
        return SemanticBlock(
            id=block_id,
            content=content,
            metadata=metadata
        )
    
    def _generate_id(self, metadata: dict, content: str) -> str:
        """生成唯一 ID
        
        Args:
            metadata: 元数据字典
            content: 内容文本
            
        Returns:
            唯一 ID 字符串
        """
        # 基于 work_id, page_start, line_range 和内容的前几个字符生成 hash
        id_string = f"{metadata['work_id']}_{metadata['page_start']}_{metadata['line_range']}_{content[:50]}"
        hash_obj = hashlib.md5(id_string.encode('utf-8'))
        short_hash = hash_obj.hexdigest()[:8]
        return f"{metadata['work_id']}_p{metadata['page_start']}-{metadata['page_end']}_l{metadata['line_range']}_{short_hash}"


# ============================================================================
# Stage 3: Normalization
# ============================================================================

class TextNormalizer:
    """阶段三：清洗与标准化 - 去除不影响语义的排版残留，统一字符集"""
    
    def __init__(self):
        """初始化标准化器"""
        # 编辑标记模式：如 [Seite 003]
        self.editorial_pattern = re.compile(r'\[Seite\s+\d+\]', re.IGNORECASE)
    
    def normalize(self, text: str) -> str:
        """主标准化流程
        
        Args:
            text: 原始文本
            
        Returns:
            标准化后的文本
        """
        # 1. HTML 实体解码
        text = self._decode_html_entities(text)
        
        # 2. 移除编辑标记
        text = self._remove_editorial_marks(text)
        
        # 3. 规范化空白字符
        text = self._normalize_whitespace(text)
        
        return text
    
    def _decode_html_entities(self, text: str) -> str:
        """HTML 实体转 Unicode
        
        Args:
            text: 包含 HTML 实体的文本
            
        Returns:
            解码后的文本
        """
        # 使用 html.unescape 解码 HTML 实体
        return unescape(text)
    
    def _remove_editorial_marks(self, text: str) -> str:
        """移除编辑标记
        
        Args:
            text: 包含编辑标记的文本
            
        Returns:
            移除标记后的文本
        """
        # 移除 [Seite 003] 等页码标记
        text = self.editorial_pattern.sub('', text)
        return text.strip()
    
    def _normalize_whitespace(self, text: str) -> str:
        """规范化空白字符
        
        Args:
            text: 原始文本
            
        Returns:
            规范化后的文本
        """
        # 将多个连续空白字符替换为单个空格
        text = re.sub(r'\s+', ' ', text)
        return text.strip()


# ============================================================================
# Pipeline: 协调三个阶段
# ============================================================================

class Pipeline:
    """主管道：协调三个阶段，按卷生成独立的 JSONL 文件"""
    
    def __init__(self, raw_dir: Path, output_dir: Path, toc_dir: Optional[Path] = None):
        """初始化管道
        
        Args:
            raw_dir: 原始 HTML 文件目录
            output_dir: 输出 JSONL 文件目录
            toc_dir: TOC 文件目录（如果为 None，则使用 raw_dir）
        """
        self.raw_dir = Path(raw_dir)
        self.output_dir = Path(output_dir)
        self.toc_dir = Path(toc_dir) if toc_dir else self.raw_dir
        
        self.extractor = AtomicExtractor()
        self.reconstructor = SemanticReconstructor()
        self.normalizer = TextNormalizer()
        
        # 加载 TOC 查找表
        self.toc_parser = TocParser()
        logger.info("Building TOC lookup table...")
        # TOC 文件在 raw_dir 的父目录（kant/）中
        toc_search_dir = self.raw_dir if self.raw_dir.name != 'aa01' else self.raw_dir.parent
        self.toc_lookup_table = self.toc_parser.build_lookup_table(toc_search_dir)
        logger.info(f"Loaded TOC for {len(self.toc_lookup_table)} volumes")
        
        # 确保输出目录存在
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def run(self) -> None:
        """主处理流程，使用生成器链模式，按卷分组输出"""
        # 1. 获取所有 HTML 文件并排序
        html_files = sorted(self.raw_dir.glob('**/*.html'))
        
        if not html_files:
            logger.warning(f"No HTML files found in {self.raw_dir}")
            return
        
        logger.info(f"Found {len(html_files)} HTML files to process")
        
        # 2. 生成原子流
        def atom_generator() -> Iterator[TextAtom]:
            """生成所有文件的原子流"""
            for html_file in tqdm(html_files, desc="Extracting atoms"):
                yield from self.extractor.extract(html_file)
        
        # 3. 按卷分组处理
        current_volume = None
        current_atoms: List[TextAtom] = []
        volume_stats: Dict[str, int] = defaultdict(int)
        
        for atom in atom_generator():
            if current_volume is None:
                current_volume = atom.volume
                current_atoms = [atom]
            elif atom.volume != current_volume:
                # 处理完一个卷，输出
                self._process_volume(current_volume, current_atoms, volume_stats)
                # 开始新卷
                current_volume = atom.volume
                current_atoms = [atom]
            else:
                current_atoms.append(atom)
        
        # 处理最后一个卷
        if current_volume and current_atoms:
            self._process_volume(current_volume, current_atoms, volume_stats)
        
        # 输出统计信息
        logger.info("Processing complete!")
        for volume, count in sorted(volume_stats.items()):
            logger.info(f"  {volume}: {count} blocks")
    
    def _process_volume(self, volume: str, atoms: List[TextAtom], stats: Dict[str, int]) -> None:
        """处理单个卷
        
        Args:
            volume: 卷号
            atoms: 该卷的所有原子
            stats: 统计信息字典
        """
        logger.info(f"Processing volume {volume} ({len(atoms)} atoms)")
        
        # 生成语义流
        semantic_stream = self.reconstructor.process_stream(iter(atoms))
        
        # 写入 JSONL 文件
        output_file = self.output_dir / f"{volume}.jsonl"
        block_count = 0
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for block in semantic_stream:
                # 应用标准化
                normalized_content = self.normalizer.normalize(block.content)
                
                # 注入 TOC 元数据（全息化）
                enhanced_metadata = self._inject_toc_metadata(
                    block.metadata,
                    volume,
                    block.metadata.get('page_start', 1)
                )
                
                # 构建输出对象
                output_obj = {
                    "id": block.id,
                    "content": normalized_content,
                    "metadata": enhanced_metadata
                }
                
                # 写入 JSONL
                f.write(json.dumps(output_obj, ensure_ascii=False) + '\n')
                block_count += 1
        
        stats[volume] = block_count
        logger.info(f"  Wrote {block_count} blocks to {output_file}")
    
    def _inject_toc_metadata(
        self,
        base_metadata: dict,
        volume: str,
        page: int
    ) -> dict:
        """注入 TOC 元数据到基础元数据中
        
        Args:
            base_metadata: 基础元数据（包含 page_start, page_end, line_range 等）
            volume: 卷号
            page: 页码（用于查找 TOC）
            
        Returns:
            增强后的元数据字典
        """
        # 查找 TOC 上下文
        context_meta = self.toc_parser.lookup_context(
            self.toc_lookup_table,
            volume,
            page
        )
        
        # 合并元数据
        # 保留原有的物理定位信息，添加语义定位信息
        enhanced = {
            # 物理定位（原有）
            "volume": base_metadata.get("work_id", volume),
            "page": page,  # 使用起始页码
            "page_start": base_metadata.get("page_start", page),
            "page_end": base_metadata.get("page_end", page),
            "line_range": base_metadata.get("line_range", ""),
            
            # 语义定位（新增）
            "year": context_meta.get("year"),
            "work_title": context_meta.get("work_title"),
            "chapter_title": context_meta.get("chapter_title"),
            "is_work_start": context_meta.get("is_work_start", False),
            
            # 其他元数据
            "language": base_metadata.get("language", "de"),
            "author": base_metadata.get("author", "Kant")
        }
        
        return enhanced


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """主入口函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='HTML to Semantic Stream Converter')
    parser.add_argument(
        '--input-dir',
        type=str,
        default='data/raw/kant',
        help='Input directory containing HTML files (default: data/raw/kant)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/cleaned',
        help='Output directory for JSONL files (default: data/cleaned)'
    )
    parser.add_argument(
        '--toc-dir',
        type=str,
        default=None,
        help='TOC directory (default: same as input-dir)'
    )
    
    args = parser.parse_args()
    
    # 创建管道并运行
    pipeline = Pipeline(
        raw_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        toc_dir=Path(args.toc_dir) if args.toc_dir else None
    )
    
    pipeline.run()


if __name__ == '__main__':
    main()
