#!/usr/bin/env python3
"""
Table of Contents (TOC) Parser

解析目录文件，构建页面到著作/章节的映射表。
"""

import re
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass

from bs4 import BeautifulSoup
from loguru import logger


@dataclass
class TocEntry:
    """TOC 条目"""
    start_page: int
    end_page: Optional[int]  # None 表示到下一个条目开始
    year: Optional[str]
    work_title: Optional[str]
    chapter_title: Optional[str]
    is_work_start: bool  # 是否为著作开始


class TocParser:
    """TOC 解析器"""
    
    def __init__(self):
        """初始化解析器"""
        self.page_pattern = re.compile(r'(\d+)\.html')
    
    def parse_toc_file(self, toc_path: Path) -> List[TocEntry]:
        """解析单个 TOC 文件
        
        Args:
            toc_path: TOC HTML 文件路径
            
        Returns:
            TOC 条目列表
        """
        try:
            encodings = ['utf-8', 'latin-1', 'iso-8859-1']
            soup = None
            
            for encoding in encodings:
                try:
                    with open(toc_path, 'r', encoding=encoding, errors='ignore') as f:
                        soup = BeautifulSoup(f, 'lxml')
                    break
                except (UnicodeDecodeError, Exception):
                    if encoding == encodings[-1]:
                        raise
                    continue
            
            if soup is None:
                logger.error(f"Failed to parse TOC {toc_path}")
                return []
            
            entries: List[TocEntry] = []
            current_year: Optional[str] = None
            current_work_title: Optional[str] = None
            current_work_start_page: Optional[int] = None
            
            # 查找所有表格行
            rows = soup.find_all('tr')
            
            for row in rows:
                # 检查是否是年份标题
                year_th = row.find('th')
                if year_th:
                    year_text = year_th.get_text(strip=True)
                    year_match = re.search(r'(\d{4})', year_text)
                    if year_match:
                        current_year = year_match.group(1)
                        current_work_title = None
                        current_work_start_page = None
                        continue
                
                # 查找页码链接
                page_link = row.find('a', href=self.page_pattern)
                if not page_link:
                    continue
                
                # 提取页码
                href = page_link.get('href', '')
                page_match = self.page_pattern.search(href)
                if not page_match:
                    continue
                
                page_num = int(page_match.group(1))
                
                # 提取标题文本
                # 查找粗体文本（可能是著作标题）
                bold_text = row.find('b')
                if bold_text:
                    title_text = bold_text.get_text(separator=' ', strip=True)
                    # 如果当前没有著作标题，或者这是新的著作开始
                    if not current_work_title or self._is_work_title(title_text, title_text):
                        current_work_title = title_text
                        current_work_start_page = page_num
                        # 创建著作开始条目
                        entries.append(TocEntry(
                            start_page=page_num,
                            end_page=None,
                            year=current_year,
                            work_title=current_work_title,
                            chapter_title=None,
                            is_work_start=True
                        ))
                        continue
                
                # 查找普通文本（可能是章节标题）
                # 查找包含文本的 td，但不是粗体
                text_tds = row.find_all('td', align='left')
                for td in text_tds:
                    # 跳过包含粗体的 td（那是著作标题）
                    if td.find('b'):
                        continue
                    chapter_text = td.get_text(separator=' ', strip=True)
                    # 清理 HTML 实体和多余空白
                    chapter_text = chapter_text.replace('&nbsp;', ' ').strip()
                    # 移除类似 "Seite X" 的页码标记
                    chapter_text = re.sub(r'\s*Seite\s+\d+\s*', '', chapter_text, flags=re.IGNORECASE)
                    chapter_text = re.sub(r'\s+', ' ', chapter_text).strip()
                    if chapter_text and len(chapter_text) > 2 and not chapter_text.startswith('<'):
                        # 创建章节条目
                        entries.append(TocEntry(
                            start_page=page_num,
                            end_page=None,
                            year=current_year,
                            work_title=current_work_title,
                            chapter_title=chapter_text,
                            is_work_start=False
                        ))
                        break
            
            # 填充 end_page：下一个条目的 start_page - 1
            for i in range(len(entries) - 1):
                if entries[i].end_page is None:
                    entries[i].end_page = entries[i + 1].start_page - 1
            
            # 最后一个条目：如果没有指定 end_page，设为 None（表示到卷末）
            
            logger.info(f"Parsed {len(entries)} TOC entries from {toc_path}")
            return entries
            
        except Exception as e:
            logger.error(f"Error parsing TOC {toc_path}: {e}")
            return []
    
    def _is_work_title(self, text: str, context: str) -> bool:
        """判断是否为著作标题（启发式）
        
        Args:
            text: 文本内容
            context: 上下文
            
        Returns:
            是否为著作标题
        """
        # 简单的启发式：如果文本较长且包含特定关键词，可能是著作标题
        if len(text) > 50:
            return True
        return False
    
    def build_lookup_table(self, toc_dir: Path) -> Dict[str, List[TocEntry]]:
        """构建所有卷的查找表
        
        Args:
            toc_dir: TOC 文件目录（包含 aa01.html, aa02.html 等）
            
        Returns:
            卷号 -> TOC 条目列表的映射
        """
        lookup_table: Dict[str, List[TocEntry]] = {}
        
        # 查找所有 TOC 文件（格式：aa01.html, aa02.html 等）
        toc_files = sorted(toc_dir.glob('aa*.html'))
        
        for toc_file in toc_files:
            # 提取卷号
            volume = toc_file.stem  # aa01
            entries = self.parse_toc_file(toc_file)
            if entries:
                lookup_table[volume] = entries
                logger.info(f"Built lookup table for {volume}: {len(entries)} entries")
        
        return lookup_table
    
    def lookup_context(
        self,
        lookup_table: Dict[str, List[TocEntry]],
        volume: str,
        page: int
    ) -> Dict[str, Optional[str]]:
        """查找页面对应的上下文信息
        
        Args:
            lookup_table: TOC 查找表
            volume: 卷号
            page: 页码
            
        Returns:
            上下文元数据字典
        """
        entries = lookup_table.get(volume, [])
        
        # 从后往前查找（因为后面的条目可能覆盖前面的）
        # 找到最后一个 start_page <= page 的条目
        matched_entry: Optional[TocEntry] = None
        
        for entry in reversed(entries):
            if entry.start_page <= page:
                if entry.end_page is None or page <= entry.end_page:
                    matched_entry = entry
                    break
        
        if matched_entry:
            return {
                "year": matched_entry.year,
                "work_title": matched_entry.work_title,
                "chapter_title": matched_entry.chapter_title,
                "is_work_start": matched_entry.is_work_start
            }
        
        # 如果没有匹配，返回默认值
        return {
            "year": None,
            "work_title": None,
            "chapter_title": None,
            "is_work_start": False
        }
