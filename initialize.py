#!/usr/bin/env python3
"""
Digital Kant Baseline v2.0 - 项目初始化脚本
初始化虚拟环境并下载 Korpora 原始数据集

Usage:
    python initialize.py [--skip-venv] [--skip-download]
"""

import os
import sys
import subprocess
import venv
import json
import re
from pathlib import Path
from typing import List, Set, Optional, Dict, DefaultDict
from collections import defaultdict, deque
from urllib.parse import urljoin, urlparse
import time

# 检查必要的依赖包
try:
    import requests
    from bs4 import BeautifulSoup
    from loguru import logger
    from tqdm import tqdm
except ImportError as e:
    print(f"错误: 缺少必要的依赖包: {e}")
    print("\n请先安装基础依赖:")
    print("  pip install requests beautifulsoup4 loguru tqdm")
    print("\n或者运行:")
    print("  pip install -r requirements.txt")
    sys.exit(1)


# 配置常量
BASE_URL = "https://korpora.org/kant/"
INDEX_URL = "https://korpora.org/kant/verzeichnisse-gesamt.html"
VENV_DIR = Path(".venv")
DATA_RAW_DIR = Path("data/raw")
REQUEST_DELAY = 1.0  # 请求间隔（秒），避免对服务器造成压力
REQUEST_TIMEOUT = 30  # 请求超时时间（秒）
MAX_RETRIES = 3  # 最大重试次数


class URLTree:
    """URL 树结构，用于标记和管理已存在的文件"""
    
    def __init__(self):
        """初始化树结构"""
        self.nodes: Dict[str, Dict] = {}  # URL -> 节点信息
        self.children: DefaultDict[str, Set[str]] = defaultdict(set)  # 父URL -> 子URL集合
        self.parent: Dict[str, str] = {}  # 子URL -> 父URL
    
    def add_node(self, url: str, file_path: str, parent_url: Optional[str] = None) -> None:
        """添加节点到树中。
        
        Args:
            url: 节点 URL
            file_path: 对应的文件路径
            parent_url: 父节点 URL（可选）
        """
        self.nodes[url] = {
            "url": url,
            "file_path": file_path,
            "exists": True,
            "visited": True
        }
        
        if parent_url:
            self.children[parent_url].add(url)
            self.parent[url] = parent_url
    
    def exists(self, url: str) -> bool:
        """检查 URL 是否已存在于树中。
        
        Args:
            url: 要检查的 URL
        
        Returns:
            是否存在
        """
        return url in self.nodes and self.nodes[url].get("exists", False)
    
    def mark_visited(self, url: str) -> None:
        """标记 URL 为已访问。
        
        Args:
            url: 要标记的 URL
        """
        if url in self.nodes:
            self.nodes[url]["visited"] = True
    
    def get_children(self, url: str) -> Set[str]:
        """获取子节点 URL 集合。
        
        Args:
            url: 父节点 URL
        
        Returns:
            子节点 URL 集合
        """
        return self.children.get(url, set())
    
    def get_stats(self) -> Dict[str, int]:
        """获取统计信息。
        
        Returns:
            包含统计信息的字典
        """
        total = len(self.nodes)
        exists = sum(1 for n in self.nodes.values() if n.get("exists", False))
        visited = sum(1 for n in self.nodes.values() if n.get("visited", False))
        
        return {
            "total": total,
            "exists": exists,
            "visited": visited
        }


class KorporaScraper:
    """Korpora 网站抓取器"""
    
    def __init__(self, base_url: str, output_dir: Path, delay: float = 1.0):
        """初始化抓取器。
        
        Args:
            base_url: Korpora 基础 URL
            output_dir: 输出目录
            delay: 请求间隔（秒）
        """
        self.base_url = base_url
        self.output_dir = output_dir
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        })
        self.visited_urls: Set[str] = set()
        self.failed_urls: List[str] = []
        self.url_to_file: Dict[str, str] = {}  # URL -> 文件路径的映射
        self.file_to_url: Dict[str, str] = {}  # 文件路径 -> URL 的反向映射
        self.mapping_file = output_dir / ".url_mapping.json"  # 映射文件路径
        self.url_tree = URLTree()  # URL 树结构
        self.skipped_count = 0  # 跳过的文件计数（已存在的文件）
        self.depth_skipped_count = 0  # 因深度限制跳过的计数
        
        # 确保输出目录存在
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载已存在的映射和已下载的文件
        self._load_existing_files()
    
    def _normalize_url(self, url: str) -> str:
        """规范化 URL（移除锚点、末尾斜杠等）。
        
        Args:
            url: 原始 URL
        
        Returns:
            规范化后的 URL
        """
        # 移除锚点部分
        if "#" in url:
            url = url.split("#")[0]
        # 移除末尾斜杠（除非是根路径）
        url = url.rstrip("/")
        # 确保 URL 格式一致
        parsed = urlparse(url)
        normalized = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
        if parsed.query:
            normalized += f"?{parsed.query}"
        return normalized
    
    def _load_existing_files(self) -> None:
        """加载已存在的文件映射，跳过已下载的文件。"""
        # 加载 URL 映射文件
        loaded_count = 0
        if self.mapping_file.exists():
            try:
                with open(self.mapping_file, "r", encoding="utf-8") as f:
                    self.url_to_file = json.load(f)
                loaded_count = len(self.url_to_file)
                logger.info(f"加载了 {loaded_count} 个已下载文件的映射")
            except Exception as e:
                logger.warning(f"加载映射文件失败: {e}，将重新扫描目录")
        
        # 验证映射中的文件是否仍然存在
        verified_count = 0
        for url, file_rel_path in list(self.url_to_file.items()):
            # 规范化 URL
            normalized_url = self._normalize_url(url)
            filepath = self.output_dir / file_rel_path
            if filepath.exists() and filepath.stat().st_size > 0:
                # 使用规范化后的 URL
                if normalized_url != url:
                    # 更新映射中的 URL
                    self.url_to_file[normalized_url] = file_rel_path
                    if url in self.url_to_file:
                        del self.url_to_file[url]
                self.visited_urls.add(normalized_url)
                self.file_to_url[file_rel_path] = normalized_url
                # 添加到树结构
                self.url_tree.add_node(normalized_url, file_rel_path)
                verified_count += 1
            else:
                # 文件不存在，从映射中移除
                del self.url_to_file[url]
                if url in self.file_to_url.values():
                    # 从反向映射中移除
                    self.file_to_url = {
                        k: v for k, v in self.file_to_url.items() if v != url
                    }
                logger.debug(f"映射中的文件已不存在，移除: {url}")
        
        # 扫描目录，找出所有 HTML 文件（补充映射中可能缺失的条目）
        html_files = list(self.output_dir.rglob("*.html")) + list(
            self.output_dir.rglob("*.htm")
        )
        
        # 从文件路径反推 URL（补充映射中可能缺失的条目）
        scanned_count = 0
        for filepath in html_files:
            # 计算相对路径
            try:
                rel_path = filepath.relative_to(self.output_dir)
                rel_path_str = str(rel_path).replace("\\", "/")  # Windows 路径转换
                
                # 检查这个文件是否已经在映射中
                already_mapped = rel_path_str in self.file_to_url
                
                if not already_mapped and filepath.exists() and filepath.stat().st_size > 0:
                    # 从相对路径构建 URL
                    # 例如: kant/aa05/016.html -> https://korpora.org/kant/aa05/016.html
                    # 或者: kant/aa06.html -> https://korpora.org/kant/aa06.html
                    url_path = rel_path_str
                    reconstructed_url = f"{self.base_url.rstrip('/')}/{url_path}"
                    # 规范化 URL
                    normalized_url = self._normalize_url(reconstructed_url)
                    
                    # 添加到映射和树结构
                    self.url_to_file[normalized_url] = rel_path_str
                    self.file_to_url[rel_path_str] = normalized_url
                    self.visited_urls.add(normalized_url)
                    self.url_tree.add_node(normalized_url, rel_path_str)
                    scanned_count += 1
            except Exception as e:
                logger.debug(f"处理文件 {filepath} 时出错: {e}")
        
        if scanned_count > 0:
            logger.info(f"扫描目录发现 {scanned_count} 个新文件，已添加到映射")
        
        total_skipped = len(self.visited_urls)
        logger.info(
            f"共发现 {total_skipped} 个已下载的文件（已验证: {verified_count}，"
            f"新扫描: {scanned_count}），将跳过这些 URL"
        )
    
    def _save_mapping(self) -> None:
        """保存 URL 到文件路径的映射。"""
        try:
            with open(self.mapping_file, "w", encoding="utf-8") as f:
                json.dump(self.url_to_file, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"保存映射文件失败: {e}")
    
    def _get_page(self, url: str) -> Optional[requests.Response]:
        """获取网页内容，带重试机制。
        
        Args:
            url: 目标 URL
        
        Returns:
            Response 对象，失败返回 None
        """
        for attempt in range(MAX_RETRIES):
            try:
                response = self.session.get(url, timeout=REQUEST_TIMEOUT)
                response.raise_for_status()
                return response
            except requests.RequestException as e:
                logger.warning(f"请求失败 (尝试 {attempt + 1}/{MAX_RETRIES}): {url} - {e}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(self.delay * (attempt + 1))
                else:
                    self.failed_urls.append(url)
                    return None
    
    def _extract_navigation_links(self, soup: BeautifulSoup, base_url: str) -> Dict[str, Optional[str]]:
        """从页面中提取导航链接（上一页、下一页、返回目录）。
        
        Args:
            soup: BeautifulSoup 对象
            base_url: 基础 URL，用于解析相对链接
        
        Returns:
            字典：{"prev": 上一页URL, "next": 下一页URL, "index": 目录页URL}
        """
        # 规范化 base_url，确保以目录结尾
        parsed_base = urlparse(base_url)
        base_path = parsed_base.path.rstrip("/")
        
        # 如果 base_path 以 .html 或 .htm 结尾，移除文件名，只保留目录
        if base_path.endswith((".html", ".htm")):
            base_dir = "/".join(base_path.split("/")[:-1])
            if base_dir:
                base_path = base_dir + "/"
            else:
                base_path = "/"
        
        # 构建基础目录 URL
        base_dir_url = f"{parsed_base.scheme}://{parsed_base.netloc}{base_path}"
        if not base_path.endswith("/"):
            base_dir_url += "/"
        
        nav_links = {"prev": None, "next": None, "index": None}
        
        # 查找所有包含"Seite"或"Inhalt"的链接
        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"]
            text = a_tag.get_text(strip=True)
            
            # 跳过锚点链接
            if href.startswith("#") or not href.strip():
                continue
            
            # 转换为绝对 URL
            absolute_url = urljoin(base_dir_url, href)
            
            # 移除锚点部分
            if "#" in absolute_url:
                absolute_url = absolute_url.split("#")[0]
            
            # 只保留 Korpora Kant 相关的链接
            if "korpora.org/kant" not in absolute_url:
                continue
            
            # 判断链接类型
            if "Inhalt" in text or "Inhaltsverzeichnis" in text or "index" in href.lower() or "Inhalt" in href:
                if nav_links["index"] is None:  # 只设置一次
                    nav_links["index"] = absolute_url
            elif "Seite" in text or re.search(r'^\d+\.html$', href):
                # 判断是上一页还是下一页
                # 从当前URL提取页码
                current_page = self._extract_page_number(base_url)
                link_page = self._extract_page_number(absolute_url)
                
                if current_page is not None and link_page is not None:
                    if link_page < current_page:
                        # 上一页（只设置一次，避免覆盖）
                        if nav_links["prev"] is None:
                            nav_links["prev"] = absolute_url
                    elif link_page > current_page:
                        # 下一页（只设置一次，避免覆盖）
                        if nav_links["next"] is None:
                            nav_links["next"] = absolute_url
        
        return nav_links
    
    def _extract_page_number(self, url: str) -> Optional[int]:
        """从 URL 中提取页码。
        
        Args:
            url: 页面 URL
        
        Returns:
            页码数字，如果无法提取返回 None
        """
        match = re.search(r'/(\d+)\.html', url)
        if match:
            return int(match.group(1))
        return None
    
    def _extract_volume_links(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """从总索引页提取所有卷的链接。
        
        Args:
            soup: BeautifulSoup 对象
            base_url: 基础 URL（如 https://korpora.org/kant/verzeichnisse-gesamt.html）
        
        Returns:
            卷目录链接列表（如 https://korpora.org/kant/aa01/, https://korpora.org/kant/aa02/ 等）
        """
        volume_links = []
        parsed_base = urlparse(base_url)
        
        # 构建基础目录 URL：从 verzeichnisse-gesamt.html 提取 /kant/ 目录
        base_path = parsed_base.path
        # 移除文件名，只保留目录部分
        if "/" in base_path:
            base_dir = "/".join(base_path.split("/")[:-1])  # 移除最后一部分（文件名）
            if not base_dir.endswith("/"):
                base_dir += "/"
        else:
            base_dir = "/"
        
        base_dir_url = f"{parsed_base.scheme}://{parsed_base.netloc}{base_dir}"
        
        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"]
            if not href.strip() or href.startswith("#"):
                continue
            
            # 转换为绝对 URL
            # href 是相对链接如 "aa01/"，base_dir_url 是 "https://korpora.org/kant/"
            absolute_url = urljoin(base_dir_url, href)
            
            # 移除锚点部分
            if "#" in absolute_url:
                absolute_url = absolute_url.split("#")[0]
            
            # 检查是否是卷目录链接（如 aa01/, aa02/ 等）
            if "korpora.org/kant" in absolute_url:
                # 匹配 /kant/aa01/ 或 /kant/aa01 格式
                match = re.search(r'/kant/aa(\d+)/?$', absolute_url)
                if match:
                    # 确保以 / 结尾
                    volume_url = absolute_url.rstrip("/") + "/"
                    volume_links.append(volume_url)
        
        return sorted(set(volume_links))  # 去重并排序
    
    def _scan_existing_files_in_volume(self, volume_name: str) -> tuple[Set[int], Optional[str]]:
        """扫描一卷中已存在的文件，并找到最后一页。
        
        Args:
            volume_name: 卷名（如 "aa01"）
        
        Returns:
            (已存在的文件页码集合, 最后一页的URL)
        """
        volume_dir = self.output_dir / "kant" / volume_name
        if not volume_dir.exists():
            return set(), None
        
        existing_pages = set()
        max_page_num = -1
        last_page_url = None
        
        for filepath in volume_dir.iterdir():
            if filepath.is_file() and filepath.suffix in [".html", ".htm"]:
                page_num = self._extract_page_number(str(filepath))
                if page_num is not None:
                    existing_pages.add(page_num)
                    if page_num > max_page_num:
                        max_page_num = page_num
                        # 构建最后一页的URL
                        filename = filepath.name
                        # 从文件路径构建URL：kant/aa01/498.html -> https://korpora.org/kant/aa01/498.html
                        rel_path = filepath.relative_to(self.output_dir)
                        rel_path_str = str(rel_path).replace("\\", "/")
                        if rel_path_str in self.file_to_url:
                            last_page_url = self.file_to_url[rel_path_str]
                        else:
                            # 如果映射中没有，从路径构建
                            last_page_url = f"{BASE_URL.rstrip('/')}/{rel_path_str}"
        
        return existing_pages, last_page_url
    
    def _get_all_pages_from_index(self, soup: BeautifulSoup, base_url: str) -> List[tuple[int, str]]:
        """从卷索引页提取所有页码链接。
        
        Args:
            soup: BeautifulSoup 对象
            base_url: 卷索引页 URL
        
        Returns:
            列表：(页码, URL) 元组列表，按页码排序
        """
        parsed_base = urlparse(base_url)
        base_path = parsed_base.path.rstrip("/")
        
        # 如果 base_path 是 aa01.html，转换为 aa01/
        if base_path.endswith(".html"):
            base_path = base_path[:-5]  # 移除 .html
        
        # 确保以 / 结尾
        if not base_path.endswith("/"):
            base_path += "/"
        
        base_dir_url = f"{parsed_base.scheme}://{parsed_base.netloc}{base_path}"
        
        # 查找所有指向数字页面的链接（如 001.html, 002.html）
        page_links = []
        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"]
            if not href.strip() or href.startswith("#"):
                continue
            
            # 检查是否是数字页面链接（如 001.html）
            match = re.search(r'^(\d+)\.html', href)
            if match:
                absolute_url = urljoin(base_dir_url, href)
                if "#" in absolute_url:
                    absolute_url = absolute_url.split("#")[0]
                # 只保留该卷目录下的链接
                if base_path.rstrip("/") in absolute_url:
                    page_links.append((int(match.group(1)), absolute_url))
        
        # 按页码排序
        page_links.sort(key=lambda x: x[0])
        return page_links
    
    def _extract_first_page_from_index(self, soup: BeautifulSoup, base_url: str) -> Optional[str]:
        """从卷索引页提取第一页的链接。
        
        Args:
            soup: BeautifulSoup 对象
            base_url: 卷索引页 URL（可能是 aa01.html 或 aa01/）
        
        Returns:
            第一页的 URL，如果找不到返回 None
        """
        page_links = self._get_all_pages_from_index(soup, base_url)
        if page_links:
            return page_links[0][1]  # 返回第一页
        return None
    
    def _save_html(self, url: str, content: bytes) -> Path:
        """保存 HTML 内容到文件。
        
        Args:
            url: 源 URL
            content: HTML 内容
        
        Returns:
            保存的文件路径
        """
        # 从 URL 生成文件名
        parsed = urlparse(url)
        # 移除开头的斜杠，替换斜杠为下划线
        path_parts = parsed.path.strip("/").split("/")
        if path_parts and path_parts[-1]:
            filename = path_parts[-1]
        else:
            filename = "index.html"
        
        # 如果文件名不是 .html，添加扩展名
        if not filename.endswith((".html", ".htm")):
            filename += ".html"
        
        # 如果有路径部分，创建子目录
        if len(path_parts) > 1:
            subdir = self.output_dir / "/".join(path_parts[:-1])
            subdir.mkdir(parents=True, exist_ok=True)
            filepath = subdir / filename
        else:
            filepath = self.output_dir / filename
        
        # 保存文件
        filepath.write_bytes(content)
        return filepath
    
    def scrape_page(self, url: str) -> Optional[List[str]]:
        """抓取单个页面并提取链接（优化版本：假设已在 scrape_recursive 中检查过）。
        
        Args:
            url: 目标 URL（已规范化）
        
        Returns:
            发现的链接列表，失败返回 None
        """
        # 规范化 URL（如果还未规范化）
        normalized_url = self._normalize_url(url)
        
        # 注意：此方法假设调用者已经在 scrape_recursive 中检查过 visited_urls
        # 因此这里直接进行下载，不再重复检查
        logger.debug(f"抓取: {normalized_url}")
        
        response = self._get_page(normalized_url)
        if response is None:
            return None
        
        # 保存 HTML
        filepath = self._save_html(normalized_url, response.content)
        # 更新映射和树结构
        rel_path = filepath.relative_to(self.output_dir)
        rel_path_str = str(rel_path).replace("\\", "/")
        self.url_to_file[normalized_url] = rel_path_str
        self.file_to_url[rel_path_str] = normalized_url
        self.url_tree.add_node(normalized_url, rel_path_str)
        logger.debug(f"已保存: {filepath}")
        
        # 解析并提取链接
        soup = BeautifulSoup(response.content, "html.parser")
        links = self._extract_links(soup, normalized_url)
        
        # 延迟，避免对服务器造成压力
        time.sleep(self.delay)
        
        return links
    
    def _scrape_volume_pages_dfs_continue(
        self,
        volume_index_url: str,
        start_url: str,
        visited_in_volume: Set[str]
    ) -> int:
        """从指定页面开始，使用 DFS 继续下载后续所有页面。
        
        用于处理从最后一页发现的后续页面。
        
        Args:
            volume_index_url: 卷索引页 URL
            start_url: 起始页面 URL
            visited_in_volume: 该卷已访问的 URL 集合
        
        Returns:
            新下载的页面数
        """
        downloaded_count = 0
        stack = [start_url]
        processed_count = 0
        
        logger.info(f"  开始从 {start_url} 继续下载...")
        
        with tqdm(desc="  继续下载", unit="页", leave=False) as page_pbar:
            while stack:
                current_url = stack.pop()
                normalized_url = self._normalize_url(current_url)
                
                # 检查是否已访问
                if normalized_url in self.visited_urls:
                    continue
                
                if normalized_url in visited_in_volume:
                    continue
                
                # 标记为已访问
                self.visited_urls.add(normalized_url)
                visited_in_volume.add(normalized_url)
                processed_count += 1
                
                # 检查文件是否已存在
                if self.url_tree.exists(normalized_url):
                    # 已存在，从文件读取导航链接
                    nav_links = self._extract_navigation_links_from_file(normalized_url)
                    self.skipped_count += 1
                    logger.debug(f"跳过已存在: {normalized_url}")
                else:
                    # 不存在，下载
                    logger.debug(f"下载: {normalized_url}")
                    response = self._get_page(normalized_url)
                    if response is None:
                        logger.warning(f"下载失败: {normalized_url}")
                        continue
                    
                    # 保存文件
                    filepath = self._save_html(normalized_url, response.content)
                    rel_path = filepath.relative_to(self.output_dir)
                    rel_path_str = str(rel_path).replace("\\", "/")
                    self.url_to_file[normalized_url] = rel_path_str
                    self.file_to_url[rel_path_str] = normalized_url
                    self.url_tree.add_node(normalized_url, rel_path_str)
                    
                    downloaded_count += 1
                    
                    # 解析 HTML 提取导航链接
                    soup = BeautifulSoup(response.content, "html.parser")
                    nav_links = self._extract_navigation_links(soup, normalized_url)
                    
                    # 延迟
                    time.sleep(self.delay)
                    page_pbar.update(1)
                
                # 将"下一页"添加到栈中（只继续下载后续页面）
                if nav_links.get("next"):
                    next_url = self._normalize_url(nav_links["next"])
                    if next_url not in visited_in_volume and next_url not in self.visited_urls:
                        stack.append(next_url)
                        logger.debug(f"  发现下一页: {next_url}")
                else:
                    # 没有"下一页"了，说明已经到达真正的最后一页
                    logger.info(f"  已到达最后一页: {normalized_url}")
        
        return downloaded_count
    
    def _scrape_volume_pages_dfs_missing_only(
        self, 
        volume_index_url: str, 
        first_page_url: str, 
        visited_in_volume: Set[str],
        missing_pages: List[int]
    ) -> int:
        """使用 DFS 遍历一卷，但只下载缺失的页面。
        
        Args:
            volume_index_url: 卷索引页 URL
            first_page_url: 第一页 URL
            visited_in_volume: 该卷已访问的 URL 集合
            missing_pages: 缺失的页码列表
        
        Returns:
            该卷新下载的页面数
        """
        # 将缺失页码转换为集合，便于快速查找
        missing_pages_set = set(missing_pages)
        
        # 如果第一页不在缺失列表中，找到第一个缺失的页面作为起始点
        first_page_num = self._extract_page_number(first_page_url)
        if first_page_num not in missing_pages_set:
            # 找到第一个缺失的页面
            if missing_pages:
                # 从第一页开始，通过导航链接找到第一个缺失的页面
                start_url = first_page_url
            else:
                return 0
        else:
            start_url = first_page_url
        
        downloaded_count = 0
        stack = [start_url]  # DFS 使用栈
        processed_count = 0
        
        with tqdm(total=len(missing_pages), desc="  下载进度", unit="页", leave=False) as page_pbar:
            while stack:
                current_url = stack.pop()
                normalized_url = self._normalize_url(current_url)
                
                # 检查是否已访问（全局检查）
                if normalized_url in self.visited_urls:
                    continue
                
                # 检查是否在该卷中已访问（避免循环）
                if normalized_url in visited_in_volume:
                    continue
                
                # 提取页码，检查是否在缺失列表中
                page_num = self._extract_page_number(normalized_url)
                if page_num is None:
                    continue
                
                # 如果不在缺失列表中，跳过下载，但仍需提取导航链接继续遍历
                is_missing = page_num in missing_pages_set
                
                # 标记为已访问
                self.visited_urls.add(normalized_url)
                visited_in_volume.add(normalized_url)
                processed_count += 1
                
                if is_missing:
                    # 需要下载
                    logger.debug(f"下载缺失页面: {normalized_url}")
                    response = self._get_page(normalized_url)
                    if response is None:
                        logger.warning(f"下载失败: {normalized_url}")
                        continue
                    
                    # 保存文件
                    filepath = self._save_html(normalized_url, response.content)
                    rel_path = filepath.relative_to(self.output_dir)
                    rel_path_str = str(rel_path).replace("\\", "/")
                    self.url_to_file[normalized_url] = rel_path_str
                    self.file_to_url[rel_path_str] = normalized_url
                    self.url_tree.add_node(normalized_url, rel_path_str)
                    
                    downloaded_count += 1
                    
                    # 解析 HTML 提取导航链接
                    soup = BeautifulSoup(response.content, "html.parser")
                    nav_links = self._extract_navigation_links(soup, normalized_url)
                    
                    # 延迟
                    time.sleep(self.delay)
                    page_pbar.update(1)
                else:
                    # 已存在，从文件读取导航链接（如果文件存在）
                    # 即使已存在，也需要继续遍历以找到其他缺失的页面
                    if self.url_tree.exists(normalized_url):
                        nav_links = self._extract_navigation_links_from_file(normalized_url)
                        self.skipped_count += 1
                    else:
                        # 文件不存在但不在缺失列表中，尝试下载以获取导航链接
                        # 这可能是因为索引页不完整
                        response = self._get_page(normalized_url)
                        if response is None:
                            continue
                        
                        # 保存文件
                        filepath = self._save_html(normalized_url, response.content)
                        rel_path = filepath.relative_to(self.output_dir)
                        rel_path_str = str(rel_path).replace("\\", "/")
                        self.url_to_file[normalized_url] = rel_path_str
                        self.file_to_url[rel_path_str] = normalized_url
                        self.url_tree.add_node(normalized_url, rel_path_str)
                        
                        # 解析 HTML 提取导航链接
                        soup = BeautifulSoup(response.content, "html.parser")
                        nav_links = self._extract_navigation_links(soup, normalized_url)
                        
                        time.sleep(self.delay)
                
                # 将"上一页"和"下一页"添加到栈中（DFS）
                new_pages = []
                if nav_links.get("next"):
                    next_url = self._normalize_url(nav_links["next"])
                    next_page_num = self._extract_page_number(next_url)
                    # 只添加缺失的页面或需要遍历的页面（用于找到其他缺失页面）
                    if next_url not in visited_in_volume and next_url not in self.visited_urls:
                        new_pages.append(next_url)
                
                if nav_links.get("prev"):
                    prev_url = self._normalize_url(nav_links["prev"])
                    prev_page_num = self._extract_page_number(prev_url)
                    # 只添加缺失的页面或需要遍历的页面
                    if prev_url not in visited_in_volume and prev_url not in self.visited_urls:
                        new_pages.append(prev_url)
                
                # 添加到栈
                stack.extend(reversed(new_pages))
        
        return downloaded_count
    
    def _scrape_volume_pages_dfs(self, volume_index_url: str, first_page_url: str, visited_in_volume: Set[str]) -> int:
        """使用 DFS 遍历一卷的所有页码。
        
        Args:
            volume_index_url: 卷索引页 URL
            first_page_url: 第一页 URL
            visited_in_volume: 该卷已访问的 URL 集合（用于避免重复）
        
        Returns:
            该卷新下载的页面数
        """
        downloaded_count = 0
        stack = [first_page_url]  # DFS 使用栈
        processed_count = 0
        
        with tqdm(desc="  页面进度", unit="页", leave=False) as page_pbar:
            while stack:
                current_url = stack.pop()
                normalized_url = self._normalize_url(current_url)
                
                # 检查是否已访问（全局检查）
                if normalized_url in self.visited_urls:
                    continue
                
                # 检查是否在该卷中已访问（避免循环）
                if normalized_url in visited_in_volume:
                    continue
                
                # 标记为已访问
                self.visited_urls.add(normalized_url)
                visited_in_volume.add(normalized_url)
                processed_count += 1
                
                # 检查文件是否已存在
                file_exists = self.url_tree.exists(normalized_url)
                if file_exists:
                    # 已存在，从文件读取导航链接
                    nav_links = self._extract_navigation_links_from_file(normalized_url)
                    self.skipped_count += 1
                    if processed_count % 10 == 0:  # 每10个页面记录一次
                        logger.debug(f"跳过已存在: {normalized_url} (已处理 {processed_count} 页)")
                else:
                    # 不存在，下载
                    if downloaded_count == 0 or downloaded_count % 10 == 0:
                        logger.debug(f"下载: {normalized_url}")
                    response = self._get_page(normalized_url)
                    if response is None:
                        logger.warning(f"下载失败: {normalized_url}")
                        continue
                    
                    # 保存文件
                    filepath = self._save_html(normalized_url, response.content)
                    rel_path = filepath.relative_to(self.output_dir)
                    rel_path_str = str(rel_path).replace("\\", "/")
                    self.url_to_file[normalized_url] = rel_path_str
                    self.file_to_url[rel_path_str] = normalized_url
                    self.url_tree.add_node(normalized_url, rel_path_str)
                    
                    downloaded_count += 1
                    
                    # 解析 HTML 提取导航链接
                    soup = BeautifulSoup(response.content, "html.parser")
                    nav_links = self._extract_navigation_links(soup, normalized_url)
                    
                    # 延迟
                    time.sleep(self.delay)
                
                # 将"上一页"和"下一页"添加到栈中（DFS）
                # 注意：先添加"下一页"，这样会先处理顺序的页面
                new_pages = []
                if nav_links.get("next"):
                    next_url = self._normalize_url(nav_links["next"])
                    if next_url not in visited_in_volume and next_url not in self.visited_urls:
                        new_pages.append(next_url)
                        logger.debug(f"  发现下一页: {next_url}")
                
                if nav_links.get("prev"):
                    prev_url = self._normalize_url(nav_links["prev"])
                    if prev_url not in visited_in_volume and prev_url not in self.visited_urls:
                        new_pages.append(prev_url)
                        logger.debug(f"  发现上一页: {prev_url}")
                
                # 如果没有任何导航链接，记录警告
                if not nav_links.get("prev") and not nav_links.get("next"):
                    logger.warning(f"  警告: {normalized_url} 没有找到导航链接")
                
                # 添加到栈（先添加下一页，后添加上一页，这样会优先处理顺序页面）
                stack.extend(reversed(new_pages))
                
                # 更新进度条
                page_pbar.update(1)
                page_pbar.set_postfix({
                    "已处理": processed_count,
                    "新下载": downloaded_count,
                    "栈大小": len(stack)
                })
        
        return downloaded_count
    
    def _extract_navigation_links_from_file(self, url: str) -> Dict[str, Optional[str]]:
        """从已存在的文件中提取导航链接。
        
        Args:
            url: 文件 URL
        
        Returns:
            导航链接字典
        """
        if url not in self.url_to_file:
            return {"prev": None, "next": None, "index": None}
        
        filepath = self.output_dir / self.url_to_file[url]
        if not filepath.exists() or filepath.stat().st_size == 0:
            return {"prev": None, "next": None, "index": None}
        
        try:
            content = filepath.read_bytes()
            soup = BeautifulSoup(content, "html.parser")
            return self._extract_navigation_links(soup, url)
        except Exception as e:
            logger.debug(f"从文件提取导航链接失败: {e}")
            return {"prev": None, "next": None, "index": None}
    
    def scrape_by_volumes(self, index_url: str) -> None:
        """按卷抓取所有页面（新逻辑：已知23卷，使用DFS遍历每卷）。
        
        Args:
            index_url: 总索引页 URL
        """
        logger.info("开始按卷抓取...")
        
        # 1. 从总索引页获取所有卷的链接
        logger.info("正在获取卷列表...")
        response = self._get_page(index_url)
        if response is None:
            logger.error("无法获取总索引页")
            return
        
        soup = BeautifulSoup(response.content, "html.parser")
        volume_links = self._extract_volume_links(soup, index_url)
        
        if not volume_links:
            logger.warning("未找到卷链接，尝试使用已知的23卷")
            # 如果提取失败，使用已知的23卷
            volume_links = [f"{BASE_URL}aa{i:02d}/" for i in range(1, 24)]
        
        logger.info(f"找到 {len(volume_links)} 卷")
        
        # 2. 对每一卷进行扫描和下载
        total_downloaded = 0
        total_volumes = len(volume_links)
        
        with tqdm(total=total_volumes, desc="卷进度", unit="卷") as volume_pbar:
            for volume_url in volume_links:
                volume_name = volume_url.rstrip("/").split("/")[-1]
                logger.info(f"\n处理卷: {volume_name}")
                
                # 步骤1: 先扫描已存在的文件
                logger.info(f"  扫描已存在的文件...")
                existing_pages, last_page_url = self._scan_existing_files_in_volume(volume_name)
                logger.info(f"  发现 {len(existing_pages)} 个已存在的文件")
                
                # 步骤1.5: 检查最后一页是否有"下一页"链接
                additional_pages = []
                if last_page_url:
                    logger.info(f"  检查最后一页: {last_page_url}")
                    normalized_last_url = self._normalize_url(last_page_url)
                    nav_links = self._extract_navigation_links_from_file(normalized_last_url)
                    
                    if nav_links.get("next"):
                        next_url = self._normalize_url(nav_links["next"])
                        next_page_num = self._extract_page_number(next_url)
                        if next_page_num is not None:
                            logger.info(f"  发现最后一页还有下一页: {next_url} (页码: {next_page_num})")
                            logger.info(f"  将从最后一页继续下载后续页面...")
                            # 将"下一页"作为起始点，继续DFS遍历
                            additional_pages.append(next_url)
                
                # 步骤2: 访问卷索引页，获取所有页码
                volume_index_url = volume_url.rstrip("/")
                response = self._get_page(volume_index_url)
                
                if response is None:
                    # 尝试 .html 格式
                    volume_index_url_html = volume_index_url + ".html"
                    response = self._get_page(volume_index_url_html)
                    if response is None:
                        logger.warning(f"无法访问卷索引页: {volume_name}")
                        volume_pbar.update(1)
                        continue
                    volume_index_url = volume_index_url_html
                else:
                    # 如果 aa01/ 返回成功，检查是否是重定向到 aa01.html
                    final_url = response.url
                    if final_url != volume_index_url:
                        volume_index_url = final_url
                
                # 从卷索引页提取所有页码
                soup = BeautifulSoup(response.content, "html.parser")
                all_pages = self._get_all_pages_from_index(soup, volume_index_url)
                
                if not all_pages:
                    logger.warning(f"无法从索引页提取页码: {volume_name}")
                    volume_pbar.update(1)
                    continue
                
                # 步骤3: 找出缺失的页码
                all_page_numbers = {page_num for page_num, _ in all_pages}
                missing_pages = sorted(all_page_numbers - existing_pages)
                
                if not missing_pages:
                    logger.info(f"  卷 {volume_name} 完整，无需下载")
                    volume_pbar.update(1)
                    continue
                
                logger.info(f"  发现 {len(missing_pages)} 个缺失的文件 (共 {len(all_page_numbers)} 页)")
                
                # 步骤4: 只下载缺失的文件
                # 从第一页开始，使用 DFS 遍历，但只访问缺失的页面
                first_page_url = all_pages[0][1]
                visited_in_volume: Set[str] = set()
                
                # 如果有从最后一页发现的额外页面，先处理这些
                if additional_pages:
                    logger.info(f"  先处理从最后一页发现的后续页面...")
                    for additional_url in additional_pages:
                        if additional_url not in visited_in_volume:
                            # 从这一页开始DFS，下载所有后续页面
                            additional_downloaded = self._scrape_volume_pages_dfs_continue(
                                volume_index_url,
                                additional_url,
                                visited_in_volume
                            )
                            logger.info(f"  从最后一页继续下载了 {additional_downloaded} 页")
                
                # 然后处理索引页中发现的缺失页面
                downloaded = self._scrape_volume_pages_dfs_missing_only(
                    volume_index_url,
                    first_page_url,
                    visited_in_volume,
                    missing_pages
                )
                
                total_downloaded += downloaded
                logger.info(f"卷 {volume_name} 完成: 新下载 {downloaded} 页，共访问 {len(visited_in_volume)} 页")
                
                volume_pbar.update(1)
                volume_pbar.set_postfix({
                    "已处理": f"{volume_pbar.n}/{total_volumes}",
                    "新下载": total_downloaded,
                    "总访问": len(self.visited_urls)
                })
        
        # 保存映射文件
        self._save_mapping()
        
        # 统计
        logger.info("\n" + "=" * 80)
        logger.info("抓取完成！")
        logger.info(f"  - 总卷数: {total_volumes}")
        logger.info(f"  - 总访问: {len(self.visited_urls)} 个页面")
        logger.info(f"  - 新下载: {total_downloaded} 个页面")
        logger.info(f"  - 跳过已存在: {self.skipped_count} 个页面")
        if self.failed_urls:
            logger.warning(f"  - 失败: {len(self.failed_urls)} 个页面")
    
    def _extract_links_from_file(self, url: str) -> Optional[List[str]]:
        """从已存在的文件中提取链接（优化：避免重复下载）。
        
        Args:
            url: 文件 URL
        
        Returns:
            链接列表，失败返回 None
        """
        # 检查映射
        if url not in self.url_to_file:
            return None
        
        filepath = self.output_dir / self.url_to_file[url]
        if not filepath.exists() or filepath.stat().st_size == 0:
            return None
        
        try:
            content = filepath.read_bytes()
            soup = BeautifulSoup(content, "html.parser")
            links = self._extract_links(soup, url)
            self.url_tree.mark_visited(url)
            return links
        except Exception as e:
            logger.debug(f"从文件提取链接失败: {e}")
            return None


def create_virtual_environment(venv_dir: Path) -> bool:
    """创建 Python 虚拟环境。
    
    Args:
        venv_dir: 虚拟环境目录路径
    
    Returns:
        是否成功创建
    """
    if venv_dir.exists():
        logger.info(f"虚拟环境已存在: {venv_dir}")
        return True
    
    try:
        logger.info(f"正在创建虚拟环境: {venv_dir}")
        venv.create(venv_dir, with_pip=True)
        logger.success(f"虚拟环境创建成功: {venv_dir}")
        return True
    except Exception as e:
        logger.error(f"创建虚拟环境失败: {e}")
        return False


def install_requirements(venv_dir: Path) -> bool:
    """在虚拟环境中安装依赖。
    
    Args:
        venv_dir: 虚拟环境目录路径
    
    Returns:
        是否成功安装
    """
    if not venv_dir.exists():
        logger.error("虚拟环境不存在，无法安装依赖")
        return False
    
    # 确定 pip 路径
    if sys.platform == "win32":
        pip_path = venv_dir / "Scripts" / "pip"
    else:
        pip_path = venv_dir / "bin" / "pip"
    
    requirements_file = Path("requirements.txt")
    if not requirements_file.exists():
        logger.warning("requirements.txt 不存在，跳过依赖安装")
        return False
    
    try:
        logger.info("正在安装依赖包...")
        subprocess.run(
            [str(pip_path), "install", "-r", str(requirements_file)],
            check=True,
            capture_output=True,
            text=True
        )
        logger.success("依赖包安装成功")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"安装依赖失败: {e}")
        logger.error(f"错误输出: {e.stderr}")
        return False


def download_korpora_data(
    index_url: str, 
    output_dir: Path,
    max_failure_rate: float = 5.0,
    max_depth: int = 15
) -> bool:
    """下载 Korpora 数据集。
    
    Args:
        index_url: 索引页面 URL
        output_dir: 输出目录
        max_failure_rate: 允许的最大失败率（百分比），默认 5.0
        max_depth: 最大递归深度，默认 15
    
    Returns:
        是否成功下载（允许少量失败，失败率 < max_failure_rate 视为成功）
    """
    try:
        scraper = KorporaScraper(
            base_url=BASE_URL,
            output_dir=output_dir,
            delay=REQUEST_DELAY
        )
        scraper.scrape_by_volumes(index_url)
        
        total_attempted = len(scraper.visited_urls)
        failed_count = len(scraper.failed_urls)
        success_count = total_attempted - failed_count
        
        if failed_count > 0:
            failure_rate = (failed_count / total_attempted) * 100
            logger.warning(
                f"部分页面下载失败: {failed_count}/{total_attempted} "
                f"({failure_rate:.2f}%)"
            )
            
            # 如果失败率小于阈值，仍然视为成功
            if failure_rate < max_failure_rate:
                logger.info(
                    f"失败率 {failure_rate:.2f}% < {max_failure_rate}%，视为成功。"
                    f"成功下载 {success_count} 个页面。"
                )
                return True
            else:
                logger.error(
                    f"失败率 {failure_rate:.2f}% >= {max_failure_rate}%，下载可能不完整。"
                )
                return False
        
        logger.success(f"全部成功！共下载 {success_count} 个页面")
        return True
    except Exception as e:
        logger.error(f"下载数据失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def main() -> int:
    """主函数。
    
    Returns:
        退出码（0 表示成功）
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Digital Kant Baseline v2.0 - 项目初始化脚本"
    )
    parser.add_argument(
        "--skip-venv",
        action="store_true",
        help="跳过虚拟环境创建"
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="跳过数据下载"
    )
    parser.add_argument(
        "--skip-install",
        action="store_true",
        help="跳过依赖安装"
    )
    parser.add_argument(
        "--max-failure-rate",
        type=float,
        default=5.0,
        help="允许的最大失败率（百分比），默认 5.0"
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=15,
        help="最大递归深度，默认 15"
    )
    
    args = parser.parse_args()
    
    # 确保日志目录存在
    Path("logs").mkdir(exist_ok=True)
    
    # 配置日志
    logger.add(
        "logs/initialize_{time}.log",
        rotation="10 MB",
        level="DEBUG",
        encoding="utf-8"
    )
    
    logger.info("=" * 60)
    logger.info("Digital Kant Baseline v2.0 - 项目初始化")
    logger.info("=" * 60)
    
    success = True
    
    # 1. 创建虚拟环境
    if not args.skip_venv:
        if not create_virtual_environment(VENV_DIR):
            success = False
            logger.error("虚拟环境创建失败，但继续执行...")
        
        # 安装依赖
        if not args.skip_install and success:
            if not install_requirements(VENV_DIR):
                logger.warning("依赖安装失败，但继续执行...")
    else:
        logger.info("跳过虚拟环境创建")
    
    # 2. 下载数据
    if not args.skip_download:
        logger.info("开始下载 Korpora 数据集...")
        if not download_korpora_data(
            INDEX_URL, 
            DATA_RAW_DIR,
            max_failure_rate=args.max_failure_rate,
            max_depth=args.max_depth
        ):
            success = False
            logger.error("数据下载失败")
    else:
        logger.info("跳过数据下载")
    
    # 总结
    logger.info("=" * 60)
    if success:
        logger.success("初始化完成！")
        logger.info(f"虚拟环境: {VENV_DIR}")
        logger.info(f"数据目录: {DATA_RAW_DIR}")
        logger.info("\n下一步:")
        logger.info("1. 激活虚拟环境:")
        if sys.platform == "win32":
            logger.info(f"   {VENV_DIR}\\Scripts\\activate")
        else:
            logger.info(f"   source {VENV_DIR}/bin/activate")
        logger.info("2. 运行数据库构建脚本:")
        logger.info("   python scripts/build_database.py")
    else:
        logger.error("初始化过程中出现错误，请检查日志")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
