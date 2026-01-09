#!/usr/bin/env python3
"""
补充抓取缺失的文件
基于 check_files.py 的检查结果，抓取所有缺失的文件
"""

import re
import sys
import time
from pathlib import Path
from typing import Set, List, Dict, Optional
from urllib.parse import urlparse
from collections import defaultdict

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
    sys.exit(1)

# 配置常量
BASE_URL = "https://korpora.org/kant/"
REQUEST_DELAY = 1.0  # 请求间隔（秒）
REQUEST_TIMEOUT = 30  # 请求超时时间（秒）
MAX_RETRIES = 3  # 最大重试次数
DATA_RAW_DIR = Path("data/raw/kant")


def extract_number(filename: str) -> int:
    """从文件名中提取数字。"""
    match = re.search(r'(\d+)', filename)
    if match:
        return int(match.group(1))
    return -1


def find_missing_files(directory: Path) -> Dict[str, List[int]]:
    """找出所有缺失的文件。
    
    Returns:
        字典：{目录名: [缺失的文件数字列表]}
    """
    missing = {}
    
    # 获取所有 aa** 目录
    aa_dirs = sorted([d for d in directory.iterdir() if d.is_dir() and d.name.startswith('aa')])
    
    for aa_dir in aa_dirs:
        # 获取目录中所有 HTML 文件
        html_files = [f.name for f in aa_dir.iterdir() if f.is_file() and f.suffix in ['.html', '.htm']]
        
        # 提取数字
        numbers = set()
        for filename in html_files:
            num = extract_number(filename)
            if num >= 0:
                numbers.add(num)
        
        if numbers:
            # 找出缺失的数字
            min_num = min(numbers)
            max_num = max(numbers)
            expected = set(range(min_num, max_num + 1))
            missing_nums = sorted(expected - numbers)
            
            if missing_nums:
                missing[aa_dir.name] = missing_nums
    
    return missing


def build_url(directory: str, file_num: int) -> str:
    """构建文件的 URL。
    
    Args:
        directory: 目录名（如 "aa01"）
        file_num: 文件数字
    
    Returns:
        完整的 URL
    """
    filename = f"{file_num:03d}.html"
    return f"{BASE_URL}{directory}/{filename}"


def normalize_url(url: str) -> str:
    """规范化 URL。"""
    if "#" in url:
        url = url.split("#")[0]
    url = url.rstrip("/")
    parsed = urlparse(url)
    normalized = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
    if parsed.query:
        normalized += f"?{parsed.query}"
    return normalized


def save_html(url: str, content: bytes, output_dir: Path) -> Path:
    """保存 HTML 内容到文件。"""
    parsed = urlparse(url)
    path_parts = parsed.path.strip("/").split("/")
    
    if path_parts and path_parts[-1]:
        filename = path_parts[-1]
    else:
        filename = "index.html"
    
    if not filename.endswith((".html", ".htm")):
        filename += ".html"
    
    if len(path_parts) > 1:
        subdir = output_dir / "/".join(path_parts[:-1])
        subdir.mkdir(parents=True, exist_ok=True)
        filepath = subdir / filename
    else:
        filepath = output_dir / filename
    
    filepath.write_bytes(content)
    return filepath


def fetch_file(session: requests.Session, url: str) -> tuple[bool, Optional[bytes], Optional[str]]:
    """抓取单个文件。
    
    Returns:
        (是否成功, 内容, 错误信息)
    """
    for attempt in range(MAX_RETRIES):
        try:
            response = session.get(url, timeout=REQUEST_TIMEOUT)
            if response.status_code == 404:
                return False, None, "404 Not Found"
            response.raise_for_status()
            return True, response.content, None
        except requests.RequestException as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(REQUEST_DELAY * (attempt + 1))
            else:
                return False, None, str(e)
    
    return False, None, "Max retries exceeded"


def main():
    """主函数"""
    # 确保输出目录存在
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    
    # 配置日志
    Path("logs").mkdir(exist_ok=True)
    logger.add(
        "logs/fetch_missing_{time}.log",
        rotation="10 MB",
        level="INFO",
        encoding="utf-8"
    )
    
    logger.info("=" * 80)
    logger.info("开始补充抓取缺失的文件")
    logger.info("=" * 80)
    
    # 找出所有缺失的文件
    logger.info("正在分析缺失的文件...")
    missing = find_missing_files(DATA_RAW_DIR)
    
    if not missing:
        logger.info("没有发现缺失的文件！")
        return
    
    # 统计
    total_missing = sum(len(nums) for nums in missing.values())
    logger.info(f"发现 {len(missing)} 个目录有缺失，共 {total_missing} 个文件需要抓取")
    
    # 创建会话
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    })
    
    # 抓取统计
    success_count = 0
    failed_count = 0
    not_found_count = 0
    failed_urls = []
    not_found_urls = []
    
    # 按目录抓取
    with tqdm(total=total_missing, desc="抓取进度", unit="文件") as pbar:
        for dir_name in sorted(missing.keys()):
            missing_nums = missing[dir_name]
            logger.info(f"\n处理目录: {dir_name} ({len(missing_nums)} 个缺失文件)")
            
            for file_num in missing_nums:
                url = build_url(dir_name, file_num)
                normalized_url = normalize_url(url)
                
                # 检查文件是否已存在（可能已经下载了）
                parsed = urlparse(normalized_url)
                path_parts = parsed.path.strip("/").split("/")
                if path_parts and path_parts[-1]:
                    filename = path_parts[-1]
                    if len(path_parts) > 1:
                        filepath = DATA_RAW_DIR / "/".join(path_parts[:-1]) / filename
                    else:
                        filepath = DATA_RAW_DIR / filename
                    
                    if filepath.exists() and filepath.stat().st_size > 0:
                        logger.debug(f"文件已存在，跳过: {url}")
                        success_count += 1
                        pbar.update(1)
                        continue
                
                # 抓取文件
                success, content, error = fetch_file(session, normalized_url)
                
                if success and content:
                    # 保存文件
                    try:
                        save_html(normalized_url, content, DATA_RAW_DIR)
                        success_count += 1
                        logger.debug(f"✓ 成功: {url}")
                    except Exception as e:
                        failed_count += 1
                        failed_urls.append((url, str(e)))
                        logger.warning(f"✗ 保存失败: {url} - {e}")
                elif error == "404 Not Found":
                    not_found_count += 1
                    not_found_urls.append(url)
                    logger.debug(f"✗ 404: {url}")
                else:
                    failed_count += 1
                    failed_urls.append((url, error))
                    logger.warning(f"✗ 失败: {url} - {error}")
                
                pbar.update(1)
                
                # 延迟
                time.sleep(REQUEST_DELAY)
    
    # 输出统计
    logger.info("\n" + "=" * 80)
    logger.info("抓取完成！")
    logger.info(f"  总缺失文件: {total_missing}")
    logger.info(f"  成功抓取: {success_count}")
    logger.info(f"  404 未找到: {not_found_count}")
    logger.info(f"  其他失败: {failed_count}")
    logger.info("=" * 80)
    
    # 保存失败列表
    if not_found_urls:
        not_found_file = Path("logs/404_urls.txt")
        with open(not_found_file, "w", encoding="utf-8") as f:
            f.write("\n".join(not_found_urls))
        logger.info(f"\n404 URL 列表已保存到: {not_found_file}")
    
    if failed_urls:
        failed_file = Path("logs/failed_urls.txt")
        with open(failed_file, "w", encoding="utf-8") as f:
            for url, error in failed_urls:
                f.write(f"{url}\t{error}\n")
        logger.info(f"失败 URL 列表已保存到: {failed_file}")
    
    # 如果有404，说明这些文件在网站上确实不存在
    if not_found_count > 0:
        logger.warning(f"\n注意: 有 {not_found_count} 个文件返回 404，这些文件可能在网站上不存在")


if __name__ == "__main__":
    main()
