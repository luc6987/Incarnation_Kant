#!/usr/bin/env python3
"""
检查 data/raw/kant/aa** 目录下文件的完整性
找出缺失的数字序列
"""

import re
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set

def extract_number(filename: str) -> int:
    """从文件名中提取数字。
    
    Args:
        filename: 文件名（如 "001.html", "016.html"）
    
    Returns:
        提取的数字，如果无法提取返回 -1
    """
    match = re.search(r'(\d+)', filename)
    if match:
        return int(match.group(1))
    return -1

def check_directory(directory: Path) -> Dict[str, Set[int]]:
    """检查目录中的文件完整性。
    
    Args:
        directory: 要检查的目录路径
    
    Returns:
        字典：{目录名: {存在的文件数字集合}}
    """
    results = {}
    
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
            results[aa_dir.name] = numbers
    
    return results

def find_gaps(numbers: Set[int]) -> List[tuple[int, int]]:
    """找出数字序列中的空缺。
    
    Args:
        numbers: 数字集合
    
    Returns:
        空缺范围列表，每个元组表示 (起始, 结束)，如果只缺一个数字，起始==结束
    """
    if not numbers:
        return []
    
    sorted_nums = sorted(numbers)
    gaps = []
    
    # 检查从最小数字到最大数字之间的空缺
    min_num = sorted_nums[0]
    max_num = sorted_nums[-1]
    
    expected = set(range(min_num, max_num + 1))
    missing = expected - numbers
    
    if missing:
        # 将连续的空缺合并为范围
        missing_sorted = sorted(missing)
        start = missing_sorted[0]
        end = missing_sorted[0]
        
        for num in missing_sorted[1:]:
            if num == end + 1:
                end = num
            else:
                gaps.append((start, end))
                start = num
                end = num
        gaps.append((start, end))
    
    return gaps

def format_gap(gap: tuple[int, int]) -> str:
    """格式化空缺范围。
    
    Args:
        gap: 空缺范围元组
    
    Returns:
        格式化的字符串
    """
    start, end = gap
    if start == end:
        return f"{start:03d}"
    else:
        return f"{start:03d}-{end:03d}"

def main():
    """主函数"""
    data_dir = Path("data/raw/kant")
    
    if not data_dir.exists():
        print(f"错误: 目录不存在: {data_dir}")
        return
    
    print("=" * 80)
    print("检查 data/raw/kant/aa** 目录下的文件完整性")
    print("=" * 80)
    print()
    
    # 检查所有目录
    results = check_directory(data_dir)
    
    if not results:
        print("未找到任何 aa** 目录")
        return
    
    # 统计信息
    total_dirs = len(results)
    total_files = sum(len(nums) for nums in results.values())
    dirs_with_gaps = 0
    total_gaps = 0
    
    # 检查每个目录
    print(f"找到 {total_dirs} 个 aa** 目录\n")
    
    for dir_name in sorted(results.keys()):
        numbers = results[dir_name]
        gaps = find_gaps(numbers)
        
        if gaps:
            dirs_with_gaps += 1
            total_gaps += sum(end - start + 1 for start, end in gaps)
            
            print(f"📁 {dir_name}:")
            print(f"   文件数: {len(numbers)}")
            print(f"   范围: {min(numbers):03d} - {max(numbers):03d}")
            print(f"   缺失: {', '.join(format_gap(g) for g in gaps)}")
            print()
        else:
            print(f"✅ {dir_name}: 完整 ({len(numbers)} 个文件, {min(numbers):03d}-{max(numbers):03d})")
    
    print("=" * 80)
    print("统计摘要:")
    print(f"  总目录数: {total_dirs}")
    print(f"  总文件数: {total_files}")
    print(f"  有缺失的目录: {dirs_with_gaps}")
    print(f"  总缺失文件数: {total_gaps}")
    print("=" * 80)
    
    # 生成详细报告
    print("\n详细缺失报告:")
    print("-" * 80)
    for dir_name in sorted(results.keys()):
        numbers = results[dir_name]
        gaps = find_gaps(numbers)
        if gaps:
            print(f"\n{dir_name}:")
            for start, end in gaps:
                if start == end:
                    print(f"  缺失: {start:03d}.html")
                else:
                    missing_list = [f"{i:03d}.html" for i in range(start, end + 1)]
                    print(f"  缺失: {', '.join(missing_list[:10])}")
                    if len(missing_list) > 10:
                        print(f"        ... 还有 {len(missing_list) - 10} 个文件")

if __name__ == "__main__":
    main()
