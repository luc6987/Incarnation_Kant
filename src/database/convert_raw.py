"""Raw HTML to TXT converter for Kant corpus.

Extracts text from table-based HTML files in data/raw/kant and saves them as line-based TXT files.
Preserves line numbers and page structure.
"""

import os
import re
from pathlib import Path
from bs4 import BeautifulSoup
from loguru import logger
from tqdm import tqdm

def extract_text_from_html(html_path: Path) -> str:
    """Extract text from a Kant HTML file.
    
    Args:
        html_path: Path to the HTML file.
        
    Returns:
        String containing line-by-line text, format "LineNo: Text".
    """
    try:
        with open(html_path, 'r', encoding='utf-8', errors='replace') as f:
            soup = BeautifulSoup(f, 'lxml')
            
        lines = []
        
        # Find all rows in the table
        rows = soup.find_all('tr')
        
        for row in rows:
            # Look for the cell containing the line number
            # Structure: <td>...LineNo...</td> <td>...Content...</td>
            cells = row.find_all('td')
            if not cells:
                continue
                
            line_no = None
            content = None
            
            # Iterate cells to find line number and content
            for i, cell in enumerate(cells):
                cell_text = cell.get_text(strip=True)
                
                # Check if this cell looks like a line number (digits, optionally followed by anchor)
                # In the HTML, it's often "01", "02" etc. inside a font tag.
                if re.match(r'^\d+$', cell_text):
                    line_no = cell_text
                    
                    # The content is usually in the next cell (or the one after next spacer?)
                    # In the observed HTML, the content cell has colspan="3" and width="60%"
                    # We look for the next cell with substantial content
                    if i + 1 < len(cells):
                        content_cell = cells[i+1]
                        # Extract text, preserving some structure? No, just pure text.
                        # Replace newlines with space to keep it one line
                        content = content_cell.get_text(" ", strip=True)
                    break
            
            if line_no and content:
                lines.append(f"{line_no}: {content}")
                
        return "\n".join(lines)
        
    except Exception as e:
        logger.error(f"Error parsing {html_path}: {e}")
        return ""

def convert_all_raw_files(raw_dir: str = "data/raw/kant", output_dir: str = "data/txt/kant"):
    """Convert all raw HTML files to TXT.
    
    Args:
        raw_dir: Source directory containing aaXX subfolders.
        output_dir: Destination directory.
    """
    raw_path = Path(raw_dir)
    out_path = Path(output_dir)
    
    if not raw_path.exists():
        logger.error(f"Raw directory not found: {raw_dir}")
        return
        
    # Find all HTML files
    html_files = sorted(list(raw_path.glob("aa*/*.html")))
    logger.info(f"Found {len(html_files)} HTML files to convert.")
    
    success_count = 0
    
    for html_file in tqdm(html_files, desc="Converting"):
        # Determine output path
        rel_path = html_file.relative_to(raw_path)
        txt_rel_path = rel_path.with_suffix('.txt')
        txt_path = out_path / txt_rel_path
        
        # Create parent directory
        txt_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Extract and write
        text_content = extract_text_from_html(html_file)
        
        if text_content:
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(text_content)
            success_count += 1
            
    logger.success(f"Successfully converted {success_count} files to {output_dir}")

if __name__ == "__main__":
    convert_all_raw_files()
