"""
PDF Parser: PDF 文本提取与切片
"""
import fitz  # PyMuPDF
from typing import List, Tuple
import config

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    从 PDF 中提取全文本
    Args:
        pdf_path: PDF 文件路径
    Returns:
        提取的文本
    """
    try:
        doc = fitz.open(pdf_path)
        text = ""
        
        for page in doc:
            text += page.get_text()
        
        doc.close()
        return text.strip()
    
    except Exception as e:
        print(f"提取 PDF 文本失败: {e}")
        return ""

def extract_text_snippet(pdf_path: str, max_chars: int = None) -> str:
    """
    提取 PDF 的前 N 个字符（用于分类）
    Args:
        pdf_path: PDF 文件路径
        max_chars: 最大字符数
    Returns:
        文本片段
    """
    if max_chars is None:
        max_chars = config.PDF_EXTRACT_MAX_CHARS
    
    full_text = extract_text_from_pdf(pdf_path)
    return full_text[:max_chars]

def split_text_into_chunks(text: str, chunk_size: int = None, 
                          chunk_overlap: int = None) -> List[str]:
    """
    将长文本切分成固定大小的片段
    Args:
        text: 输入文本
        chunk_size: 每个片段的大小
        chunk_overlap: 片段之间的重叠大小
    Returns:
        文本片段列表
    """
    if chunk_size is None:
        chunk_size = config.CHUNK_SIZE
    
    if chunk_overlap is None:
        chunk_overlap = config.CHUNK_OVERLAP
    
    if len(text) == 0:
        return []
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        
        # 只添加非空 chunk
        if chunk.strip():
            chunks.append(chunk)
        
        # 移动到下一个片段（考虑重叠）
        start = end - chunk_overlap
        
        # 防止无限循环
        if start >= len(text):
            break
    
    return chunks

def extract_and_chunk_pdf(pdf_path: str) -> Tuple[str, List[str]]:
    """
    提取 PDF 并切分成片段
    Args:
        pdf_path: PDF 文件路径
    Returns:
        (完整文本, 文本片段列表)
    """
    full_text = extract_text_from_pdf(pdf_path)
    chunks = split_text_into_chunks(full_text)
    
    return full_text, chunks

def get_pdf_metadata(pdf_path: str) -> dict:
    """
    获取 PDF 元数据
    Args:
        pdf_path: PDF 文件路径
    Returns:
        元数据字典
    """
    try:
        doc = fitz.open(pdf_path)
        metadata = doc.metadata
        
        # 基础信息
        result = {
            'title': metadata.get('title', ''),
            'author': metadata.get('author', ''),
            'subject': metadata.get('subject', ''),
            'keywords': metadata.get('keywords', ''),
            'creator': metadata.get('creator', ''),
            'producer': metadata.get('producer', ''),
            'num_pages': len(doc)
        }
        
        doc.close()
        return result
    
    except Exception as e:
        print(f"获取 PDF 元数据失败: {e}")
        return {}
