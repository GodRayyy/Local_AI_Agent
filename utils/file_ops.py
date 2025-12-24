"""
File Operations: 文件移动、路径管理等工具函数
"""
import os
import shutil
from typing import List
from pathlib import Path

def ensure_dir_exists(dir_path: str):
    """确保目录存在，如果不存在则创建"""
    os.makedirs(dir_path, exist_ok=True)

def move_file(src_path: str, dst_dir: str, create_dir: bool = True) -> str:
    """
    移动文件到目标目录
    Args:
        src_path: 源文件路径
        dst_dir: 目标目录
        create_dir: 如果目录不存在是否创建
    Returns:
        新的文件路径
    """
    if create_dir:
        ensure_dir_exists(dst_dir)
    
    file_name = os.path.basename(src_path)
    dst_path = os.path.join(dst_dir, file_name)
    
    # 如果目标文件已存在，添加编号
    base_name, ext = os.path.splitext(file_name)
    counter = 1
    while os.path.exists(dst_path):
        new_name = f"{base_name}_{counter}{ext}"
        dst_path = os.path.join(dst_dir, new_name)
        counter += 1
    
    shutil.move(src_path, dst_path)
    print(f"文件已移动: {src_path} -> {dst_path}")
    
    return dst_path

def copy_file(src_path: str, dst_dir: str, create_dir: bool = True) -> str:
    """
    复制文件到目标目录
    Args:
        src_path: 源文件路径
        dst_dir: 目标目录
        create_dir: 如果目录不存在是否创建
    Returns:
        新的文件路径
    """
    if create_dir:
        ensure_dir_exists(dst_dir)
    
    file_name = os.path.basename(src_path)
    dst_path = os.path.join(dst_dir, file_name)
    
    # 如果目标文件已存在，添加编号
    base_name, ext = os.path.splitext(file_name)
    counter = 1
    while os.path.exists(dst_path):
        new_name = f"{base_name}_{counter}{ext}"
        dst_path = os.path.join(dst_dir, new_name)
        counter += 1
    
    shutil.copy2(src_path, dst_path)
    print(f"文件已复制: {src_path} -> {dst_path}")
    
    return dst_path

def scan_files(directory: str, extensions: List[str]) -> List[str]:
    """
    扫描目录下指定扩展名的所有文件
    Args:
        directory: 目录路径
        extensions: 文件扩展名列表（如 ['.pdf', '.txt']）
    Returns:
        文件路径列表
    """
    if not os.path.exists(directory):
        print(f"目录不存在: {directory}")
        return []
    
    files = []
    for root, dirs, filenames in os.walk(directory):
        for filename in filenames:
            if any(filename.lower().endswith(ext) for ext in extensions):
                files.append(os.path.join(root, filename))
    
    return files

def get_relative_path(path: str, base_path: str) -> str:
    """
    获取相对路径
    Args:
        path: 完整路径
        base_path: 基准路径
    Returns:
        相对路径
    """
    return os.path.relpath(path, base_path)

def organize_files_by_category(files: List[str], categories: dict, 
                               base_dir: str, move: bool = True):
    """
    按分类组织文件
    Args:
        files: 文件路径列表
        categories: 文件到分类的映射 {file_path: category}
        base_dir: 基础目录
        move: True 为移动，False 为复制
    """
    for file_path in files:
        if file_path not in categories:
            continue
        
        category = categories[file_path]
        target_dir = os.path.join(base_dir, category)
        
        if move:
            move_file(file_path, target_dir)
        else:
            copy_file(file_path, target_dir)

def get_file_size(file_path: str) -> int:
    """获取文件大小（字节）"""
    return os.path.getsize(file_path)

def format_file_size(size_bytes: int) -> str:
    """
    格式化文件大小
    Args:
        size_bytes: 字节数
    Returns:
        格式化的字符串（如 "1.5 MB"）
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"
