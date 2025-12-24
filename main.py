#!/usr/bin/env python3
"""
Local Multimodal AI Agent - 本地多模态智能助手
主入口文件：命令行参数解析与功能分发

支持的命令：
1. add_paper: 添加并分类论文
2. search_paper: 搜索论文（语义搜索）
3. ask_paper: 基于论文回答问题（RAG）
4. index_images: 索引图片库
5. search_image: 以文搜图
6. chat_image: 与图像对话（多模态）
7. organize_library: 批量整理论文库
8. info: 查看系统信息
"""

import argparse
import os
import sys

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from core.llm_engine import get_llm_engine
from core.vlm_engine import get_vlm_engine
from core.embedding import get_embedding_engine
from core.db_manager import get_db_manager
from utils.pdf_parser import extract_text_snippet, extract_and_chunk_pdf
from utils.file_ops import move_file, scan_files
from utils.gpu_utils import clear_vram, print_gpu_memory_info, check_cuda_available


def add_paper_command(args):
    """添加并分类论文"""
    print(f"\n{'='*60}")
    print("功能：添加并分类论文")
    print(f"{'='*60}\n")
    
    pdf_path = args.path
    
    # 检查文件是否存在
    if not os.path.exists(pdf_path):
        print(f"错误：文件不存在 - {pdf_path}")
        return
    
    # 检查是否为 PDF 文件
    if not pdf_path.lower().endswith('.pdf'):
        print(f"错误：只支持 PDF 文件")
        return
    
    print(f"正在处理论文: {os.path.basename(pdf_path)}")
    
    # 1. 提取文本片段用于分类
    print("\n[1/4] 提取论文内容...")
    text_snippet = extract_text_snippet(pdf_path)
    
    if not text_snippet:
        print("错误：无法从 PDF 中提取文本")
        return
    
    print(f"提取了 {len(text_snippet)} 个字符")
    
    # 2. 使用 LLM 进行分类
    print("\n[2/4] 使用 LLM 进行分类...")
    llm = get_llm_engine()
    
    # 解析用户指定的主题（如果有）
    if args.topics:
        topics = [t.strip() for t in args.topics.split(',')]
    else:
        topics = config.PAPER_TOPICS
    
    category = llm.classify_paper(text_snippet, topics)
    print(f"分类结果: {category}")
    
    # 3. 移动文件到对应目录
    print("\n[3/4] 移动文件...")
    target_dir = os.path.join(config.LIBRARY_PATH, category)
    new_path = move_file(pdf_path, target_dir)
    
    # 4. 提取全文并存入向量数据库
    print("\n[4/4] 建立向量索引...")
    full_text, chunks = extract_and_chunk_pdf(new_path)
    
    if not chunks:
        print("警告：未能提取到有效文本片段")
        return
    
    db = get_db_manager()
    paper_id = os.path.basename(new_path)
    metadata = {
        'file_path': new_path,
        'file_name': paper_id,
        'category': category
    }
    
    db.add_paper(paper_id, chunks, metadata, category)
    
    print(f"\n✓ 论文处理完成！")
    print(f"  分类: {category}")
    print(f"  位置: {new_path}")
    print(f"  索引片段数: {len(chunks)}")


def search_paper_command(args):
    """搜索论文"""
    print(f"\n{'='*60}")
    print("功能：语义搜索论文")
    print(f"{'='*60}\n")
    
    query = args.query
    print(f"查询: {query}\n")
    
    # 初始化嵌入模型和数据库
    db = get_db_manager()
    
    # 执行检索
    print("正在检索...")
    results = db.search_papers(query, n_results=args.top_k)
    
    if not results['documents'] or len(results['documents'][0]) == 0:
        print("未找到相关论文")
        return
    
    # 显示结果
    print(f"\n找到 {len(results['documents'][0])} 个相关片段:\n")
    
    for i, (doc, meta, distance) in enumerate(zip(
        results['documents'][0], 
        results['metadatas'][0],
        results['distances'][0]
    )):
        print(f"[{i+1}] 相似度: {1 - distance:.4f}")
        print(f"    论文: {meta['file_name']}")
        print(f"    分类: {meta['category']}")
        print(f"    片段 {meta['chunk_index']}: {doc[:200]}...")
        print()


def ask_paper_command(args):
    """基于论文回答问题（RAG）"""
    print(f"\n{'='*60}")
    print("功能：基于论文回答问题 (RAG)")
    print(f"{'='*60}\n")
    
    question = args.question
    print(f"问题: {question}\n")
    
    # 1. 检索相关文档
    print("[1/2] 检索相关文献...")
    db = get_db_manager()
    results = db.search_papers(question, n_results=args.top_k)
    
    if not results['documents'] or len(results['documents'][0]) == 0:
        print("未找到相关论文，无法回答")
        return
    
    # 2. 构造上下文
    context = "\n\n".join(results['documents'][0])
    
    print(f"找到 {len(results['documents'][0])} 个相关片段")
    
    # 3. 使用 LLM 生成答案
    print("\n[2/2] 生成答案...\n")
    llm = get_llm_engine()
    answer = llm.answer_question(question, context)
    
    print("回答:")
    print(f"{answer}\n")
    
    # 4. 显示参考来源
    print("参考来源:")
    for i, meta in enumerate(results['metadatas'][0][:3]):
        print(f"  [{i+1}] {meta['file_name']} (分类: {meta['category']})")


def index_images_command(args):
    """索引图片库"""
    print(f"\n{'='*60}")
    print("功能：索引图片库")
    print(f"{'='*60}\n")
    
    image_dir = args.path if args.path else config.IMAGES_PATH
    
    if not os.path.exists(image_dir):
        print(f"错误：目录不存在 - {image_dir}")
        return
    
    # 扫描图片文件
    print(f"正在扫描目录: {image_dir}")
    image_files = scan_files(image_dir, ['.jpg', '.jpeg', '.png', '.bmp', '.gif'])
    
    if not image_files:
        print("未找到图片文件")
        return
    
    print(f"找到 {len(image_files)} 张图片\n")
    
    # 批量添加到数据库
    db = get_db_manager()
    
    # 分批处理（避免一次性加载太多图片）
    batch_size = 50
    for i in range(0, len(image_files), batch_size):
        batch = image_files[i:i+batch_size]
        print(f"正在处理第 {i+1}-{min(i+batch_size, len(image_files))} 张图片...")
        db.add_images(batch)
    
    print(f"\n✓ 图片索引完成！共索引 {len(image_files)} 张图片")


def search_image_command(args):
    """以文搜图"""
    print(f"\n{'='*60}")
    print("功能：以文搜图")
    print(f"{'='*60}\n")
    
    query = args.query
    print(f"查询: {query}\n")
    
    # 初始化数据库
    db = get_db_manager()
    
    # 执行检索
    print("正在检索...")
    results = db.search_images(query, n_results=args.top_k)
    
    if not results['documents'] or len(results['documents'][0]) == 0:
        print("未找到相关图片")
        return
    
    # 显示结果
    print(f"\n找到 {len(results['documents'][0])} 张相关图片:\n")
    
    for i, (image_path, meta, distance) in enumerate(zip(
        results['documents'][0],
        results['metadatas'][0],
        results['distances'][0]
    )):
        print(f"[{i+1}] 相似度: {1 - distance:.4f}")
        print(f"    路径: {image_path}")
        print(f"    文件名: {meta['file_name']}")
        if 'description' in meta:
            print(f"    描述: {meta['description']}")
        print()


def chat_image_command(args):
    """与图像对话（多模态）"""
    print(f"\n{'='*60}")
    print("功能：与图像对话 (多模态 VLM)")
    print(f"{'='*60}\n")
    
    image_path = args.image
    question = args.question
    
    # 检查文件是否存在
    if not os.path.exists(image_path):
        print(f"错误：图片不存在 - {image_path}")
        return
    
    print(f"图片: {image_path}")
    print(f"问题: {question}\n")
    
    # 清理显存（卸载可能已加载的 LLM）
    clear_vram()
    
    # 使用 VLM 进行推理
    print("正在分析图片...")
    vlm = get_vlm_engine()
    answer = vlm.chat_with_image(image_path, question)
    
    print("\n回答:")
    print(answer)


def organize_library_command(args):
    """批量整理论文库"""
    print(f"\n{'='*60}")
    print("功能：批量整理论文库")
    print(f"{'='*60}\n")
    
    source_dir = args.path
    
    if not os.path.exists(source_dir):
        print(f"错误：目录不存在 - {source_dir}")
        return
    
    # 扫描 PDF 文件
    print(f"正在扫描目录: {source_dir}")
    pdf_files = scan_files(source_dir, ['.pdf'])
    
    if not pdf_files:
        print("未找到 PDF 文件")
        return
    
    print(f"找到 {len(pdf_files)} 篇论文\n")
    
    # 解析主题
    if args.topics:
        topics = [t.strip() for t in args.topics.split(',')]
    else:
        topics = config.PAPER_TOPICS
    
    print(f"分类主题: {', '.join(topics)}\n")
    
    # 初始化 LLM
    llm = get_llm_engine()
    db = get_db_manager()
    
    # 逐个处理
    success_count = 0
    for i, pdf_path in enumerate(pdf_files):
        print(f"\n[{i+1}/{len(pdf_files)}] 处理: {os.path.basename(pdf_path)}")
        
        try:
            # 提取文本
            text_snippet = extract_text_snippet(pdf_path)
            if not text_snippet:
                print("  ✗ 无法提取文本，跳过")
                continue
            
            # 分类
            category = llm.classify_paper(text_snippet, topics)
            print(f"  分类: {category}")
            
            # 移动文件
            target_dir = os.path.join(config.LIBRARY_PATH, category)
            new_path = move_file(pdf_path, target_dir)
            
            # 索引
            full_text, chunks = extract_and_chunk_pdf(new_path)
            if chunks:
                paper_id = os.path.basename(new_path)
                metadata = {
                    'file_path': new_path,
                    'file_name': paper_id,
                    'category': category
                }
                db.add_paper(paper_id, chunks, metadata, category)
                print(f"  ✓ 完成（{len(chunks)} 个片段）")
                success_count += 1
            else:
                print("  ✗ 无法提取文本片段")
        
        except Exception as e:
            print(f"  ✗ 处理失败: {e}")
    
    print(f"\n{'='*60}")
    print(f"整理完成！成功: {success_count}/{len(pdf_files)}")
    print(f"{'='*60}")


def info_command(args):
    """查看系统信息"""
    print(f"\n{'='*60}")
    print("系统信息")
    print(f"{'='*60}\n")
    
    # CUDA 信息
    print("1. CUDA 状态:")
    check_cuda_available()
    print()
    
    # 显存信息
    print("2. GPU 显存:")
    print_gpu_memory_info()
    print()
    
    # 配置信息
    print("3. 模型配置:")
    print(f"  LLM: {config.LLM_MODEL_PATH}")
    print(f"  VLM: {config.VLM_MODEL_PATH}")
    print(f"  Text Embedding: {config.TEXT_EMBEDDING_MODEL}")
    print(f"  Image Embedding: {config.IMAGE_EMBEDDING_MODEL}")
    print()
    
    # 数据库信息
    print("4. 数据库统计:")
    db = get_db_manager()
    
    try:
        papers_collection = db.get_or_create_papers_collection()
        paper_count = papers_collection.count()
        print(f"  论文片段数: {paper_count}")
    except:
        print(f"  论文片段数: 0")
    
    try:
        images_collection = db.get_or_create_images_collection()
        image_count = images_collection.count()
        print(f"  图片数: {image_count}")
    except:
        print(f"  图片数: 0")
    
    print()
    
    # 路径信息
    print("5. 路径配置:")
    print(f"  论文库: {config.LIBRARY_PATH}")
    print(f"  图片库: {config.IMAGES_PATH}")
    print(f"  数据库: {config.DB_PATH}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="Local Multimodal AI Agent - 本地多模态智能助手",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 添加并分类论文
  python main.py add_paper paper.pdf
  
  # 搜索论文
  python main.py search_paper "Transformer architecture"
  
  # 基于论文回答问题
  python main.py ask_paper "What is attention mechanism?"
  
  # 索引图片库
  python main.py index_images ./Images
  
  # 以文搜图
  python main.py search_image "sunset by the sea"
  
  # 与图像对话
  python main.py chat_image image.jpg "Describe this image"
  
  # 批量整理论文库
  python main.py organize_library ./messy_papers
  
  # 查看系统信息
  python main.py info
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # add_paper 命令
    parser_add = subparsers.add_parser('add_paper', help='添加并分类论文')
    parser_add.add_argument('path', help='PDF 文件路径')
    parser_add.add_argument('--topics', help='候选主题（逗号分隔）', default=None)
    
    # search_paper 命令
    parser_search = subparsers.add_parser('search_paper', help='搜索论文')
    parser_search.add_argument('query', help='搜索查询')
    parser_search.add_argument('--top_k', type=int, default=5, help='返回结果数')
    
    # ask_paper 命令
    parser_ask = subparsers.add_parser('ask_paper', help='基于论文回答问题')
    parser_ask.add_argument('question', help='问题')
    parser_ask.add_argument('--top_k', type=int, default=5, help='检索片段数')
    
    # index_images 命令
    parser_index = subparsers.add_parser('index_images', help='索引图片库')
    parser_index.add_argument('path', nargs='?', help='图片目录路径', default=None)
    
    # search_image 命令
    parser_img_search = subparsers.add_parser('search_image', help='以文搜图')
    parser_img_search.add_argument('query', help='搜索查询')
    parser_img_search.add_argument('--top_k', type=int, default=5, help='返回结果数')
    
    # chat_image 命令
    parser_chat = subparsers.add_parser('chat_image', help='与图像对话')
    parser_chat.add_argument('image', help='图片路径')
    parser_chat.add_argument('question', help='问题')
    
    # organize_library 命令
    parser_organize = subparsers.add_parser('organize_library', help='批量整理论文库')
    parser_organize.add_argument('path', help='源目录路径')
    parser_organize.add_argument('--topics', help='候选主题（逗号分隔）', default=None)
    
    # info 命令
    parser_info = subparsers.add_parser('info', help='查看系统信息')
    
    # 解析参数
    args = parser.parse_args()
    
    # 如果没有指定命令，显示帮助
    if not args.command:
        parser.print_help()
        return
    
    # 执行对应命令
    command_map = {
        'add_paper': add_paper_command,
        'search_paper': search_paper_command,
        'ask_paper': ask_paper_command,
        'index_images': index_images_command,
        'search_image': search_image_command,
        'chat_image': chat_image_command,
        'organize_library': organize_library_command,
        'info': info_command
    }
    
    try:
        command_map[args.command](args)
    except KeyboardInterrupt:
        print("\n\n操作已取消")
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
