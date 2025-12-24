"""
Database Manager: ChromaDB 向量数据库管理
"""
import chromadb
from chromadb.config import Settings
import os
from typing import List, Dict, Optional
import config
from core.embedding import get_embedding_engine

class DatabaseManager:
    """
    向量数据库管理器，负责 ChromaDB 的增删改查
    """
    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            db_path = config.DB_PATH
        
        self.db_path = db_path
        
        # 确保数据库目录存在
        os.makedirs(db_path, exist_ok=True)
        
        # 初始化 ChromaDB 客户端
        self.client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # 论文集合
        self.papers_collection = None
        # 图像集合
        self.images_collection = None
        
        # 获取嵌入引擎
        self.embedding_engine = get_embedding_engine()
    
    def get_or_create_papers_collection(self):
        """获取或创建论文集合"""
        if self.papers_collection is None:
            self.papers_collection = self.client.get_or_create_collection(
                name="papers",
                metadata={"description": "学术论文向量数据库"}
            )
        return self.papers_collection
    
    def get_or_create_images_collection(self):
        """获取或创建图像集合"""
        if self.images_collection is None:
            self.images_collection = self.client.get_or_create_collection(
                name="images",
                metadata={"description": "图像向量数据库"}
            )
        return self.images_collection
    
    def add_paper(self, paper_id: str, chunks: List[str], 
                 metadata: Dict, category: str):
        """
        添加论文到数据库
        Args:
            paper_id: 论文唯一标识符（通常为文件名）
            chunks: 论文文本切片列表
            metadata: 元数据（文件路径、标题等）
            category: 分类标签
        """
        collection = self.get_or_create_papers_collection()
        
        # 为每个 chunk 生成唯一 ID
        chunk_ids = [f"{paper_id}_chunk_{i}" for i in range(len(chunks))]
        
        # 生成嵌入向量
        print(f"正在为论文 {paper_id} 生成嵌入向量...")
        embeddings = self.embedding_engine.encode_text(chunks)
        
        # 准备元数据
        metadatas = []
        for i, chunk in enumerate(chunks):
            meta = metadata.copy()
            meta['chunk_index'] = i
            meta['category'] = category
            meta['paper_id'] = paper_id
            metadatas.append(meta)
        
        # 添加到数据库
        collection.add(
            ids=chunk_ids,
            embeddings=embeddings.tolist(),
            documents=chunks,
            metadatas=metadatas
        )
        
        print(f"论文 {paper_id} 已添加到数据库（{len(chunks)} 个切片）")
    
    def search_papers(self, query: str, n_results: int = None) -> Dict:
        """
        搜索论文
        Args:
            query: 查询文本
            n_results: 返回结果数量
        Returns:
            检索结果字典
        """
        if n_results is None:
            n_results = config.RETRIEVAL_TOP_K
        
        collection = self.get_or_create_papers_collection()
        
        # 生成查询向量
        query_embedding = self.embedding_engine.encode_text(query)
        
        # 检索
        results = collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=n_results
        )
        
        return results
    
    def add_images(self, image_paths: List[str], descriptions: Optional[List[str]] = None):
        """
        批量添加图像到数据库
        Args:
            image_paths: 图像路径列表
            descriptions: 图像描述列表（可选）
        """
        collection = self.get_or_create_images_collection()
        
        # 生成图像嵌入向量
        print(f"正在为 {len(image_paths)} 张图像生成嵌入向量...")
        embeddings = self.embedding_engine.encode_image(image_paths)
        
        # 生成 ID
        ids = [os.path.basename(path) for path in image_paths]
        
        # 准备元数据
        metadatas = []
        for i, path in enumerate(image_paths):
            meta = {
                'file_path': path,
                'file_name': os.path.basename(path)
            }
            if descriptions and i < len(descriptions):
                meta['description'] = descriptions[i]
            metadatas.append(meta)
        
        # 使用文件路径作为文档（方便后续检索显示）
        documents = image_paths
        
        # 添加到数据库
        collection.add(
            ids=ids,
            embeddings=embeddings.tolist(),
            documents=documents,
            metadatas=metadatas
        )
        
        print(f"{len(image_paths)} 张图像已添加到数据库")
    
    def search_images(self, query: str, n_results: int = 5) -> Dict:
        """
        以文搜图
        Args:
            query: 查询文本
            n_results: 返回结果数量
        Returns:
            检索结果字典
        """
        collection = self.get_or_create_images_collection()
        
        # 使用 CLIP 的文本编码器生成查询向量
        query_embedding = self.embedding_engine.encode_text_for_image_search(query)
        
        # 检索
        results = collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=n_results
        )
        
        return results
    
    def delete_paper(self, paper_id: str):
        """删除论文"""
        collection = self.get_or_create_papers_collection()
        
        # 查找所有相关的 chunk IDs
        results = collection.get(where={"paper_id": paper_id})
        
        if results['ids']:
            collection.delete(ids=results['ids'])
            print(f"论文 {paper_id} 已从数据库中删除")
        else:
            print(f"未找到论文 {paper_id}")
    
    def get_all_papers(self) -> List[Dict]:
        """获取所有论文的元数据"""
        collection = self.get_or_create_papers_collection()
        
        # 获取所有数据
        results = collection.get()
        
        # 按 paper_id 去重
        papers_dict = {}
        for i, paper_id in enumerate(results['metadatas']):
            pid = paper_id.get('paper_id')
            if pid and pid not in papers_dict:
                papers_dict[pid] = results['metadatas'][i]
        
        return list(papers_dict.values())
    
    def reset_database(self):
        """重置数据库（删除所有数据）"""
        self.client.reset()
        print("数据库已重置")
        self.papers_collection = None
        self.images_collection = None


# 全局单例
_db_manager = None

def get_db_manager() -> DatabaseManager:
    """获取数据库管理器单例"""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager
