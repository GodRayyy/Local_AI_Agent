"""
Embedding Engine: 封装文本和图像的嵌入模型
"""
import torch
from sentence_transformers import SentenceTransformer
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
from typing import List, Union
import config
import numpy as np

class EmbeddingEngine:
    """
    嵌入模型引擎，负责文本和图像的向量化
    """
    def __init__(self):
        self.text_model = None
        self.image_model = None
        self.image_processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def load_text_model(self):
        """加载文本嵌入模型（BGE-M3）"""
        if self.text_model is not None:
            print("文本嵌入模型已加载，跳过重复加载")
            return
        
        print(f"正在加载文本嵌入模型: {config.TEXT_EMBEDDING_MODEL}")
        self.text_model = SentenceTransformer(
            config.TEXT_EMBEDDING_MODEL,
            device=self.device
        )
        print("文本嵌入模型加载完成！")
    
    def load_image_model(self):
        """加载图像嵌入模型（CLIP）"""
        if self.image_model is not None:
            print("图像嵌入模型已加载，跳过重复加载")
            return
        
        print(f"正在加载图像嵌入模型: {config.IMAGE_EMBEDDING_MODEL}")
        # 使用 transformers 直接加载 CLIP 模型
        self.image_model = CLIPModel.from_pretrained(config.IMAGE_EMBEDDING_MODEL).to(self.device)
        self.image_processor = CLIPProcessor.from_pretrained(config.IMAGE_EMBEDDING_MODEL)
        print("图像嵌入模型加载完成！")
    
    def encode_text(self, texts: Union[str, List[str]], 
                   normalize: bool = True) -> np.ndarray:
        """
        编码文本为向量
        Args:
            texts: 单个文本或文本列表
            normalize: 是否归一化向量
        Returns:
            文本向量（numpy array）
        """
        if self.text_model is None:
            self.load_text_model()
        
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = self.text_model.encode(
            texts,
            normalize_embeddings=normalize,
            show_progress_bar=False
        )
        
        return embeddings
    
    def encode_image(self, image_paths: Union[str, List[str]], 
                    normalize: bool = True) -> np.ndarray:
        """
        编码图像为向量
        Args:
            image_paths: 单个图像路径或图像路径列表
            normalize: 是否归一化向量
        Returns:
            图像向量（numpy array）
        """
        if self.image_model is None:
            self.load_image_model()
        
        if isinstance(image_paths, str):
            image_paths = [image_paths]
        
        # 加载图像
        images = [Image.open(img_path).convert('RGB') for img_path in image_paths]
        
        # 使用 CLIP processor 处理图像
        inputs = self.image_processor(images=images, return_tensors="pt").to(self.device)
        
        # 获取图像特征
        with torch.no_grad():
            image_features = self.image_model.get_image_features(**inputs)
        
        # 转换为 numpy 数组
        embeddings = image_features.cpu().numpy()
        
        # 归一化
        if normalize:
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        return embeddings
    
    def encode_text_for_image_search(self, texts: Union[str, List[str]], 
                                     normalize: bool = True) -> np.ndarray:
        """
        编码文本用于图像搜索（使用 CLIP 的文本编码器）
        Args:
            texts: 单个文本或文本列表
            normalize: 是否归一化向量
        Returns:
            文本向量（numpy array）
        """
        if self.image_model is None:
            self.load_image_model()
        
        if isinstance(texts, str):
            texts = [texts]
        
        # 使用 CLIP processor 处理文本
        inputs = self.image_processor(text=texts, return_tensors="pt", padding=True).to(self.device)
        
        # 获取文本特征
        with torch.no_grad():
            text_features = self.image_model.get_text_features(**inputs)
        
        # 转换为 numpy 数组
        embeddings = text_features.cpu().numpy()
        
        # 归一化
        if normalize:
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        return embeddings
    
    def unload_models(self):
        """卸载所有模型，释放显存"""
        if self.text_model is not None:
            del self.text_model
            self.text_model = None
        
        if self.image_model is not None:
            del self.image_model
            self.image_model = None
        
        if self.image_processor is not None:
            del self.image_processor
            self.image_processor = None
        
        torch.cuda.empty_cache()
        print("嵌入模型已卸载")


# 全局单例
_embedding_engine = None

def get_embedding_engine() -> EmbeddingEngine:
    """获取嵌入引擎单例"""
    global _embedding_engine
    if _embedding_engine is None:
        _embedding_engine = EmbeddingEngine()
    return _embedding_engine
