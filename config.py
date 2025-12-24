"""
配置文件：模型路径、数据库路径、Prompt模板

使用说明:
1. 修改模型路径部分，指向你本地的模型目录
2. 如果使用不同版本的模型，请确保代码兼容性
3. 可以根据需求调整推理参数（温度、采样等）
"""
import os

# ==================== 路径配置 ====================
# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# 文献库路径（论文将按分类存储在此目录下）
LIBRARY_PATH = os.path.join(PROJECT_ROOT, "Library")

# 图片库路径（待索引的图片存放位置）
IMAGES_PATH = os.path.join(PROJECT_ROOT, "Images")

# 向量数据库路径（ChromaDB 持久化存储位置）
DB_PATH = os.path.join(PROJECT_ROOT, "data", "chroma_db")

# ==================== 模型配置 ====================
# 注意：请根据实际情况修改以下路径

# LLM 模型路径（用于论文分类和问答生成）
LLM_MODEL_PATH = "/data0/tygao/models/Qwen3-4B-Instruct-2507"

# VLM 模型路径（用于图像理解和多模态对话）
VLM_MODEL_PATH = "/data0/tygao/models/Qwen3-VL-4B-Instruct"

# Text Embedding 模型（用于文本语义检索）
TEXT_EMBEDDING_MODEL = "/data0/tygao/models/bge-m3"

# Image Embedding 模型（用于图像语义检索）
IMAGE_EMBEDDING_MODEL = "/data0/tygao/models/clip-vit-large-patch14-336"

# ==================== 分类主题配置 ====================
# 论文分类主题列表（可自定义）
# 添加新主题时，建议在 CLASSIFICATION_PROMPT 中添加对应的分类规则
PAPER_TOPICS = [
    "Computer_Vision",              # 计算机视觉
    "Natural_Language_Processing",  # 自然语言处理
    "Reinforcement_Learning",       # 强化学习
    "Machine_Learning",             # 机器学习
    "Robotics",                     # 机器人
    "Other"                         # 其他
]

# ==================== Prompt 模板 ====================
# 论文分类 Prompt
CLASSIFICATION_PROMPT = """你是一个专业的学术档案管理员。
请阅读以下论文片段，并将其归类到以下类别之一：{topics}。

分类规则：
- Computer_Vision：计算机视觉、图像处理、目标检测、图像分割等
- Natural_Language_Processing：自然语言处理、文本分析、机器翻译、对话系统等
- Reinforcement_Learning：强化学习、智能体、策略优化等
- Machine_Learning：机器学习、深度学习的通用方法
- Robotics：机器人、运动规划、控制等
- Other：不属于以上任何类别

输出格式要求：仅输出类别名称，不要包含任何其他解释或标点符号。

论文片段：
{content}

分类结果："""

# RAG 问答 Prompt
RAG_PROMPT = """你是一个专业的学术助手。请基于以下参考资料回答用户的问题。

参考资料：
{context}

用户问题：{question}

请给出详细且准确的回答，如果参考资料中没有相关信息，请明确说明。"""

# 多模态对话 Prompt
VLM_PROMPT = """你是一个专业的图像分析助手。请仔细观察这张图片并回答用户的问题。
请给出详细且专业的回答。"""

# ==================== 模型推理配置 ====================
# 生成参数
MAX_NEW_TOKENS = 512        # 最大生成 token 数
TEMPERATURE = 0.7           # 采样温度（0.0-1.0，越高越随机）
TOP_P = 0.9                 # 核采样参数
REPETITION_PENALTY = 1.05   # 重复惩罚

# 检索参数
RETRIEVAL_TOP_K = 5         # 检索返回的文档数量

# PDF 处理参数
PDF_EXTRACT_MAX_CHARS = 2000  # 用于分类的最大字符数
CHUNK_SIZE = 500              # 文本分块大小
CHUNK_OVERLAP = 50            # 分块重叠大小

# ==================== 日志配置 ====================
LOG_LEVEL = "INFO"
