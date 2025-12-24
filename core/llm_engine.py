"""
LLM Engine: 封装 Qwen3-7B-Instruct 模型的加载与推理
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional
import config

class LLMEngine:
    """
    大语言模型引擎，负责加载和推理 Qwen3 模型
    """
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def load_model(self):
        """加载 Qwen3 模型（懒加载）"""
        if self.model is not None:
            print("LLM 模型已加载，跳过重复加载")
            return
        
        print(f"正在加载 LLM 模型: {config.LLM_MODEL_PATH}")
        
        # 加载 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.LLM_MODEL_PATH,
            trust_remote_code=True
        )
        
        # 使用 FP16 加载模型
        self.model = AutoModelForCausalLM.from_pretrained(
            config.LLM_MODEL_PATH,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
        
        self.model.eval()
        print("✓ LLM 模型加载完成！")
    
    def generate(self, prompt: str, max_new_tokens: Optional[int] = None) -> str:
        """
        生成文本
        Args:
            prompt: 输入提示词
            max_new_tokens: 最大生成长度
        Returns:
            生成的文本
        """
        if self.model is None:
            self.load_model()
        
        if max_new_tokens is None:
            max_new_tokens = config.MAX_NEW_TOKENS
        
        # 构造消息格式
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": prompt}
        ]
        
        # 应用聊天模板
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # 编码输入
        inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        
        # 生成
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=config.TEMPERATURE,
                top_p=config.TOP_P,
                do_sample=True
            )
        
        # 解码输出（只保留新生成的部分）
        generated_ids = [
            output_ids[len(input_ids):] 
            for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response.strip()
    
    def classify_paper(self, text_snippet: str, topics: list) -> str:
        """
        对论文进行分类
        Args:
            text_snippet: 论文片段
            topics: 候选主题列表
        Returns:
            分类结果
        """
        prompt = config.CLASSIFICATION_PROMPT.format(
            topics=", ".join(topics),
            content=text_snippet
        )
        
        response = self.generate(prompt, max_new_tokens=50)
        
        # 提取分类结果（防止模型输出额外内容）
        for topic in topics:
            if topic in response:
                return topic
        
        # 如果没有匹配到，返回第一个候选主题或 Other
        print(f"警告：无法从响应中提取分类，原始响应: {response}")
        return "Other"
    
    def answer_question(self, question: str, context: str) -> str:
        """
        基于上下文回答问题（RAG）
        Args:
            question: 用户问题
            context: 检索到的上下文
        Returns:
            答案
        """
        prompt = config.RAG_PROMPT.format(
            context=context,
            question=question
        )
        
        return self.generate(prompt)
    
    def unload_model(self):
        """卸载模型，释放显存"""
        if self.model is not None:
            del self.model
            del self.tokenizer
            self.model = None
            self.tokenizer = None
            torch.cuda.empty_cache()
            print("LLM 模型已卸载")


# 全局单例
_llm_engine = None

def get_llm_engine() -> LLMEngine:
    """获取 LLM 引擎单例"""
    global _llm_engine
    if _llm_engine is None:
        _llm_engine = LLMEngine()
    return _llm_engine
