"""
VLM Engine: 封装 Qwen3-VL 模型的加载与推理
"""
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
from qwen_vl_utils import process_vision_info
from typing import Optional
import config

class VLMEngine:
    """
    视觉语言模型引擎，负责加载和推理 Qwen3-VL 模型
    """
    def __init__(self):
        self.model = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def load_model(self):
        """加载 Qwen3-VL 模型（懒加载）"""
        if self.model is not None:
            print("VLM 模型已加载，跳过重复加载")
            return
        
        print(f"正在加载 VLM 模型: {config.VLM_MODEL_PATH}")
        
        # 加载 processor
        self.processor = AutoProcessor.from_pretrained(
            config.VLM_MODEL_PATH,
            trust_remote_code=True
        )
        
        # 使用 FP16 加载模型，使用 AutoModelForVision2Seq 以支持 Qwen3-VL
        self.model = AutoModelForVision2Seq.from_pretrained(
            config.VLM_MODEL_PATH,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
        
        self.model.eval()
        print("✓ VLM 模型加载完成！")
    
    def chat_with_image(self, image_path: str, user_question: str, 
                       max_new_tokens: Optional[int] = None) -> str:
        """
        与图像进行对话
        Args:
            image_path: 图像路径
            user_question: 用户问题
            max_new_tokens: 最大生成长度
        Returns:
            模型回答
        """
        if self.model is None:
            self.load_model()
        
        if max_new_tokens is None:
            max_new_tokens = config.MAX_NEW_TOKENS
        
        # 构造多模态消息
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path,
                    },
                    {
                        "type": "text", 
                        "text": user_question
                    }
                ],
            }
        ]
        
        # 应用聊天模板
        text = self.processor.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # 处理图像和文本
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)
        
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
        generated_ids_trimmed = [
            out_ids[len(in_ids):] 
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        
        return output_text[0].strip()
    
    def describe_image(self, image_path: str) -> str:
        """
        生成图像描述
        Args:
            image_path: 图像路径
        Returns:
            图像描述
        """
        return self.chat_with_image(
            image_path, 
            "请详细描述这张图片的内容。"
        )
    
    def unload_model(self):
        """卸载模型，释放显存"""
        if self.model is not None:
            del self.model
            del self.processor
            self.model = None
            self.processor = None
            torch.cuda.empty_cache()
            print("VLM 模型已卸载")


# 全局单例
_vlm_engine = None

def get_vlm_engine() -> VLMEngine:
    """获取 VLM 引擎单例"""
    global _vlm_engine
    if _vlm_engine is None:
        _vlm_engine = VLMEngine()
    return _vlm_engine
