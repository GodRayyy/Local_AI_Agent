"""
GPU Utils: 显存管理工具
"""
import torch
import gc

def clear_vram():
    """
    清理显存
    在切换不同功能模式时调用，释放不需要的模型
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    print("显存已清理")

def get_gpu_memory_info():
    """
    获取 GPU 显存信息
    Returns:
        字典，包含已分配和总显存（GB）
    """
    if not torch.cuda.is_available():
        return {"message": "CUDA 不可用"}
    
    allocated = torch.cuda.memory_allocated() / (1024 ** 3)
    reserved = torch.cuda.memory_reserved() / (1024 ** 3)
    total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    
    return {
        "allocated_gb": f"{allocated:.2f}",
        "reserved_gb": f"{reserved:.2f}",
        "total_gb": f"{total:.2f}",
        "free_gb": f"{total - allocated:.2f}"
    }

def print_gpu_memory_info():
    """打印 GPU 显存信息"""
    info = get_gpu_memory_info()
    
    if "message" in info:
        print(info["message"])
    else:
        print(f"GPU 显存信息:")
        print(f"  已分配: {info['allocated_gb']} GB")
        print(f"  已预留: {info['reserved_gb']} GB")
        print(f"  总容量: {info['total_gb']} GB")
        print(f"  可用: {info['free_gb']} GB")

def check_cuda_available():
    """检查 CUDA 是否可用"""
    if torch.cuda.is_available():
        print(f"✓ CUDA 可用")
        print(f"  设备数量: {torch.cuda.device_count()}")
        print(f"  当前设备: {torch.cuda.current_device()}")
        print(f"  设备名称: {torch.cuda.get_device_name(0)}")
        return True
    else:
        print("✗ CUDA 不可用，将使用 CPU 运行（速度会很慢）")
        return False
