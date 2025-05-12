# import os

# import torch
# import yaml

# from models.multimodal_encoder.t5_encoder import T5Embedder


# GPU = 0
# MODEL_PATH = "google/t5-v1_1-xxl"
# CONFIG_PATH = "configs/base.yaml"
# SAVE_DIR = "outs/"

# # Modify this to your task name and instruction
# TASK_NAME = "handover_pan"
# INSTRUCTION = "Pick up the black marker on the right and put it into the packaging box on the left."

# # Note: if your GPU VRAM is less than 24GB, 
# # it is recommended to enable offloading by specifying an offload directory.
# OFFLOAD_DIR = None  # Specify your offload directory here, ensuring the directory exists.

# def main():
#     with open(CONFIG_PATH, "r") as fp:
#         config = yaml.safe_load(fp)
    
#     device = torch.device(f"cuda:{GPU}")
#     text_embedder = T5Embedder(
#         from_pretrained=MODEL_PATH, 
#         model_max_length=config["dataset"]["tokenizer_max_length"], 
#         device=device,
#         use_offload_folder=OFFLOAD_DIR
#     )
#     tokenizer, text_encoder = text_embedder.tokenizer, text_embedder.model

#     tokens = tokenizer(
#         INSTRUCTION, return_tensors="pt",
#         padding="longest",
#         truncation=True
#     )["input_ids"].to(device)

#     tokens = tokens.view(1, -1)
#     with torch.no_grad():
#         pred = text_encoder(tokens).last_hidden_state.detach().cpu()
    
#     save_path = os.path.join(SAVE_DIR, f"{TASK_NAME}.pt")
#     # We save the embeddings in a dictionary format
#     torch.save({
#             "name": TASK_NAME,
#             "instruction": INSTRUCTION,
#             "embeddings": pred
#         }, save_path
#     )
    
#     print(f'\"{INSTRUCTION}\" from \"{TASK_NAME}\" is encoded by \"{MODEL_PATH}\" into shape {pred.shape} and saved to \"{save_path}\"')


# if __name__ == "__main__":
#     main()

import os
import sys

# 添加项目根目录到 Python 路径
sys.path.insert(0, '/home/ubuntu/桌面/RoboticsDiffusionTransformer')

import torch
import yaml
from models.multimodal_encoder.t5_encoder import T5Embedder

# 配置参数
GPU = 0
MODEL_PATH = "google/t5-v1_1-xxl"
CONFIG_PATH = "configs/base.yaml"
SAVE_DIR = "outs/"  # 保存目录
TASK_NAME = "push_cube_train"
INSTRUCTION = "Push and move a cube to a goal region in front of it."
OFFLOAD_DIR = None

def main():
    # 确保保存目录存在
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
        print(f"创建目录: {SAVE_DIR}")
    
    # 如果设置了卸载目录，确保它存在
    if OFFLOAD_DIR and not os.path.exists(OFFLOAD_DIR):
        os.makedirs(OFFLOAD_DIR)
        print(f"创建卸载目录: {OFFLOAD_DIR}")
    
    with open(CONFIG_PATH, "r") as fp:
        config = yaml.safe_load(fp)
    
    device = torch.device(f"cuda:{GPU}")
    print(f"使用设备: {device}")
    
    print(f"正在加载文本编码器: {MODEL_PATH}")
    text_embedder = T5Embedder(
        from_pretrained=MODEL_PATH,
        model_max_length=config["dataset"]["tokenizer_max_length"],
        device=device,
        use_offload_folder=OFFLOAD_DIR
    )
    
    tokenizer, text_encoder = text_embedder.tokenizer, text_embedder.model
    print("文本编码器加载完成")
    
    print(f"正在编码指令: \"{INSTRUCTION}\"")
    tokens = tokenizer(
        INSTRUCTION, return_tensors="pt",
        padding="longest",
        truncation=True
    )["input_ids"].to(device)
    
    tokens = tokens.view(1, -1)
    print(f"令牌形状: {tokens.shape}")
    
    with torch.no_grad():
        print("生成嵌入...")
        pred = text_encoder(tokens).last_hidden_state.detach().cpu()
    
    save_path = os.path.join(SAVE_DIR, f"{TASK_NAME}.pt")
    print(f"保存嵌入到: {save_path}")
    
    # 保存嵌入
    torch.save({
        "name": TASK_NAME,
        "instruction": INSTRUCTION,
        "embeddings": pred
    }, save_path)
    
    print(f'\"{INSTRUCTION}\" 从 \"{TASK_NAME}\" 被编码为形状 {pred.shape} 并保存到 \"{save_path}\"')

if __name__ == "__main__":
    main()