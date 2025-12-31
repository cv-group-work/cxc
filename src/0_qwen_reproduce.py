"""
Qwen3-VL模型复现演示 (LangChain API版本)
==========================================

本文件演示了使用LangChain API调用Qwen3-VL模型，无需本地GPU显存。
主要功能：
1. 使用LangChain ChatOpenAI统一接口
2. 无需本地模型加载
3. 适合快速原型开发
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from config import DATA_IMAGES, DATA_RESULTS, API_KEY
from vqa_common import save_vqa_result, create_vqa_visualization_comparison, load_model, image_to_base64

from langchain_core.messages import HumanMessage

print("初始化 Qwen3-VL API 客户端...")
llm, model_name = load_model(API_KEY)

image_path = DATA_IMAGES / "5.jpg"

base64_image = image_to_base64(image_path)
if not base64_image:
    print("无法处理图片")
    exit(1)

message = HumanMessage(
    content=[
        {"type": "text", "text": "这个图片是什么？请简短回答。"},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
    ]
)

response = llm.invoke([message])
answer = response.content.strip()

print("=== Qwen3-VL 视觉问答演示 ===")
print(f"图像文件: {os.path.basename(image_path)}")
print(f"问题: 这个图片是什么?")
print(f"回答: {answer}")

qwen_output_dir = DATA_RESULTS / "qwen_demo"
os.makedirs(qwen_output_dir, exist_ok=True)
save_vqa_result(image_path, "这个图片是什么?", answer, qwen_output_dir)
create_vqa_visualization_comparison(image_path, "这个图片是什么?", answer, os.path.join(qwen_output_dir, "vqa_comparison.png"))
print(f"\n所有结果已保存至: {qwen_output_dir}")
