"""
CLIP模型基础演示 (Hugging Face Pipeline版本)
=============================================

本文件演示了使用Hugging Face pipeline简化CLIP模型调用。
主要功能：
1. 零样本图像分类 - 使用pipeline一行代码完成
2. 图文检索 - 使用特征提取和相似度计算
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from config import DATA_IMAGES, DATA_RESULTS
from vqa_common import (
    save_classification_result, save_retrieval_result,
    create_visualization_comparison, create_retrieval_comparison
)

from transformers import pipeline, AutoProcessor, AutoModelForZeroShotImageClassification
import torch
from PIL import Image
import numpy as np

classifier = pipeline("zero-shot-image-classification", model="openai/clip-vit-base-patch32")
print("CLIP分类器加载完成")

processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model = AutoModelForZeroShotImageClassification.from_pretrained("openai/clip-vit-base-patch32")
clip_model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model.to(device)
print(f"CLIP模型已加载到: {device}")

def zero_shot_classification(image_path, labels):
    """使用pipeline进行零样本分类"""
    if not isinstance(image_path, str):
        image_path = str(image_path)
    result = classifier(image_path, candidate_labels=labels)
    probs = [r["score"] for r in result]
    print(f"\n图像: {os.path.basename(image_path)}")
    for label, prob in zip(labels, probs):
        print(f"  {label}: {prob:.3f}")
    return {
        "image": image_path,
        "top_label": result[0]["label"],
        "top_prob": result[0]["score"],
        "all_probs": {r["label"]: r["score"] for r in result}
    }

def text_image_retrieval(text, image_folder, top_k=3):
    """使用CLIP特征提取进行图文检索"""
    images = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not images:
        print("没有找到图像！")
        return None

    image_paths = [os.path.join(image_folder, f) for f in images]
    pil_images = [Image.open(p).convert("RGB") for p in image_paths]

    with torch.no_grad():
        text_inputs = processor(text=[text], return_tensors="pt", padding=True).to(device)
        text_features = clip_model.get_text_features(**text_inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        image_inputs = processor(images=pil_images, return_tensors="pt").to(device)
        image_features = clip_model.get_image_features(**image_inputs)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        similarities = (text_features @ image_features.T).squeeze(0)

    top_k = min(top_k, len(images))
    top_indices = torch.topk(similarities, top_k).indices.tolist()

    results = []
    print(f"\n查询文本: '{text}'")
    print(f"检索结果（从 {len(images)} 张图像中）:")
    for i, idx in enumerate(top_indices):
        similarity = float(similarities[idx])
        path = image_paths[idx]
        results.append({"image": path, "similarity": similarity, "rank": i + 1})
        print(f"  {i + 1}. {os.path.basename(path)} (相似度: {similarity:.3f})")

    return {"query": text, "results": results}

if __name__ == "__main__":
    clip_output_dir = DATA_RESULTS / "clip_demo"
    os.makedirs(clip_output_dir, exist_ok=True)
    print("=== CLIP 基础复现演示 ===")

    print("\n1. 零样本图像分类:")
    test_image = DATA_IMAGES / "0.jpg"
    if os.path.exists(test_image):
        labels = ["a cat", "a dog", "a car", "a tree", "a person", "a phone"]
        result = zero_shot_classification(test_image, labels)
        probs = [result["all_probs"][label] for label in labels]
        save_classification_result(test_image, labels, np.array(probs), clip_output_dir)
        create_visualization_comparison(test_image, labels, probs, os.path.join(clip_output_dir, "classification_comparison.png"))
    else:
        print(f"测试图像不存在: {test_image}")

    print("\n2. 图文检索:")
    if os.path.exists(DATA_IMAGES) and len(os.listdir(DATA_IMAGES)) > 0:
        retrieval_result = text_image_retrieval("a person", DATA_IMAGES, top_k=3)
        if retrieval_result:
            save_retrieval_result(retrieval_result["query"], retrieval_result["results"], clip_output_dir)
            create_retrieval_comparison(retrieval_result["query"], retrieval_result["results"], DATA_IMAGES, os.path.join(clip_output_dir, "retrieval_comparison.png"))

    print(f"\n所有结果已保存至: {clip_output_dir}")
