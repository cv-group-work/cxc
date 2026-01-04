"""
CLIP重排序答案生成器
基于Qwen3-VL生成候选答案，CLIP评分排序
"""
import os
from PIL import Image
import torch
import torch.nn.functional as F
from langchain_openai import ChatOpenAI
from transformers import AutoProcessor, AutoModel
from dotenv import load_dotenv

load_dotenv()

MODELS = {
    "clip": "openai/clip-vit-base-patch32",
    "qwen": "qwen3-vl-8b-instruct"
}
API_KEY = os.getenv("DASHSCOPE_API_KEY")
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


def image_to_base64(image_path):
    """图片转base64"""
    try:
        with Image.open(image_path) as img:
            import base64
            from io import BytesIO
            buffer = BytesIO()
            img.save(buffer, format='JPEG')
            return base64.b64encode(buffer.getvalue()).decode('utf-8')
    except Exception:
        return None


def generate_candidates(image_path, question, num=5):
    """调用Qwen生成多个候选答案"""
    base64_image = image_to_base64(image_path)
    if not base64_image:
        return []

    prompt = f"{question}\n请用最简短的词或短语回答，1-3个词足矣，不要任何解释。"
    candidates = []

    for i in range(num):
        temp = 0.3 + i * 0.3  # 温度从0.3到1.5，变化更大
        llm = ChatOpenAI(api_key=API_KEY, base_url=BASE_URL, model=MODELS["qwen"], temperature=temp, max_tokens=128)
        from langchain_core.messages import HumanMessage
        message = HumanMessage(content=[
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
        ])
        response = llm.invoke([message])
        answer = response.content.strip()
        if answer and not answer.startswith("错误:"):
            candidates.append(answer)

    return candidates[:10]  # 保留前10个，不去重


def clip_rank(image_path, candidates):
    """CLIP评分排序"""
    if not candidates:
        return []

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(MODELS["clip"])
    model = AutoModel.from_pretrained(MODELS["clip"]).eval().to(device)

    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, text=candidates, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        image_embeds = F.normalize(outputs.image_embeds, p=2, dim=1)
        text_embeds = F.normalize(outputs.text_embeds, p=2, dim=1)
        similarities = (image_embeds @ text_embeds.T).softmax(dim=1).cpu().numpy().flatten()

    ranked = sorted(zip(candidates, similarities), key=lambda x: x[1], reverse=True)
    return [{"answer": ans, "score": float(score)} for ans, score in ranked]


def main(image_path, question):
    """主函数：生成并排序候选答案"""

    print(f"图片: {image_path}")
    print(f"问题: {question}\n")

    candidates = generate_candidates(image_path, question, num=5)
    if not candidates:
        print("错误：无法生成候选答案")
        return

    print(f"生成了 {len(candidates)} 个候选答案\n")
    ranked_results = clip_rank(image_path, candidates)

    print("CLIP评分排序结果：")
    print("=" * 40)
    for i, item in enumerate(ranked_results, 1):
        print(f"{i}. {item['answer']} (CLIP: {item['score']:.4f})")

    return ranked_results


if __name__ == "__main__":
    image_path = input("请输入图片路径")
    question = input("请输入问题")
    main(image_path, question)
