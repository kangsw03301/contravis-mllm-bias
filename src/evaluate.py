import argparse
import json
import os
import random
import re
import numpy as np
import torch
from PIL import Image
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig,
    LlavaForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
)

PROMPT_TEMPLATES = {
    "p0": "{query} Answer in one or two words.",
    "p_soft": "Look closely at the image before answering. {query} Answer in one or two words.",
    "p_strict": (
        "Ignore your prior knowledge and real-world common sense. "
        "The image may depict unusual or edited objects. "
        "Base your answer strictly and exclusively on the visual pixels shown. "
        "{query} Answer in one or two words."
    ),
}

CANONICAL_SYNONYMS = {
    "yellow": ["yellow", "golden", "amber"],
    "blue": ["blue", "cobalt", "navy", "azure", "cyan"],
    "red": ["red", "crimson", "scarlet", "maroon"],
    "purple": ["purple", "violet", "lavender", "magenta"],
    "green": ["green", "emerald", "lime", "olive"],
    "pink": ["pink", "rose", "coral"],
    "round": ["round", "circular", "circle", "spherical", "sphere", "curved"],
    "square": ["square", "rectangular", "boxy", "quadrilateral"],
    "triangular": ["triangular", "triangle", "pyramid"],
    "cubic": ["cubic", "cube", "cuboid"],
    "octagonal": ["octagonal", "octagon"],
    "oval": ["oval", "elliptical", "ellipse"],
    "metal": ["metal", "steel", "iron", "aluminum", "metallic"],
    "glass": ["glass", "crystal", "transparent"],
    "wood": ["wood", "wooden", "timber", "oak"],
    "wool": ["wool", "woolen", "knitted", "yarn", "fabric"],
    "ice": ["ice", "frozen", "water"],
    "rubber": ["rubber", "silicone", "latex"],
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_label(raw_text: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z\s]", " ", raw_text.lower()).strip()
    tokens = cleaned.split()
    for canonical, variants in CANONICAL_SYNONYMS.items():
        for variant in variants:
            if variant in tokens or variant in cleaned:
                return canonical
    return tokens[0] if tokens else "unknown"


def load_model_and_processor(model_name: str):
    if "Qwen2.5-VL" in model_name:
        processor = AutoProcessor.from_pretrained(model_name)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
        )
    elif "llava-1.5" in model_name:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
        processor = AutoProcessor.from_pretrained(model_name)
        model = LlavaForConditionalGeneration.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
        )
    elif "InternVL2_5" in model_name:
        processor = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map="auto",
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    model.eval()
    return model, processor


@torch.no_grad()
def generate_response(model, processor, model_name: str, image: Image.Image, prompt_text: str, max_new_tokens: int) -> str:
    if "Qwen2.5-VL" in model_name:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]
        text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text_input], images=[image], return_tensors="pt").to(model.device)
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        trimmed_ids = output_ids[:, inputs.input_ids.shape[1]:]
        return processor.batch_decode(trimmed_ids, skip_special_tokens=True)[0].strip()

    if "llava-1.5" in model_name:
        formatted_prompt = f"USER: <image>\n{prompt_text}\nASSISTANT:"
        inputs = processor(text=formatted_prompt, images=image, return_tensors="pt").to(model.device)
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        trimmed_ids = output_ids[:, inputs.input_ids.shape[1]:]
        return processor.batch_decode(trimmed_ids, skip_special_tokens=True)[0].strip()

    if "InternVL2_5" in model_name:
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.Resize((448, 448)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        pixel_values = transform(image).unsqueeze(0).to(torch.float16).to(model.device)
        generation_config = {"max_new_tokens": max_new_tokens, "do_sample": False}
        response = model.chat(processor, pixel_values, prompt_text, generation_config)
        return response.strip()

    return ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--meta_path", type=str, default="data/contravis_meta.json")
    parser.add_argument("--image_dir", type=str, default="data/images")
    parser.add_argument("--condition", type=str, choices=["standard", "conflict"], required=True)
    parser.add_argument("--prompt_key", type=str, choices=["p0", "p_soft", "p_strict"], required=True)
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.meta_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    items = [d for d in dataset if d["condition"] == args.condition]
    model, processor = load_model_and_processor(args.model_name)

    records = []
    for item in items:
        img_path = os.path.join(args.image_dir, item["image_file"])
        image = Image.open(img_path).convert("RGB")
        prompt_text = PROMPT_TEMPLATES[args.prompt_key].format(query=item["query"])

        raw_pred = generate_response(
            model=model,
            processor=processor,
            model_name=args.model_name,
            image=image,
            prompt_text=prompt_text,
            max_new_tokens=args.max_new_tokens,
        )
        norm_pred = normalize_label(raw_pred)
        ground_truth = item["ground_truth"].lower()
        prior = item["prior"].lower()

        is_correct = int(norm_pred == ground_truth)
        is_biased = int(norm_pred == prior) if args.condition == "conflict" else 0

        records.append({
            "id": item["id"],
            "category": item["category"],
            "condition": args.condition,
            "prompt_key": args.prompt_key,
            "ground_truth": ground_truth,
            "prior": prior,
            "raw_prediction": raw_pred,
            "normalized_prediction": norm_pred,
            "is_correct": is_correct,
            "is_biased": is_biased,
        })

    model_short = args.model_name.split("/")[-1]
    out_file = os.path.join(args.output_dir, f"{model_short}_{args.condition}_{args.prompt_key}.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    acc = np.mean([r["is_correct"] for r in records]) * 100.0
    bias = np.mean([r["is_biased"] for r in records]) * 100.0
    print(f"Saved {out_file} | Acc_V: {acc:.1f}% | Bias_L: {bias:.1f}%")


if __name__ == "__main__":
    main()
