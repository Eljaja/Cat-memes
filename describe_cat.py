import re
import pandas as pd
from pathlib import Path
from functools import lru_cache
from PIL import Image
import torch
from transformers import (
    LlavaNextProcessor,
    LlavaNextForConditionalGeneration,
    BitsAndBytesConfig,
)
from tqdm.auto import tqdm       #  ← индикатор прогресса

MODEL_ID = "llava-hf/llava-v1.6-mistral-7b-hf"


# ---------- 1. ОДНОКРАТНАЯ ЗАГРУЗКА МОДЕЛИ -------------------------------------
@lru_cache(maxsize=1)
def load_llava(model_id: str = MODEL_ID):
    bnb_conf = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    processor = LlavaNextProcessor.from_pretrained(model_id, use_fast=False)
    model = LlavaNextForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=bnb_conf,
        torch_dtype=torch.float16,
        device_map="auto",
        attn_implementation="sdpa",
    ).eval()
    return processor, model


# ---------- 2. ОПИСАНИЕ ОДНОГО ИЗОБРАЖЕНИЯ -------------------------------------
def describe_cat_meme(image_path: str) -> str:
    processor, model = load_llava()      # ← кэшировано

    img = Image.open(image_path).convert("RGB")
    prompt = (
        "<image>\n"
        "Describe this meme cat in the format:\n"
        "Emotion: <one word>\n"
        "Description: <15 tags about the meme>"
    )
    inputs = processor(text=prompt, images=img, return_tensors="pt").to(model.device)
    out_ids = model.generate(**inputs, max_new_tokens=60)
    output = processor.decode(out_ids[0], skip_special_tokens=True)

    # удаляем префикс промпта, оставляем «Emotion: …»
    return re.sub(
        r"(?s).*?Emotion:",
        "Emotion:",
        output.strip(), count=1
    )


# ---------- 3. ПАРСИНГ РЕЗУЛЬТАТА ----------------------------------------------
def parse_output(text: str):
    emotion = description = ""
    for line in (l.strip() for l in text.splitlines() if l.strip()):
        if line.lower().startswith("emotion:"):
            val = line.split(":", 1)[1].strip()
            if val and "<" not in val:        # ⚠️ пропускаем заглушку
                emotion = val
        elif line.lower().startswith("description:"):
            val = line.split(":", 1)[1].strip()
            if val and "<" not in val:        # ⚠️ пропускаем заглушку
                description = val
    return emotion, description


# ---------- 4. ОСНОВНОЙ ЦИКЛ ----------------------------------------------------
def build_csv(img_dir="cats_from_memes", csv_path="cats_from_memes.csv"):
    img_paths = sorted(Path(img_dir).glob("cat-*.png"))
    rows = []

    for img_path in tqdm(img_paths, desc="Processing cats", unit="img"):
        try:
            raw = describe_cat_meme(img_path)
            emo, desc = parse_output(raw)

            tqdm.write(f"{img_path.name:35} → {emo:10} | {desc}")
            rows.append({"Filename": img_path.name, "Emotion": emo, "Description": desc})

        except Exception as e:
            tqdm.write(f"ERROR on {img_path.name}: {e}")

    pd.DataFrame(rows).to_csv(csv_path, index=False, encoding="utf-8-sig")
    tqdm.write(f"\nSaved {len(rows)} records to {csv_path}")


if __name__ == "__main__":
    build_csv()
