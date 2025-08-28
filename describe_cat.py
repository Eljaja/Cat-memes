import re
import pandas as pd
from pathlib import Path
from functools import lru_cache
from PIL import Image
import base64
import io
from vllm import LLM, SamplingParams
from vllm.multimodal.llm import MultiModalLLM
from tqdm.auto import tqdm

MODEL_ID = "llava-hf/llava-v1.6-mistral-7b-hf"


# ---------- 1. ОДНОКРАТНАЯ ЗАГРУЗКА МОДЕЛИ -------------------------------------
@lru_cache(maxsize=1)
def load_llava(model_id: str = MODEL_ID):
    # Инициализация vLLM с поддержкой мультимодальности
    llm = MultiModalLLM(
        model=model_id,
        trust_remote_code=True,
        tensor_parallel_size=1,  # Увеличьте для многогпоточности
        gpu_memory_utilization=0.8,
        max_model_len=4096,
        quantization="awq",  # Или "gptq", "squeezellm" для квантизации
    )
    
    # Параметры семплирования
    sampling_params = SamplingParams(
        temperature=0.1,
        top_p=0.9,
        max_tokens=60,
        stop=["</s>", "\n\n"]
    )
    
    return llm, sampling_params


# ---------- 2. КОНВЕРТАЦИЯ ИЗОБРАЖЕНИЯ В BASE64 --------------------------------
def image_to_base64(image_path: str) -> str:
    """Конвертирует изображение в base64 для vLLM"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


# ---------- 3. ОПИСАНИЕ ОДНОГО ИЗОБРАЖЕНИЯ -------------------------------------
def describe_cat_meme(image_path: str) -> str:
    llm, sampling_params = load_llava()
    
    # Конвертируем изображение в base64
    image_base64 = image_to_base64(image_path)
    
    # Формируем промпт для vLLM
    prompt = (
        "<image>\n"
        "Describe this meme cat in the format:\n"
        "Emotion: <one word>\n"
        "Description: <15 tags about the meme>"
    )
    
    # Создаем сообщения для vLLM
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
                {"type": "text", "text": prompt}
            ]
        }
    ]
    
    # Генерируем ответ
    outputs = llm.chat(messages, sampling_params=sampling_params)
    output = outputs[0].outputs[0].text
    
    # Удаляем префикс промпта, оставляем «Emotion: …»
    return re.sub(
        r"(?s).*?Emotion:",
        "Emotion:",
        output.strip(), count=1
    )


# ---------- 4. ПАРСИНГ РЕЗУЛЬТАТА ----------------------------------------------
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


# ---------- 5. ОСНОВНОЙ ЦИКЛ ----------------------------------------------------
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


# ---------- 6. АЛЬТЕРНАТИВНЫЙ ВАРИАНТ С BATCH ОБРАБОТКОЙ ------------------------
def build_csv_batch(img_dir="cats_from_memes", csv_path="cats_from_memes.csv", batch_size=4):
    """Обработка изображений батчами для лучшей производительности"""
    img_paths = sorted(Path(img_dir).glob("cat-*.png"))
    rows = []
    
    llm, sampling_params = load_llava()
    
    # Обрабатываем батчами
    for i in tqdm(range(0, len(img_paths), batch_size), desc="Processing batches"):
        batch_paths = img_paths[i:i + batch_size]
        batch_messages = []
        
        # Подготавливаем батч сообщений
        for img_path in batch_paths:
            image_base64 = image_to_base64(img_path)
            prompt = (
                "<image>\n"
                "Describe this meme cat in the format:\n"
                "Emotion: <one word>\n"
                "Description: <15 tags about the meme>"
            )
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            batch_messages.append(messages)
        
        # Обрабатываем батч
        try:
            outputs = llm.chat(batch_messages, sampling_params=sampling_params)
            
            for j, output in enumerate(outputs):
                img_path = batch_paths[j]
                raw = output.outputs[0].text
                raw = re.sub(r"(?s).*?Emotion:", "Emotion:", raw.strip(), count=1)
                emo, desc = parse_output(raw)
                
                tqdm.write(f"{img_path.name:35} → {emo:10} | {desc}")
                rows.append({"Filename": img_path.name, "Emotion": emo, "Description": desc})
                
        except Exception as e:
            tqdm.write(f"ERROR in batch {i//batch_size + 1}: {e}")
            # Fallback к индивидуальной обработке
            for img_path in batch_paths:
                try:
                    raw = describe_cat_meme(img_path)
                    emo, desc = parse_output(raw)
                    tqdm.write(f"{img_path.name:35} → {emo:10} | {desc}")
                    rows.append({"Filename": img_path.name, "Emotion": emo, "Description": desc})
                except Exception as e2:
                    tqdm.write(f"ERROR on {img_path.name}: {e2}")

    pd.DataFrame(rows).to_csv(csv_path, index=False, encoding="utf-8-sig")
    tqdm.write(f"\nSaved {len(rows)} records to {csv_path}")


if __name__ == "__main__":
    # Используйте build_csv() для обычной обработки
    # или build_csv_batch() для батчевой обработки
    build_csv()
    # build_csv_batch(batch_size=4)  # Раскомментируйте для батчевой обработки
