#!/usr/bin/env python3
"""
Zero-shot ретривер кото-мемов «по вайбу» (Mac M4)
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import warnings

import numpy as np
import torch
from PIL import Image
import open_clip
from tqdm import tqdm

# Подавляем предупреждения
warnings.filterwarnings("ignore")

# Константы
DEFAULT_VIBES = {
    "cute": "cute cozy wholesome cat, warm soft light, endearing",
    "irony": "sarcastic ironic tongue-in-cheek cat, playful witty",
    "cringe": "awkward cringe offbeat weird cat, uncomfortable",
    "absurd": "surreal absurd bizarre cat, nonsensical, odd",
    "creepy": "creepy eerie unsettling cat, dark moody",
    "calm": "calm zen serene relaxed cat, peaceful, cozy",
    "angry": "angry annoyed hissing cat, aggressive",
    "sad": "sad gloomy melancholic cat, depressed vibe",
    "surprised": "surprised startled confused cat, wide eyes"
}

# Коэффициенты для скоринга
ALPHA = 0.65  # вес контекста
BETA = 0.35   # вес вайбов
LAMBDA_MMR = 0.5  # для диверсификации

# Поддерживаемые расширения
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}


class MemeRetriever:
    def __init__(self, device: str = None):
        """Инициализация ретривера с CLIP моделью"""
        if device is None:
            if torch.backends.mps.is_available():
                self.device = "mps"
            elif torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = device
            
        print(f"Using device: {self.device}", file=sys.stderr)
        
        # Загружаем модель OpenCLIP
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            'ViT-B-16', 
            pretrained='laion2b_s34b_b88k',
            device=self.device
        )
        self.model.eval()

        # Кэш текстовых эмбеддингов, чтобы не пересчитывать одинаковые строки
        self._text_embedding_cache: Dict[str, np.ndarray] = {}
        
        # Получаем размер эмбеддинга
        with torch.no_grad():
            dummy_text = open_clip.tokenize(["dummy"])
            dummy_text = dummy_text.to(self.device)
            self.embedding_dim = self.model.encode_text(dummy_text).shape[1]
        
        print(f"Embedding dimension: {self.embedding_dim}", file=sys.stderr)

    def get_image_files(self, images_dir: Path) -> List[Path]:
        """Рекурсивно собирает все изображения из директории"""
        image_files = []
        for ext in IMAGE_EXTENSIONS:
            image_files.extend(images_dir.rglob(f"*{ext}"))
            image_files.extend(images_dir.rglob(f"*{ext.upper()}"))
        
        # Исключаем скрытые файлы
        image_files = [f for f in image_files if not any(part.startswith('.') for part in f.parts)]
        
        return sorted(image_files)

    def load_and_preprocess_image(self, image_path: Path) -> Tuple[Optional[torch.Tensor], Dict]:
        """Загружает и предобрабатывает изображение, возвращает тензор и метаданные"""
        try:
            with Image.open(image_path) as img:
                # Конвертируем в RGB
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Вычисляем яркость (среднее значение канала L/grayscale)
                # Используем PIL 'L' для близкого эквивалента канала L => [0..255]
                brightness = float(np.array(img.convert('L')).mean()) / 255.0
                
                # Получаем размеры
                width, height = img.size
                
                # Предобрабатываем для модели
                img_tensor = self.preprocess(img).unsqueeze(0).to(self.device)
                
                metadata = {
                    "path": str(image_path),
                    "width": width,
                    "height": height,
                    "brightness": brightness
                }
                
                return img_tensor, metadata
                
        except Exception as e:
            print(f"Warning: Failed to process {image_path}: {e}", file=sys.stderr)
            return None, None

    def get_image_embedding(self, image_tensor: torch.Tensor, use_tta: bool = False) -> np.ndarray:
        """Получает эмбеддинг изображения. TTA: усреднение с эмбеддингом горизонтального флипа."""
        with torch.no_grad():
            emb_orig = self.model.encode_image(image_tensor)

            if use_tta:
                # Отражаем по ширине: тензор формы [1, C, H, W], ширина = -1
                flipped = torch.flip(image_tensor, dims=[-1])
                emb_flip = self.model.encode_image(flipped)

                # Нормализуем по отдельности, усредняем и снова нормализуем
                emb_orig = emb_orig / torch.norm(emb_orig, dim=1, keepdim=True)
                emb_flip = emb_flip / torch.norm(emb_flip, dim=1, keepdim=True)
                emb = (emb_orig + emb_flip) / 2.0
                emb = emb / torch.norm(emb, dim=1, keepdim=True)
            else:
                emb = emb_orig
                emb = emb / torch.norm(emb, dim=1, keepdim=True)

            return emb.cpu().numpy().flatten()

    def get_text_embedding(self, text: str) -> np.ndarray:
        """Получает эмбеддинг текста с кэшированием на уровне процесса"""
        if text in self._text_embedding_cache:
            return self._text_embedding_cache[text]

        with torch.no_grad():
            text_tokens = open_clip.tokenize([text]).to(self.device)
            embedding = self.model.encode_text(text_tokens)
            embedding = embedding / torch.norm(embedding, dim=1, keepdim=True)
            arr = embedding.cpu().numpy().flatten()
            self._text_embedding_cache[text] = arr
            return arr

    def index_images(self, images_dir: Path, index_dir: Path, use_tta: bool = False):
        """Индексирует все изображения в директории"""
        print(f"Indexing images from {images_dir}", file=sys.stderr)
        
        # Создаем директорию индекса
        index_dir.mkdir(parents=True, exist_ok=True)
        
        # Получаем список изображений
        image_files = self.get_image_files(images_dir)
        print(f"Found {len(image_files)} images", file=sys.stderr)
        
        if not image_files:
            print("No images found!", file=sys.stderr)
            return
        
        # Инициализируем массивы
        embeddings = []
        metadata_list = []
        
        # Обрабатываем изображения
        for image_path in tqdm(image_files, desc="Processing images"):
            img_tensor, metadata = self.load_and_preprocess_image(image_path)
            
            if img_tensor is not None and metadata is not None:
                embedding = self.get_image_embedding(img_tensor, use_tta=use_tta)
                embeddings.append(embedding)
                metadata_list.append(metadata)
        
        if not embeddings:
            print("No valid images processed!", file=sys.stderr)
            return
        
        # Сохраняем эмбеддинги
        embeddings_array = np.array(embeddings, dtype=np.float32)
        np.save(index_dir / "embeddings.npy", embeddings_array)
        
        # Сохраняем метаданные
        with open(index_dir / "meta.jsonl", "w") as f:
            for meta in metadata_list:
                f.write(json.dumps(meta) + "\n")
        
        # Сохраняем вайбы
        with open(index_dir / "vibes.json", "w") as f:
            json.dump(DEFAULT_VIBES, f, indent=2, ensure_ascii=False)
        
        # Создаем пустой файл для текстовых промптов
        with open(index_dir / "text_prompts.json", "w") as f:
            json.dump([], f)
        
        print(f"Indexed {len(embeddings)} images → {index_dir}", file=sys.stderr)

    def load_index(self, index_dir: Path) -> Tuple[np.ndarray, List[Dict], Dict]:
        """Загружает индекс из директории"""
        if not index_dir.exists():
            print(f"Error: Index directory {index_dir} does not exist", file=sys.stderr)
            sys.exit(2)
        
        # Загружаем эмбеддинги
        embeddings_path = index_dir / "embeddings.npy"
        if not embeddings_path.exists():
            print(f"Error: embeddings.npy not found in {index_dir}", file=sys.stderr)
            sys.exit(2)
        
        embeddings = np.load(embeddings_path)
        
        # Загружаем метаданные
        meta_path = index_dir / "meta.jsonl"
        if not meta_path.exists():
            print(f"Error: meta.jsonl not found in {index_dir}", file=sys.stderr)
            sys.exit(2)
        
        metadata = []
        with open(meta_path, "r") as f:
            for line in f:
                metadata.append(json.loads(line.strip()))
        
        # Загружаем вайбы
        vibes_path = index_dir / "vibes.json"
        if not vibes_path.exists():
            print(f"Error: vibes.json not found in {index_dir}", file=sys.stderr)
            sys.exit(2)
        
        with open(vibes_path, "r") as f:
            vibes = json.load(f)
        
        return embeddings, metadata, vibes

    def parse_vibes(self, vibes_str: str, available_vibes: Dict) -> List[Tuple[str, float]]:
        """Парсит строку вайбов в список (имя, вес)"""
        if not vibes_str:
            return []
        
        vibes = []
        for pair in vibes_str.split(','):
            if ':' not in pair:
                continue
            name, weight_str = pair.split(':', 1)
            name = name.strip()
            weight = float(weight_str.strip())
            
            if name not in available_vibes:
                print(f"Warning: Unknown vibe '{name}', skipping", file=sys.stderr)
                continue
            
            vibes.append((name, weight))
        
        # Нормализуем веса
        total_weight = sum(w for _, w in vibes)
        if total_weight > 0:
            vibes = [(name, w / total_weight) for name, w in vibes]
        
        return vibes

    def apply_filters(self, metadata: List[Dict], orientation: str, 
                     min_brightness: float, max_brightness: float) -> List[int]:
        """Применяет фильтры и возвращает индексы подходящих изображений"""
        valid_indices = []
        
        for i, meta in enumerate(metadata):
            # Фильтр по ориентации
            if orientation != "any":
                is_landscape = meta["width"] >= meta["height"]
                if orientation == "landscape" and not is_landscape:
                    continue
                if orientation == "portrait" and is_landscape:
                    continue
            
            # Фильтр по яркости
            brightness = meta["brightness"]
            if brightness < min_brightness or brightness > max_brightness:
                continue
            
            valid_indices.append(i)
        
        return valid_indices

    def calculate_penalties(self, metadata: List[Dict], indices: List[int],
                          orientation: str, min_brightness: float, max_brightness: float) -> np.ndarray:
        """Вычисляет штрафы за несоответствие фильтрам"""
        penalties = np.zeros(len(indices))
        
        for i, idx in enumerate(indices):
            meta = metadata[idx]
            
            # Штраф за ориентацию
            if orientation != "any":
                is_landscape = meta["width"] >= meta["height"]
                if orientation == "landscape" and not is_landscape:
                    penalties[i] += 0.25
                elif orientation == "portrait" and is_landscape:
                    penalties[i] += 0.25
            
            # Штраф за яркость
            brightness = meta["brightness"]
            if brightness < min_brightness:
                penalties[i] += 0.10
            if brightness > max_brightness:
                penalties[i] += 0.10
        
        return penalties

    def mmr_diversify(self, scores: np.ndarray, embeddings: np.ndarray, 
                     indices: List[int], k: int, lambda_mmr: float) -> List[int]:
        """Применяет MMR диверсификацию для уменьшения схожести результатов"""
        if len(indices) <= k:
            return indices
        
        selected = []
        remaining = list(range(len(indices)))
        
        # Выбираем первый элемент с максимальным скором
        first_idx = np.argmax(scores)
        selected.append(first_idx)
        remaining.remove(first_idx)
        
        # Выбираем остальные элементы
        while len(selected) < k and remaining:
            best_score = -float('inf')
            best_idx = None
            
            for idx in remaining:
                # Скор с учетом диверсификации
                relevance = scores[idx]
                
                # Максимальная схожесть с уже выбранными
                max_similarity = 0.0
                if selected:
                    similarities = np.dot(embeddings[indices[idx]], 
                                        embeddings[[indices[i] for i in selected]].T)
                    max_similarity = np.max(similarities)
                
                mmr_score = lambda_mmr * relevance - (1 - lambda_mmr) * max_similarity
                
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx
            
            if best_idx is not None:
                selected.append(best_idx)
                remaining.remove(best_idx)
            else:
                break
        
        return [indices[i] for i in selected]

    def search(self, index_dir: Path, context: str, vibes: str = None, k: int = 10,
              orientation: str = "any", min_brightness: float = 0.0, 
              max_brightness: float = 1.0, alpha: float = ALPHA, beta: float = BETA,
              mmr_lambda: float = LAMBDA_MMR) -> List[Dict]:
        """Выполняет поиск по индексу"""
        print(f"Searching in {index_dir}", file=sys.stderr)
        
        # Загружаем индекс
        embeddings, metadata, available_vibes = self.load_index(index_dir)
        
        if len(embeddings) == 0:
            print("Error: Empty index", file=sys.stderr)
            sys.exit(2)
        
        # Применяем фильтры
        valid_indices = self.apply_filters(metadata, orientation, min_brightness, max_brightness)
        
        if not valid_indices:
            print("No images match the filters", file=sys.stderr)
            return []
        
        print(f"Found {len(valid_indices)} images matching filters", file=sys.stderr)
        
        # Получаем эмбеддинг контекста
        context_embedding = self.get_text_embedding(context)
        
        # Вычисляем скоринг
        scores = np.zeros(len(valid_indices))
        
        # Контекстный скор
        context_scores = np.dot(embeddings[valid_indices], context_embedding)
        scores += alpha * context_scores
        
        # Вайб-скор
        parsed_vibes: List[Tuple[str, float]] = []
        vibe_embeds: Dict[str, np.ndarray] = {}
        if vibes:
            parsed_vibes = self.parse_vibes(vibes, available_vibes)
            # Предрассчитываем эмбеддинги вайбов один раз
            for vibe_name, _ in parsed_vibes:
                vibe_text = available_vibes[vibe_name]
                vibe_embeds[vibe_name] = self.get_text_embedding(vibe_text)

            for vibe_name, weight in parsed_vibes:
                vibe_scores = np.dot(embeddings[valid_indices], vibe_embeds[vibe_name])
                scores += beta * weight * vibe_scores
        
        # Применяем штрафы
        penalties = self.calculate_penalties(metadata, valid_indices, 
                                           orientation, min_brightness, max_brightness)
        scores -= penalties
        
        # Применяем MMR диверсификацию
        final_indices = self.mmr_diversify(scores, embeddings, valid_indices, k, lambda_mmr=mmr_lambda)
        
        # Формируем результат
        results = []
        for idx in final_indices:
            meta = metadata[idx]
            score = scores[valid_indices.index(idx)]
            
            # Вычисляем компоненты скора для объяснения
            context_score = context_scores[valid_indices.index(idx)]
            vibe_score = 0.0
            if parsed_vibes:
                for vibe_name, weight in parsed_vibes:
                    vibe_score += weight * float(np.dot(embeddings[idx], vibe_embeds[vibe_name]))
            
            penalty = penalties[valid_indices.index(idx)]
            
            result = {
                "path": meta["path"],
                "score": float(score),
                "explain": {
                    "cos_ctx": float(context_score),
                    "cos_vibes": float(vibe_score),
                    "quality": 0.0,  # Заглушка для будущего расширения
                    "penalty": float(penalty)
                }
            }
            results.append(result)
        
        return results


def main():
    parser = argparse.ArgumentParser(description="Zero-shot ретривер кото-мемов «по вайбу»")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Команда index
    index_parser = subparsers.add_parser('index', help='Index images')
    index_parser.add_argument('--images-dir', type=Path, required=True,
                             help='Directory with images to index')
    index_parser.add_argument('--index-dir', type=Path, default=Path('./cats_index'),
                             help='Directory to save index (default: ./cats_index)')
    index_parser.add_argument('--tta', action='store_true', default=False,
                             help='Use horizontal flip TTA when computing image embeddings')
    
    # Команда search
    search_parser = subparsers.add_parser('search', help='Search images')
    search_parser.add_argument('--index-dir', type=Path, required=True,
                              help='Directory with index')
    search_parser.add_argument('--context', type=str, required=True,
                              help='Text description/context')
    search_parser.add_argument('--vibes', type=str,
                              help='Vibes in format "name:weight,name:weight"')
    search_parser.add_argument('--k', type=int, default=10,
                              help='Number of results (default: 10)')
    search_parser.add_argument('--orientation', choices=['any', 'landscape', 'portrait'],
                              default='any', help='Image orientation filter')
    search_parser.add_argument('--min-brightness', type=float, default=0.0,
                              help='Minimum brightness [0..1]')
    search_parser.add_argument('--max-brightness', type=float, default=1.0,
                              help='Maximum brightness [0..1]')
    search_parser.add_argument('--alpha', type=float, default=ALPHA,
                              help='Weight for context cosine (default: 0.65)')
    search_parser.add_argument('--beta', type=float, default=BETA,
                              help='Weight for vibes cosine total (default: 0.35)')
    search_parser.add_argument('--mmr-lambda', dest='mmr_lambda', type=float, default=LAMBDA_MMR,
                              help='MMR diversification lambda (default: 0.5)')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Инициализируем ретривер
    retriever = MemeRetriever()
    
    if args.command == 'index':
        if not args.images_dir.exists():
            print(f"Error: Images directory {args.images_dir} does not exist", file=sys.stderr)
            sys.exit(2)
        
        retriever.index_images(args.images_dir, args.index_dir, use_tta=args.tta)
    
    elif args.command == 'search':
        results = retriever.search(
            index_dir=args.index_dir,
            context=args.context,
            vibes=args.vibes,
            k=args.k,
            orientation=args.orientation,
            min_brightness=args.min_brightness,
            max_brightness=args.max_brightness,
            alpha=args.alpha,
            beta=args.beta,
            mmr_lambda=args.mmr_lambda
        )
        
        # Выводим результат в stdout как одну строку JSON
        print(json.dumps(results, ensure_ascii=False))


if __name__ == "__main__":
    main()
