import os
import uuid
from PIL import Image

# Папка с котами
folder = './cats_from_memes'

for filename in os.listdir(folder):
    file_path = os.path.join(folder, filename)
    # Пропускаем папки
    if not os.path.isfile(file_path):
        continue

    # Генерируем UUID
    new_uuid = uuid.uuid4()
    new_filename = f'cat-{new_uuid}.png'
    new_file_path = os.path.join(folder, new_filename)

    try:
        # Открываем изображение и конвертируем в PNG
        with Image.open(file_path) as img:
            img.convert('RGBA').save(new_file_path, 'PNG')
        # Удаляем старый файл
        os.remove(file_path)
        print(f'Converted and renamed: {filename} -> {new_filename}')
    except Exception as e:
        print(f'Error processing {filename}: {e}')