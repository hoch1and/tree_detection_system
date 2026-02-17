import pickle
import pandas as pd
import numpy as np
import cv2
import os
from collections import defaultdict

def create_masks_for_all_images(data, output_dir='masks'):
    os.makedirs(output_dir, exist_ok=True)
    images_dict = defaultdict(list)
    
    for idx, row in data.iterrows():
        file_name = row['file_name']
        segmentation = row['segmentation']
        class_id = row['category_id']
        
        images_dict[file_name].append({
            'segmentation': segmentation,
            'class_id': class_id,
            'index': idx
        })
    
    print(f'Найдено {len(images_dict)} уникальных изображений')
    print(f'Всего объектов: {len(data)}')
    
    masks_info = []
    print_every = 250
    
    for idx, (file_name, objects) in enumerate(images_dict.items()):
        img_path = f'../train/{file_name}'
        img = cv2.imread(img_path)
        
        height, width = img.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        
        for obj in objects:
            segmentation = obj['segmentation']
            class_id = obj['class_id']
            
            points = np.array(segmentation, dtype=np.int32).reshape((-1, 2))
            points_poly = points.reshape((-1, 1, 2))

            if class_id == 1:
                cv2.fillPoly(mask, [points_poly], 1)
            elif class_id == 2:
                cv2.fillPoly(mask, [points_poly], 2)
            else:
                print(f"Неизвестный class_id: {class_id}")
        
        mask_filename = file_name.replace('.jpg', '.png')
        mask_path = os.path.join(output_dir, mask_filename)
        cv2.imwrite(mask_path, mask)

        masks_info.append({
            'file_name': file_name,
            'mask_name': mask_filename,
            'num_objects': len(objects),
            'classes': [obj['class_id'] for obj in objects],
            'image_size': (height, width)
        })

        if idx % print_every == 0:
            print(f'Создана маска для {file_name}: {len(objects)} объектов, классы: {set([obj["class_id"] for obj in objects])}')
    
    print(f'\nВсего создано масок: {len(masks_info)}')
    return masks_info

def save_masks_info(masks_info, output_path='../masks/masks_info.pkl'):
    with open(output_path, 'wb') as f:
        pickle.dump(masks_info, f)
        
    print(f'Информация о масках сохранена в {output_path}')

if __name__ == '__main__':
    data = pd.read_pickle('../data/dataset.pkl')
    masks_info = create_masks_for_all_images(data, output_dir='../masks')
    save_masks_info(masks_info, '../masks/masks_info.pkl')
    
    print('Маски сгенерированы')