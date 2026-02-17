import json
import pandas as pd

annotations_path = '../train/_annotations.coco.json'

def parse_json_annts(path):
    with open(path, 'r', encoding='utf-8') as file:
        annotations = json.loads(file.read())

    images = pd.DataFrame(annotations['images'])
    annotations = pd.DataFrame(annotations['annotations'])
    final_annts = annotations.merge(
        images, 
        how='left', 
        left_on='image_id', 
        right_on='id'
    )

    print(f'keys: {list(annotations.keys())}')
    print(f'n images: {images.shape[0]}')
    print(f'n objects overall: {annotations.shape[0]}')
    
    return final_annts

data_path = '../data/raw_dataset.csv'
if __name__ == '__main__':
    annotations = parse_json_annts(annotations_path)
    annotations.to_csv(data_path, index=False)

    print(f'complete & saved in {data_path}')