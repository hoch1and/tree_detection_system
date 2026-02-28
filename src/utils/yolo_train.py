import argparse
from ultralytics import YOLO
from datetime import datetime
import torch
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description='обучение YOLO сегментации')
    
    parser.add_argument('--project-name', type=str, default='tree_segmentation', help='название проекта')
    parser.add_argument('--experiment-dir', type=str, default=None, help='директория для сохранения (если не указана, создается автоматически)')
    parser.add_argument('--data', type=str, default='dataset/data.yaml', help='путь к data.yaml')
    parser.add_argument('--model', type=str, default='models/yolo26m-seg.pt', help='базовая модель (yolo26m-seg.pt)')
    parser.add_argument('--epochs', type=int, default=120, help='количество эпох')
    parser.add_argument('--batch', type=int, default=16, help='размер батча')
    parser.add_argument('--imgsz', type=int, default=640, help='размер изображения')
    parser.add_argument('--device', type=str, default=None, help='cuda устройство (0,1,2,3 или cpu)')
    parser.add_argument('--patience', type=int, default=30, help='early stopping')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    if args.experiment_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        experiment_name = f'run_{timestamp}'
    else:
        experiment_name = args.experiment_dir
    
    if args.device is None:
        device = 0 if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    if device != 'cpu':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    model = YOLO(args.model)
    
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        patience=args.patience,
        batch=args.batch,
        imgsz=args.imgsz,
        device=device,
        workers=8,
        project=args.project_name,
        name=experiment_name,
        exist_ok=True,
        
        pretrained=True,
        optimizer='auto',
        lr0=0.01,
        weight_decay=0.0005,
        
        seed=42,
        deterministic=True,
        
        save=True,
        save_period=10,
        
        plots=True,
        amp=True,
        fraction=1.0,
        profile=False,
        cache=True,
        
        cos_lr=True,
        warmup_epochs=3,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        
        dropout=0.0,
        
        rect=False,
        overlap_mask=True,
        
        augment=True,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0,
        copy_paste=0.0,
    )
    
    best_model_path = Path(args.project_name) / experiment_name / 'weights' / 'best.pt'
    print(f'\nОбучение завершено')
    print(f'Лучшая модель: {best_model_path}')
    print(f'Метрики сохранены в {Path(args.project_name) / experiment_name}')

if __name__ == '__main__':
    main()