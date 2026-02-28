```python
[![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/opencv-%23white.svg?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-%23FF6F00.svg?style=for-the-badge&logo=YOLO&logoColor=white)](https://ultralytics.com/)
[![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
[![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)](https://matplotlib.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![scikit-image](https://img.shields.io/badge/scikit--image-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-image.org/)

# Система обнаружения деревьев для автономного робота

## Описание проекта
Разработка системы компьютерного зрения для обнаружения деревьев в лесной местности с целью обеспечения навигации автономного робота. Система позволяет распознавать деревья и дороги (классы: tree, road) для планирования безопасного маршрута.

## Реализованный функционал

### 1. Подготовка данных
- **Загрузка датасета**: импорт изображений с размеченными объектами из `dataset.pkl`
- **Конвертация bbox**: преобразование разрозненных bbox в YOLO-формат для обучения детекции
- **Визуализация**: анализ качества разметки, выявление проблем с bbox

### 2. Сегментация
- **Генерация масок**: создание бинарных масок для каждого изображения (3 класса: background, tree, road)
- **Конвертация в YOLO-seg**: преобразование PNG масок в полигоны для YOLO сегментации
- **Структурирование**: подготовка датасета в формате train/val с data.yaml

### 3. Обучение
- **Модель**: YOLO26m-seg (предобученная на сегментацию)
- **Метрики**: mAP50-95 = 0.837, mAP50 = 0.988
- **Классы**: tree (mAP50-95=0.834), road (mAP50-95=0.840)
- **Логи**: сохранение всех графиков и метрик в `runs/segment/`

### 4. Визуализация
- Графики обучения (precision, recall, F1, confusion matrix)
- Примеры предсказаний: оригинал + ground truth + предсказанная маска

## Структура репозитория
```markdown
src
 ├── dataset/                  # Датасет для YOLO
 │   ├── train/
 │   │   ├── images/           # тренировочные изображения
 │   │   └── labels/           # полигоны сегментации
 │   ├── val/
 │   │   ├── images/           # валидационные изображения
 │   │   └── labels/           # полигоны сегментации
 │   └── data.yaml             # конфиг датасета
 ├── data/
 │   └── dataset.pkl           # исходные аннотации
 ├── train/                    # исходные изображения
 ├── masks/                    # PNG маски сегментации
 ├── yolo_seg_labels/          # сконвертированные полигоны
 ├── bboxes/                   # YOLO bbox (для детекции)
 ├── runs/                     # результаты обучения
 ├── models/                   # модели
 ├── notebooks/                # ноутбуки с тестами
 └── utils/
requirements.txt
README.md
.gitignore
LICENSE
```

## Использование

### Обучение
```bash
python utils/yolo_train.py --project-name tree_segmentation --experiment-dir yolo_seg_training
```

### Визуализация
```bash
jupyter notebook notebooks/yolo_test.ipynb
```