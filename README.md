[![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/opencv-%23white.svg?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)](https://www.tensorflow.org/)
[![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
[![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)](https://matplotlib.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)

# Система обнаружения деревьев для автономного робота

## Описание проекта
Разработка системы компьютерного зрения для обнаружения деревьев в лесной местности с целью обеспечения навигации автономного робота. Система позволяет распознавать деревья и препятствия в режиме реального времени для планирования безопасного маршрута.

## Реализованный функционал

### 1. Подготовка данных
- **Загрузка датасета**: импорт изображений с размеченными объектами (деревья)
- **Очистка данных**: удаление дубликатов, проверка целостности аннотаций
- **Обработка изображений**: приведение к единому формату, нормализация

### 2. Сегментация
- **Генерация масок**: создание бинарных масок для каждого изображения
- **Мультиклассовая сегментация**: разделение объектов по классам (класс 1, класс 2, фон 0)
- **Группировка объектов**: объединение всех объектов с одного изображения в единую маску
- **Сохранение**: экспорт масок в отдельную директорию `../masks/`

### 3. Структура данных
- Изображения: `../train/`
- Маски сегментации: `../masks/`
- Информация о масках: `../masks/masks_info.pkl`
- Датасет с аннотациями: `../data/dataset.pkl`

### 4. Визуализация
- Функции для отображения изображений с наложенными масками
- Сравнение оригинальных изображений и сгенерированных масок

## Планируемые работы

### 1. Исследование моделей сегментации
Планируется протестировать следующие архитектуры:
- **U-Net** — базовая архитектура для медицинской и семантической сегментации
- **DeepLabV3+** — современная модель с расширенными свертками
- **Mask R-CNN** — для instance-сегментации отдельных деревьев
- **YOLOv26** — для быстрой сегментации в реальном времени

### 2. Обучение и валидация
- Разделение датасета на train/val/test
- Аугментация данных
- Подбор гиперпараметров
- Валидация на тестовой выборке

### 3. Интеграция с роботом
- Оптимизация моделей для работы в реальном времени
- Интеграция с системой навигации ROS
- Тестирование на реальном роботе в лесных условиях

## Структура репозитория
```markdown
src
 ├── data/
 │   ├── raw_dataset.csv        # Сырые изображения с аннотациями 
 │   └── dataset.pkl            # Обработанные аннотации и метаданные
 ├── train/                     # Исходные изображения
 ├── masks/                     # Сгенерированные маски
 ├── notebooks/                 # notebooks для экспериментов
 ├── utils/
 │   ├── mask_generator.py      # Модуль генерации масок
 │   └── annotations_parser.py  # Модуль парсинга аннотаций
README.md
.gitignore
```