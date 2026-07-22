# BERT_Comparsion

## Описание проекта

Данный проект посвящен исследованию методов интерпретации моделей глубокого обучения на примере языковой модели BERT.

---

# Используемые технологии

## Machine Learning

- Python 3.11
- PyTorch
- Hugging Face Transformers
- BERT


## Explainable AI

- SHAP
- LIME
- Captum Integrated Gradients


## Обработка данных

- Hugging Face Datasets
- Pandas
- Scikit-learn


## Визуализация

- Matplotlib
- Seaborn


---

# Архитектура проекта

```text
BERT_Comparsion/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── splits/
│
├── models/
│   └── bert_imdb/
│
├── results/
│   ├── explanations/
│   ├── comparison/
│   ├── figures/
│   └── metrics/
│
├── download.py
├── requirements.txt
├── preprocess.py
├── split.py
├── train.py
├── explain_shap.py
├── explain_lime.py
├── explain_ig.py
├── compare_methods.py
├── run_experiment.py
├── config.py
└── README.md
```

---

# Подготовка окружения


Создание виртуального окружения:

- python -m venv venv

Активация: 

- Windows: venv\Scripts\activate

Установка зависимостей:

- pip install -r requirements.txt

При ошибке в установке зависимостей, в свойствах сети лучше отключить IP версии 6.

---

# Запуск проекта

- python run_experiment.py

Потом для проверки, загрузилось ли все, проработали ли все методы запустить:

- python check_project.py


---

