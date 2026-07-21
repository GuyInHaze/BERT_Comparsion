from pathlib import Path

import torch



ROOT_DIR = Path(__file__).resolve().parent


DATA_DIR = ROOT_DIR / "data"


RAW_DATA_DIR = (

    DATA_DIR /

    "raw" /

    "imdb"

)


PROCESSED_DATA_DIR = (

    DATA_DIR /

    "processed" /

    "imdb"

)


SPLIT_DATA_DIR = (

    DATA_DIR /

    "splits" /

    "imdb"

)



MODEL_DIR = (

    ROOT_DIR /

    "models" /

    "bert_imdb"

)



RESULTS_DIR = (

    ROOT_DIR /

    "results"

)



FIGURES_DIR = (

    RESULTS_DIR /

    "figures"

)



METRICS_DIR = (

    RESULTS_DIR /

    "metrics"

)



EXPLANATIONS_DIR = (

    RESULTS_DIR /

    "explanations"

)



COMPARISON_DIR = (

    RESULTS_DIR /

    "comparison"

)



MODEL_NAME = (

    "bert-base-uncased"

)



NUM_CLASSES = 2



MAX_LENGTH = 128



BATCH_SIZE = 16


EPOCHS = 1


LEARNING_RATE = 2e-5


WEIGHT_DECAY = 0.01



SEED = 42



SHAP_SAMPLES = 100


LIME_SAMPLES = 500


IG_STEPS = 50


TOP_K_TOKENS = 10



DEVICE = torch.device(

    "cuda"

    if torch.cuda.is_available()

    else "cpu"

)


LABELS = {

    0: "NEGATIVE",

    1: "POSITIVE"

}