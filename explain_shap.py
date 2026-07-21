from pathlib import Path

import json


import torch

import shap


from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)


from datasets import load_from_disk



import matplotlib.pyplot as plt



ROOT_DIR = Path(__file__).resolve().parent



MODEL_PATH = (
    ROOT_DIR /
    "models" /
    "bert_imdb"
)



DATASET_PATH = (
    ROOT_DIR /
    "data" /
    "splits" /
    "imdb"
)



OUTPUT_PATH = (
    ROOT_DIR /
    "results" /
    "explanations"
)



OUTPUT_PATH.mkdir(

    parents=True,

    exist_ok=True

)




MAX_LENGTH = 256


DEVICE = torch.device(

    "cuda"
    if torch.cuda.is_available()
    else "cpu"

)



LABELS = {

    0: "NEGATIVE",

    1: "POSITIVE"

}




def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model directory not found:\n{MODEL_PATH}\n\nRun train.py first."
        )


    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        use_fast=True
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH
    )


    model.to(

        DEVICE

    )


    model.eval()



    return (

        model,

        tokenizer

    )




def model_predict(

        texts,

        model,

        tokenizer

):


    encoded = tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_attention_mask=True,
        return_tensors="pt"
    )



    encoded = {

        key: value.to(DEVICE)

        for key, value

        in encoded.items()

    }



    with torch.no_grad():


        outputs = model(

            **encoded

        )


        probabilities = torch.softmax(

            outputs.logits,

            dim=1

        )



    return (

        probabilities

        .cpu()

        .numpy()

    )



def explain_text(

        text,

        model,

        tokenizer

):


    print()

    print(

        "Creating SHAP explainer..."

    )



    predict_fn = lambda texts: model_predict(
        texts,
        model,
        tokenizer
    )

    explainer = shap.Explainer(
        predict_fn,
        tokenizer
    )



    shap_values = explainer(

        [text]

    )



    return shap_values




def save_explanation(

        shap_values,

        text

):


    tokens = (

        shap_values.data[0]

        .tolist()

    )


    values = (

        shap_values.values[0]

    )



    result = []



    for token, value in zip(

        tokens,

        values[:,1]

    ):


        result.append({
            "token": str(token),
            "importance": float(value)
        })



    path = (

        OUTPUT_PATH /

        "shap_explanation.json"

    )


    with open(

        path,

        "w",

        encoding="utf-8"

    ) as file:


        json.dump(

            {

                "text":

                    text,


                "tokens":

                    result

            },

            file,

            indent=4,

            ensure_ascii=False

        )



    print(

        f"Saved: {path}"

    )




def visualize(

        shap_values

):


    shap.plots.text(

        shap_values[0]

    )




def main():


    model, tokenizer = load_model()



    dataset = load_from_disk(

        str(DATASET_PATH)

    )



    example = dataset["test"][0]



    text = example["text"]



    print()

    print(

        "Text example:"

    )

    print(text[:500])



    print()

    print(

        "True label:",

        LABELS[example["label"].item()]

    )



    shap_values = explain_text(

        text,

        model,

        tokenizer

    )

    prediction = model_predict(
        [text],
        model,
        tokenizer
    )

    predicted_class = prediction.argmax(axis=1)[0]

    print()
    print("Predicted label:", LABELS[predicted_class])
    print(
        f"Confidence: {prediction[0][predicted_class]:.4f}"
    )



    save_explanation(

        shap_values,

        text

    )



    visualize(

        shap_values

    )





if __name__ == "__main__":

    main()