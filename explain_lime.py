from pathlib import Path

import json


import torch


import numpy as np


import matplotlib.pyplot as plt



from lime.lime_text import LimeTextExplainer



from transformers import (

    AutoTokenizer,

    AutoModelForSequenceClassification

)



from datasets import load_from_disk




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




MAX_LENGTH = 128



DEVICE = torch.device(

    "cuda"

    if torch.cuda.is_available()

    else "cpu"

)



CLASS_NAMES = [

    "NEGATIVE",

    "POSITIVE"

]




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




def predict_proba(

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

        key:

        value.to(DEVICE)

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


    explainer = LimeTextExplainer(

        class_names=CLASS_NAMES

    )



    predict_fn = lambda texts: predict_proba(
        texts,
        model,
        tokenizer
    )

    explanation = explainer.explain_instance(
        text,
        predict_fn,
        num_features=20,
        num_samples=200
    )


    return explanation



def save_explanation(

        explanation,

        text

):


    values = explanation.as_list()



    result = []



    for word, weight in values:


        result.append({
            "word": str(word),
            "importance": float(weight)
        })



    output = (

        OUTPUT_PATH /

        "lime_explanation.json"

    )



    with open(

        output,

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

        f"Saved: {output}"

    )




def visualize(

        explanation

):


    values = (

        explanation.as_list()

    )


    words = [

        item[0]

        for item in values

    ]


    weights = [

        item[1]

        for item in values

    ]



    plt.figure(

        figsize=(10,6)

    )


    colors = [
        "green" if value > 0 else "red"
        for value in weights
    ]

    plt.barh(
        words,
        weights,
        color=colors
    )


    plt.xlabel(

        "Importance"

    )


    plt.title(

        "LIME explanation"

    )


    plt.tight_layout()



    plt.savefig(

        OUTPUT_PATH /

        "lime_visualization.png",

        dpi=300

    )


    plt.close()



def main():


    model, tokenizer = load_model()



    dataset = load_from_disk(

        str(DATASET_PATH)

    )



    example = dataset["test"][1]



    text = example["text"]



    print()

    print(

        "Text:"

    )

    print(

        text[:500]

    )



    print()

    prediction = predict_proba(
        [text],
        model,
        tokenizer
    )

    predicted_class = prediction.argmax(axis=1)[0]

    print()
    print("Predicted class:", CLASS_NAMES[predicted_class])
    print(
        f"Confidence: {prediction[0][predicted_class]:.4f}"
    )



    explanation = explain_text(

        text,

        model,

        tokenizer

    )



    print()

    print(

        explanation.as_list()

    )



    save_explanation(

        explanation,

        text

    )



    visualize(

        explanation

    )





if __name__ == "__main__":

    main()