from pathlib import Path

import json


import torch


import matplotlib.pyplot as plt



from transformers import (

    AutoTokenizer,

    AutoModelForSequenceClassification

)



from captum.attr import (

    IntegratedGradients

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



def forward_func(

        input_ids,

        attention_mask,

        model

):



    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask
    )

    return outputs.logits[:, 1]




def get_embedding_layer(model):


    return (

        model

        .bert

        .embeddings

        .word_embeddings

    )




def explain_text(

        text,

        model,

        tokenizer

):


    encoded = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        return_attention_mask=True,
        return_tensors="pt"
    )



    input_ids = (

        encoded["input_ids"]

        .to(DEVICE)

    )


    attention_mask = (

        encoded["attention_mask"]

        .to(DEVICE)

    )



    embedding_layer = get_embedding_layer(

        model

    )



    ig = IntegratedGradients(

        lambda embeddings,
               mask:

            forward_from_embeddings(

                embeddings,

                mask,

                model

            )

    )



    embeddings = embedding_layer(

        input_ids

    )



    baseline = torch.zeros_like(

        embeddings

    )



    attributions = ig.attribute(

        embeddings,

        baselines=baseline,

        additional_forward_args=(

            attention_mask

        ),

        n_steps=50

    )



    token_importance = (

        attributions

        .sum(dim=-1)

        .squeeze(0)

    )



    tokens = tokenizer.convert_ids_to_tokens(

        input_ids.squeeze(0)

    )



    scores = (
        token_importance
        .detach()
        .cpu()
        .numpy()
    )

    return tokens, scores



def forward_from_embeddings(

        embeddings,

        attention_mask,

        model

):


    outputs = model(

        inputs_embeds=embeddings,

        attention_mask=attention_mask

    )


    probabilities = torch.softmax(

        outputs.logits,

        dim=1

    )


    return probabilities[:,1]




def save_result(

        text,

        tokens,

        scores

):


    data = []



    for token, score in zip(

            tokens,

            scores

    ):


        data.append({

            "token": str(token),


            "importance": float(score)

        })



    output = (

        OUTPUT_PATH /

        "integrated_gradients.json"

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

                    data

            },

            file,

            indent=4,

            ensure_ascii=False

        )



    print(

        f"Saved: {output}"

    )




def visualize(

        tokens,

        scores

):


    tokens = tokens[:25]

    scores = scores[:25]



    plt.figure(

        figsize=(12,6)

    )



    colors = [
        "green" if value > 0 else "red"
        for value in scores
    ]

    plt.bar(
        tokens,
        scores,
        color=colors
    )



    plt.xticks(

        rotation=90

    )


    plt.title(

        "Integrated Gradients token importance"

    )


    plt.tight_layout()



    plt.savefig(

        OUTPUT_PATH /

        "integrated_gradients.png",

        dpi=300

    )


    plt.close()




def main():


    model, tokenizer = load_model()



    dataset = load_from_disk(

        str(DATASET_PATH)

    )



    example = dataset["test"][2]


    text = example["text"]



    print()

    print(

        "Text:"

    )

    print(

        text[:500]

    )


    print()

    print(

        "True label:",

        LABELS[example["label"].item()]

    )

    encoded = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )

    encoded = {
        k: v.to(DEVICE)
        for k, v in encoded.items()
    }   

    with torch.no_grad():
        logits = model(**encoded).logits
        probs = torch.softmax(logits, dim=1)

    predicted = probs.argmax(dim=1).item()

    print()
    print("Predicted label:", LABELS[predicted])
    print(f"Confidence: {probs[0, predicted]:.4f}")


    tokens, scores = explain_text(

        text,

        model,

        tokenizer

    )



    save_result(

        text,

        tokens,

        scores

    )



    visualize(

        tokens,

        scores

    )





if __name__ == "__main__":

    main()