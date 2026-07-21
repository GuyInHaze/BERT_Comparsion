import json

import random

import numpy as np

import pandas as pd


import torch

from tqdm import tqdm


from sklearn.metrics import (

    accuracy_score,

    precision_score,

    recall_score,

    f1_score,

    roc_auc_score,

    confusion_matrix,

    roc_curve

)


import matplotlib.pyplot as plt



from torch.utils.data import DataLoader



from transformers import (

    BertForSequenceClassification,

    AdamW,

    AutoTokenizer,

    get_linear_schedule_with_warmup

)

from config import *

from utils import load_dataset_splits



def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    
    torch.backends.cudnn.benchmark = False




set_seed(SEED)



MODEL_DIR.mkdir(

    parents=True,

    exist_ok=True

)


METRICS_DIR.mkdir(

    parents=True,

    exist_ok=True

)


FIGURES_DIR.mkdir(

    parents=True,

    exist_ok=True

)



print(

    "Loading dataset..."

)



train_dataset, val_dataset, test_dataset = load_dataset_splits()





train_loader = DataLoader(

    train_dataset,

    batch_size=BATCH_SIZE,

    shuffle=True

)



val_loader = DataLoader(

    val_dataset,

    batch_size=BATCH_SIZE,

    shuffle=True

)



test_loader = DataLoader(

    test_dataset,

    batch_size=BATCH_SIZE,

    shuffle=True

)




model = BertForSequenceClassification.from_pretrained(

    MODEL_NAME,

    num_labels=NUM_CLASSES

)

tokenizer = AutoTokenizer.from_pretrained(

    MODEL_NAME

)


model.to(

    DEVICE

)




optimizer = AdamW(

    model.parameters(),

    lr=LEARNING_RATE,

    weight_decay=WEIGHT_DECAY

)





total_steps = (

    len(train_loader)

    *

    EPOCHS

)



scheduler = get_linear_schedule_with_warmup(

    optimizer,

    num_warmup_steps=0,

    num_training_steps=total_steps

)




def train_epoch():


    model.train()


    total_loss = 0



    for batch in tqdm(

        train_loader,

        desc="Training"

    ):


        optimizer.zero_grad()



        input_ids = batch["input_ids"].to(

            DEVICE

        )


        mask = batch["attention_mask"].to(

            DEVICE

        )


        labels = batch["label"].to(

            DEVICE

        )



        output = model(

            input_ids=input_ids,

            attention_mask=mask,

            labels=labels

        )



        loss = output.loss



        total_loss += loss.item()



        loss.backward()



        torch.nn.utils.clip_grad_norm_(

            model.parameters(),

            1.0

        )



        optimizer.step()


        scheduler.step()



    return total_loss / len(train_loader)




def evaluate(loader):


    model.eval()



    predictions = []

    probabilities = []

    labels_all = []



    loss_total = 0



    with torch.no_grad():


        for batch in tqdm(

            loader,

            desc="Evaluation"

        ):


            input_ids = batch["input_ids"].to(

                DEVICE

            )


            mask = batch["attention_mask"].to(

                DEVICE

            )


            labels = batch["label"].to(

                DEVICE

            )



            output = model(

                input_ids=input_ids,

                attention_mask=mask,

                labels=labels

            )



            loss_total += output.loss.item()



            probs = torch.softmax(

                output.logits,

                dim=1

            )



            preds = torch.argmax(

                probs,

                dim=1

            )



            predictions.extend(

                preds.cpu().numpy()

            )


            probabilities.extend(

                probs[:,1]

                .cpu()

                .numpy()

            )


            labels_all.extend(

                labels.cpu().numpy()

            )



    metrics = {


        "loss":

        loss_total / len(loader),



        "accuracy":

        accuracy_score(

            labels_all,

            predictions

        ),



        "precision":

        precision_score(

            labels_all,

            predictions,

            zero_division=0

        ),



        "recall":

        recall_score(

            labels_all,

            predictions,

            zero_division=0

        ),



        "f1":

        f1_score(

            labels_all,

            predictions,

            zero_division=0

        ),



        "roc_auc":

        roc_auc_score(

            labels_all,

            probabilities

        ) if len(set(labels_all)) > 1 else 0

    }



    return (

        metrics,

        predictions,

        probabilities,

        labels_all

    )




history = []


best_f1 = 0



for epoch in range(EPOCHS):


    print()

    print(

        f"EPOCH {epoch+1}/{EPOCHS}"

    )


    train_loss = train_epoch()



    val_metrics, _, _, _ = evaluate(

        val_loader

    )



    val_metrics["train_loss"] = train_loss



    history.append(

        val_metrics

    )



    print(val_metrics)



    if val_metrics["f1"] > best_f1:


        best_f1 = val_metrics["f1"]



        print(

            "Saving best model..."

        )



        model.save_pretrained(

            MODEL_DIR

        )

        tokenizer.save_pretrained(

            MODEL_DIR

        )
        
    else:
        
        print(

            "Sorry, couldn't find the best model"

        )

if not (MODEL_DIR / "config.json").exists():

    print(

        "Saving final model..."

    )


    model.save_pretrained(

        MODEL_DIR

    )

    tokenizer.save_pretrained(

        MODEL_DIR

    )
    




print(

    "Loading best checkpoint..."

)


if (MODEL_DIR / "config.json").exists():

    model = BertForSequenceClassification.from_pretrained(

        MODEL_DIR

    )

else:

    print(

        "Best checkpoint not found. Using current model."

    )


model.to(

    DEVICE

)



test_metrics, preds, probs, labels = evaluate(

    test_loader

)



print()

print(

    "TEST RESULTS"

)



for k,v in test_metrics.items():

    print(

        k,

        ":",

        round(v,4)

    )




with open(

    METRICS_DIR /

    "test_metrics.json",

    "w"

) as f:


    json.dump(

        test_metrics,

        f,

        indent=4

    )





pd.DataFrame(

    history

).to_csv(

    METRICS_DIR /

    "training_history.csv",

    index=False

)




cm = confusion_matrix(

    labels,

    preds

)



plt.figure(

    figsize=(6,5)

)


plt.imshow(cm)



plt.title(

    "Confusion Matrix"

)



plt.savefig(

    FIGURES_DIR /

    "confusion_matrix.png",

    dpi=300

)



plt.close()





print()

print(

    "Training completed"

)