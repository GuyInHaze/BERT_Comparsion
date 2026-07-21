from pathlib import Path


from datasets import load_from_disk


from transformers import AutoTokenizer

import re



from config import (

    RAW_DATA_DIR,

    PROCESSED_DATA_DIR,

    MODEL_NAME,

    MAX_LENGTH,
    
    MODEL_DIR

)




PROCESSED_DATA_DIR.mkdir(

    parents=True,

    exist_ok=True

)



def clean_text(text):

    text = re.sub(
        r"<br\s*/?>",
        " ",
        text
    )

    text = text.strip()

    text = " ".join(text.split())

    return text




def tokenize_dataset(dataset):


    print()

    print(

        "Loading tokenizer..."

    )



    tokenizer = AutoTokenizer.from_pretrained(

        MODEL_NAME,
        use_fast=True

    )





    def process(batch):


        texts = [

            clean_text(text)

            for text in batch["text"]

        ]



        tokens = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
            return_attention_mask=True,
            return_token_type_ids=True
        )



        tokens["label"] = batch["label"]



        return tokens





    tokenized = dataset.map(

        process,

        batched=True,

    )



    tokenized.set_format(

        "torch"

    )

    tokenizer.save_pretrained(

        MODEL_DIR

    )



    return tokenized




def main():


    print()

    print(" " * 60)

    print(

        "PREPROCESS IMDb DATASET"

    )

    print(" " * 60)



    dataset = load_from_disk(

        str(RAW_DATA_DIR)

    )



    print()

    print(

        "Original dataset:"

    )

    print(f"Train samples : {len(dataset['train'])}")
    print(f"Test samples  : {len(dataset['test'])}")





    processed = tokenize_dataset(

        dataset

    )





    processed.save_to_disk(

        str(PROCESSED_DATA_DIR)

    )

    loaded = load_from_disk(str(PROCESSED_DATA_DIR))

    assert len(loaded["train"]) == len(processed["train"])
    assert len(loaded["test"]) == len(processed["test"])



    print()

    print(

        "Processed dataset saved:"

    )

    print(

        PROCESSED_DATA_DIR

    )





if __name__ == "__main__":

    main()