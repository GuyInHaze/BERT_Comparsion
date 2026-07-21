from pathlib import Path


import torch



from torch.utils.data import Dataset



from datasets import load_from_disk



from transformers import BertTokenizer



from config import (
    SPLIT_DATA_DIR,
    MODEL_NAME,
    MAX_LENGTH
)




class IMDbTorchDataset(Dataset):


    def __init__(

            self,

            dataset

    ):


        self.dataset = dataset




    def __len__(self):


        return len(

            self.dataset

        )





    def __getitem__(

            self,

            index

    ):


        item = self.dataset[index]



        return {


            "input_ids":

                torch.tensor(

                    item["input_ids"],

                    dtype=torch.long

                ),



            "attention_mask":

                torch.tensor(

                    item["attention_mask"],

                    dtype=torch.long

                ),



            "label":

                torch.tensor(

                    item["label"],

                    dtype=torch.long

                )

        }





def load_dataset_splits():


    train_path = (

        SPLIT_DATA_DIR /

        "train"

    )


    val_path = (

        SPLIT_DATA_DIR /

        "validation"

    )


    test_path = (

        SPLIT_DATA_DIR /

        "test"

    )



    train_dataset = load_from_disk(

        str(train_path)

    )


    val_dataset = load_from_disk(

        str(val_path)

    )


    test_dataset = load_from_disk(

        str(test_path)

    )



    return (


        IMDbTorchDataset(

            train_dataset

        ),



        IMDbTorchDataset(

            val_dataset

        ),



        IMDbTorchDataset(

            test_dataset

        )

    )




def tokenize_dataset(dataset):


    tokenizer = BertTokenizer.from_pretrained(

        MODEL_NAME

    )



    def tokenize(batch):


        return tokenizer(

            batch["text"],

            padding="max_length",

            truncation=True,

            max_length=MAX_LENGTH

        )



    tokenized = dataset.map(

        tokenize,

        batched=True

    )



    tokenized = tokenized.remove_columns(

        [

            "text"

        ]

    )



    tokenized.set_format(

        "torch"

    )



    return tokenized