from pathlib import Path


from datasets import load_dataset, load_from_disk, DatasetDict



from config import RAW_DATA_DIR, SEED





# -------------------------------------------------
# Создание директории
# -------------------------------------------------

RAW_DATA_DIR.mkdir(

    parents=True,

    exist_ok=True

)



def download_imdb():


    print()

    print(" " * 60)

    print(

        "DOWNLOADING IMDb DATASET"

    )

    print(" " * 60)



    dataset_path = (

        RAW_DATA_DIR

    )



    if any(dataset_path.iterdir()):


        print(

            "Dataset already exists"

        )


        return





    dataset = load_dataset(
        "imdb"
    )

    dataset = dataset.shuffle(SEED)


    dataset["train"] = dataset["train"].select(
        range(3000)
    )


    dataset["test"] = dataset["test"].select(
        range(500)
    )


    print()

    print(

        dataset

    )



    dataset.save_to_disk(

        str(dataset_path)

    )



    print()

    print(

        "Dataset saved:"

    )

    print(

        dataset_path

    )




def show_info(dataset):


    print()

    print(" " * 60)

    print(

        "DATASET INFORMATION"

    )

    print(" " * 60)



    for split in dataset:


        print(

            split,

            len(dataset[split])

        )




def main():


    download_imdb()



    dataset = load_from_disk(

        str(RAW_DATA_DIR)

    )



    show_info(

        dataset

    )





if __name__ == "__main__":

    main()