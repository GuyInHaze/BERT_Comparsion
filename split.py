from pathlib import Path


from datasets import load_from_disk, DatasetDict


from config import (

    PROCESSED_DATA_DIR,

    SPLIT_DATA_DIR,

    SEED

)




SPLIT_DATA_DIR.mkdir(

    parents=True,

    exist_ok=True

)




def create_split():


    print()

    print(" " * 60)

    print(

        "CREATING DATA SPLITS"

    )

    print(" " * 60)



    dataset = load_from_disk(

        str(PROCESSED_DATA_DIR)

    )



    print()

    print(

        "Original dataset:"

    )

    print(

        dataset

    )





    train_data = dataset["train"]





    split = train_data.train_test_split(

        test_size=0.2,

        seed=SEED

    )



    train_dataset = split["train"]


    validation_dataset = split["test"]


    test_dataset = dataset["test"]


    split_dataset = DatasetDict({

        "train": train_dataset,

        "validation": validation_dataset,

        "test": test_dataset

    })


    split_dataset.save_to_disk(

        str(SPLIT_DATA_DIR)

    )




    print()

    print(" " * 60)

    print(

        "SPLIT RESULTS"

    )

    print(" " * 60)



    print(

        "Train:",

        len(train_dataset)

    )


    print(

        "Validation:",

        len(validation_dataset)

    )


    print(

        "Test:",

        len(test_dataset)

    )





def main():


    create_split()



if __name__ == "__main__":

    main()