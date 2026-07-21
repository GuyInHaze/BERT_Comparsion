from pathlib import Path
import importlib




ROOT_DIR = Path(__file__).resolve().parent


DATA_DIR = ROOT_DIR / "data"

MODEL_DIR = ROOT_DIR / "models" / "bert_imdb"

RESULTS_DIR = ROOT_DIR / "results"




REQUIRED_PACKAGES = [

    "torch",

    "transformers",

    "datasets",

    "shap",

    "lime",

    "captum",

    "sklearn",

    "pandas",

    "numpy",

    "matplotlib"

]





def check_packages():

    print()

    print("=" * 60)

    print("CHECKING PACKAGES")

    print("=" * 60)



    success = True



    for package in REQUIRED_PACKAGES:


        try:

            importlib.import_module(

                package

            )

            print(

                f"[OK] {package}"

            )


        except ImportError:


            print(

                f"[MISSING] {package}"

            )


            success = False



    return success




def check_directories():

    print()

    print(" " * 60)

    print("CHECKING DIRECTORIES")

    print(" " * 60)



    directories = [

        DATA_DIR,

        MODEL_DIR,

        RESULTS_DIR

    ]



    success = True



    for directory in directories:


        if directory.exists():


            print(

                f"[OK] {directory}"

            )


        else:


            print(

                f"[MISSING] {directory}"

            )


            success = False



    return success




def check_dataset():

    print()

    print(" " * 60)

    print("CHECKING DATASET")

    print(" " * 60)



    required_paths = [

        DATA_DIR /

        "raw" /

        "imdb",


        DATA_DIR /

        "processed" /

        "imdb",


        DATA_DIR /

        "splits" /

        "imdb"

    ]



    success = True



    for path in required_paths:


        if path.exists():


            print(

                f"[OK] {path}"

            )


        else:


            print(

                f"[MISSING] {path}"

            )


            success = False



    return success



def check_model():

    print()

    print(" " * 60)

    print("CHECKING MODEL")

    print(" " * 60)



    required_files = [

        "config.json",

        "model.safetensors",

        "tokenizer.json"

    ]



    success = True



    for file in required_files:


        path = MODEL_DIR / file



        if path.exists():


            print(

                f"[OK] {file}"

            )


        else:


            print(

                f"[MISSING] {file}"

            )


            success = False



    return success




def check_results():

    print()

    print(" " * 60)

    print("CHECKING RESULTS")

    print(" " * 60)



    if RESULTS_DIR.exists():


        files = list(

            RESULTS_DIR.rglob("*")

        )


        files = [

            f for f in files

            if f.is_file()

        ]



        print(

            f"[OK] Found {len(files)} result files"

        )


        return True



    else:


        print(

            "[MISSING] results folder"

        )


        return False



def main():


    results = []



    results.append(

        check_packages()

    )


    results.append(

        check_directories()

    )


    results.append(

        check_dataset()

    )


    results.append(

        check_model()

    )


    results.append(

        check_results()

    )



    print()

    print(" " * 60)

    print("PROJECT STATUS")

    print(" " * 60)



    if all(results):


        print(

            "PROJECT READY"

        )


    else:


        print(

            "PROJECT NOT READY"

        )


    print(" " * 60)





if __name__ == "__main__":

    main()