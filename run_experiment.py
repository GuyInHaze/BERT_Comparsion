import subprocess

import sys

from pathlib import Path





ROOT_DIR = Path(__file__).resolve().parent




def run_script(script):

    print()

    print(" " * 60)

    print(

        f"RUNNING: {script}"

    )

    print(" " * 60)



    result = subprocess.run(

        [

            sys.executable,

            script

        ],

        cwd=ROOT_DIR

    )



    if result.returncode != 0:


        raise RuntimeError(

            f"Error running {script}"

        )




def main():


    pipeline = [

        "download.py",

        "preprocess.py",

        "split.py",

        "train.py",

        "explain_shap.py",

        "explain_lime.py",

        "explain_ig.py",

        "compare_methods.py"

    ]



    print()

    print(" " * 60)

    print(

        "EXPLAINABLE BERT FULL EXPERIMENT"

    )

    print(" " * 60)



    for script in pipeline:


        run_script(

            script

        )



    print()

    print(" " * 60)

    print(

        "EXPERIMENT COMPLETED SUCCESSFULLY"

    )

    print(" " * 60)





if __name__ == "__main__":

    main()