from pathlib import Path

import json


import pandas as pd


import matplotlib.pyplot as plt




ROOT_DIR = Path(__file__).resolve().parent



EXPLANATION_PATH = (

    ROOT_DIR /

    "results" /

    "explanations"

)



OUTPUT_PATH = (

    ROOT_DIR /

    "results" /

    "comparison"

)



OUTPUT_PATH.mkdir(

    parents=True,

    exist_ok=True

)




TOP_K = 10




def load_json(path):

    if not path.exists():

        raise FileNotFoundError(

            f"Explanation file not found:\n{path}\n\nRun the corresponding explain_*.py script first."

        )

    with open(

        path,

        "r",

        encoding="utf-8"

    ) as file:


        return json.load(file)




def get_top_tokens(

        data,

        token_key

):


    tokens = data["tokens"]



    sorted_tokens = sorted(

        tokens,

        key=lambda x:

            abs(

                x["importance"]

            ),

        reverse=True

    )



    result = []



    for item in sorted_tokens[:TOP_K]:

        token = str(item[token_key]).strip()

        if token:
            result.append(token)    


    return result




def calculate_similarity(

        first,

        second

):


    set1 = set(first)

    set2 = set(second)



    union = (

        set1 |

        set2

    )


    intersection = (

        set1 &

        set2

    )



    if len(union)==0:

        return 0



    return (

        len(intersection)

        /

        len(union)

    )




def load_explanations():


    shap = load_json(

        EXPLANATION_PATH /

        "shap_explanation.json"

    )


    lime = load_json(

        EXPLANATION_PATH /

        "lime_explanation.json"

    )


    ig = load_json(

        EXPLANATION_PATH /

        "integrated_gradients.json"

    )



    return (

        shap,

        lime,

        ig

    )




def compare():


    shap_data, lime_data, ig_data = load_explanations()



    shap_tokens = get_top_tokens(

        shap_data,

        "token"

    )



    lime_tokens = get_top_tokens(

        lime_data,

        "word"

    )



    ig_tokens = get_top_tokens(

        ig_data,

        "token"

    )



    results = {


        "SHAP_vs_LIME":

            calculate_similarity(

                shap_tokens,

                lime_tokens

            ),



        "SHAP_vs_IG":

            calculate_similarity(

                shap_tokens,

                ig_tokens

            ),



        "LIME_vs_IG":

            calculate_similarity(

                lime_tokens,

                ig_tokens

            )

    }

    best_method = max(

        results,
        key=results.get

    )

    results["BEST_MATCH"] = best_method



    dataframe = pd.DataFrame(

        [

            results

        ]

    )



    dataframe.to_csv(

        OUTPUT_PATH /

        "method_similarity.csv",

        index=False

    )



    return (

        shap_tokens,

        lime_tokens,

        ig_tokens,

        results

    )




def create_token_table(

        shap_tokens,

        lime_tokens,

        ig_tokens

):


    max_len = max(

        len(shap_tokens),

        len(lime_tokens),

        len(ig_tokens)

    )



    table = []



    for i in range(max_len):


        table.append({

            "position":

                i+1,


            "SHAP":

                shap_tokens[i]

                if i < len(shap_tokens)

                else "",



            "LIME":

                lime_tokens[i]

                if i < len(lime_tokens)

                else "",



            "Integrated_Gradients":

                ig_tokens[i]

                if i < len(ig_tokens)

                else ""

        })



    df = pd.DataFrame(

        table

    )



    df.to_csv(

        OUTPUT_PATH /

        "important_tokens.csv",

        index=False

    )




def plot_similarity(results):


    names = [

        key for key in results
        if key != "BEST_MATCH"

    ]

    values = [

        results[key]
        for key in names

    ]



    plt.figure(

        figsize=(8,5)

    )



    colors = [

        "steelblue",
        "darkorange",
        "forestgreen"

    ]

    plt.bar(

        names,
        values,
        color=colors

    )



    plt.ylim(

        0,

        1

    )


    plt.ylabel(

        "Jaccard similarity"

    )


    plt.title(

        "Agreement between XAI methods"

    )



    plt.xticks(

        rotation=20

    )


    plt.tight_layout()



    plt.savefig(

        OUTPUT_PATH /

        "method_similarity.png",

        dpi=300

    )


    plt.close()




def main():


    print(

        "Comparing explanation methods..."

    )



    (

        shap_tokens,

        lime_tokens,

        ig_tokens,

        results

    ) = compare()



    create_token_table(

        shap_tokens,

        lime_tokens,

        ig_tokens

    )



    plot_similarity(

        results

    )



    print()

    print(

        "Results:"

    )

    print(" " * 40)



    for key, value in results.items():

        if isinstance(value, (int, float)):
            print(key, ":", round(value, 3))
        else:
            print(key, ":", value)
        
    print()
    print("Top tokens")
    print("SHAP :", shap_tokens)
    print("LIME :", lime_tokens)
    print("IG   :", ig_tokens)



    print()

    print(" " * 40)

    print(

        "Comparison completed."

    )





if __name__ == "__main__":

    main()