# import libs
import pandas as pd
import json

# set variables
data = pd.read_csv("desafio1.csv")
mean = data[["pontuacao_credito", "estado_residencia"]].groupby("estado_residencia").mean().astype("float")
median = data[["pontuacao_credito", "estado_residencia"]].groupby("estado_residencia").median().astype("float")
std = data[["pontuacao_credito", "estado_residencia"]].groupby("estado_residencia").std().astype("float")
mode = data[["pontuacao_credito", "estado_residencia"]].groupby("estado_residencia").apply(lambda x: x.mode())


# set dict
states = data.groupby(["estado_residencia"]).nunique()
answers = {f"{x}": {} for x in states.iloc[:, 0].keys()}


# defining function to add values ​​in dict
def output(state):
    if answers:
        answers[state].update({"media": mean.loc[:, x][0] for x in mean})
        answers[state].update({"mediana": median.loc[:, x][0] for x in median})
        answers[state].update({"moda": float(mode[mode["estado_residencia"] == state]["pontuacao_credito"][0])})
        answers[state].update({"desvio_padrao": std.loc[:, x][0] for x in std})


# using the function
for i in states.iloc[:, 0].keys():
    output(i)

# saving as json formatsaving as json format
with open("submission.json", "w") as json_file:
    json.dump(answers, json_file, indent=4)
