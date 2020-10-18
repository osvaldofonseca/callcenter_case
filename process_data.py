import pandas
from sys import argv, exit
from collections import defaultdict
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

def labelEncoding(categories):
    id_count = 0
    category_mapping = dict()
    for category in categories:
        category_mapping[category] = id_count
        id_count += 1

    return category_mapping


def readDatabase():
    input_file = "data/callcenter_case.csv"
    case_data = pandas.read_csv(input_file)

    for key in case_data:
        print(">>>", key)
        print(case_data[key].value_counts())

def test():

    # get information from categorical features
    categories_by_feature = defaultdict(lambda: set())
    input_file = "data/callcenter_case.csv"
    f_in = open(input_file, "r")
    headers = f_in.readline().strip().split(",")
    for line in f_in:
        tokens = line.strip().split(",")
        categories_by_feature["profissao"].add(tokens[1])
        categories_by_feature["estado_civil"].add(tokens[2])
        categories_by_feature["educacao"].add(tokens[3])
        categories_by_feature["inadimplente"].add(tokens[4])
        categories_by_feature["emprestimo_moradia"].add(tokens[5])
        categories_by_feature["emprestimo_pessoal"].add(tokens[6])
        categories_by_feature["meio_contato"].add(tokens[7])
        categories_by_feature["mes"].add(tokens[8])
        categories_by_feature["dia_da_semana"].add(tokens[9])




        #case_data.append(tokens[:2])
        #case_target.append(tokens[18])
    f_in.close()

    label_encod_by_feature = dict()
    label_encod_by_feature["profissao"] = labelEncoding(categories_by_feature["profissao"])
    label_encod_by_feature["estado_civil"] = labelEncoding(categories_by_feature["estado_civil"])
    label_encod_by_feature["educacao"] = labelEncoding(categories_by_feature["educacao"])
    label_encod_by_feature["inadimplente"] = labelEncoding(categories_by_feature["inadimplente"])
    label_encod_by_feature["emprestimo_moradia"] = labelEncoding(categories_by_feature["emprestimo_moradia"])
    label_encod_by_feature["emprestimo_pessoal"] = labelEncoding(categories_by_feature["emprestimo_pessoal"])
    label_encod_by_feature["meio_contato"] = labelEncoding(categories_by_feature["meio_contato"])
    label_encod_by_feature["mes"] = labelEncoding(categories_by_feature["mes"])
    label_encod_by_feature["dia_da_semana"] = labelEncoding(categories_by_feature["dia_da_semana"])

    print(categories_by_feature["dia_da_semana"])

    input_file = "data/callcenter_case.csv"
    count = 0
    case_data = []
    case_target = []
    f_in = open(input_file, "r")
    headers = f_in.readline().strip().split(",")
    for line in f_in:
        tokens = line.split(",")
        client_features = []
        # idade
        client_features.append(tokens[0])
        # profissao
        client_features.append(label_encod_by_feature["profissao"][tokens[1]])
        # estado civil
        client_features.append(label_encod_by_feature["estado_civil"][tokens[2]])
        # educacao
        client_features.append(label_encod_by_feature["educacao"][tokens[3]])
        # inadimplente
        client_features.append(label_encod_by_feature["inadimplente"][tokens[4]])
        # emprestimo moradia
        client_features.append(label_encod_by_feature["emprestimo_moradia"][tokens[5]])
        # emprestimo pessoal
        client_features.append(label_encod_by_feature["emprestimo_pessoal"][tokens[6]])
        # meio contato
        client_features.append(label_encod_by_feature["meio_contato"][tokens[7]])
        # mes
        client_features.append(label_encod_by_feature["mes"][tokens[8]])
        # dia da semana
        client_features.append(label_encod_by_feature["dia_da_semana"][tokens[9]])


        #print(client_features)
        case_data.append(client_features)
        case_target.append(tokens[18])
        #print(len(tokens[:18]),tokens[18])
        # if count > 1000:
        #     break
        count += 1


    clf = DecisionTreeClassifier(random_state=0)
    clf.fit(case_data, case_target)
    iris = load_iris()
    print(cross_val_score(clf, case_data, case_target, cv=10))

if __name__ == "__main__":
    readDatabase()
    test()
