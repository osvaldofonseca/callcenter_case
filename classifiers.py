import numpy as np
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import recall_score, precision_score,\
        accuracy_score, f1_score

def readDatabase(input_file):

    records_list = []
    f_in = open(input_file, "r")
    header = f_in.readline().strip().split(",")
    for line in f_in:
        tokens = line.strip().split(",")
        records_list.append(tokens)
    f_in.close()

    return (header, records_list)

def getCompleteData(header, records_list):

    case_data = []
    case_target = []
    for record in records_list:
        data = record[:1]
        target = record[-1]
        case_data.append(data)
        case_target.append(target)

    return (case_data, case_target)

def getDataFromBestKFeaturesSelection(header, records_list, kv=20):

    case_data = []
    case_target = []
    for record in records_list:
        data = record[:-1]
        target = record[-1]
        case_data.append(data)
        case_target.append(target)

    new_case_data = SelectKBest(k=kv).fit_transform(case_data,\
            case_target)

    return (new_case_data, case_target)

def informScoresInfo(scores):

    acc = scores['test_accuracy'].mean()
    precision = scores['test_precision_macro'].mean()
    recall = scores['test_recall_macro'].mean()
    f1_macro = scores['test_f1_macro'].mean()
    print("accuracy,precision,recall,f1_macro")
    print("%.2f\t%.2f\t%.2f\t%.2f"%(acc,precision,recall,f1_macro))

def runRandomForest(case_data, case_target, max_depth_value):

    scoring_metrics = ["accuracy", "recall_macro", "f1_macro", "precision_macro"]
    clf = RandomForestClassifier(n_estimators=1000, max_depth=max_depth_value)
    scores = cross_validate(clf, case_data, case_target, cv=5,\
            scoring=scoring_metrics)
    
    informScoresInfo(scores)


def runSVM(case_data, case_target, kernel_value):

    scoring_metrics = ["accuracy", "recall_macro", "f1_macro", "precision_macro"]
    clf = SVC(kernel=kernel_value)
    scores = cross_validate(clf, case_data, case_target, cv=10,\
            scoring=scoring_metrics)

    informScoresInfo(scores)

def runDecisionTree(header, records_list):

    case_data = []
    case_target = []
    for record in records_list:
        data = record[:1]
        target = record[-1]
        case_data.append(data)
        case_target.append(target)

    clf = DecisionTreeClassifier()
    print(cross_val_score(clf, case_data, case_target, cv=2))

# def runRandomForest(header, records_list):

#     case_data = []
#     case_target = []
#     for record in records_list:
#         data = record[:-1]
#         target = record[-1]
#         case_data.append(data)
#         case_target.append(target)

#     scoring_metrics = ["accuracy", "recall_macro", "f1_macro", "precision_macro"]
#     clf = RandomForestClassifier(n_estimators=100, max_depth=2)
#     scores = cross_validate(clf, case_data, case_target, cv=5,\
#             scoring=scoring_metrics)
    
#     informScoresInfo(scores)

def selectBestFeatures(header, records_list):

    case_data = []
    case_target = []
    for record in records_list:
        data = record[:-1]
        target = record[-1]
        case_data.append(data)
        case_target.append(target)

    new_case_data = SelectKBest(k=20).fit_transform(case_data, case_target)
    print(len(new_case_data[0]))

    clf = DecisionTreeClassifier()
    print(np.mean(cross_val_score(clf, new_case_data, case_target, cv=100)))

if __name__ == "__main__":
    input_file = "data/callcenter_case_fixed.csv"
    header, records_list = readDatabase(input_file)
    #runDecisionTree(header, records_list)
    #runSVM(header, records_list)   
    #selectBestFeatures(header, records_list)
    #runRandomForest(header, records_list)

