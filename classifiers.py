import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, cross_validate, cross_val_predict
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import recall_score, precision_score,\
        accuracy_score, f1_score, confusion_matrix, precision_recall_fscore_support
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB


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
        data = record[:-1]
        data = [float(x) for x in data]
        target = record[-1]
        case_data.append(data)
        case_target.append(target)

    return (case_data, case_target)

def getDataFromBestKFeaturesSelection(header, records_list, kv=20, flag_chi2=False):

    case_data = []
    case_target = []
    for record in records_list:
        data = record[:-1]
        data = [float(x) for x in data]
        target = record[-1]
        case_data.append(data)
        case_target.append(target)

    if flag_chi2:
        new_case_data = SelectKBest(chi2, k=kv).fit_transform(case_data,\
                case_target)
    else: 
        new_case_data = SelectKBest(k=kv).fit_transform(case_data,\
               case_target)

    return (new_case_data, case_target)

def getDataPCATransformed(header, records_list, number_components):

    case_data = []
    case_target = []
    for record in records_list:
        data = record[:-1]
        data = [float(x) for x in data]
        target = record[-1]
        case_data.append(data)
        case_target.append(target)

    pca = PCA(n_components=number_components)
    pca.fit_transform(case_data, case_target)

    return (case_data, case_target)

def informScoresInfo(scores):

    acc = scores['test_accuracy'].mean()
    precision = scores['test_precision_macro'].mean()
    recall = scores['test_recall_macro'].mean()
    f1_macro = scores['test_f1_macro'].mean()
    print("accuracy,precision,recall,f1_macro")
    print("%.2f\t%.2f\t%.2f\t%.2f"%(acc,precision,recall,f1_macro))

def runRandomForest(case_data, case_target, max_depth_value, conf_mat_flag=False):

    scoring_metrics = ["accuracy", "recall_macro", "f1_macro", "precision_macro"]
    clf = RandomForestClassifier(n_estimators=1000, max_depth=max_depth_value)
    scores = cross_validate(clf, case_data, case_target, cv=5,\
            scoring=scoring_metrics)
    
    informScoresInfo(scores)

    if conf_mat_flag:
        y_pred = cross_val_predict(clf, case_data, case_target, cv=10)
        conf_mat = confusion_matrix(case_target, y_pred)
        print(conf_mat)


def runSVM(case_data, case_target, kernel_value, c_value, conf_mat_flag=False):

    scoring_metrics = ["accuracy", "recall_macro", "f1_macro", "precision_macro"]
    clf = SVC(kernel=kernel_value, C=c_value, cache_size=1000)
    scores = cross_validate(clf, case_data, case_target, cv=10,\
            scoring=scoring_metrics)

    informScoresInfo(scores)

    if conf_mat_flag:
        y_pred = cross_val_predict(clf, case_data, case_target, cv=10)
        conf_mat = confusion_matrix(case_target, y_pred)
        print(conf_mat)

def runNaiveBayes(case_data, case_target, conf_mat_flag=False):

    scoring_metrics = ["accuracy", "recall_macro", "f1_macro", "precision_macro"]
    clf = GaussianNB()
    scores = cross_validate(clf, case_data, case_target, cv=10,\
            scoring=scoring_metrics)

    informScoresInfo(scores)

    if conf_mat_flag:
        y_pred = cross_val_predict(clf, case_data, case_target, cv=10)
        conf_mat = confusion_matrix(case_target, y_pred)
        print(conf_mat)


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

