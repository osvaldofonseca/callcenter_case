import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sys import argv, exit
from collections import defaultdict

def checkForIncompleteRegisters(input_file):

    total_num_registers = 0
    num_registers_missing_variables = 0

    f_in = open(input_file, "r")
    f_in.readline()
    for line in f_in:
        tokens = line.strip().split(',')
        if '' in tokens:
            num_registers_missing_variables += 1
        total_num_registers += 1
    f_in.close()

    print("Total number of registers: ", total_num_registers)
    print("Number of registers missing variables: ",\
            num_registers_missing_variables)

def checkMissingVariablesByFeature():
    
    missing_variables_by_feature = defaultdict(lambda: 0)

    input_file = "data/callcenter_case.csv"
    f_in = open(input_file, "r")
    headers = f_in.readline().strip().split(",")
    for line in f_in:
        tokens = line.strip().split(',')
        
        # calculate the number of missing variables by feature
        for index in range(len(headers)):
            if tokens[index] == '':
                missing_variables_by_feature[headers[index]] += 1        
    f_in.close()

    for feature in missing_variables_by_feature:
        print(feature, missing_variables_by_feature[feature])

def featureDistribution(feat_position):
    
    categories_count = defaultdict(lambda: 0)

    input_file = "data/callcenter_case.csv"
    f_in = open(input_file, "r")
    headers = f_in.readline().strip().split(",")
    for line in f_in:
        tokens = line.strip().split(',')
        if tokens[feat_position] == '':
            continue
        categories_count[tokens[feat_position]] += 1
    f_in.close()

    print("Study about feature: ", headers[feat_position])
    for category in categories_count:
        print(category, categories_count[category])

    # create bar graph
    labels = list(categories_count.keys())
    ys = []
    for label in labels:
        occur_count = categories_count[label]
        ys.append(occur_count)
    
    x = np.arange(len(labels))
    width = 0.15

    fig, ax = plt.subplots()
    ax.bar(labels, ys, width)
    ax.set_title("Study about feature: " + headers[feat_position])
    ax.set_ylabel("Number of Occurrences")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    len(fig)

def removeSpecificFeature(feat_position, input_file, output_file):

    f_in = open(input_file, "r")
    f_out = open(output_file, "w")
    header = f_in.readline().strip().split(",")
    new_header = header[:feat_position] + header[feat_position+1:]
    str_line = ",".join(new_header) + "\n"
    f_out.write(str_line)
    for line in f_in:
        tokens = line.strip().split(',')
        new_list = tokens[:feat_position] + tokens[feat_position+1:]
        str_line = ",".join(new_list) + "\n"
        f_out.write(str_line)    
    f_in.close()
    f_out.close()

    print("Feature \"" + header[feat_position] + "\" removed from data.")

def removeRegistersWithMissingVariables(input_file, output_file):

    f_in = open(input_file, "r")
    f_out = open(output_file, "w")
    header = f_in.readline().strip()
    str_line = header + "\n"
    f_out.write(str_line)
    for line in f_in:
        tokens = line.strip().split(',')
        if '' in tokens:
            continue
        f_out.write(line)    
    f_in.close()
    f_out.close()






if __name__ == "__main__":
    removeSpecificFeature(4, "data/callcenter_case.csv",\
            "data/callcenter_case_no-inadimplente.csv")