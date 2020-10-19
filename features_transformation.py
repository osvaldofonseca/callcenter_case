from collections import defaultdict
from feature_encoding import findAndReplace, oneHotEncoding,\
        fixHeaderBasedOnOneHotEncoding

def readDatabase(input_file):

    records_list = []
    f_in = open(input_file, "r")
    header = f_in.readline().strip().split(",")
    for line in f_in:
        tokens = line.strip().split(",")
        records_list.append(tokens)
    f_in.close()

    return (header, records_list)

def getFeaturePosition(header, feature):

    curr_position = 0
    feat_position = None
    for feat_name in header:
        if feat_name == feature:
            feat_position = curr_position
            return feat_position
        curr_position += 1

def treatSpecificBinaryFeature(feature, header, records_list,\
        replace_mapping):

    feat_position = getFeaturePosition(header, feature)
    for record in records_list:
        feat_value = record[feat_position]
        record[feat_position] = findAndReplace(replace_mapping, feat_value)
        

def treatCategoricalBinaryFeatures(header, records_list):

    # treat the feature "emprestimo_moradia"
    feat_name = "emprestimo_moradia"
    replace_mapping = dict()
    replace_mapping["nao"] = 0
    replace_mapping["sim"] = 1
    treatSpecificBinaryFeature(feat_name, header, records_list,\
            replace_mapping)

    # treat the feature "emprestimo_pessoal"
    feat_name = "emprestimo_pessoal"
    replace_mapping = dict()
    replace_mapping["nao"] = 0
    replace_mapping["sim"] = 1
    treatSpecificBinaryFeature(feat_name, header, records_list,\
            replace_mapping)

    # treat the feature "meio_contato"
    feat_name = "meio_contato"
    replace_mapping = dict()
    replace_mapping["telefone"] = 0
    replace_mapping["celular"] = 1
    treatSpecificBinaryFeature(feat_name, header, records_list,\
            replace_mapping)
    
    # treat the feature "aderencia_campanha"
    # feat_name = "aderencia_campanha"
    # replace_mapping = dict()
    # replace_mapping["nao"] = 0
    # replace_mapping["sim"] = 1
    # treatSpecificBinaryFeature(feat_name, header, records_list,\
    #         replace_mapping)

def getCategoriesByFeature():

    categories_by_feat = defaultdict(lambda: set())
    input_file = "data/callcenter_case_no-inadimplente_no-missing.csv"
    
    f_in = open(input_file, "r")
    header = f_in.readline().strip().split(",")
    for line in f_in:
        tokens = line.strip().split(",")
        for index in range(len(header)):
            feat_value = tokens[index]
            feat_name = header[index]
            categories_by_feat[feat_name].add(feat_value)
    f_in.close()

    return categories_by_feat

def treatSpecificCategoricalFeature(feature, header, records_list,\
            categories_by_feat):
    
    feat_position = getFeaturePosition(header, feature)
    feat_categories = sorted(list(categories_by_feat[feature]))
    # fix header based on the addition of more features
    # due to one hot encoding
    new_header = fixHeaderBasedOnOneHotEncoding(header, feat_position,\
            feat_categories)
    
    new_records_list = []
    for record in records_list:
        category = record[feat_position]
        category_encoding = oneHotEncoding(feat_categories, category)
        new_record = record[:feat_position] + category_encoding +\
                record[feat_position+1:]
        new_records_list.append(new_record)
        # print("---")
        # print(record)
        # print(new_record)

    return (new_header, new_records_list)

def treatCategoricalFeatures(header, records_list):
    
    categories_by_feat = getCategoriesByFeature()

    # treat the feature "estado_civil"
    feat_name = "estado_civil"
    header, records_list = treatSpecificCategoricalFeature(feat_name,\
            header, records_list, categories_by_feat)
    # print(header)
    # print(len(header))
    
    # treat the feature "profissao"
    feat_name = "profissao"
    header, records_list = treatSpecificCategoricalFeature(feat_name,\
            header, records_list, categories_by_feat)
    # print(header)
    # print(len(header))

    # treat the feature "educacao"
    feat_name = "educacao"
    header, records_list = treatSpecificCategoricalFeature(feat_name,\
            header, records_list, categories_by_feat)
    # print(header)
    # print(len(header))

    # treat the feature "campanha_anterior"
    feat_name = "campanha_anterior"
    header, records_list = treatSpecificCategoricalFeature(feat_name,\
            header, records_list, categories_by_feat)
    # print(header)
    # print(len(header))

    # treat the feature "mes"
    feat_name = "mes"
    header, records_list = treatSpecificCategoricalFeature(feat_name,\
            header, records_list, categories_by_feat)
    # print(header)
    # print(len(header))
    
    # treat the feature "dia_da_semana"
    feat_name = "dia_da_semana"
    header, records_list = treatSpecificCategoricalFeature(feat_name,\
            header, records_list, categories_by_feat)
    # print(header)
    # print(len(header))

    return (header, records_list)

def normalizeNumericFeatures():

    categories_by_feat = getCategoriesByFeature()

    # treat the feature "idade"
    feat_name = "idade"
    feat_values = list(categories_by_feat[feat_name])
    min_value = min(feat_values)
    max_value = max(feat_values)
    print(min_value, max_value) 



def dumpDatasetBasedOnTransformedRecords(output_file, header,\
        records_list):
    
    f_out = open(output_file, "w")
    str_line = ",".join(header) + "\n"
    f_out.write(str_line)
    for record in records_list:
        str_record = [str(x) for x in record]
        str_line = ",".join(str_record) + "\n"
        f_out.write(str_line)
    f_out.close()


if __name__ == "__main__":

    header_, records_list_ = readDatabase("data/callcenter_case_no-inadimplente_no-missing.csv")
    # treatCategoricalBinaryFeatures(header_, records_list_)
    # feat_name = "estado_civil"
    
    # header, records_list = treatCategoricalFeatures(header_,\
    #     records_list_)
    normalizeNumericFeatures()
    
    
    
