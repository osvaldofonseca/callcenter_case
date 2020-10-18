from collections import defaultdict

def labelEncoding(categories):
    id_count = 0
    category_mapping = dict()
    for category in categories:
        category_mapping[category] = id_count
        id_count += 1

    return category_mapping

def oneHotEncoding(categories_list, register_category):

    category_encoding = []
    for category in categories_list:
        if category == register_category:
            category_encoding.append(1)
        else:
            category_encoding.append(0)

    return category_encoding

def findAndReplace(replace_mapping, register_category):

    register_mapping = replace_mapping[register_category]

    return register_mapping

def fixHeaderBasedOnOneHotEncoding(header_list, feat_position,\
        categories_list):

    sorted_categories_list = sorted(categories_list)
    new_header_list = header_list[:feat_position] +\
            sorted_categories_list + header_list[feat_position+1:]

    return new_header_list


