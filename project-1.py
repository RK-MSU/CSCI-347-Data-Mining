# project-01

import numpy as np
import pandas as pd
from src import multiDimensionalMean, rangeNormalize, zScoreNormalize, covariance, covarianceMatrix, labelEncodeMatrix

DATA_FILENAME = 'https://archive.ics.uci.edu/ml/machine-learning-databases/cpu-performance/machine.data'
DATA_COL_NAMES = ['Vendor Name', 'Model Name', 'MYCT', 'MMIN', 'MMAX', 'CACH', 'CHMIN', 'CHMAX', 'PRP', 'ERP']

df = pd.read_csv(DATA_FILENAME,
        names=DATA_COL_NAMES,
        delimiter=',',
        encoding="utf-8",
        skipinitialspace=True
    )

df_sample = df.sample(n=100, replace=False, random_state=1)



def convertDataFrameToTwoDimensionalList(df):
    data = list()
    for index, row in df.iterrows():
        data_row = []
        for col_name in DATA_COL_NAMES:
            data_row.append(row[col_name])
        data.append(data_row)
    return data

def labelEncodeDataFrameToNumpyArray(df):
    data = convertDataFrameToTwoDimensionalList(df)
    data = labelEncodeMatrix(data)
    np_array = np.ndarray((len(data), len(DATA_COL_NAMES)), dtype=int)
    for i in range(len(data)):
        np_array[i] = np.array(data[i], dtype=int)
    return np_array

def convertNumpyNdArrayToDataFrame(m):
    df_data = {}
    for col_name_index in range(len(DATA_COL_NAMES)):
        col_name = DATA_COL_NAMES[col_name_index]
        df_data[col_name] = m[:,col_name_index]
    return pd.DataFrame(df_data)

def labelEncodeDataFrame(df):
    data = labelEncodeDataFrameToNumpyArray(df)
    return convertNumpyNdArrayToDataFrame(data)


label_encoded_matrix = labelEncodeDataFrameToNumpyArray(df)
df_label_encoded = labelEncodeDataFrame(df)
multi_di_mean = multiDimensionalMean(label_encoded_matrix)
covar_matrix = covarianceMatrix(label_encoded_matrix)
range_norm_matrix = rangeNormalize(label_encoded_matrix)
zscore_norm_matrix = zScoreNormalize(label_encoded_matrix)

def calculateGreatestCoVar(m, okay_indices):

    covar_data_dict = {}
    okay_indices = [2,3,4,5,6,7,8,9]

    for i in okay_indices:
        for j in okay_indices:
            if i == j: continue # do not need to check covariance of same column
            if j in covar_data_dict: continue # no need for duplicates
            # ensure dictionary value is available
            if i not in covar_data_dict: covar_data_dict[i] = {}
            # get col i
            col_i = m[:,i]
            # get col v
            col_j = m[:,j]
            # calculate covariance between columns i and j
            var = covariance(col_i, col_j)
            covar_data_dict[i][j] = var
    
    covar_data_list = list()
    
    for i in covar_data_dict:
        for j in covar_data_dict[i]:
            # covar = covar_data_dict[i][j]
            covar_data_list.append({
                "i": i,
                "j": j,
                "covar": covar_data_dict[i][j]
            })

    covar_data_list.sort(key=lambda x: x['covar'], reverse=True)

    highest_covar_pair_data = covar_data_list[0]
    
    return {
        "col_1" : DATA_COL_NAMES[highest_covar_pair_data['i']],
        "col_1" : DATA_COL_NAMES[highest_covar_pair_data['j']],
        "covar_value": highest_covar_pair_data['covar']
    }

covar_data_dict = {}
numerical_attribute_indecies = [2,3,4,5,6,7,8,9]
for i in numerical_attribute_indecies:
    for j in numerical_attribute_indecies:
        if i == j: continue # do not need to check covariance of same column
        if j in covar_data_dict: continue # no need for duplicates
        # ensure dictionary value is available
        if i not in covar_data_dict: covar_data_dict[i] = {}
        # get col i
        col_i = range_norm_matrix[:,i]
        # get col v
        col_j = range_norm_matrix[:,j]
        # calculate covariance between columns i and j
        var = covariance(col_i, col_j)
        covar_data_dict[i][j] = var


# for i in covar_data_list:
#     print(i)

# max_covar = None
# for i in covar_data:
#     for j in covar_data[i]:
#         if max_covar is None:
#             max_covar = {
#                 "i": i,
#                 "j": j,
#                 "covar": covar_data[i][j]
#             }
#         else:
#             if max_covar['covar'] < covar_data[i][j]:
#                 max_covar = {
#                     "i": i,
#                     "j": j,
#                     "covar": covar_data[i][j]
#                 }
#         # print(i, covar_data[i])
# print(max_covar)




# END
