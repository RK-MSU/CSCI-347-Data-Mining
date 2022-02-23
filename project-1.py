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
print(covar_matrix)


# END
